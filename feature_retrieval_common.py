#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from run_top_p_retrieval import build_text_embedding, build_video_embedding, compute_anomaly_scores


WORKSPACE_ROOT = Path("/workspace")
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "VadCLIP" / "data" / "model_ucf.pth"
DEFAULT_FEATURE_ROOT = WORKSPACE_ROOT / "VadCLIP" / "data" / "UCFClipFeatures"
DEFAULT_GT_JSON = WORKSPACE_ROOT / "ablation1" / "UCF-AR" / "ucf_crime_test.json"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "ablation1" / "output2"
DEFAULT_EMBEDDING_CACHE_ROOT = WORKSPACE_ROOT / "ablation1" / "output"
DEFAULT_CLIP_CACHE_DIR = WORKSPACE_ROOT / "ablation1" / "clip_cache"

FEATURE_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+?)__(?P<index>\d+)$")


def canonical_video_key(video_name: str) -> str:
    normalized = video_name.replace("\\", "/").strip()
    if normalized.startswith("Testing_Anomaly_Videos/"):
        normalized = normalized.split("/", 1)[1]
    return normalized

@dataclass(frozen=True)
class IndexedFeaturePath:
    path: Path
    relative_parent: str
    base_stem: str
    full_stem: str
    variant_index: int


class FeaturePathResolver:
    """Resolve a canonical video name like `Abuse/Abuse028_x264.mp4` to one `.npy` file.

    Matching rule:
    1. Remove the video suffix and keep the category folder if present.
    2. Search recursively under `feature_root`.
    3. Prefer the exact variant `<stem>__0.npy` inside the same category folder because it is
       the most direct one-to-one match for a video stem.
    4. If `__0` is missing, fall back to the lowest indexed `<stem>__k.npy` in the same folder.
    5. If the folder-specific lookup fails, repeat the same preference globally.
    6. Fail loudly for missing or ambiguous matches.
    """

    def __init__(self, feature_root: Path) -> None:
        self.feature_root = feature_root
        self._indexed = False
        self._by_base_stem: Dict[str, List[IndexedFeaturePath]] = {}
        self._cache: Dict[str, Path] = {}

    def _index_feature_paths(self) -> None:
        if self._indexed:
            return
        if not self.feature_root.exists():
            raise FileNotFoundError(f"Feature root does not exist: {self.feature_root}")

        indexed_paths: Dict[str, List[IndexedFeaturePath]] = {}
        for path in sorted(self.feature_root.rglob("*.npy")):
            relative_parent = path.parent.relative_to(self.feature_root).as_posix()
            match = FEATURE_SUFFIX_PATTERN.match(path.stem)
            if match is None:
                base_stem = path.stem
                variant_index = 0
            else:
                base_stem = match.group("base")
                variant_index = int(match.group("index"))

            indexed = IndexedFeaturePath(
                path=path,
                relative_parent=relative_parent,
                base_stem=base_stem,
                full_stem=path.stem,
                variant_index=variant_index,
            )
            indexed_paths.setdefault(base_stem, []).append(indexed)

        for base_stem, items in indexed_paths.items():
            self._by_base_stem[base_stem] = sorted(
                items,
                key=lambda item: (item.relative_parent, item.variant_index, item.path.as_posix()),
            )
        self._indexed = True

    def canonical_video_to_feature_path(self, video_name: str) -> Path:
        canonical_name = canonical_video_key(video_name)
        cached = self._cache.get(canonical_name)
        if cached is not None:
            return cached

        self._index_feature_paths()
        video_path = Path(canonical_name)
        category = video_path.parent.as_posix()
        stem = video_path.stem
        matches = self._by_base_stem.get(stem, [])
        if not matches:
            raise FileNotFoundError(
                f"Missing CLIP feature file for video '{video_name}' under '{self.feature_root}'. "
                f"Expected a recursive match for stem '{stem}'."
            )

        same_category = [item for item in matches if item.relative_parent == category]
        selected = self._select_preferred_match(video_name=video_name, stem=stem, matches=same_category)
        if selected is None:
            selected = self._select_preferred_match(video_name=video_name, stem=stem, matches=matches)
        if selected is None:
            candidate_list = ", ".join(item.path.as_posix() for item in matches[:5])
            raise RuntimeError(
                f"Ambiguous CLIP feature mapping for video '{video_name}'. "
                f"Found {len(matches)} candidates, for example: {candidate_list}"
            )

        self._cache[canonical_name] = selected.path
        return selected.path

    @staticmethod
    def _select_preferred_match(
        video_name: str,
        stem: str,
        matches: Sequence[IndexedFeaturePath],
    ) -> IndexedFeaturePath | None:
        if not matches:
            return None

        exact_direct = [item for item in matches if item.full_stem == f"{stem}__0"]
        if len(exact_direct) == 1:
            return exact_direct[0]
        if len(exact_direct) > 1:
            sample_paths = ", ".join(item.path.as_posix() for item in exact_direct[:5])
            raise RuntimeError(
                f"Multiple exact '__0' feature files found for video '{video_name}': {sample_paths}"
            )

        if len(matches) == 1:
            return matches[0]

        lowest_index = matches[0].variant_index
        lowest_matches = [item for item in matches if item.variant_index == lowest_index]
        if len(lowest_matches) == 1:
            return lowest_matches[0]
        return None


def canonical_video_to_feature_path(video_name: str, feature_root: Path) -> Path:
    return FeaturePathResolver(feature_root).canonical_video_to_feature_path(video_name)


def load_temporal_clip_features(video_name: str, resolver: FeaturePathResolver, device: str) -> tuple[torch.Tensor, Path]:
    feature_path = resolver.canonical_video_to_feature_path(video_name)
    array = np.load(feature_path)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D temporal CLIP feature array for '{video_name}', got shape {array.shape} from {feature_path}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"Feature file '{feature_path}' is empty for video '{video_name}'.")

    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    return tensor, feature_path


def load_gt_items(gt_json: Path) -> List[dict]:
    with gt_json.open() as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON list in {gt_json}, got {type(items).__name__}.")
    return items


def summarize_ranks(ranks: Sequence[dict]) -> dict:
    if not ranks:
        raise ValueError("Cannot summarize an empty rank list.")

    rank_values = [item["rank"] for item in ranks]
    return {
        "num_queries": len(rank_values),
        "top1": sum(item["top1_hit"] for item in ranks) / len(rank_values),
        "top5": sum(item["top5_hit"] for item in ranks) / len(rank_values),
        "top10": sum(item["top10_hit"] for item in ranks) / len(rank_values),
        "mean_rank": sum(rank_values) / len(rank_values),
        "median_rank": statistics.median(rank_values),
        "mrr": sum(1.0 / item["rank"] for item in ranks) / len(rank_values),
    }


def build_gallery_video_keys(gt_items: Sequence[dict]) -> List[str]:
    return sorted({canonical_video_key(item["Video Name"]) for item in gt_items})


def build_text_matrix(model: Any, gt_items: Sequence[dict], device: str) -> torch.Tensor:
    text_embeddings = [
        build_text_embedding(model.clipmodel, item["English Text"], device) for item in gt_items
    ]
    return F.normalize(torch.stack(text_embeddings).to(device), dim=-1)


def evaluate_bidirectional_retrieval(
    gt_items: Sequence[dict],
    gallery_video_keys: Sequence[str],
    gallery_embeddings: Dict[str, torch.Tensor],
    text_matrix: torch.Tensor,
    device: str,
) -> tuple[dict, dict]:
    video_matrix = torch.stack([gallery_embeddings[key] for key in gallery_video_keys]).to(device)
    video_matrix = F.normalize(video_matrix, dim=-1)
    gallery_index_by_key = {key: index for index, key in enumerate(gallery_video_keys)}

    text_to_video_ranks: List[dict] = []
    for text_index, item in enumerate(gt_items):
        video_key = canonical_video_key(item["Video Name"])
        gt_gallery_index = gallery_index_by_key[video_key]
        similarities = video_matrix @ text_matrix[text_index]
        ranked_indices = torch.argsort(similarities, descending=True)
        rank = int((ranked_indices == gt_gallery_index).nonzero(as_tuple=False)[0].item()) + 1
        text_to_video_ranks.append(
            {
                "video_name": video_key,
                "query": item["English Text"],
                "gt_gallery_index": gt_gallery_index,
                "rank": rank,
                "top1_hit": rank <= 1,
                "top5_hit": rank <= 5,
                "top10_hit": rank <= 10,
            }
        )

    text_indices_by_video: Dict[str, List[int]] = {}
    for text_index, item in enumerate(gt_items):
        text_indices_by_video.setdefault(canonical_video_key(item["Video Name"]), []).append(text_index)

    similarity_matrix = video_matrix @ text_matrix.T
    video_to_text_ranks: List[dict] = []
    for video_key in gallery_video_keys:
        gt_text_indices = text_indices_by_video[video_key]
        ranked_indices = torch.argsort(similarity_matrix[gallery_index_by_key[video_key]], descending=True)
        gt_index_tensor = torch.tensor(gt_text_indices, device=ranked_indices.device)
        candidate_ranks = (ranked_indices.unsqueeze(1) == gt_index_tensor.unsqueeze(0)).nonzero(as_tuple=False)
        if candidate_ranks.numel() == 0:
            raise RuntimeError(f"Failed to find text rank for gallery video '{video_key}'.")
        rank = int(candidate_ranks[:, 0].min().item()) + 1
        matched_text_index = gt_text_indices[0]
        video_to_text_ranks.append(
            {
                "video_name": video_key,
                "query": gt_items[matched_text_index]["English Text"],
                "gt_text_indices": gt_text_indices,
                "rank": rank,
                "top1_hit": rank <= 1,
                "top5_hit": rank <= 5,
                "top10_hit": rank <= 10,
            }
        )

    metrics = {
        "text_to_video": summarize_ranks(text_to_video_ranks),
        "video_to_text": summarize_ranks(video_to_text_ranks),
    }
    ranks = {
        "gallery_video_keys": list(gallery_video_keys),
        "gallery_index_by_key": gallery_index_by_key,
        "text_to_video": text_to_video_ranks,
        "video_to_text": video_to_text_ranks,
    }
    return metrics, ranks


def validate_gallery_coverage(gt_items: Sequence[dict], gallery_embeddings: Dict[str, torch.Tensor]) -> None:
    missing = [
        canonical_video_key(item["Video Name"])
        for item in gt_items
        if canonical_video_key(item["Video Name"]) not in gallery_embeddings
    ]
    if missing:
        preview = ", ".join(sorted(set(missing))[:5])
        raise KeyError(f"GT videos missing from gallery embeddings: {preview}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_embedding_cache(cache_path: Path, expected_meta: dict) -> Dict[str, torch.Tensor] | None:
    if not cache_path.exists():
        return None
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache.get("meta") != expected_meta:
        return None
    embeddings = cache.get("embeddings")
    if not isinstance(embeddings, dict):
        return None
    return {key: value.float() for key, value in embeddings.items()}


def _save_embedding_cache(cache_path: Path, meta: dict, embeddings: Dict[str, torch.Tensor]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "embeddings": embeddings}, cache_path)


def build_feature_embedding_index(
    *,
    gallery_video_keys: Sequence[str],
    resolver: FeaturePathResolver,
    device: str,
    embedding_cache: Path,
    cache_meta: dict,
    embedding_builder: Any,
) -> Dict[str, torch.Tensor]:
    cached = _load_embedding_cache(embedding_cache, cache_meta)
    if cached is not None:
        return cached

    embeddings: Dict[str, torch.Tensor] = {}
    for video_key in tqdm(gallery_video_keys, desc="Encoding gallery videos"):
        frame_features, _ = load_temporal_clip_features(video_key, resolver, device)
        video_embedding = embedding_builder(frame_features).detach().cpu()
        embeddings[video_key] = F.normalize(video_embedding.float(), dim=0)

    _save_embedding_cache(embedding_cache, cache_meta, embeddings)
    return embeddings


def describe_feature_sources(
    resolver: FeaturePathResolver,
    video_keys: Iterable[str],
) -> Dict[str, str]:
    return {video_key: str(resolver.canonical_video_to_feature_path(video_key)) for video_key in video_keys}
