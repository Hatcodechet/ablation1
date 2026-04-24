#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


WORKSPACE_ROOT = Path("/workspace")
VADCLIP_SRC = WORKSPACE_ROOT / "VadCLIP" / "src"
CLIP_MODULE = None
CLIPVAD_CLASS = None


def ensure_vadclip_imports():
    global CLIP_MODULE, CLIPVAD_CLASS
    if CLIP_MODULE is not None and CLIPVAD_CLASS is not None:
        return CLIP_MODULE, CLIPVAD_CLASS

    if str(VADCLIP_SRC) not in sys.path:
        sys.path.insert(0, str(VADCLIP_SRC))

    from clip import clip as clip_module  # noqa: E402
    from model import CLIPVAD as clipvad_class  # noqa: E402

    CLIP_MODULE = clip_module
    CLIPVAD_CLASS = clipvad_class
    return CLIP_MODULE, CLIPVAD_CLASS


@dataclass
class VideoResult:
    rank: int
    video_path: str
    similarity: float
    selected_frames: int
    total_frames: int
    top_p_mass: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-p anomaly-guided text-video retrieval with VadCLIP on raw videos."
    )
    parser.add_argument("--video-root", type=Path, default=WORKSPACE_ROOT / "test")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE_ROOT / "model_ucf.pth")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--snippet-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=WORKSPACE_ROOT / "ablation_study" / "output" / "retrieval_results.csv")
    parser.add_argument("--clip-cache-dir", type=Path, default=WORKSPACE_ROOT / "ablation_study" / "clip_cache")
    parser.add_argument("--min-selected-frames", type=int, default=1)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"])
    return parser.parse_args()


def build_vadclip(device: str, checkpoint_path: Path, clip_cache_dir: Path):
    clip_module, clipvad_class = ensure_vadclip_imports()
    clip_cache_dir.mkdir(parents=True, exist_ok=True)
    model = clipvad_class(
        num_class=14,
        embed_dim=512,
        visual_length=256,
        visual_width=512,
        visual_head=1,
        visual_layers=2,
        attn_window=8,
        prompt_prefix=10,
        prompt_postfix=10,
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # CLIPVAD already loads ViT-B/16 in its constructor. Reuse that instance to avoid
    # downloading/loading the CLIP backbone twice per run.
    clip_model = model.clipmodel
    preprocess = clip_module._transform(clip_model.visual.input_resolution)
    model.to(device)
    model.eval()
    return model, preprocess


def list_videos(video_root: Path, extensions: Sequence[str]) -> List[Path]:
    normalized_exts = {ext.lower() for ext in extensions}
    videos = [path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in normalized_exts]
    return sorted(videos)


def decode_sampled_frames(video_path: Path, sample_fps: float, max_frames: int) -> List[Image.Image]:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0 when using ffmpeg decoding.")

    with tempfile.TemporaryDirectory(prefix="top_p_frames_") as tmp_dir:
        output_pattern = os.path.join(tmp_dir, "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sample_fps}",
            "-frames:v",
            str(max_frames),
            output_pattern,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {completed.stderr.strip()}")

        frame_paths = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frames sampled from video: {video_path}")

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        return frames


def encode_frames(
    frames: Sequence[Image.Image],
    preprocess,
    clip_model,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    encoded_batches: List[torch.Tensor] = []
    for start in range(0, len(frames), batch_size):
        frame_batch = frames[start : start + batch_size]
        image_tensor = torch.stack([preprocess(frame) for frame in frame_batch]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
        encoded_batches.append(image_features.float())
    return torch.cat(encoded_batches, dim=0)


def split_temporal_features(features: torch.Tensor, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape {tuple(features.shape)}.")
    if max_length <= 0:
        raise ValueError("max_length must be > 0.")

    total_length, feature_dim = features.shape
    chunk_count = (total_length + max_length - 1) // max_length
    padded = torch.zeros(
        chunk_count,
        max_length,
        feature_dim,
        device=features.device,
        dtype=features.dtype,
    )
    lengths = torch.zeros(chunk_count, device=features.device, dtype=torch.long)

    for chunk_index in range(chunk_count):
        start = chunk_index * max_length
        end = min(start + max_length, total_length)
        chunk = features[start:end]
        padded[chunk_index, : chunk.shape[0]] = chunk
        lengths[chunk_index] = chunk.shape[0]

    return padded, lengths


def build_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"Expected 1D lengths tensor, got shape {tuple(lengths.shape)}.")
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def compute_anomaly_scores(model: Any, frame_features: torch.Tensor, snippet_size: int = 16) -> torch.Tensor:
    """Compute anomaly scores following VadCLIP test-time behavior.

    The precomputed UCF CLIP features are already temporal-segment features, so to stay aligned
    with `VadCLIP/src/ucf_test.py` we score them directly and only split them into chunks of
    `model.visual_length`. The `snippet_size` argument is kept for CLI compatibility but is not
    used in this code path.
    """
    with torch.no_grad():
        if frame_features.ndim != 2:
            raise ValueError(f"Expected a 2D temporal feature tensor, got shape {tuple(frame_features.shape)}.")
        if frame_features.shape[0] == 0:
            raise ValueError("Received an empty temporal feature sequence.")

        feature_batch, lengths = split_temporal_features(frame_features, model.visual_length)
        padding_mask = build_padding_mask(lengths, model.visual_length)
        temporal_features = model.encode_video(feature_batch, padding_mask, lengths)
        logits = model.classifier(temporal_features + model.mlp2(temporal_features)).squeeze(-1)

        valid_logits = [logits[chunk_index, :chunk_length] for chunk_index, chunk_length in enumerate(lengths.tolist())]
        return torch.sigmoid(torch.cat(valid_logits, dim=0))


def top_p_select_indices(
    anomaly_scores: torch.Tensor,
    top_p: float,
    min_selected_frames: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if anomaly_scores.ndim != 1:
        raise ValueError("Expected 1D anomaly score tensor.")

    scaled_scores = anomaly_scores / max(temperature, 1e-6)
    probabilities = torch.softmax(scaled_scores, dim=0)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    threshold_index = torch.nonzero(cumulative >= top_p, as_tuple=False)
    if len(threshold_index) == 0:
        selected_count = len(sorted_indices)
    else:
        selected_count = int(threshold_index[0].item()) + 1
    selected_count = max(selected_count, min_selected_frames)
    selected_indices = sorted_indices[:selected_count]
    selected_indices, _ = torch.sort(selected_indices)
    return selected_indices, probabilities


def build_video_embedding(frame_features: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
    selected_features = frame_features[selected_indices]
    selected_features = F.normalize(selected_features, dim=-1)
    video_embedding = selected_features.mean(dim=0)
    return F.normalize(video_embedding, dim=0)


def build_text_embedding(clip_model, query: str, device: str) -> torch.Tensor:
    clip_module, _ = ensure_vadclip_imports()
    with torch.no_grad():
        text_tokens = clip_module.tokenize([query]).to(device)
        word_embeddings = clip_model.encode_token(text_tokens)
        text_embedding = clip_model.encode_text(word_embeddings, text_tokens)
    text_embedding = text_embedding[0].float()
    return F.normalize(text_embedding, dim=0)


def retrieve(
    model: Any,
    preprocess,
    video_paths: Sequence[Path],
    query: str,
    device: str,
    batch_size: int,
    top_p: float,
    min_selected_frames: int,
    score_temperature: float,
    sample_fps: float,
    max_frames: int,
    snippet_size: int,
):
    clip_model = model.clipmodel
    text_embedding = build_text_embedding(clip_model, query, device)
    rows = []

    for video_path in tqdm(video_paths, desc="Processing videos"):
        frames = decode_sampled_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
        frame_features = encode_frames(frames, preprocess, clip_model, device, batch_size)
        anomaly_scores = compute_anomaly_scores(model, frame_features, snippet_size=snippet_size)
        selected_indices, anomaly_distribution = top_p_select_indices(
            anomaly_scores,
            top_p=top_p,
            min_selected_frames=min_selected_frames,
            temperature=score_temperature,
        )
        video_embedding = build_video_embedding(frame_features, selected_indices)
        similarity = torch.dot(video_embedding, text_embedding).item()
        top_p_mass = anomaly_distribution[selected_indices].sum().item()
        rows.append(
            {
                "video_path": str(video_path),
                "similarity": similarity,
                "selected_frames": int(selected_indices.numel()),
                "total_frames": int(frame_features.shape[0]),
                "top_p_mass": top_p_mass,
            }
        )

    rows.sort(key=lambda item: item["similarity"], reverse=True)
    results = [
        VideoResult(
            rank=index + 1,
            video_path=row["video_path"],
            similarity=row["similarity"],
            selected_frames=row["selected_frames"],
            total_frames=row["total_frames"],
            top_p_mass=row["top_p_mass"],
        )
        for index, row in enumerate(rows)
    ]
    return results


def write_results(output_csv: Path, results: Sequence[VideoResult]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "video_path", "similarity", "selected_frames", "total_frames", "top_p_mass"])
        for result in results:
            writer.writerow(
                [
                    result.rank,
                    result.video_path,
                    f"{result.similarity:.6f}",
                    result.selected_frames,
                    result.total_frames,
                    f"{result.top_p_mass:.6f}",
                ]
            )


def print_top_results(results: Sequence[VideoResult], limit: int = 10) -> None:
    print("")
    print(f"Top {min(limit, len(results))} retrieval results")
    print("-" * 100)
    for result in results[:limit]:
        print(
            f"{result.rank:>3} | sim={result.similarity:>8.4f} | "
            f"selected={result.selected_frames:>3}/{result.total_frames:<3} | "
            f"mass={result.top_p_mass:>6.4f} | {result.video_path}"
        )


if __name__ == "__main__":
    args = parse_args()
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in the range (0, 1].")

    videos = list_videos(args.video_root, args.extensions)
    if not videos:
        raise RuntimeError(f"No videos found under {args.video_root}")

    model, preprocess = build_vadclip(args.device, args.checkpoint, args.clip_cache_dir)
    results = retrieve(
        model=model,
        preprocess=preprocess,
        video_paths=videos,
        query=args.query,
        device=args.device,
        batch_size=args.batch_size,
        top_p=args.top_p,
        min_selected_frames=args.min_selected_frames,
        score_temperature=args.score_temperature,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        snippet_size=args.snippet_size,
    )
    write_results(args.output_csv, results)
    print_top_results(results)
    print("")
    print(f"Saved CSV to: {args.output_csv}")
