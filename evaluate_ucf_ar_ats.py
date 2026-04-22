#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from tqdm import tqdm

from run_top_p_retrieval import (
    build_text_embedding,
    build_vadclip,
    build_video_embedding,
    compute_anomaly_scores,
    decode_sampled_frames,
    encode_frames,
    list_videos,
)


WORKSPACE_ROOT = Path("/workspace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UCF-CrimeAR with ATS-guided frame selection.")
    parser.add_argument("--video-root", type=Path, default=WORKSPACE_ROOT / "test")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE_ROOT / "model_ucf.pth")
    parser.add_argument("--gt-json", type=Path, default=WORKSPACE_ROOT / "ablation_study" / "UCF-AR" / "ucf_crime_test.json")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--snippet-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--select-frames", type=int, default=16)
    parser.add_argument("--ats-tau", type=float, default=0.1)
    parser.add_argument("--clip-cache-dir", type=Path, default=WORKSPACE_ROOT / "ablation_study" / "clip_cache")
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--ranks-json", type=Path, default=None)
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"])
    return parser.parse_args()


def ats_tag(select_frames: int, tau: float) -> str:
    tau_int = int(round(tau * 100))
    return f"ats_f{select_frames}_tau{tau_int:02d}"


def resolve_output_paths(args: argparse.Namespace) -> None:
    suffix = ats_tag(args.select_frames, args.ats_tau)
    output_root = WORKSPACE_ROOT / "ablation_study" / "output"
    if args.embedding_cache is None:
        args.embedding_cache = output_root / f"video_embedding_cache_{suffix}.pt"
    if args.metrics_json is None:
        args.metrics_json = output_root / f"ucf_ar_metrics_{suffix}.json"
    if args.ranks_json is None:
        args.ranks_json = output_root / f"ucf_ar_ranks_{suffix}.json"


def canonical_video_key(video_path: Path, video_root: Path) -> str:
    relative = video_path.relative_to(video_root)
    parts = relative.parts
    if parts[0] == "Testing_Anomaly_Videos":
        return "/".join(parts[1:])
    return "/".join(parts)


def ats_density_aware_sample(anomaly_scores: torch.Tensor, select_frames: int, tau: float) -> List[int]:
    scores = anomaly_scores.detach().cpu().numpy().astype(np.float64)
    num_frames = scores.shape[0]
    if num_frames <= select_frames:
        sampled = np.rint(np.linspace(0, max(num_frames - 1, 0), select_frames)).astype(int)
        return [int(idx) for idx in sampled]

    if np.sum(scores) < 1:
        sampled = np.rint(np.linspace(0, num_frames - 1, select_frames)).astype(int)
        return [int(idx) for idx in sampled]

    dense_scores = scores + tau
    score_cumsum = np.concatenate((np.zeros((1,), dtype=np.float64), np.cumsum(dense_scores)), axis=0)
    max_score_cumsum = int(np.round(score_cumsum[-1]))
    if max_score_cumsum <= 0:
        sampled = np.rint(np.linspace(0, num_frames - 1, select_frames)).astype(int)
        return [int(idx) for idx in sampled]

    sampler = interpolate.interp1d(
        score_cumsum,
        np.arange(num_frames + 1),
        kind="linear",
        axis=0,
        fill_value="extrapolate",
    )
    scale_x = np.linspace(1, max_score_cumsum, select_frames)
    sampled = sampler(scale_x)
    sampled = [min(num_frames - 1, max(0, int(idx))) for idx in sampled]
    return sampled


def build_ats_video_embedding(frame_features: torch.Tensor, sampled_indices: List[int]) -> torch.Tensor:
    sampled_index_tensor = torch.tensor(sampled_indices, device=frame_features.device, dtype=torch.long)
    return build_video_embedding(frame_features, sampled_index_tensor)


def build_video_embedding_index(args: argparse.Namespace, model, preprocess) -> Dict[str, torch.Tensor]:
    if args.embedding_cache.exists():
        cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
        cache_meta = cache.get("meta", {})
        expected_meta = {
            "video_root": str(args.video_root.resolve()),
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
            "snippet_size": args.snippet_size,
            "select_frames": args.select_frames,
            "ats_tau": args.ats_tau,
        }
        if cache_meta == expected_meta:
            return {key: value.float() for key, value in cache["embeddings"].items()}

    video_paths = list_videos(args.video_root, args.extensions)
    embeddings = {}
    for video_path in tqdm(video_paths, desc="Encoding videos (ATS)"):
        frames = decode_sampled_frames(video_path, sample_fps=args.sample_fps, max_frames=args.max_frames)
        frame_features = encode_frames(frames, preprocess, model.clipmodel, args.device, args.batch_size)
        anomaly_scores = compute_anomaly_scores(model, frame_features, snippet_size=args.snippet_size)
        sampled_indices = ats_density_aware_sample(anomaly_scores, args.select_frames, args.ats_tau)
        video_embedding = build_ats_video_embedding(frame_features, sampled_indices).detach().cpu()
        embeddings[canonical_video_key(video_path, args.video_root)] = video_embedding

    args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {
                "video_root": str(args.video_root.resolve()),
                "sample_fps": args.sample_fps,
                "max_frames": args.max_frames,
                "snippet_size": args.snippet_size,
                "select_frames": args.select_frames,
                "ats_tau": args.ats_tau,
            },
            "embeddings": embeddings,
        },
        args.embedding_cache,
    )
    return embeddings


def load_gt(gt_json: Path) -> List[dict]:
    with gt_json.open() as handle:
        return json.load(handle)


def summarize_ranks(ranks: List[dict]) -> dict:
    rank_values = [item["rank"] for item in ranks]
    sorted_ranks = sorted(rank_values)
    return {
        "num_queries": len(ranks),
        "top1": sum(item["top1_hit"] for item in ranks) / len(ranks),
        "top5": sum(item["top5_hit"] for item in ranks) / len(ranks),
        "top10": sum(item["top10_hit"] for item in ranks) / len(ranks),
        "mean_rank": sum(rank_values) / len(ranks),
        "median_rank": sorted_ranks[len(ranks) // 2],
        "mrr": sum(1.0 / item["rank"] for item in ranks) / len(ranks),
    }


def evaluate(args: argparse.Namespace) -> tuple[dict, dict]:
    model, preprocess = build_vadclip(args.device, args.checkpoint, args.clip_cache_dir)
    video_embeddings = build_video_embedding_index(args, model, preprocess)
    gt_items = load_gt(args.gt_json)
    gt_video_keys = [item["Video Name"] for item in gt_items]

    missing_videos = [key for key in gt_video_keys if key not in video_embeddings]
    if missing_videos:
        raise KeyError(f"GT videos not found in current video set: {missing_videos[:5]}")

    video_matrix = torch.stack([video_embeddings[key] for key in gt_video_keys]).to(args.device)
    video_matrix = F.normalize(video_matrix, dim=-1)

    text_embeddings = []
    for item in tqdm(gt_items, desc="Encoding texts"):
        text_embeddings.append(build_text_embedding(model.clipmodel, item["English Text"], args.device))
    text_matrix = torch.stack(text_embeddings).to(args.device)
    text_matrix = F.normalize(text_matrix, dim=-1)

    text_to_video_ranks = []
    for index, item in enumerate(tqdm(gt_items, desc="Scoring text-to-video")):
        similarities = video_matrix @ text_matrix[index]
        ranked_indices = torch.argsort(similarities, descending=True)
        rank = int((ranked_indices == index).nonzero(as_tuple=False)[0].item()) + 1
        text_to_video_ranks.append(
            {
                "video_name": item["Video Name"],
                "query": item["English Text"],
                "rank": rank,
                "top1_hit": rank <= 1,
                "top5_hit": rank <= 5,
                "top10_hit": rank <= 10,
            }
        )

    video_to_text_ranks = []
    similarity_matrix = video_matrix @ text_matrix.T
    for index, item in enumerate(tqdm(gt_items, desc="Scoring video-to-text")):
        ranked_indices = torch.argsort(similarity_matrix[index], descending=True)
        rank = int((ranked_indices == index).nonzero(as_tuple=False)[0].item()) + 1
        video_to_text_ranks.append(
            {
                "video_name": item["Video Name"],
                "query": item["English Text"],
                "rank": rank,
                "top1_hit": rank <= 1,
                "top5_hit": rank <= 5,
                "top10_hit": rank <= 10,
            }
        )

    metrics = {
        "sampling": {
            "strategy": "ats_guided",
            "select_frames": args.select_frames,
            "ats_tau": args.ats_tau,
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
        },
        "text_to_video": summarize_ranks(text_to_video_ranks),
        "video_to_text": summarize_ranks(video_to_text_ranks),
    }
    ranks = {
        "text_to_video": text_to_video_ranks,
        "video_to_text": video_to_text_ranks,
    }
    return metrics, ranks


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    metrics, ranks = evaluate(args)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2))
    args.ranks_json.write_text(json.dumps(ranks, indent=2))

    print(json.dumps(metrics, indent=2))
    print("")
    for direction in ["text_to_video", "video_to_text"]:
        print(direction)
        best = sorted(ranks[direction], key=lambda item: item["rank"])[:5]
        worst = sorted(ranks[direction], key=lambda item: item["rank"], reverse=True)[:5]
        print("Best-5 ranks:")
        for item in best:
            print(f"rank={item['rank']:>3} | {item['video_name']} | {item['query']}")
        print("")
        print("Worst-5 ranks:")
        for item in worst:
            print(f"rank={item['rank']:>3} | {item['video_name']} | {item['query']}")
        print("")


if __name__ == "__main__":
    main()
