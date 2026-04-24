#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy import interpolate

from feature_retrieval_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CLIP_CACHE_DIR,
    DEFAULT_EMBEDDING_CACHE_ROOT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_GT_JSON,
    DEFAULT_OUTPUT_ROOT,
    FeaturePathResolver,
    build_feature_embedding_index,
    build_gallery_video_keys,
    build_text_matrix,
    compute_anomaly_scores,
    describe_feature_sources,
    evaluate_bidirectional_retrieval,
    load_gt_items,
    validate_gallery_coverage,
    write_json,
)
from run_top_p_retrieval import build_vadclip, build_video_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UCF-CrimeAR ATS retrieval using precomputed CLIP feature sequences."
    )
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt-json", type=Path, default=DEFAULT_GT_JSON)
    parser.add_argument("--snippet-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--select-frames", type=int, default=12)
    parser.add_argument("--ats-tau", type=float, default=0.1)
    parser.add_argument("--clip-cache-dir", type=Path, default=DEFAULT_CLIP_CACHE_DIR)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--ranks-json", type=Path, default=None)
    return parser.parse_args()


def ats_tag(select_frames: int, tau: float) -> str:
    tau_int = int(round(tau * 100))
    return f"ats_f{select_frames}_tau{tau_int:02d}"


def resolve_output_paths(args: argparse.Namespace) -> None:
    suffix = ats_tag(args.select_frames, args.ats_tau)
    if args.embedding_cache is None:
        args.embedding_cache = DEFAULT_EMBEDDING_CACHE_ROOT / f"ats_features_embedding_cache_{suffix}.pt"
    if args.metrics_json is None:
        args.metrics_json = DEFAULT_OUTPUT_ROOT / f"ats_features_metrics_{suffix}.json"
    if args.ranks_json is None:
        args.ranks_json = DEFAULT_OUTPUT_ROOT / f"ats_features_ranks_{suffix}.json"


def ats_density_aware_sample(anomaly_scores: torch.Tensor, select_frames: int, tau: float) -> List[int]:
    """Mirror HolmesVAU ATS sampler behavior as closely as possible.

    Reference:
    - `/workspace/HolmesVAU/holmesvau/holmesvau_utils.py`
    - `/workspace/HolmesVAU/holmesvau/ATS/Temporal_Sampler.py`
    """
    scores = anomaly_scores.detach().cpu().numpy().astype(np.float64)
    num_frames = int(scores.shape[0])
    if num_frames == 0:
        raise ValueError("ATS sampling received an empty anomaly score sequence.")
    if select_frames <= 0:
        raise ValueError("--select-frames must be > 0.")
    if num_frames <= select_frames or float(scores.sum()) < 1.0:
        sampled = np.rint(np.linspace(0, num_frames - 1, select_frames)).astype(int)
        return [int(index) for index in sampled.tolist()]

    dense_scores = [float(score) + tau for score in scores]
    score_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(dense_scores)), axis=0)
    max_score_cumsum = int(np.round(score_cumsum[-1]))
    sampler = interpolate.interp1d(
        score_cumsum,
        np.arange(num_frames + 1),
        kind="linear",
        axis=0,
        fill_value="extrapolate",
    )
    scale_x = np.linspace(1, max_score_cumsum, select_frames)
    sampled = sampler(scale_x)
    sampled = [min(num_frames - 1, max(0, int(index))) for index in sampled]
    return sampled


def evaluate(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.snippet_size <= 0:
        raise ValueError("--snippet-size must be > 0.")
    if args.select_frames <= 0:
        raise ValueError("--select-frames must be > 0.")

    gt_items = load_gt_items(args.gt_json)
    gallery_video_keys = build_gallery_video_keys(gt_items)
    resolver = FeaturePathResolver(args.feature_root)
    model, _ = build_vadclip(args.device, args.checkpoint, args.clip_cache_dir)

    def embedding_builder(frame_features: torch.Tensor) -> torch.Tensor:
        anomaly_scores = compute_anomaly_scores(model, frame_features, snippet_size=args.snippet_size)
        sampled_indices = ats_density_aware_sample(anomaly_scores, args.select_frames, args.ats_tau)
        sampled_index_tensor = torch.tensor(sampled_indices, device=frame_features.device, dtype=torch.long)
        return build_video_embedding(frame_features, sampled_index_tensor)

    cache_meta = {
        "feature_root": str(args.feature_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "gt_json": str(args.gt_json.resolve()),
        "snippet_size": args.snippet_size,
        "select_frames": args.select_frames,
        "ats_tau": args.ats_tau,
    }
    video_embeddings = build_feature_embedding_index(
        gallery_video_keys=gallery_video_keys,
        resolver=resolver,
        device=args.device,
        embedding_cache=args.embedding_cache,
        cache_meta=cache_meta,
        embedding_builder=embedding_builder,
    )
    validate_gallery_coverage(gt_items, video_embeddings)
    text_matrix = build_text_matrix(model, gt_items, args.device)

    metrics, ranks = evaluate_bidirectional_retrieval(
        gt_items=gt_items,
        gallery_video_keys=gallery_video_keys,
        gallery_embeddings=video_embeddings,
        text_matrix=text_matrix,
        device=args.device,
    )
    metrics["sampling"] = {
        "strategy": "ats_features",
        "feature_root": str(args.feature_root),
        "snippet_size": args.snippet_size,
        "vadclip_style_segment_scoring": True,
        "select_frames": args.select_frames,
        "ats_tau": args.ats_tau,
    }
    ranks["feature_paths"] = describe_feature_sources(resolver, gallery_video_keys)
    return metrics, ranks


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    metrics, ranks = evaluate(args)
    write_json(args.metrics_json, metrics)
    write_json(args.ranks_json, ranks)
    print(args.metrics_json)
    print(args.ranks_json)


if __name__ == "__main__":
    main()
