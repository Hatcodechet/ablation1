#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

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
from run_top_p_retrieval import build_vadclip, build_video_embedding, top_p_select_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UCF-CrimeAR Top-P retrieval using precomputed CLIP feature sequences."
    )
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt-json", type=Path, default=DEFAULT_GT_JSON)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--snippet-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-selected-frames", type=int, default=1)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--clip-cache-dir", type=Path, default=DEFAULT_CLIP_CACHE_DIR)
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--ranks-json", type=Path, default=None)
    return parser.parse_args()


def top_p_tag(top_p: float) -> str:
    return f"p{int(round(top_p * 100)):02d}"


def resolve_output_paths(args: argparse.Namespace) -> None:
    suffix = top_p_tag(args.top_p)
    if args.embedding_cache is None:
        args.embedding_cache = DEFAULT_EMBEDDING_CACHE_ROOT / f"top_p_features_embedding_cache_{suffix}.pt"
    if args.metrics_json is None:
        args.metrics_json = DEFAULT_OUTPUT_ROOT / f"top_p_features_metrics_{suffix}.json"
    if args.ranks_json is None:
        args.ranks_json = DEFAULT_OUTPUT_ROOT / f"top_p_features_ranks_{suffix}.json"


def evaluate(args: argparse.Namespace) -> tuple[dict, dict]:
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in the range (0, 1].")
    if args.snippet_size <= 0:
        raise ValueError("--snippet-size must be > 0.")
    if args.min_selected_frames <= 0:
        raise ValueError("--min-selected-frames must be > 0.")

    gt_items = load_gt_items(args.gt_json)
    gallery_video_keys = build_gallery_video_keys(gt_items)
    resolver = FeaturePathResolver(args.feature_root)
    model, _ = build_vadclip(args.device, args.checkpoint, args.clip_cache_dir)

    def embedding_builder(frame_features: torch.Tensor) -> torch.Tensor:
        anomaly_scores = compute_anomaly_scores(model, frame_features, snippet_size=args.snippet_size)
        selected_indices, _ = top_p_select_indices(
            anomaly_scores,
            top_p=args.top_p,
            min_selected_frames=args.min_selected_frames,
            temperature=args.score_temperature,
        )
        return build_video_embedding(frame_features, selected_indices)

    cache_meta = {
        "feature_root": str(args.feature_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "gt_json": str(args.gt_json.resolve()),
        "top_p": args.top_p,
        "snippet_size": args.snippet_size,
        "min_selected_frames": args.min_selected_frames,
        "score_temperature": args.score_temperature,
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
        "strategy": "top_p_features",
        "feature_root": str(args.feature_root),
        "top_p": args.top_p,
        "snippet_size": args.snippet_size,
        "vadclip_style_segment_scoring": True,
        "score_temperature": args.score_temperature,
        "min_selected_frames": args.min_selected_frames,
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
