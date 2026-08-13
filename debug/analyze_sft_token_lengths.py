#!/usr/bin/env python3
"""Summarize token-length and image-count distributions in an HF dataset cache.

This is intended for the tokenized dataset produced by LlamaFactory, for
example ``cache/opensearch_sft_tokenized_8k_32768``.  It reads ``input_ids``
and ``attention_mask`` directly, so the reported lengths include the training
template and multimodal input processing rather than only raw character
counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


DEFAULT_BUCKETS = (
    (0, 4096),
    (4097, 8192),
    (8193, 16384),
    (16385, 24576),
    (24577, 32767),
)


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _select_split(dataset: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"split {split!r} is not available; available={list(dataset)}")
        return dataset[split]
    return dataset


def _sequence_length(row: dict[str, Any]) -> int:
    attention_mask = row.get("attention_mask")
    if attention_mask:
        return int(sum(int(value) for value in attention_mask))
    return len(row.get("input_ids") or [])


def analyze(dataset_path: str | Path, *, split: str, cutoff_len: int) -> dict[str, Any]:
    dataset = _select_split(load_from_disk(str(dataset_path)), split)
    lengths: list[int] = []
    image_counts: list[int] = []

    for row in tqdm(dataset, desc=f"Reading {split}", unit="sample"):
        lengths.append(_sequence_length(row))
        images = row.get("images") or []
        image_counts.append(len(images) if isinstance(images, list) else 0)

    sorted_lengths = sorted(lengths)
    sorted_images = sorted(image_counts)
    buckets: dict[str, int] = {
        f"{lower}-{upper}": sum(lower <= value <= upper for value in lengths)
        for lower, upper in DEFAULT_BUCKETS
    }
    buckets[f">={cutoff_len}"] = sum(value >= cutoff_len for value in lengths)

    result: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "split": split,
        "samples": len(lengths),
        "cutoff_len": cutoff_len,
        "token_length": {
            "min": min(lengths) if lengths else 0,
            "mean": mean(lengths) if lengths else 0.0,
            "median": median(lengths) if lengths else 0.0,
            "p90": _percentile(sorted_lengths, 90),
            "p95": _percentile(sorted_lengths, 95),
            "p99": _percentile(sorted_lengths, 99),
            "max": max(lengths) if lengths else 0,
            "at_or_above_cutoff": sum(value >= cutoff_len for value in lengths),
        },
        "token_length_buckets": buckets,
        "images_per_sample": {
            "with_images": sum(value > 0 for value in image_counts),
            "total_image_references": sum(image_counts),
            "min": min(image_counts) if image_counts else 0,
            "mean": mean(image_counts) if image_counts else 0.0,
            "median": median(image_counts) if image_counts else 0.0,
            "max": max(image_counts) if image_counts else 0,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-path", required=True, help="LlamaFactory load_from_disk dataset directory.")
    parser.add_argument("--split", default="train", help="Dataset split to inspect (default: train).")
    parser.add_argument("--cutoff-len", type=int, default=32768, help="Configured cutoff length.")
    parser.add_argument("--output-json", help="Optional path for the JSON summary.")
    args = parser.parse_args()

    result = analyze(args.tokenized_path, split=args.split, cutoff_len=args.cutoff_len)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
