#!/usr/bin/env python3
"""Check cutoff-length tokenized rows for Qwen3-VL image-token alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
from datasets import load_from_disk
from PIL import Image
from transformers import AutoProcessor

from llamafactory.data.mm_plugin import get_mm_plugin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-max-pixels", type=int, required=True)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--cutoff-len", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    done: set[int] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["cache_index"])
                except (json.JSONDecodeError, KeyError):
                    continue

    dataset = load_from_disk(args.tokenized_path)["train"]
    lengths = pc.list_value_length(dataset._data.table.column("input_ids")).to_numpy(
        zero_copy_only=False
    )
    candidates = [int(index) for index in (lengths >= args.cutoff_len).nonzero()[0] if index not in done]
    if args.limit:
        candidates = candidates[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.image_max_pixels = args.image_max_pixels
    processor.image_min_pixels = args.image_min_pixels
    plugin = get_mm_plugin("qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
    image_token_id = processor.image_token_id

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for cache_index in candidates:
            row = dataset[cache_index]
            result: dict[str, object] = {
                "cache_index": cache_index,
                "sequence_length": len(row["input_ids"]),
                "images": row["images"],
            }
            try:
                image_paths = row["images"] or []
                if not image_paths:
                    actual = row["input_ids"].count(image_token_id)
                    result.update(expected_image_tokens=0, actual_image_tokens=actual, aligned=(actual == 0))
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"cache_index={cache_index} aligned={result['aligned']} actual={actual} expected=0", flush=True)
                    continue

                images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
                grid = plugin._get_mm_inputs(images, [], [], processor)["image_grid_thw"]
                expected = int(grid.prod(dim=1).sum()) // (processor.image_processor.merge_size**2)
                actual = row["input_ids"].count(image_token_id)
                result.update(
                    expected_image_tokens=expected,
                    actual_image_tokens=actual,
                    aligned=(expected == actual),
                )
            except Exception as exc:  # retain image-path diagnostics instead of failing the scan
                result.update(aligned=False, error=repr(exc))
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"cache_index={cache_index} aligned={result.get('aligned')} "
                f"actual={result.get('actual_image_tokens')} expected={result.get('expected_image_tokens')}",
                flush=True,
            )

    print(json.dumps({"checked_now": len(candidates), "previously_checked": len(done)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
