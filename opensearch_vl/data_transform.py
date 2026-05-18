#!/usr/bin/env python3
"""Convert VDR-Bench-testmini into an OpenSearch-VL-friendly parquet file."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def _decode_image(image_field) -> bytes:
    if isinstance(image_field, bytes):
        return image_field
    if isinstance(image_field, str):
        return base64.b64decode(image_field)
    raise TypeError(f"Unsupported image field type: {type(image_field)!r}")


def _infer_category(sample_id: str) -> str:
    parts = sample_id.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def build_dataframe(dataset_name: str, split: str) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    rows = []
    for item in ds:
        sample_id = str(item["id"])
        rows.append(
            {
                "data_id": sample_id,
                "category": _infer_category(sample_id),
                "data_source": "VDR-Bench",
                "prompt": [{"content": item["question"]}],
                "images": [{"bytes": _decode_image(item["image"])}],
                "answer": item.get("answer", ""),
            }
        )
    return pd.DataFrame(rows)


def write_parquet_via_buffer(df: pd.DataFrame, output_path: str) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="Osilly/VDR-Bench-testmini",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--output",
        default="vdr_testmini_for_opensearch.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    df = build_dataframe(args.dataset, args.split)
    write_parquet_via_buffer(df, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
