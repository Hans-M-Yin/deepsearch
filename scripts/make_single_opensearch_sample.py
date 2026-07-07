#!/usr/bin/env python3
"""Build a one-row parquet dataset for OpenSearch-VL inference smoke tests.

Example:
    python scripts/make_single_opensearch_sample.py
    python scripts/make_single_opensearch_sample.py --output scripts/single_sample_q_000001.parquet
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

SAMPLE = {
    "answer": (
        "Smaller, rolled-up bundles of incense sticks that are tied with a "
        "light-colored material."
    ),
    "draft_question": (
        "The stock photography agency identified by the watermark in the lower "
        "right corner of this image sources public domain content from a specific "
        "media repository. In the winning photograph for this repository's "
        "Picture of the Year 2023 contest, what items are stacked in two rows "
        "to the right of the person arranging the incense?"
    ),
    "final_question": (
        "The stock photography agency whose branding appears on this image is "
        "known to source content from a large, freely licensed media repository. "
        "That repository holds an annual contest to select its best image, and "
        "the most recently announced winner depicts a scene of traditional labor "
        "in a Vietnamese village. In that specific photograph, what items are "
        "stacked in two rows to the right of the person shown?"
    ),
    "image_url": "/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/synthesis/.image_cache/068989c3df92593ffe74d3d6.jpg",
    "path_id": "path_a44e8e244ce4f352",
    "polished_question": (
        "The stock photography agency identified by the watermark in the lower "
        "right corner of this image sources public domain content from a specific "
        "media repository. In the winning photograph for a major annual "
        "photography competition held by this repository in 2023, what items are "
        "stacked in two rows to the right of the person in the photograph?"
    ),
    "question_id": "q_000001",
    "sample_id": "sample_path_a44e8e244ce4f352",
    "status": "verified",
}


def _normalize_image_reference(image_url: str) -> str:
    """Convert local absolute paths to ``file://`` URLs for vLLM/OpenAI APIs."""

    if image_url.startswith(("http://", "https://", "data:", "file://")):
        return image_url

    candidate = Path(image_url).expanduser()
    if candidate.is_absolute():
        return candidate.resolve().as_uri()
    return image_url


def build_row() -> dict[str, object]:
    question = SAMPLE["final_question"]
    image_url = _normalize_image_reference(SAMPLE["image_url"])
    return {
        "data_id": SAMPLE["sample_id"],
        "question_id": SAMPLE["question_id"],
        "sample_id": SAMPLE["sample_id"],
        "path_id": SAMPLE["path_id"],
        "category": "manual_smoke_test",
        "data_source": "manual_single_sample",
        "question": question,
        "prompt": [{"content": question}],
        "images": [{"url": image_url}],
        "answer": SAMPLE["answer"],
        "draft_question": SAMPLE["draft_question"],
        "final_question": SAMPLE["final_question"],
        "polished_question": SAMPLE["polished_question"],
        "status": SAMPLE["status"],
        "source_metadata": json.dumps(
            {**SAMPLE, "image_url": image_url}, ensure_ascii=False
        ),
    }


def write_parquet(row: dict[str, object], output_path: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing parquet requires pandas plus a parquet backend such as "
            "pyarrow or fastparquet."
        ) from exc

    df = pd.DataFrame([row])
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet. Make sure pyarrow or fastparquet is installed."
        ) from exc
    buffer.seek(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def write_json(row: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="scripts/single_sample_q_000001.parquet",
        help="Output dataset path.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json"],
        default="parquet",
        help="Dataset format to write.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    row = build_row()
    if args.format == "json":
        write_json(row, output_path)
    else:
        write_parquet(row, output_path)
    print(f"Wrote 1 row to {output_path}")


if __name__ == "__main__":
    main()
