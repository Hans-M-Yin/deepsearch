#!/usr/bin/env python3
"""Build a one-row parquet dataset for OpenSearch-VL inference smoke tests.

Example:
    python scripts/make_single_opensearch_sample.py
    python scripts/make_single_opensearch_sample.py --output scripts/single_sample_q_000003.parquet
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

SAMPLE = {
    "answer": (
        "For a subsequent act or service, bronze, silver, and gold palm devices could "
        "be awarded. While some non-U.S. citizens are known to have received these "
        "devices, there is no evidence of any U.S. citizens having received them."
    ),
    "draft_question": (
        "The singer shown in this image received an award for her contributions to music "
        "and culture. This award is the successor to an earlier medal established in 1945 "
        "by President Harry S. Truman to honor civilian service. According to the executive "
        "order establishing that earlier medal, what suitable devices could be awarded for "
        "a subsequent act of service, and what distinction has been noted regarding the "
        "citizenship of the recipients of these devices?"
    ),
    "final_question": (
        "The singer shown in the image and her husband were both recipients of a high U.S. "
        "civilian honor, the modern form of which was established by a 1963 executive "
        "order, reviving and substantially altering an award originally created in the "
        "1940s. Regarding the executive order that established that original 1940s award, "
        "what devices were authorized for subsequent acts of service, and what was the rule "
        "concerning the citizenship of recipients?"
    ),
    "image_url": (
        "https://search-hans.oss-cn-beijing.aliyuncs.com/vision_deepresearch/2026-06-16/"
        "synthesis_2026-06-16_synthesis/opensearch-vl/"
        "synthesis_0_trajectory_turn0_image_cache_33e76e867c17c025834cbf6b.png"
    ),
    "path_id": "path_d395543d7a89cf4c",
    "polished_question": (
        "The singer shown in this image received an award for her contributions to music "
        "and culture. This award is the successor to an earlier, now-defunct medal honoring "
        "civilian service. According to the executive order establishing that earlier medal, "
        "what suitable devices could be awarded for a subsequent act of service, and what "
        "was specified about the citizenship of those who received these devices?"
    ),
    "question_id": "q_000003",
    "sample_id": "sample_path_d395543d7a89cf4c",
    "status": "verified",
}


def build_row() -> dict[str, object]:
    question = SAMPLE["final_question"]
    return {
        "data_id": SAMPLE["sample_id"],
        "question_id": SAMPLE["question_id"],
        "sample_id": SAMPLE["sample_id"],
        "path_id": SAMPLE["path_id"],
        "category": "manual_smoke_test",
        "data_source": "manual_single_sample",
        "question": question,
        "prompt": [{"content": question}],
        "images": [{"url": SAMPLE["image_url"]}],
        "answer": SAMPLE["answer"],
        "draft_question": SAMPLE["draft_question"],
        "final_question": SAMPLE["final_question"],
        "polished_question": SAMPLE["polished_question"],
        "status": SAMPLE["status"],
        "source_metadata": json.dumps(SAMPLE, ensure_ascii=False),
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
        default="scripts/single_sample_q_000003.parquet",
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
