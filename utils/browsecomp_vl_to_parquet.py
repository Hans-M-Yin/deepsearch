#!/usr/bin/env python3
"""Convert BrowseComp-VL JSON/JSONL data into an OpenSearch-VL-friendly parquet.

Expected input layout from the official WebWatcher repository:

    browsecomp-vl/
    ├── bc_vl_level1.jsonl
    ├── bc_vl_level2.jsonl
    └── images/
        ├── level1/
        └── level2/

The upstream files are named ``*.jsonl``, but in practice they may contain
either one JSON object per line or multiple concatenated JSON objects with
embedded newlines inside string fields. This script handles both forms.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterable

def _load_records(input_path: Path) -> list[dict[str, Any]]:
    """Parse either standard JSONL or concatenated JSON objects."""

    text = input_path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []

    # Fast path: regular JSONL.
    parse_failed = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            parse_failed = True
            records = []
            break
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object per line, got {type(obj)!r}")
        records.append(obj)

    if records and not parse_failed:
        return records

    # Fallback: concatenated JSON objects with arbitrary whitespace/newlines.
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj)!r}")
        records.append(obj)
        idx = next_idx

    return records


def _resolve_image_bytes(dataset_root: Path, image_path: str) -> bytes:
    image_file = dataset_root / image_path
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file}")
    return image_file.read_bytes()


def _infer_case_id(image_path: str, fallback_prefix: str, index: int) -> str:
    stem = Path(image_path).stem
    return stem or f"{fallback_prefix}_{index}"


def _iter_rows(
    records: Iterable[dict[str, Any]],
    dataset_root: Path,
    source_name: str,
) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(records):
        question = str(item.get("question", "")).strip()
        image_path = str(item.get("image_path", "")).strip()
        answers = item.get("answers", [])
        domain = str(item.get("domain", "unknown")).strip() or "unknown"

        if not question:
            raise ValueError(f"Record {index} is missing 'question'")
        if not image_path:
            raise ValueError(f"Record {index} is missing 'image_path'")
        if not isinstance(answers, list):
            answers = [str(answers)]

        case_id = _infer_case_id(image_path, source_name, index)
        image_bytes = _resolve_image_bytes(dataset_root, image_path)
        image_format = Path(image_path).suffix.lower().lstrip(".") or "jpg"

        yield {
            "data_id": case_id,
            "category": domain,
            "data_source": source_name,
            "question": question,
            "answers": answers,
            "original_data": {
                "question": question,
                "answers": answers,
                "image_path": image_path,
                "domain": domain,
                "source": source_name,
            },
            "prompt": [{"role": "user", "content": question}],
            "images": [{"bytes": image_bytes, "format": image_format}],
        }


def build_dataframe(input_path: Path, dataset_root: Path, source_name: str):
    import pandas as pd

    records = _load_records(input_path)
    rows = list(_iter_rows(records, dataset_root=dataset_root, source_name=source_name))
    return pd.DataFrame(rows)


def write_parquet_via_buffer(df, output_path: Path) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official BrowseComp-VL JSONL data to parquet for OpenSearch-VL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to bc_vl_level1.jsonl or bc_vl_level2.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory containing the browsecomp-vl folder and images/",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path",
    )
    parser.add_argument(
        "--source-name",
        default="BrowseComp-VL",
        help="Value to write into the data_source column",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    df = build_dataframe(
        input_path=input_path,
        dataset_root=dataset_root,
        source_name=args.source_name,
    )
    write_parquet_via_buffer(df, output_path)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
