#!/usr/bin/env python3
"""Build a small, deterministic mixed evaluation dataset.

The five benchmark sources are sampled by row index at approximately uniform
intervals.  The output keeps the common OpenSearch-VL schema and adds one
extra top-level field, ``source``, for per-benchmark scoring.

Example:
    python scripts/build_evaluation_dataset.py
    python scripts/build_evaluation_dataset.py --samples-per-benchmark 40 \
        --output ../data/benchmarks/evaluation/test.parquet
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


COMMON_COLUMNS = (
    "data_id",
    "question_id",
    "sample_id",
    "category",
    "data_source",
    "question",
    "prompt",
    "images",
    "answer",
    "source_metadata",
)
OUTPUT_COLUMNS = COMMON_COLUMNS + ("source",)


@dataclass(frozen=True)
class BenchmarkSpec:
    source: str
    relative_path: str


BENCHMARKS = (
    BenchmarkSpec("FVQA", "fvqa/test.parquet"),
    BenchmarkSpec("SimpleVQA", "simplevqa/test.parquet"),
    BenchmarkSpec("MMSearch", "mmsearch/end2end/data.parquet"),
    BenchmarkSpec("VDR-Bench", "vdr_bench/train.parquet"),
    BenchmarkSpec("BrowseComp-VL", "browsecomp_vl/test.parquet"),
)


def _uniform_indices(total: int, count: int) -> list[int]:
    """Return ``count`` deterministic, evenly spaced row indices."""

    if count <= 0:
        raise ValueError(f"samples-per-benchmark must be positive, got {count}")
    if total < count:
        raise ValueError(f"Dataset has {total} rows, fewer than requested {count}")
    if count == 1:
        return [0]

    # Include both endpoints.  Since total >= count, these indices are unique.
    return [int(index * (total - 1) / (count - 1)) for index in range(count)]


def _to_python(value: Any) -> Any:
    """Convert pandas/numpy containers to ordinary Python containers."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, str, int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]

    # pandas Series cells containing nested parquet lists are commonly numpy
    # arrays or numpy scalar objects.
    if hasattr(value, "tolist"):
        return _to_python(value.tolist())
    if hasattr(value, "item"):
        return _to_python(value.item())
    return value


def _text(value: Any, default: str = "") -> str:
    value = _to_python(value)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalise_images(value: Any) -> list[dict[str, Any]]:
    value = _to_python(value)
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    images: list[dict[str, Any]] = []
    for item in value:
        item = _to_python(item)
        if isinstance(item, dict):
            images.append(item)
    return images


def _normalise_source_metadata(value: Any) -> str:
    value = _to_python(value)
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    import json

    return json.dumps(value, ensure_ascii=False)


def _read_sample(path: Path, count: int) -> tuple[list[dict[str, Any]], int]:
    """Read only the common columns, then select uniformly spaced rows."""

    parquet_columns = set(pq.read_schema(path).names)
    columns = [column for column in COMMON_COLUMNS if column in parquet_columns]
    frame = pd.read_parquet(path, columns=columns)
    indices = _uniform_indices(len(frame), count)

    rows: list[dict[str, Any]] = []
    for index in indices:
        raw = frame.iloc[index]
        question = _text(raw.get("question"))
        rows.append(
            {
                "data_id": _text(raw.get("data_id")),
                "question_id": _text(raw.get("question_id")),
                "sample_id": _text(raw.get("sample_id")),
                "category": _text(raw.get("category"), "unknown"),
                "data_source": _text(raw.get("data_source")),
                "question": question,
                "prompt": [{"content": question}],
                "images": _normalise_images(raw.get("images")),
                "answer": _text(raw.get("answer")),
                "source_metadata": _normalise_source_metadata(raw.get("source_metadata")),
            }
        )
    return rows, len(frame)


def _make_ids_unique(rows: list[dict[str, Any]]) -> int:
    """Avoid trajectory filename collisions while retaining original IDs when possible."""

    seen: set[str] = set()
    changed = 0
    for row_index, row in enumerate(rows):
        base_id = _text(row.get("data_id")) or f"evaluation_{row_index:04d}"
        unique_id = base_id
        suffix = 1
        while unique_id in seen:
            unique_id = f"{base_id}__evaluation_{suffix}"
            suffix += 1
        seen.add(unique_id)
        if unique_id != base_id:
            changed += 1
        row["data_id"] = unique_id
        row["question_id"] = unique_id
        row["sample_id"] = unique_id
    return changed


def build_evaluation_dataset(
    benchmark_root: Path,
    output_path: Path,
    samples_per_benchmark: int = 40,
) -> None:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for spec in BENCHMARKS:
        input_path = benchmark_root / spec.relative_path
        if not input_path.exists():
            raise FileNotFoundError(f"Missing {spec.source} dataset: {input_path}")

        sampled_rows, total = _read_sample(input_path, samples_per_benchmark)
        for row in sampled_rows:
            row["source"] = spec.source
        rows.extend(sampled_rows)
        counts[spec.source] = total

    changed_ids = _make_ids_unique(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_parquet(output_path, index=False)

    print(f"Wrote {len(rows)} rows to {output_path}")
    for source, total in counts.items():
        print(f"  {source}: sampled={samples_per_benchmark}, available={total}")
    print(f"  duplicate IDs disambiguated: {changed_ids}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_benchmark_root = repo_root.parent / "data" / "benchmarks"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=default_benchmark_root,
        help=f"Benchmark directory (default: {default_benchmark_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_benchmark_root / "evaluation" / "test.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--samples-per-benchmark",
        type=int,
        default=40,
        help="Number of uniformly sampled rows from each benchmark",
    )
    args = parser.parse_args()

    build_evaluation_dataset(
        benchmark_root=args.benchmark_root.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        samples_per_benchmark=args.samples_per_benchmark,
    )


if __name__ == "__main__":
    main()
