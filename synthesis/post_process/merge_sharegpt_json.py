#!/usr/bin/env python3
"""Merge disjoint ShareGPT JSON arrays in a canonical record-id order."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="ShareGPT JSON arrays to merge.")
    parser.add_argument(
        "--order-from",
        required=True,
        help="ShareGPT JSON whose record-id order defines the merged output order.",
    )
    parser.add_argument("--output", required=True, help="New JSON output path; must not already exist.")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return payload


def atomic_json_dump(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, output)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    records_by_id: dict[str, dict[str, Any]] = {}
    for raw_path in args.input:
        path = Path(raw_path)
        for record in read_records(path):
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Record without a non-empty id in {path}")
            if record_id in records_by_id:
                raise ValueError(f"Duplicate record id across merge inputs: {record_id}")
            records_by_id[record_id] = record

    ordered_ids: list[str] = []
    seen_order_ids: set[str] = set()
    for record in read_records(Path(args.order_from)):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Order source contains invalid id: {record_id!r}")
        if record_id in seen_order_ids:
            raise ValueError(f"Duplicate record id in order source: {record_id}")
        seen_order_ids.add(record_id)
        if record_id in records_by_id:
            ordered_ids.append(record_id)

    missing = set(records_by_id) - set(ordered_ids)
    if missing:
        example = next(iter(missing))
        raise ValueError(f"{len(missing)} input records are absent from --order-from (e.g. {example})")

    merged = [records_by_id[record_id] for record_id in ordered_ids]
    atomic_json_dump(merged, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(merged),
                "first_id": merged[0]["id"] if merged else None,
                "last_id": merged[-1]["id"] if merged else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
