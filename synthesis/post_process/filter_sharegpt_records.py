#!/usr/bin/env python3
"""Remove explicit record IDs from a large ShareGPT JSON array atomically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from assemble_rewritten_sharegpt_dataset import iter_json_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--remove-id", action="append", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    remove_ids = set(args.remove_id)
    temp = source.with_name(f".{source.name}.cutoff_filter.tmp")
    if temp.exists():
        raise FileExistsError(f"Remove stale temporary output first: {temp}")

    removed: list[str] = []
    kept = 0
    with temp.open("w", encoding="utf-8") as output:
        output.write("[")
        for record in iter_json_array(source):
            record_id = record.get("id")
            if record_id in remove_ids:
                removed.append(record_id)
                continue
            if kept:
                output.write(",")
            json.dump(record, output, ensure_ascii=False)
            kept += 1
        output.write("]\n")

    missing = remove_ids - set(removed)
    if missing:
        temp.unlink()
        raise ValueError(f"Requested IDs absent from dataset: {sorted(missing)}")
    os.replace(temp, source)
    print(json.dumps({"input": str(source), "kept": kept, "removed": removed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
