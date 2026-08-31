#!/usr/bin/env python3
"""Atomically relocate legacy terminal-evidence image references into extra_images/."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from assemble_rewritten_sharegpt_dataset import iter_json_array, normalize_terminal_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    source = dataset_dir / "trajectories_sharegpt.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = source.with_name(f".{source.name}.normalize_terminal_images.tmp")
    if temporary.exists():
        raise FileExistsError(f"Remove stale temporary output first: {temporary}")

    written = 0
    rewritten_images = 0
    with temporary.open("w", encoding="utf-8") as output:
        output.write("[")
        for record in iter_json_array(source):
            before = list(record.get("images") or [])
            result = normalize_terminal_images(record, dataset_dir)
            if result is None:
                raise FileNotFoundError(
                    f"Missing terminal evidence image for record {record.get('id')!r}: {before}"
                )
            rewritten_images += result
            if written:
                output.write(",")
            json.dump(record, output, ensure_ascii=False)
            written += 1
        output.write("]\n")
    os.replace(temporary, source)
    print(json.dumps({"dataset_dir": str(dataset_dir), "records": written, "terminal_images_relocated": rewritten_images}))


if __name__ == "__main__":
    main()
