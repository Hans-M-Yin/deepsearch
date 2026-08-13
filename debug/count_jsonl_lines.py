#!/usr/bin/env python3
"""Count records/lines in a large JSONL file without loading it into memory."""

from __future__ import annotations

import argparse
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def count_lines(path: Path, *, non_empty_only: bool) -> int:
    count = 0
    pending = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            parts = (pending + chunk).split(b"\n")
            pending = parts.pop()
            if non_empty_only:
                count += sum(bool(part.strip()) for part in parts)
            else:
                count += len(parts)

    if pending:
        count += bool(pending.strip()) if non_empty_only else 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="JSONL file to count")
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="Count blank lines too; default counts non-empty JSONL records only.",
    )
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"file not found: {args.jsonl}")

    count = count_lines(args.jsonl, non_empty_only=not args.all_lines)
    mode = "all lines" if args.all_lines else "non-empty records"
    print(f"{mode}: {count}")


if __name__ == "__main__":
    main()
