#!/usr/bin/env python3
"""Concatenate already-valid ShareGPT JSON arrays without loading them in RAM."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--dataset-info-from", required=True)
    parser.add_argument("--images-from", required=True)
    parser.add_argument("--extra-images-from", action="append", default=[])
    return parser.parse_args()


def copy_array_payload(source: Path, destination, first: bool) -> tuple[bool, int]:
    size = source.stat().st_size
    with source.open("rb") as handle:
        if handle.read(1) != b"[":
            raise ValueError(f"Not a JSON array: {source}")
        handle.seek(max(0, size - 1024))
        tail = handle.read()
    trimmed = tail.rstrip()
    if not trimmed.endswith(b"]"):
        raise ValueError(f"Not a complete JSON array: {source}")
    end = size - (len(tail) - len(trimmed)) - 1
    if end <= 1:
        return first, 0
    if not first:
        destination.write(b",")
    remaining = end - 1
    with source.open("rb") as handle:
        handle.seek(1)
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                raise ValueError(f"Unexpected EOF: {source}")
            destination.write(chunk)
            remaining -= len(chunk)
    return False, end - 1


def link_extra_images(source_dir: Path, destination_dir: Path) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    destination_dir.mkdir(exist_ok=True)
    count = 0
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        link = destination_dir / source.name
        if link.exists() or link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError(f"Conflicting extra image: {link.name}")
            continue
        link.symlink_to(source)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    try:
        (output_dir / "images").symlink_to(Path(args.images_from).resolve())
        shutil.copy2(args.dataset_info_from, output_dir / "dataset_info.json")
        extra_count = sum(
            link_extra_images(Path(raw), output_dir / "extra_images")
            for raw in args.extra_images_from
        )
        first = True
        with (output_dir / "trajectories_sharegpt.json").open("wb") as destination:
            destination.write(b"[")
            for raw in args.input_json:
                first, _ = copy_array_payload(Path(raw), destination, first)
            destination.write(b"]\n")
        print({"output_dir": str(output_dir), "extra_image_links": extra_count})
    except Exception:
        raise


if __name__ == "__main__":
    main()
