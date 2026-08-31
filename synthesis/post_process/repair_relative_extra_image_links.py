#!/usr/bin/env python3
"""Convert relative extra_images symlink targets to absolute existing paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    root = Path(args.dataset_dir).resolve()
    extra = root / "extra_images"
    if not extra.is_dir():
        raise FileNotFoundError(extra)

    repaired = 0
    broken = []
    for link in extra.iterdir():
        if not link.is_symlink():
            continue
        raw_target = link.readlink()
        if raw_target.is_absolute():
            continue
        # Legacy assembler stored repository-relative targets. Resolve those
        # relative to the caller's repository cwd, not link.parent.
        candidate = (Path.cwd() / raw_target).resolve()
        if not candidate.is_file():
            broken.append((str(link), str(candidate)))
            continue
        link.unlink()
        link.symlink_to(candidate)
        repaired += 1
    if broken:
        raise RuntimeError(f"{len(broken)} relative links could not be repaired: {broken[:3]}")
    print({"dataset_dir": str(root), "repaired_links": repaired})


if __name__ == "__main__":
    main()
