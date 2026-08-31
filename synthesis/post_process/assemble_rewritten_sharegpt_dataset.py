#!/usr/bin/env python3
"""Assemble successful ShareGPT rewrite shards into a new training dataset.

Each --source consists of a ShareGPT JSON array and, optionally, its rewrite
audit JSONL. When an audit is supplied, only records with ``status == "ok"``
are included. This deliberately excludes fallback/original records emitted for
failed rewrite calls.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-from", required=True, help="Canonical record order JSON.")
    parser.add_argument("--output-dir", required=True, help="Must not already exist.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="JSON[:AUDIT_JSONL]",
        help="Input JSON, optionally followed by its audit JSONL.",
    )
    parser.add_argument(
        "--dataset-info-from",
        required=True,
        help="Existing dataset_info.json to copy into the new directory.",
    )
    parser.add_argument(
        "--images-from",
        required=True,
        help="Existing images directory/symlink to reuse as images/.",
    )
    parser.add_argument(
        "--extra-images-from",
        help="Existing extra_images directory whose referenced files should be linked into output.",
    )
    return parser.parse_args()


def iter_json_array(path: Path):
    """Yield elements of a large JSON array without loading it all into memory."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        while True:
            chunk = handle.read(1 << 20)
            if chunk:
                buffer += chunk
            eof = not chunk
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"Expected JSON array: {path}")
                    position += 1
                    started = True
                    continue
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position < len(buffer) and buffer[position] == ",":
                    position += 1
                    continue
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield value
                position = end
            if position:
                buffer = buffer[position:]
                position = 0
            if eof:
                if buffer.strip():
                    raise ValueError(f"Incomplete JSON array: {path}")
                raise ValueError(f"Missing closing ]: {path}")


def successful_ids(audit_path: Path) -> set[str]:
    result: set[str] = set()
    with audit_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            record = json.loads(line)
            record_id = record.get("record_id")
            if not isinstance(record_id, str):
                raise ValueError(f"Missing record_id in {audit_path}:{line_no}")
            if record.get("status") == "ok":
                result.add(record_id)
    return result


def atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def add_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            # The terminal-image filename is content-addressed. The same
            # downloaded image can legitimately have been materialized in two
            # rewrite workdirs, so retain an existing link when bytes match.
            if link.is_file() and target.is_file() and filecmp.cmp(link, target, shallow=False):
                return
            raise FileExistsError(f"Conflicting image name: {link}")
        return
    # A relative target is interpreted relative to `link.parent`, not the
    # process working directory. Store an absolute target so assembled
    # datasets remain valid after moving into a new directory.
    link.symlink_to(target.resolve())


def link_existing_extra_images(source_dir: Path, output_dir: Path) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    output_extra = output_dir / "extra_images"
    output_extra.mkdir(exist_ok=True)
    count = 0
    for source in source_dir.iterdir():
        if source.is_file():
            add_link(output_extra / source.name, source)
            count += 1
    return count


def normalize_terminal_images(record: dict[str, Any], output_dir: Path) -> int | None:
    """Link terminal images; return None when an absolute image is missing."""
    images = record.get("images")
    if not isinstance(images, list):
        return 0
    extra_dir = output_dir / "extra_images"
    replacements = 0
    normalized: list[Any] = []
    for image in images:
        if not isinstance(image, str):
            normalized.append(image)
            continue
        # Only rewritten terminal evidence is relocated. Other absolute paths
        # may be externally managed data paths and are left untouched. Earlier
        # rewrite outputs stored these paths relative to the repository root,
        # so handle both absolute and repository-relative forms here.
        if "terminal_visual_evidence" not in image:
            normalized.append(image)
            continue
        source = Path(image)
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_file():
            return None
        extra_dir.mkdir(exist_ok=True)
        target = extra_dir / source.name
        add_link(target, source)
        normalized.append(f"extra_images/{source.name}")
        replacements += 1
    record["images"] = normalized
    return replacements


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")

    canonical_index: dict[str, int] = {}
    for index, record in enumerate(iter_json_array(Path(args.order_from))):
        record_id = record.get("id")
        if not isinstance(record_id, str):
            raise ValueError("Invalid canonical record id")
        if record_id in canonical_index:
            raise ValueError(f"Duplicate canonical id: {record_id}")
        canonical_index[record_id] = index

    output_dir.mkdir(parents=True)
    try:
        images_from = Path(args.images_from)
        if not images_from.exists() and not images_from.is_symlink():
            raise FileNotFoundError(images_from)
        (output_dir / "images").symlink_to(images_from.resolve())
        shutil.copy2(args.dataset_info_from, output_dir / "dataset_info.json")
        inherited_extra_images = (
            link_existing_extra_images(Path(args.extra_images_from), output_dir)
            if args.extra_images_from
            else 0
        )
        terminal_image_links = 0
        missing_image_records = 0
        written = 0
        last_canonical_index = -1
        seen_ids: set[str] = set()
        source_counts: list[dict[str, Any]] = []
        with (output_dir / "trajectories_sharegpt.json").open("w", encoding="utf-8") as handle:
            handle.write("[")
            for spec in args.source:
                json_text, sep, audit_text = spec.partition(":")
                json_path = Path(json_text)
                audit_path = Path(audit_text) if sep else None
                allowed = successful_ids(audit_path) if audit_path else None
                included = 0
                skipped_missing_images = 0
                for record in iter_json_array(json_path):
                    record_id = record.get("id")
                    if not isinstance(record_id, str):
                        raise ValueError(f"Invalid record id in {json_path}")
                    if allowed is not None and record_id not in allowed:
                        continue
                    if record_id in seen_ids:
                        raise ValueError(f"Duplicate record id: {record_id}")
                    order = canonical_index.get(record_id)
                    if order is None:
                        raise ValueError(f"Record absent from --order-from: {record_id}")
                    if order <= last_canonical_index:
                        raise ValueError(f"Sources are not in canonical order near: {record_id}")
                    last_canonical_index = order
                    seen_ids.add(record_id)
                    linked_images = normalize_terminal_images(record, output_dir)
                    if linked_images is None:
                        missing_image_records += 1
                        skipped_missing_images += 1
                        continue
                    terminal_image_links += linked_images
                    if written:
                        handle.write(",")
                    json.dump(record, handle, ensure_ascii=False)
                    written += 1
                    included += 1
                source_counts.append({"json": str(json_path), "included": included, "skipped_missing_images": skipped_missing_images, "audit": str(audit_path) if audit_path else None})
            handle.write("]\n")
        atomic_json_dump(
            {
                "records": written,
                "skipped_missing_image_records": missing_image_records,
                "inherited_extra_image_links": inherited_extra_images,
                "terminal_image_links": terminal_image_links,
                "sources": source_counts,
            },
            output_dir / "assembly_manifest.json",
        )
    except Exception:
        # Keep partial output for diagnosis. Large FUSE-backed directories can
        # take a long time to remove and otherwise hide the actual exception.
        raise

    print(json.dumps({"output_dir": str(output_dir), "records": written, "skipped_missing_image_records": missing_image_records, "inherited_extra_image_links": inherited_extra_images, "terminal_image_links": terminal_image_links}, ensure_ascii=False))


if __name__ == "__main__":
    main()
