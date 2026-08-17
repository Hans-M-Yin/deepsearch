#!/usr/bin/env python3
"""Merge converted Llama-Factory ShareGPT dataset directories.

The converter writes a directory rather than a JSONL file.  A converted
dataset contains a JSON array in ``trajectories_sharegpt.json`` and image
paths relative to the dataset directory, so concatenating the JSON files (or
copying only ``dataset_info.json``) is not sufficient.  This utility merges
any number of converted dataset directories and rewrites every image path to
the new output directory.

Example::

    python -m synthesis.sft.merge_sharegpt_datasets \
        --input-dir data/sharegpt_dataset_8k \
        data/sharegpt_dataset_part3 \
        data/sharegpt_dataset_part4 \
        --output-dir data/sharegpt_dataset_18k

The input order is preserved.  Records are never deduplicated: if two input
datasets contain different trajectories for the same question, both remain
in the merged dataset.  The output directory must not already exist.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any

from tqdm import tqdm


ROWS_FILE = "trajectories_sharegpt.json"
DATASET_INFO_FILE = "dataset_info.json"
METADATA_DIR = ".metadata"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return component[:80] or "dataset"


def _relative_image_source(dataset_dir: Path, image_ref: Any) -> tuple[Path, Path]:
    """Resolve an image reference and return ``(source_path, relative_path)``.

    Converted rows normally contain paths such as ``images/foo.jpg``.  The
    extra handling for absolute paths is useful for older datasets, while the
    traversal checks prevent a malformed row from copying an arbitrary file.
    """

    if not isinstance(image_ref, str) or not image_ref.strip():
        raise ValueError(f"invalid image reference: {image_ref!r}")
    raw = image_ref.strip().replace("\\", "/")
    candidate = Path(raw)
    root = dataset_dir.resolve()

    if candidate.is_absolute():
        source = candidate.resolve()
        try:
            relative = source.relative_to(root)
        except ValueError:
            # Absolute references from old conversion runs can point outside
            # the dataset directory.  Keep them safe by using only the file
            # name in the merged output.
            relative = Path(source.name)
    else:
        relative = Path(raw)
        if any(part == ".." for part in relative.parts):
            raise ValueError(f"image reference escapes dataset directory: {image_ref!r}")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"image reference escapes dataset directory: {image_ref!r}") from exc

    if not source.is_file():
        raise FileNotFoundError(f"image referenced by trajectory does not exist: {source}")

    # Keep the path beneath images/ in the usual case.  For absolute or old
    # references outside that subtree, use a basename to avoid reproducing an
    # arbitrary filesystem hierarchy in the output dataset.
    if relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])
    elif relative.parts and relative.parts[0] not in {".", ""}:
        relative = Path(*relative.parts)
    if not relative.parts or str(relative) == ".":
        relative = Path(source.name)
    return source, relative


def _copy_one(job: tuple[Path, Path]) -> None:
    source, destination = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _load_rejected(dataset_dir: Path, source_index: int) -> list[dict[str, Any]]:
    path = dataset_dir / METADATA_DIR / "rejected.jsonl"
    if not path.is_file():
        return []
    rejected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"rejection entry in {path}:{line_number} is not an object")
            item = dict(value)
            item["source_dataset_index"] = source_index
            item["source_dataset"] = str(dataset_dir)
            rejected.append(item)
    return rejected


def merge_datasets(
    input_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Merge converted ShareGPT dataset directories into a new directory."""

    if not input_dirs:
        raise ValueError("at least one input dataset directory is required")
    if workers <= 0:
        raise ValueError("workers must be positive")

    sources = [Path(value).expanduser().resolve() for value in input_dirs]
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"input dataset directory does not exist: {source}")
        for required in (ROWS_FILE, DATASET_INFO_FILE):
            if not (source / required).is_file():
                raise FileNotFoundError(f"missing {required} in input dataset: {source}")

    dataset_info: dict[str, Any] | None = None
    dataset_info_mismatches: list[str] = []
    merged_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    # The second item is kept relative until the temporary output directory
    # has been created.  Copying directly into ``destination`` would create
    # the final directory before the atomic directory rename below.
    copy_jobs: list[tuple[Path, Path]] = []
    image_mapping: dict[tuple[int, str], str] = {}
    seen_ids: set[str] = set()
    duplicate_ids: list[dict[str, Any]] = []

    for source_index, source in enumerate(sources):
        rows_value = _read_json(source / ROWS_FILE)
        if not isinstance(rows_value, list) or any(not isinstance(row, dict) for row in rows_value):
            raise ValueError(f"{source / ROWS_FILE} must contain a JSON array of objects")

        info_value = _read_json(source / DATASET_INFO_FILE)
        if not isinstance(info_value, dict):
            raise ValueError(f"{source / DATASET_INFO_FILE} must contain a JSON object")
        if dataset_info is None:
            dataset_info = info_value
        elif info_value != dataset_info:
            dataset_info_mismatches.append(str(source))

        tag = f"dataset{source_index:02d}_{_safe_component(source.name)}"
        source_image_count = 0
        source_unique_images: set[str] = set()
        for row_index, original_row in enumerate(rows_value):
            row = deepcopy(original_row)
            row_id = row.get("id")
            if row_id is not None:
                row_id_text = str(row_id)
                if row_id_text in seen_ids:
                    duplicate_ids.append(
                        {
                            "id": row_id_text,
                            "source_dataset_index": source_index,
                            "source_row_index": row_index,
                        }
                    )
                seen_ids.add(row_id_text)

            image_refs = row.get("images") or []
            if not isinstance(image_refs, list):
                raise ValueError(f"images must be a list in {source / ROWS_FILE} row {row_index}")
            rewritten_images: list[str] = []
            for image_ref in image_refs:
                source_image, relative_image = _relative_image_source(source, image_ref)
                cache_key = (source_index, str(source_image))
                destination_relative = image_mapping.get(cache_key)
                if destination_relative is None:
                    # Keep each source dataset in its own namespace.  This
                    # prevents same-named files from overwriting each other,
                    # including duplicate question IDs across parts.
                    relative_destination = Path("images") / tag / relative_image
                    destination_relative = relative_destination.as_posix()
                    image_mapping[cache_key] = destination_relative
                    copy_jobs.append((source_image, Path(relative_destination)))
                rewritten_images.append(destination_relative)
                source_image_count += 1
                source_unique_images.add(str(source_image))
            row["images"] = rewritten_images
            merged_rows.append(row)

        source_rejected = _load_rejected(source, source_index)
        rejected.extend(source_rejected)
        source_summary_path = source / METADATA_DIR / "summary.json"
        source_summary = _read_json(source_summary_path) if source_summary_path.is_file() else {}
        source_summaries.append(
            {
                "dataset_index": source_index,
                "path": str(source),
                "records": len(rows_value),
                "image_references": source_image_count,
                "unique_image_files": len(source_unique_images),
                "rejected_records": len(source_rejected),
                "summary": source_summary,
            }
        )

    if dataset_info is None:  # defensive; inputs are validated above
        raise ValueError("no dataset_info.json was loaded")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent)))
    try:
        # Copy images in parallel, while preserving deterministic row order.
        if copy_jobs:
            actual_copy_jobs = [(source, temporary / relative) for source, relative in copy_jobs]
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sharegpt-merge") as executor:
                iterator = executor.map(_copy_one, actual_copy_jobs)
                for _ in tqdm(iterator, total=len(actual_copy_jobs), desc="Copying images", unit="image"):
                    pass

        _write_json(temporary / ROWS_FILE, merged_rows)
        _write_json(temporary / DATASET_INFO_FILE, dataset_info)
        metadata = temporary / METADATA_DIR
        metadata.mkdir(parents=True, exist_ok=True)

        with (metadata / "rejected.jsonl").open("w", encoding="utf-8") as handle:
            for item in rejected:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "operation": "merge_sharegpt_datasets",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(destination),
            "input_datasets": source_summaries,
            "written_records": len(merged_rows),
            "image_references": sum(len(row.get("images") or []) for row in merged_rows),
            "copied_image_files": len(copy_jobs),
            "rejected_records_from_inputs": len(rejected),
            "duplicate_ids_preserved": len(duplicate_ids),
            "dataset_info_mismatches": dataset_info_mismatches,
            "workers": workers,
        }
        _write_json(metadata / "summary.json", summary)
        _write_json(
            metadata / "merge_manifest.json",
            {
                "input_dirs": [str(source) for source in sources],
                "output_dir": str(destination),
                "duplicate_ids": duplicate_ids,
                "image_namespace_mapping": {
                    str(index): f"dataset{index:02d}_{_safe_component(source.name)}"
                    for index, source in enumerate(sources)
                },
            },
        )

        os.replace(temporary, destination)
        temporary = None  # type: ignore[assignment]
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        nargs="+",
        required=True,
        help="Converted ShareGPT dataset directories, merged in the given order.",
    )
    parser.add_argument("--output-dir", required=True, help="New output dataset directory.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent image-copy workers (default: 4).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = merge_datasets(args.input_dir, args.output_dir, workers=args.workers)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"merge failed: {exc}") from exc

    print("Merged ShareGPT datasets")
    print(f"  Input datasets : {len(summary['input_datasets'])}")
    print(f"  Records        : {summary['written_records']}")
    print(f"  Image files    : {summary['copied_image_files']}")
    print(f"  Image refs     : {summary['image_references']}")
    print(f"  Duplicate IDs  : {summary['duplicate_ids_preserved']} (preserved)")
    print(f"  Output         : {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
