#!/usr/bin/env python3
"""Create an independent, train-ready contiguous subset of a ShareGPT dataset.

The subset keeps record order, copies ``dataset_info.json`` when present, and
links the source ``images`` directory instead of duplicating image files.

Example:
    python synthesis/post_process/create_sharegpt_subset.py \
        --input data/sharegpt_dataset_filtered_v1/trajectories_sharegpt.json \
        --output-dir data/sharegpt_dataset_filtered_v1_first7688 \
        --offset 0 --limit 7688 \
        --exclude-filter-recommended-audit \
          data/sharegpt_dataset_filtered_v1/trajectory_rewrite_gpt54_window2_test_audit.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more disjoint rewritten ShareGPT JSON arrays, kept in the supplied order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New standalone dataset directory.")
    parser.add_argument("--offset", type=int, default=0, help="Zero-based first source record.")
    parser.add_argument("--limit", type=int, required=True, help="Maximum number of records to retain.")
    parser.add_argument(
        "--exclude-filter-recommended-audit",
        type=Path,
        nargs="+",
        help=(
            "Optional rewrite audit JSONL. Records with any "
            "changes[*].filter_recommended=true are excluded after slicing."
        ),
    )
    parser.add_argument(
        "--require-successful-rewrite-audit",
        type=Path,
        nargs="+",
        help=(
            "Optional rewrite audit JSONL files. When supplied, retain only records "
            "that have at least one status=ok audit entry; this drops unresolved rewrite errors."
        ),
    )
    return parser.parse_args()


def _recommended_record_ids(audit_paths: list[Path]) -> set[str]:
    recommended: set[str] = set()
    for audit_path in audit_paths:
        if not audit_path.is_file():
            raise SystemExit(f"Rewrite audit does not exist: {audit_path}")
        with audit_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    audit = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid audit JSON at {audit_path}:{line_number}: {exc}") from exc
                record_id = str(audit.get("record_id") or "").strip()
                if not record_id:
                    continue
                if any(
                    isinstance(change, dict) and bool(change.get("filter_recommended"))
                    for change in (audit.get("changes") or [])
                ):
                    recommended.add(record_id)
    return recommended


def _successful_record_ids(audit_paths: list[Path]) -> set[str]:
    successful: set[str] = set()
    for audit_path in audit_paths:
        if not audit_path.is_file():
            raise SystemExit(f"Rewrite audit does not exist: {audit_path}")
        with audit_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    audit = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid audit JSON at {audit_path}:{line_number}: {exc}") from exc
                if str(audit.get("status") or "").strip().lower() == "ok":
                    record_id = str(audit.get("record_id") or "").strip()
                    if record_id:
                        successful.add(record_id)
    return successful


def _resolve_local_image_path(image_path: str, source_dir: Path) -> Path | None:
    """Resolve a nonstandard local image path stored by an earlier rewrite run."""

    candidate = Path(image_path)
    candidates = [candidate] if candidate.is_absolute() else [
        source_dir / candidate,
        Path(__file__).resolve().parents[2] / candidate,
        Path.cwd() / candidate,
    ]
    for item in candidates:
        if item.is_file():
            return item.resolve()
    return None


def _link_external_images(records: list[dict[str, Any]], source_dir: Path, output_dir: Path) -> int:
    """Make rewrite-added local images portable under the new dataset root.

    Normal trajectory images already use ``images/...`` and resolve through the
    shared image-library symlink.  Older terminal-image repair runs stored cache
    paths such as ``data/.../terminal_visual_evidence/...`` instead; LLaMA-
    Factory interprets those relative to its working directory and fails.  Link
    such files into ``extra_images/`` and rewrite only their record-local paths.
    """

    external_dir = output_dir / "extra_images"
    linked: dict[Path, str] = {}
    for record in records:
        images = record.get("images")
        if not isinstance(images, list):
            continue
        for index, raw_path in enumerate(images):
            image_path = str(raw_path or "")
            if not image_path or image_path.startswith("images/") or image_path.startswith(("http://", "https://")):
                continue
            source = _resolve_local_image_path(image_path, source_dir)
            if source is None:
                raise SystemExit(
                    "A nonstandard local image path cannot be resolved while packaging the subset: "
                    f"{image_path}"
                )
            relative = linked.get(source)
            if relative is None:
                digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
                destination = external_dir / f"{digest}{source.suffix.lower() or '.img'}"
                external_dir.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source)
                relative = destination.relative_to(output_dir).as_posix()
                linked[source] = relative
            images[index] = relative
    return len(linked)


def _has_unresolvable_external_image(record: dict[str, Any], source_dir: Path) -> bool:
    """Whether a record contains a rewrite-added image that can no longer be packaged."""

    for raw_path in record.get("images") or []:
        image_path = str(raw_path or "")
        if not image_path or image_path.startswith("images/") or image_path.startswith(("http://", "https://")):
            continue
        if _resolve_local_image_path(image_path, source_dir) is None:
            return True
    return False


def main() -> int:
    args = _parse_args()
    if args.offset < 0 or args.limit < 0:
        raise SystemExit("--offset and --limit must be non-negative.")
    for input_path in args.input:
        if not input_path.is_file():
            raise SystemExit(f"Input does not exist: {input_path}")
    if args.output_dir.exists():
        raise SystemExit(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new --output-dir so an existing training dataset is never overwritten."
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for input_path in args.input:
        with input_path.open("r", encoding="utf-8") as handle:
            part: Any = json.load(handle)
        if not isinstance(part, list):
            raise SystemExit(f"Expected a JSON-array ShareGPT dataset: {input_path}")
        for record in part:
            if not isinstance(record, dict):
                raise SystemExit(f"Non-object record in {input_path}")
            record_id = str(record.get("id") or "").strip()
            if not record_id:
                raise SystemExit(f"Record without id in {input_path}")
            if record_id in seen_ids:
                raise SystemExit(f"Duplicate record id across inputs: {record_id}")
            seen_ids.add(record_id)
            records.append(record)

    sliced_records = records[args.offset : args.offset + args.limit]
    recommended_ids = (
        _recommended_record_ids(args.exclude_filter_recommended_audit)
        if args.exclude_filter_recommended_audit
        else set()
    )
    successful_ids = (
        _successful_record_ids(args.require_successful_rewrite_audit)
        if args.require_successful_rewrite_audit
        else None
    )
    subset_before_missing_image_filter = [
        record
        for record in sliced_records
        if str(record.get("id") or "").strip() not in recommended_ids
    ]
    excluded_recommended = len(sliced_records) - len(subset_before_missing_image_filter)
    subset_before_success_filter = subset_before_missing_image_filter
    if successful_ids is not None:
        subset_before_missing_image_filter = [
            record
            for record in subset_before_success_filter
            if str(record.get("id") or "").strip() in successful_ids
        ]
    excluded_unsuccessful_rewrites = len(subset_before_success_filter) - len(subset_before_missing_image_filter)
    source_dir = args.input[0].parent
    subset = [
        record
        for record in subset_before_missing_image_filter
        if not _has_unresolvable_external_image(record, source_dir)
    ]
    excluded_unavailable_image_records = len(subset_before_missing_image_filter) - len(subset)
    args.output_dir.mkdir(parents=True)
    linked_external_images = _link_external_images(subset, source_dir, args.output_dir)
    output_json = args.output_dir / "trajectories_sharegpt.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle, ensure_ascii=False)
        handle.write("\n")

    source_info = source_dir / "dataset_info.json"
    if source_info.is_file():
        shutil.copy2(source_info, args.output_dir / "dataset_info.json")

    source_images = source_dir / "images"
    if source_images.exists() or source_images.is_symlink():
        image_link = args.output_dir / "images"
        image_link.symlink_to(source_images.resolve())

    manifest = {
        "sources": [str(path.resolve()) for path in args.input],
        "source_total_records": len(records),
        "offset": args.offset,
        "limit": args.limit,
        "selected_before_rewrite_recommendations": len(sliced_records),
        "rewrite_filter_recommendation_audits": (
            [str(path.resolve()) for path in args.exclude_filter_recommended_audit]
            if args.exclude_filter_recommended_audit
            else None
        ),
        "successful_rewrite_audits": (
            [str(path.resolve()) for path in args.require_successful_rewrite_audit]
            if args.require_successful_rewrite_audit
            else None
        ),
        "excluded_filter_recommended_records": excluded_recommended,
        "excluded_unsuccessful_rewrite_records": excluded_unsuccessful_rewrites,
        "excluded_unavailable_external_image_records": excluded_unavailable_image_records,
        "linked_external_images": linked_external_images,
        "written_records": len(subset),
        "output": str(output_json.resolve()),
        "images": str((args.output_dir / "images").resolve()) if (args.output_dir / "images").exists() else None,
    }
    with (args.output_dir / "subset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
