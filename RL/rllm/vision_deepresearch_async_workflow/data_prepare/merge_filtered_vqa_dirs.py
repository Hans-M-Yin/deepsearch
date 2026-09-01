#!/usr/bin/env python3
"""Merge filtered VQA directories without copying their image files.

The filtered VQA directories contain ``questions.jsonl`` and ``samples.jsonl``
plus intermediate filtering artifacts.  This utility creates a new, compact
VQA directory containing only the two JSONL files and a manifest.  Image
references are intentionally left untouched, so local image paths continue to
point into the original directories' image caches.

Each source must be passed as ``PREFIX=PATH``.  ``question_id`` is rewritten to
``PREFIX__<original-question-id>`` because question IDs are local to a VQA run
and can collide across runs.  ``sample_id`` is retained: it is already derived
from the path ID and is globally unique in the two current inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def _parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Source must have the form PREFIX=PATH, got: {raw!r}")
    prefix, raw_path = raw.split("=", 1)
    prefix = prefix.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid source prefix {prefix!r}; use letters, digits, '_' or '-'")
    if not path.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {path}")
    for filename in ("questions.jsonl", "samples.jsonl"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing {filename} under source directory: {path}")
    return prefix, path


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _merge_questions(source_prefix: str, source_dir: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in _iter_jsonl(source_dir / "questions.jsonl"):
        old_id = str(record.get("question_id") or record.get("id") or "")
        if not old_id:
            raise ValueError(f"Question record in {source_dir} has no question_id/id")
        updated = dict(record)
        updated["question_id"] = f"{source_prefix}__{old_id}"
        # Keep the original local ID and source label for later audits.  These
        # fields are harmless to the VQA converter and avoid losing provenance.
        updated["source_question_id"] = old_id
        updated["source_dataset"] = source_prefix
        merged.append(updated)
    return merged


def _merge_samples(source_prefix: str, source_dir: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in _iter_jsonl(source_dir / "samples.jsonl"):
        updated = dict(record)
        updated["source_dataset"] = source_prefix
        merged.append(updated)
    return merged


def _check_unique(records: Iterable[dict[str, Any]], field: str) -> None:
    seen: set[str] = set()
    for index, record in enumerate(records):
        value = record.get(field)
        if value is None:
            continue
        value = str(value)
        if value in seen:
            raise ValueError(f"Duplicate {field} after merge at record {index}: {value}")
        seen.add(value)


def merge_sources(sources: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}; "
            "choose another path or remove it explicitly before rerunning"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)

    try:
        all_questions: list[dict[str, Any]] = []
        all_samples: list[dict[str, Any]] = []
        source_stats: list[dict[str, Any]] = []
        for prefix, source_dir in sources:
            questions = _merge_questions(prefix, source_dir)
            samples = _merge_samples(prefix, source_dir)
            all_questions.extend(questions)
            all_samples.extend(samples)
            source_stats.append(
                {
                    "prefix": prefix,
                    "source_dir": str(source_dir),
                    "questions": len(questions),
                    "samples": len(samples),
                }
            )

        _check_unique(all_questions, "question_id")
        _check_unique(all_questions, "sample_id")
        _check_unique(all_samples, "sample_id")

        questions_count = _write_jsonl(staging_dir / "questions.jsonl", all_questions)
        samples_count = _write_jsonl(staging_dir / "samples.jsonl", all_samples)
        manifest = {
            "format": "merged_filtered_vqa",
            "image_policy": "preserve_source_paths",
            "question_id_policy": "<source_prefix>__<original_question_id>",
            "sample_id_policy": "preserve_source_sample_id",
            "sources": source_stats,
            "total_questions": questions_count,
            "total_samples": samples_count,
        }
        (staging_dir / "merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge filtered VQA directories while preserving source image paths"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="PREFIX=DIR",
        help="Source filtered VQA directory; repeat once per source",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [_parse_source(raw) for raw in args.source]
    output_dir = args.output_dir.expanduser().resolve()
    manifest = merge_sources(sources, output_dir)
    print(f"Merged {manifest['total_questions']} questions and {manifest['total_samples']} samples")
    print(f"Output: {output_dir}")
    print("Images were not copied; source image paths were preserved.")
    for source in manifest["sources"]:
        print(f"  {source['prefix']}: {source['questions']} questions, {source['samples']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
