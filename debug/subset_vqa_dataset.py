"""Create a VQA subset directory from selected sample/question identifiers.

Examples:
  python debug/subset_vqa_dataset.py \
    --vqa-dir runs/my_graph/vqa/0719_120000 \
    --list q_000001,q_000007,sample_path_abc \
    --new-vqa-dir /tmp/vqa_subset

  python debug/subset_vqa_dataset.py \
    --vqa-dir runs/my_graph/vqa/0719_120000 \
    --list-file keep_ids.txt \
    --new-vqa-dir runs/my_graph/vqa/0719_subset

The selector list is ordered. Each selector may be a question_id, sample_id,
path_id, or a 1-based row number from questions.jsonl. A JSON list, JSONL file,
or plain text file with one selector per line is accepted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _parse_selector_item(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("question_id", "sample_id", "path_id", "id", "index", "line"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        raise ValueError(f"selector object has no supported id field: {value!r}")
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty selector")
    return text


def _selectors_from_text(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError("JSON selector payload must be a list")
        return [_parse_selector_item(item) for item in payload]
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            for key in ("ids", "selectors", "items", "samples", "questions"):
                if isinstance(payload.get(key), list):
                    return [_parse_selector_item(item) for item in payload[key]]
        raise ValueError("JSON selector object must contain a list field such as ids/selectors/items")

    selectors: list[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            selectors.append(_parse_selector_item(json.loads(line)))
            continue
        for part in line.split(","):
            item = part.strip()
            if item:
                selectors.append(item)
    return selectors


def _load_selectors(raw_list: str | None, list_file: Path | None) -> list[str]:
    selectors: list[str] = []
    if raw_list:
        selectors.extend(_selectors_from_text(raw_list))
    if list_file is not None:
        selectors.extend(_selectors_from_text(list_file.read_text(encoding="utf-8")))
    if not selectors:
        raise ValueError("provide at least one selector via --list or --list-file")
    return selectors


def _question_row_key(index: int) -> str:
    return str(index + 1)


def _build_question_lookup(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(questions):
        keys = [
            _question_row_key(index),
            str(record.get("question_id") or "").strip(),
            str(record.get("sample_id") or "").strip(),
            str(record.get("path_id") or "").strip(),
        ]
        for key in keys:
            if key and key not in lookup:
                lookup[key] = record
    return lookup


def _renumber_question(record: dict[str, Any], index: int) -> dict[str, Any]:
    updated = dict(record)
    updated["question_id"] = f"q_{index + 1:06d}"
    return updated


def _copy_optional_sidecar(vqa_dir: Path, new_vqa_dir: Path, name: str) -> None:
    source = vqa_dir / name
    if source.exists() and source.is_file():
        shutil.copy2(source, new_vqa_dir / name)


def _write_subset_metadata(
    *,
    vqa_dir: Path,
    new_vqa_dir: Path,
    selectors: list[str],
    selected_questions: list[dict[str, Any]],
    missing: list[str],
    renumber_questions: bool,
) -> None:
    payload = {
        "created_at": _utc_now(),
        "source_vqa_dir": str(vqa_dir.resolve()),
        "new_vqa_dir": str(new_vqa_dir.resolve()),
        "requested_selectors": selectors,
        "missing_selectors": missing,
        "selected_count": len(selected_questions),
        "renumber_questions": renumber_questions,
        "selected": [
            {
                "question_id": record.get("question_id"),
                "sample_id": record.get("sample_id"),
                "path_id": record.get("path_id"),
            }
            for record in selected_questions
        ],
    }
    (new_vqa_dir / "subset_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_subset(
    *,
    vqa_dir: Path,
    selectors: list[str],
    new_vqa_dir: Path,
    allow_missing: bool,
    overwrite: bool,
    renumber_questions: bool,
    copy_metadata: bool,
) -> dict[str, Any]:
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    questions = _load_jsonl(questions_path)
    samples = _load_jsonl(samples_path)
    samples_by_id = {
        str(record.get("sample_id") or "").strip(): record
        for record in samples
        if str(record.get("sample_id") or "").strip()
    }
    question_lookup = _build_question_lookup(questions)

    selected_questions: list[dict[str, Any]] = []
    selected_samples: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_question_keys: set[str] = set()

    for selector in selectors:
        question = question_lookup.get(selector)
        if question is None:
            missing.append(selector)
            continue
        stable_key = str(question.get("question_id") or question.get("sample_id") or question.get("path_id") or selector)
        if stable_key in seen_question_keys:
            continue
        seen_question_keys.add(stable_key)
        sample_id = str(question.get("sample_id") or "").strip()
        sample = samples_by_id.get(sample_id)
        if sample is None:
            missing.append(f"{selector} (missing sample_id={sample_id})")
            continue
        selected_questions.append(dict(question))
        selected_samples.append(dict(sample))

    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise ValueError(f"{len(missing)} selector(s) could not be resolved: {preview}{suffix}")

    if new_vqa_dir.exists() and any(new_vqa_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"new-vqa-dir is not empty: {new_vqa_dir} (pass --overwrite to replace files)")
    new_vqa_dir.mkdir(parents=True, exist_ok=True)

    if renumber_questions:
        selected_questions = [
            _renumber_question(record, index)
            for index, record in enumerate(selected_questions)
        ]

    _write_jsonl(new_vqa_dir / "questions.jsonl", selected_questions)
    _write_jsonl(new_vqa_dir / "samples.jsonl", selected_samples)
    _write_subset_metadata(
        vqa_dir=vqa_dir,
        new_vqa_dir=new_vqa_dir,
        selectors=selectors,
        selected_questions=selected_questions,
        missing=missing,
        renumber_questions=renumber_questions,
    )
    if copy_metadata:
        _copy_optional_sidecar(vqa_dir, new_vqa_dir, "question_metadata.json")
        _copy_optional_sidecar(vqa_dir, new_vqa_dir, "sampler_state.json")

    return {
        "source_vqa_dir": str(vqa_dir),
        "new_vqa_dir": str(new_vqa_dir),
        "requested": len(selectors),
        "selected": len(selected_questions),
        "missing": missing,
        "questions_path": str(new_vqa_dir / "questions.jsonl"),
        "samples_path": str(new_vqa_dir / "samples.jsonl"),
        "manifest_path": str(new_vqa_dir / "subset_manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", type=Path, required=True, help="Source VQA directory containing questions.jsonl and samples.jsonl.")
    parser.add_argument("--list", dest="raw_list", default=None, help="Comma-separated selectors or a JSON list. Selectors can be question_id, sample_id, path_id, or 1-based row number.")
    parser.add_argument("--list-file", type=Path, default=None, help="Text/JSON/JSONL file containing selectors in desired output order.")
    parser.add_argument("--new-vqa-dir", type=Path, required=True, help="Destination VQA subset directory.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip selectors that cannot be resolved instead of failing.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty destination directory.")
    parser.add_argument("--renumber-questions", action="store_true", help="Rewrite question_id values as q_000001... in the subset questions.jsonl.")
    parser.add_argument("--no-copy-metadata", action="store_true", help="Do not copy optional question_metadata.json and sampler_state.json sidecars.")
    args = parser.parse_args()

    selectors = _load_selectors(args.raw_list, args.list_file)
    summary = build_subset(
        vqa_dir=args.vqa_dir.resolve(),
        selectors=selectors,
        new_vqa_dir=args.new_vqa_dir.resolve(),
        allow_missing=args.allow_missing,
        overwrite=args.overwrite,
        renumber_questions=args.renumber_questions,
        copy_metadata=not args.no_copy_metadata,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
