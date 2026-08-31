#!/usr/bin/env python3
"""Filter ShareGPT trajectories with the RL evidence-consistency judge.

The input is a ShareGPT-style JSONL file (a JSON array is also accepted for
convenience).  Each record must retain ``sample_id`` so its question can be
joined to ``--vqa-dir/samples.jsonl``.  The latter provides the construction
hops and supporting facts used by the same v3 evidence-consistency prompt as
``filter_rl_vqa_dataset.py``.

Only records judged ``reject`` are removed. ``accept`` and ``review`` are kept.
Judge/API errors are kept as well, so a transient infrastructure failure never
silently removes training data. Every decision is recorded in ``--audit-jsonl``.

Examples:

    python -u synthesis/post_process/filter_sharegpt_question_consistency.py \
      --input data/sharegpt_dataset_final/trajectories_sharegpt.json \
      --output data/sharegpt_dataset_consistency_filtered/trajectories_sharegpt.jsonl \
      --audit-jsonl data/sharegpt_dataset_consistency_filtered/audit.jsonl \
      --vqa-dir runs/.../vqa/0803_batch_1 \
      --model-alias gpt54_internal_azure --workers 24

    # Audit only; no filtered training-data file is written.
    python -u synthesis/post_process/filter_sharegpt_question_consistency.py \
      --input data/sharegpt_dataset_final/trajectories_sharegpt.json \
      --audit-jsonl synthesis/.ignore/sharegpt_consistency_dry_run.jsonl \
      --vqa-dir runs/.../vqa/0803_batch_1 \
      --model-alias gpt54_internal_azure --workers 24 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
from synthesis.post_process.filter_rl_vqa_dataset import (
    QUALITY_SYSTEM_PROMPT,
    _compact_hop_chain,
    _parse_quality_judge,
    _sample_by_id,
)


logger = logging.getLogger("filter_sharegpt_question_consistency")


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records, also accepting the repository's JSON-array files."""

    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield item


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_json_records(path))


def _record_key(record: dict[str, Any], index: int) -> str:
    for key in ("id", "sample_id", "question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"row_{index:08d}"


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def _question_from_record(record: dict[str, Any]) -> str:
    for key in ("question", "prompt", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages") or record.get("conversations") or []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role in {"user", "human"}:
                text = _text_from_content(message.get("content", message.get("value")))
                if text:
                    return text
    return ""


def _prompt(record: dict[str, Any], sample: dict[str, Any], question: str) -> str:
    path = sample.get("path") or {}
    trajectory = path.get("trajectory") or {}
    hop_chain = _compact_hop_chain(sample)
    lines: list[str] = []
    for position, hop in enumerate(hop_chain):
        lines.append(
            "Hop {position}:\n"
            "  source: {source}\n"
            "  relation: {relation}\n"
            "  statement: {statement}\n"
            "  target: {target}\n"
            "  supporting_facts: {facts}".format(
                position=position,
                source=hop.get("source") or "",
                relation=hop.get("relation") or "",
                statement=hop.get("statement") or "(no statement provided)",
                target=hop.get("target") or "",
                facts=json.dumps(hop.get("supporting_facts") or [], ensure_ascii=False),
            )
        )
    if not lines:
        lines.append("(construction hop chain is missing)")
    return (
        "Audit the following candidate question against its construction evidence.\n"
        "Apply the system decision policy exactly; this task is about factual and "
        "relational consistency, not generic question difficulty.\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Construction hop chain and intermediate statements]\n"
        + "\n\n".join(lines)
        + "\n\n[Path summary]\n"
        + json.dumps(
            {
                "node_types": path.get("node_types") or [],
                "hop_count": trajectory.get("hop_count"),
                "modality_sequence": trajectory.get("modality_sequence") or [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _fingerprint(record: dict[str, Any], question: str) -> str:
    payload = {
        "id": record.get("id"),
        "sample_id": record.get("sample_id"),
        "question_id": record.get("question_id"),
        "question": question,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _load_audit(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("record_key") and item.get("fingerprint"):
            results[str(item["record_key"])] = item
    return results


def _judge_one(
    record: dict[str, Any], *, index: int, sample: dict[str, Any], model_alias: str
) -> dict[str, Any]:
    record_key = _record_key(record, index)
    question = _question_from_record(record)
    audit: dict[str, Any] = {
        "record_key": record_key,
        "source_index": index,
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "fingerprint": _fingerprint(record, question),
        "question": question,
        "model_alias": model_alias,
        "system_prompt_sha256": hashlib.sha256(QUALITY_SYSTEM_PROMPT.encode()).hexdigest(),
    }
    if not question:
        return {
            **audit,
            "parse_ok": False,
            "decision": "error",
            "keep": True,
            "reason": "missing_question",
        }
    if not sample:
        return {
            **audit,
            "parse_ok": False,
            "decision": "error",
            "keep": True,
            "reason": "sample_id_not_found_in_vqa_dir",
        }
    try:
        response = LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=QUALITY_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=_prompt(record, sample, question)),
                ],
                metadata={"trace_label": "sharegpt_question_consistency_judge"},
            )
        )
        parsed = _parse_quality_judge(str(response.content or ""))
        decision = str(parsed.get("decision") or "reject")
        return {
            **audit,
            **parsed,
            "keep": decision in {"accept", "review"} or not parsed.get("parse_ok", False),
            "hop_chain": _compact_hop_chain(sample),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **audit,
            "parse_ok": False,
            "decision": "error",
            "keep": True,
            "reason": f"judge_error:{type(exc).__name__}",
            "error": str(exc),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ShareGPT JSONL or JSON-array input.")
    parser.add_argument("--output", type=Path, help="Filtered ShareGPT JSONL output.")
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--vqa-dir", type=Path, required=True, help="VQA directory containing samples.jsonl.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Judge and write audit only; do not write --output.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        parser.error("--offset and --limit must be non-negative")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    return args


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s")
    input_path = args.input.expanduser().resolve()
    audit_path = args.audit_jsonl.expanduser().resolve()
    vqa_dir = args.vqa_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else None
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    samples_path = vqa_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    records = list(_iter_json_records(input_path))
    selected = records[args.offset : None if args.limit is None else args.offset + args.limit]
    samples_by_id = _sample_by_id(_load_jsonl(samples_path))
    prior = _load_audit(audit_path) if args.resume else {}

    tasks: list[tuple[int, dict[str, Any], str]] = []
    results: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(selected, start=args.offset):
        key = _record_key(record, index)
        question = _question_from_record(record)
        fingerprint = _fingerprint(record, question)
        cached = prior.get(key)
        if cached and cached.get("fingerprint") == fingerprint:
            results[index] = cached
        else:
            tasks.append((index, record, key))

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as audit_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _judge_one,
                    record,
                    index=index,
                    sample=samples_by_id.get(str(record.get("sample_id")), {}),
                    model_alias=args.model_alias,
                ): index
                for index, record, _ in tasks
            }
            with tqdm(total=len(selected), initial=len(results), desc="Judging ShareGPT", unit="trajectory") as progress:
                for future in as_completed(futures):
                    index = futures[future]
                    result = future.result()
                    results[index] = result
                    audit_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    audit_handle.flush()
                    progress.update(1)

    ordered = [(index, results[index]) for index in range(args.offset, args.offset + len(selected))]
    counts = Counter(result.get("decision", "error") for _, result in ordered)
    kept_indices = {index for index, result in ordered if result.get("keep", True)}
    if not args.dry_run:
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(selected, start=args.offset):
                if index in kept_indices:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input": str(input_path),
        "vqa_dir": str(vqa_dir),
        "selected": len(selected),
        "resumed": len(selected) - len(tasks),
        "decisions": dict(counts),
        "kept": len(kept_indices),
        "filtered": len(selected) - len(kept_indices),
        "dry_run": args.dry_run,
        "audit_jsonl": str(audit_path),
        "output": None if args.dry_run else str(output_path),
    }
    logger.info("ShareGPT consistency filter complete: %s", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
