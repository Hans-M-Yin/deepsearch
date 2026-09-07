#!/usr/bin/env python3
"""Build a balanced multimodal VQA benchmark from a ``run_batch`` output.

The input is a VQA run directory containing the normal ``run_batch`` files
(``questions.jsonl`` and ``samples.jsonl``).  Samples are classified from
``samples.jsonl.path.node_types``:

* ``image-start``: exactly one image node, at the first path position;
* ``image-end``: exactly one image node, at the last path position;
* ``image-interleave``: every other path containing an image node (including
  paths with an image in the middle or images at both ends).

By default the script targets 300 samples per category, evenly split across
2-, 3-, and 4-hop paths (100 per category/hop stratum).  Every structurally
Candidates in each stratum are shuffled and sent through the evidence-
consistency LLM judge used by ``filter_rl_vqa_dataset.py`` only until that
stratum's quota is full.  Explicit judge rejects and judge parse/errors are
excluded and replaced by more candidates; ``review`` is retained, matching
the existing filter's policy.  If a stratum is too small or has too many
quality failures, all of its candidates are evaluated and the shortfall is
reported.

The script is resumable.  ``quality_judge_details.jsonl`` is an append-only
cache, so rerunning the same command only evaluates candidates that do not
already have a quality result.  The source VQA directory is never modified.

Pass ``--emit-index path/to/question_ids.txt`` to exclude the listed
``question_id`` values before quality judging and sampling.  Blank lines in
the index are ignored.

Example::

    python -m synthesis.benchmarks.build_vqa_benchmark \
        --vqa-dir runs/.../vqa/0803_batch_1 \
        --output-dir runs/.../vqa/0803_batch_1/benchmark_900 \
        --quality-judge-model-alias gpt54_internal_azure \
        --quality-judge-workers 16 \
        --seed 42

Use ``--allow-incomplete`` only when a partial benchmark is useful.  Without
it, a shortfall in any category/hop stratum leaves diagnostics in the output
directory and exits with status 2 instead of publishing an incomplete VQA
directory as if it met the requested contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.post_process.filter_rl_vqa_dataset import (  # noqa: E402
    QUALITY_SYSTEM_PROMPT,
    _compact_task_record,
    _run_quality_judge,
)


LOGGER = logging.getLogger("build_vqa_benchmark")
CATEGORIES = ("image-start", "image-end", "image-interleave")
DEFAULT_HOP_COUNTS = (2, 3, 4)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _jsonl_write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _normalise_node_types(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    return [str(item).strip().lower() for item in values if str(item).strip()]


def classify_image_path(node_types: Any) -> str | None:
    """Classify an image-containing path using its ordered node types.

    A path with no image is outside this benchmark.  A one-node image path is
    deliberately treated as interleave because it is simultaneously at both
    boundaries; normal benchmark candidates have at least two hops anyway.
    """

    normalised = _normalise_node_types(node_types)
    image_positions = [
        index for index, node_type in enumerate(normalised) if node_type == "image"
    ]
    if not image_positions:
        return None
    if len(image_positions) == 1 and len(normalised) > 1:
        if image_positions[0] == 0:
            return "image-start"
        if image_positions[0] == len(normalised) - 1:
            return "image-end"
    return "image-interleave"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_hop_count(sample: dict[str, Any]) -> int | None:
    """Read hop count with fallbacks for older compact VQA records."""

    path = sample.get("path") or {}
    trajectory = path.get("trajectory") or {}
    for value in (
        trajectory.get("hop_count"),
        path.get("hop_count"),
        sample.get("hop_count"),
    ):
        hop_count = _int_or_none(value)
        if hop_count is not None:
            return hop_count

    edge_ids = path.get("edge_ids") or []
    if isinstance(edge_ids, (list, tuple)) and edge_ids:
        return len(edge_ids)
    node_types = _normalise_node_types(path.get("node_types"))
    if len(node_types) > 1:
        return len(node_types) - 1
    return None


def _sample_path(sample: dict[str, Any]) -> dict[str, Any]:
    path = sample.get("path")
    return path if isinstance(path, dict) else {}


def _source_identity(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path)}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_emit_index(path: Path | None) -> set[str]:
    """Load question IDs that must be excluded from benchmark construction."""

    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Missing --emit-index file: {path}")
    question_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            question_id = line.strip()
            if question_id:
                question_ids.add(question_id)
    return question_ids


def _question_text_for_duplicate(record: dict[str, Any]) -> str:
    for key in ("final_question", "question", "enhanced_question", "polished_question", "draft_question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value.casefold()).strip()
    return ""


def _index_samples(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    by_sample_id: dict[str, dict[str, Any]] = {}
    by_path_id: dict[str, dict[str, Any]] = {}
    duplicate_sample_ids: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "").strip()
        path_id = str(_sample_path(sample).get("path_id") or "").strip()
        if sample_id:
            if sample_id in by_sample_id:
                duplicate_sample_ids.add(sample_id)
            else:
                by_sample_id[sample_id] = sample
        if path_id and path_id not in by_path_id:
            by_path_id[path_id] = sample
    return by_sample_id, by_path_id, duplicate_sample_ids


def _match_sample(
    question: dict[str, Any],
    *,
    by_sample_id: dict[str, dict[str, Any]],
    by_path_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    sample_id = str(question.get("sample_id") or "").strip()
    if sample_id and sample_id in by_sample_id:
        return by_sample_id[sample_id]
    path_id = str(question.get("path_id") or "").strip()
    if path_id and path_id in by_path_id:
        return by_path_id[path_id]
    return None


def _target_counts(total: int, hops: tuple[int, ...]) -> dict[int, int]:
    base, remainder = divmod(total, len(hops))
    return {
        hop_count: base + int(index < remainder)
        for index, hop_count in enumerate(hops)
    }


def _quality_passes(
    result: dict[str, Any] | None,
    keep_decisions: set[str],
) -> bool:
    if not result or not result.get("parse_ok"):
        return False
    return str(result.get("decision") or "").strip().lower() in keep_decisions


def _stratum_key(category: str, hop_count: int) -> str:
    return f"{category}/{hop_count}hop"


def _judge_and_select_incrementally(
    *,
    all_records: list[dict[str, Any]],
    compact_records: list[dict[str, Any]],
    quality_samples_by_id: dict[str, dict[str, Any]],
    quality_cache_path: Path,
    target_by_hop: dict[int, int],
    hop_counts: tuple[int, ...],
    seed: int,
    keep_decisions: set[str],
    model_alias: str,
    max_tokens: int,
    workers: int,
    batch_size: int,
    resume: bool,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    Counter[str],
    Counter[str],
    dict[str, int],
    dict[str, int],
]:
    """Judge random candidates until each category/hop quota is filled.

    The returned counters are, respectively, global quality decisions and
    per-stratum judged counts, followed by per-stratum quality-pass counts and
    selected counts.  Candidates after a filled quota are intentionally left
    unjudged and marked in the manifest.
    """

    compact_by_id = {str(record["id"]): record for record in compact_records}
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        if record["base_rejection_reasons"]:
            continue
        key = _stratum_key(str(record["category"]), int(record["hop_count"]))
        pools[key].append(record)

    quality_by_id: dict[str, dict[str, Any]] = {}
    decision_counts: Counter[str] = Counter()
    judged_by_stratum: Counter[str] = Counter()
    quality_pass_by_stratum: dict[str, int] = {}
    selected_by_stratum: dict[str, int] = {}
    selected_records: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for category in CATEGORIES:
        for hop_count in hop_counts:
            key = _stratum_key(category, hop_count)
            pool = pools.get(key, [])
            rng.shuffle(pool)
            quota = target_by_hop[hop_count]
            selected_in_stratum: list[dict[str, Any]] = []
            quality_pass_count = 0

            if quota <= 0:
                for record in pool:
                    record["selection_rejection_reasons"].append(
                        "zero_stratum_quota_not_judged"
                    )
                quality_pass_by_stratum[key] = 0
                selected_by_stratum[key] = 0
                continue

            for batch_start in range(0, len(pool), batch_size):
                batch = pool[batch_start : batch_start + batch_size]
                batch_compact = [compact_by_id[str(record["id"])] for record in batch]
                batch_results = _run_quality_judge(
                    batch_compact,
                    quality_samples_by_id,
                    quality_cache_path,
                    model_alias=model_alias,
                    max_tokens=max_tokens,
                    workers=workers,
                    resume=resume,
                    retry_failures=True,
                )

                for record in batch:
                    result = batch_results.get(str(record["id"]))
                    quality_by_id[str(record["id"])] = result or {}
                    record["quality_judge"] = result
                    judged_by_stratum[key] += 1
                    decision = str((result or {}).get("decision") or "error").lower()
                    decision_counts[decision] += 1
                    if _quality_passes(result, keep_decisions):
                        record["quality_pass"] = True
                        quality_pass_count += 1
                    elif not result or not result.get("parse_ok"):
                        record["selection_rejection_reasons"].append(
                            "quality_judge_error_or_parse_failure"
                        )
                    else:
                        record["selection_rejection_reasons"].append(
                            f"quality_decision_{str(result.get('decision')).lower()}"
                        )

                remaining = quota - len(selected_in_stratum)
                passing_batch = [record for record in batch if record["quality_pass"]]
                selected_batch = passing_batch[:remaining]
                selected_in_stratum.extend(selected_batch)
                for record in selected_batch:
                    record["selected"] = True
                for record in passing_batch[remaining:]:
                    record["selection_rejection_reasons"].append(
                        "stratum_quota_not_selected"
                    )

                if len(selected_in_stratum) >= quota:
                    for record in pool[batch_start + len(batch) :]:
                        record["selection_rejection_reasons"].append(
                            "quality_quota_filled_without_judging"
                        )
                    break

            quality_pass_by_stratum[key] = quality_pass_count
            selected_by_stratum[key] = len(selected_in_stratum)
            selected_records.extend(selected_in_stratum)

    return (
        quality_by_id,
        selected_records,
        decision_counts,
        judged_by_stratum,
        quality_pass_by_stratum,
        selected_by_stratum,
    )


def _benchmark_signature(
    *,
    vqa_dir: Path,
    questions_path: Path,
    samples_path: Path,
    category_count: int,
    hop_counts: tuple[int, ...],
    seed: int,
    keep_decisions: set[str],
    quality_judge_model_alias: str,
    quality_judge_max_tokens: int,
    quality_judge_batch_size: int,
    emit_index_path: Path | None,
) -> str:
    return _sha256(
        {
            "vqa_dir": str(vqa_dir),
            "questions": _source_identity(questions_path),
            "samples": _source_identity(samples_path),
            "category_count": category_count,
            "hop_counts": list(hop_counts),
            "seed": seed,
            "keep_decisions": sorted(keep_decisions),
            "quality_judge_model_alias": quality_judge_model_alias,
            "quality_judge_max_tokens": quality_judge_max_tokens,
            "quality_judge_batch_size": quality_judge_batch_size,
            "emit_index": _source_identity(emit_index_path) if emit_index_path else None,
            "quality_system_prompt_sha256": _sha256(QUALITY_SYSTEM_PROMPT),
        }
    )


def _prepare_output_dir(
    output_dir: Path,
    *,
    signature: str,
    resume: bool,
    force_rerun: bool,
) -> None:
    if force_rerun:
        if output_dir.is_symlink():
            raise ValueError("Refusing to force-rerun through a symlink output directory")
        if output_dir in {Path("/"), ROOT}:
            raise ValueError(f"Refusing to delete unsafe output directory: {output_dir}")
        if output_dir.exists():
            LOGGER.warning("Force rerun: deleting output directory %s", output_dir)
            shutil.rmtree(output_dir)
        resume = False

    state_path = output_dir / "benchmark_state.json"
    previous = _read_json(state_path)
    if resume and previous.get("signature") and previous["signature"] != signature:
        raise RuntimeError(
            f"Benchmark resume signature mismatch in {output_dir}. Use a new "
            "--output-dir or --force-rerun when changing input/parameters."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        state_path,
        {
            "signature": signature,
            "status": "running",
            "updated_at": _utc_now(),
        },
    )


def _build_records(
    questions: list[dict[str, Any]],
    *,
    by_sample_id: dict[str, dict[str, Any]],
    by_path_id: dict[str, dict[str, Any]],
    duplicate_sample_ids: set[str],
    hop_counts: set[int],
    emit_question_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Create compact records, structural candidate records, and all manifests."""

    compact_by_id: dict[str, dict[str, Any]] = {}
    quality_samples_by_id: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    for source_index, question in enumerate(questions):
        sample = _match_sample(question, by_sample_id=by_sample_id, by_path_id=by_path_id)
        sample_for_compact = sample or {}
        compact = _compact_task_record(question, sample_for_compact, source_index)
        sample_id = str(compact.get("sample_id") or "").strip()
        if not sample_id and sample:
            sample_id = str(sample.get("sample_id") or "").strip()
            compact["sample_id"] = sample_id or None
        path = _sample_path(sample_for_compact)
        node_types = _normalise_node_types(path.get("node_types"))
        category = classify_image_path(node_types)
        hop_count = extract_hop_count(sample_for_compact)
        reasons: list[str] = []

        question_id = str(question.get("question_id") or "").strip()
        if question_id in emit_question_ids:
            reasons.append("question_id_in_emit_index")

        if question.get("status") != "verified":
            reasons.append("source_status_not_verified")
        if not compact.get("question"):
            reasons.append("empty_question")
        if not compact.get("answer"):
            reasons.append("empty_answer")
        if sample is None:
            reasons.append("matching_sample_not_found")
        elif sample_id in duplicate_sample_ids:
            reasons.append("duplicate_sample_id")
        if not compact.get("sample_id"):
            reasons.append("missing_sample_id")

        case_id = str(compact["id"])
        duplicate_question = _question_text_for_duplicate(question)
        if case_id in seen_ids:
            reasons.append("duplicate_id")
        if duplicate_question and duplicate_question in seen_questions:
            reasons.append("duplicate_question")

        if category is None:
            reasons.append("path_has_no_image_node")
        elif hop_count not in hop_counts:
            reasons.append("hop_count_not_requested")

        # A duplicate ID/question is rejected after the first occurrence, just
        # like the existing RL filter.  Only a record without base errors can
        # reach the LLM quality judge.
        if not reasons:
            seen_ids.add(case_id)
            if duplicate_question:
                seen_questions.add(duplicate_question)

        item = {
            **compact,
            "category": category,
            "hop_count": hop_count,
            "base_rejection_reasons": reasons,
            "quality_judge": None,
            "quality_pass": False,
            "selected": False,
            "selection_rejection_reasons": [],
        }
        records.append(item)
        if not reasons:
            compact_by_id[case_id] = compact
            if sample_id:
                quality_samples_by_id[sample_id] = sample_for_compact

    return list(compact_by_id.values()), quality_samples_by_id, records


def _write_selected_vqa_dir(
    output_dir: Path,
    *,
    source_questions: list[dict[str, Any]],
    source_samples: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    report: dict[str, Any],
    source_metadata: dict[str, Any],
    source_vqa_dir: Path,
) -> None:
    selected_questions: list[dict[str, Any]] = []
    selected_samples: list[dict[str, Any]] = []
    selected_sample_ids: set[str] = set()
    sample_by_id = {
        str(sample.get("sample_id")): sample
        for sample in source_samples
        if sample.get("sample_id") is not None
    }
    sample_by_path_id = {
        str(_sample_path(sample).get("path_id")): sample
        for sample in source_samples
        if _sample_path(sample).get("path_id") is not None
    }

    for record in sorted(selected_records, key=lambda item: int(item["source_index"])):
        selected_questions.append(dict(source_questions[int(record["source_index"])]))
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id and sample_id not in selected_sample_ids:
            sample = sample_by_id.get(sample_id)
            if sample is None:
                path_id = str(record.get("path_id") or "").strip()
                sample = sample_by_path_id.get(path_id)
            if sample is not None:
                actual_sample_id = str(sample.get("sample_id") or sample_id)
                if actual_sample_id not in selected_sample_ids:
                    selected_samples.append(dict(sample))
                    selected_sample_ids.add(actual_sample_id)

    _jsonl_write(output_dir / "questions.jsonl", selected_questions)
    _jsonl_write(output_dir / "samples.jsonl", selected_samples)
    with (output_dir / "source_indices.txt").open("w", encoding="utf-8") as handle:
        for record in sorted(selected_records, key=lambda item: int(item["source_index"])):
            handle.write(f"{record['source_index']}\n")

    metadata = dict(source_metadata)
    metadata["benchmark"] = {
        "source_vqa_dir": str(source_vqa_dir),
        "report": str(output_dir / "report.json"),
        "created_at": _utc_now(),
    }
    _json_write(output_dir / "question_metadata.json", metadata)
    _json_write(output_dir / "selection_report.json", report)


def _write_diagnostics(
    output_dir: Path,
    *,
    all_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    _jsonl_write(output_dir / "candidate_manifest.jsonl", all_records)
    selected_ids = {str(record["id"]) for record in selected_records}
    _jsonl_write(
        output_dir / "selection_manifest.jsonl",
        [record for record in all_records if str(record["id"]) in selected_ids],
    )
    _json_write(output_dir / "report.json", report)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--emit-index",
        type=Path,
        default=None,
        help="TXT file with one question_id per line; matching samples are skipped.",
    )
    parser.add_argument(
        "--category-count",
        type=int,
        default=300,
        help="Target number of selected samples in each image category (default: 300).",
    )
    parser.add_argument(
        "--hop-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_HOP_COUNTS),
        help="Requested hop counts, split as evenly as possible (default: 2 3 4).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality-judge-model-alias",
        default=os.environ.get("RL_QUALITY_JUDGE_MODEL_ALIAS", "gpt54_internal_azure"),
        help="Model alias used by filter_rl_vqa_dataset's quality judge.",
    )
    parser.add_argument("--quality-judge-max-tokens", type=int, default=1024)
    parser.add_argument("--quality-judge-workers", type=int, default=8)
    parser.add_argument(
        "--quality-judge-batch-size",
        type=int,
        default=32,
        help=(
            "Maximum number of new candidates judged per batch in each stratum "
            "(default: 32)."
        ),
    )
    parser.add_argument(
        "--keep-decisions",
        nargs="+",
        choices=("accept", "review"),
        default=["accept", "review"],
        help="Quality decisions eligible for selection (default: accept review).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse quality results already present in the output directory (default: true).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Delete the output directory and rerun all quality judgments.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Publish a partial VQA directory when one or more strata are short.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[int, tuple[int, ...], set[str]]:
    if args.category_count <= 0:
        raise ValueError("--category-count must be positive")
    if args.quality_judge_workers <= 0:
        raise ValueError("--quality-judge-workers must be positive")
    if args.quality_judge_batch_size <= 0:
        raise ValueError("--quality-judge-batch-size must be positive")
    if args.quality_judge_max_tokens <= 0:
        raise ValueError("--quality-judge-max-tokens must be positive")
    hop_counts = tuple(dict.fromkeys(int(value) for value in args.hop_counts))
    if not hop_counts or any(value <= 0 for value in hop_counts):
        raise ValueError("--hop-counts must contain positive integers")
    keep_decisions = {str(value).lower() for value in args.keep_decisions}
    if not keep_decisions:
        raise ValueError("--keep-decisions must not be empty")
    return args.category_count, hop_counts, keep_decisions


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    category_count, hop_counts, keep_decisions = _validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
    )

    vqa_dir = args.vqa_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == vqa_dir:
        raise ValueError("--output-dir must be different from --vqa-dir")
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing {questions_path}")
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Missing {samples_path}; run_batch VQA directories must include samples.jsonl"
        )

    questions = _jsonl_load(questions_path)
    samples = _jsonl_load(samples_path)
    emit_index_path = args.emit_index.expanduser().resolve() if args.emit_index else None
    emit_question_ids = _load_emit_index(emit_index_path)
    by_sample_id, by_path_id, duplicate_sample_ids = _index_samples(samples)
    signature = _benchmark_signature(
        vqa_dir=vqa_dir,
        questions_path=questions_path,
        samples_path=samples_path,
        category_count=category_count,
        hop_counts=hop_counts,
        seed=args.seed,
        keep_decisions=keep_decisions,
        quality_judge_model_alias=args.quality_judge_model_alias,
        quality_judge_max_tokens=args.quality_judge_max_tokens,
        quality_judge_batch_size=args.quality_judge_batch_size,
        emit_index_path=emit_index_path,
    )
    _prepare_output_dir(
        output_dir,
        signature=signature,
        resume=args.resume,
        force_rerun=args.force_rerun,
    )

    compact_records, quality_samples_by_id, all_records = _build_records(
        questions,
        by_sample_id=by_sample_id,
        by_path_id=by_path_id,
        duplicate_sample_ids=duplicate_sample_ids,
        hop_counts=set(hop_counts),
        emit_question_ids=emit_question_ids,
    )
    # _build_records intentionally returns only base-valid records here.  The
    # quality helper looks up samples by compact.sample_id, matching the
    # existing filter's cache and audit schema.
    target_by_hop = _target_counts(category_count, hop_counts)
    quality_cache_path = output_dir / "quality_judge_details.jsonl"
    if not args.resume or args.force_rerun:
        # ``--no-resume`` means a clean quality audit even when the caller
        # intentionally reuses the same output directory.
        quality_cache_path.write_text("", encoding="utf-8")
    (
        quality_by_id,
        selected_records,
        decision_counts,
        judged_by_stratum,
        quality_pass_by_stratum,
        selected_by_stratum,
    ) = _judge_and_select_incrementally(
        all_records=all_records,
        compact_records=compact_records,
        quality_samples_by_id=quality_samples_by_id,
        quality_cache_path=quality_cache_path,
        target_by_hop=target_by_hop,
        hop_counts=hop_counts,
        seed=args.seed,
        keep_decisions=keep_decisions,
        model_alias=args.quality_judge_model_alias,
        max_tokens=args.quality_judge_max_tokens,
        workers=args.quality_judge_workers,
        batch_size=args.quality_judge_batch_size,
        resume=args.resume and not args.force_rerun,
    )

    shortfalls: dict[str, int] = {}
    for category in CATEGORIES:
        for hop_count in hop_counts:
            key = _stratum_key(category, hop_count)
            shortfalls[key] = target_by_hop[hop_count] - selected_by_stratum.get(key, 0)

    for record in all_records:
        if record["selected"]:
            continue
        if record["base_rejection_reasons"]:
            continue
        if not record["quality_judge"] and not record["selection_rejection_reasons"]:
            record["selection_rejection_reasons"].append(
                "quality_quota_filled_without_judging"
            )
        elif record["quality_pass"] and not record["selection_rejection_reasons"]:
            record["selection_rejection_reasons"].append("not_selected")

    status = "complete" if not any(shortfalls.values()) else "incomplete"
    report = {
        "status": status,
        "created_at": _utc_now(),
        "vqa_dir": str(vqa_dir),
        "output_dir": str(output_dir),
        "category_definitions": {
            "image-start": "exactly one image node at path position 0",
            "image-end": "exactly one image node at the final path position",
            "image-interleave": "all other image-containing paths",
        },
        "category_count_target": category_count,
        "hop_counts": list(hop_counts),
        "target_per_category_hop": target_by_hop,
        "seed": args.seed,
        "emit_index": {
            "path": str(emit_index_path) if emit_index_path else None,
            "question_id_count": len(emit_question_ids),
            "skipped_question_count": sum(
                "question_id_in_emit_index" in record["base_rejection_reasons"]
                for record in all_records
            ),
        },
        "keep_quality_decisions": sorted(keep_decisions),
        "quality_judge_model_alias": args.quality_judge_model_alias,
        "quality_judge_batch_size": args.quality_judge_batch_size,
        "quality_judge_system_prompt_sha256": _sha256(QUALITY_SYSTEM_PROMPT),
        "input_counts": {
            "questions": len(questions),
            "samples": len(samples),
            "base_valid_candidates": len(compact_records),
            "quality_judged": sum(judged_by_stratum.values()),
            "quality_not_judged_after_quota": len(compact_records) - sum(judged_by_stratum.values()),
            "quality_decisions": dict(decision_counts),
        },
        "structural_candidate_counts": dict(
            Counter(
                _stratum_key(str(record["category"]), int(record["hop_count"]))
                for record in all_records
                if not record["base_rejection_reasons"]
            )
        ),
        "base_rejection_reason_counts": dict(
            Counter(
                reason
                for record in all_records
                for reason in record["base_rejection_reasons"]
            )
        ),
        "quality_judged_counts": dict(judged_by_stratum),
        "quality_pass_counts": quality_pass_by_stratum,
        "selected_counts": dict(selected_by_stratum),
        "shortfall_counts": shortfalls,
        "selection_rejection_reason_counts": dict(
            Counter(
                reason
                for record in all_records
                for reason in record["selection_rejection_reasons"]
            )
        ),
        "selected_total": len(selected_records),
        "signature": signature,
        "paths": {
            "questions": str(output_dir / "questions.jsonl"),
            "samples": str(output_dir / "samples.jsonl"),
            "candidate_manifest": str(output_dir / "candidate_manifest.jsonl"),
            "selection_manifest": str(output_dir / "selection_manifest.jsonl"),
            "quality_judge_details": str(output_dir / "quality_judge_details.jsonl"),
            "source_indices": str(output_dir / "source_indices.txt"),
        },
    }

    _write_diagnostics(
        output_dir,
        all_records=all_records,
        selected_records=selected_records,
        report=report,
    )
    if status == "incomplete" and not args.allow_incomplete:
        _json_write(
            output_dir / "benchmark_state.json",
            {
                "signature": signature,
                "status": "incomplete",
                "shortfall_counts": shortfalls,
                "updated_at": _utc_now(),
            },
        )
        LOGGER.error(
            "Benchmark is incomplete; refusing to publish VQA files. Shortfalls: %s",
            json.dumps(shortfalls, ensure_ascii=False, sort_keys=True),
        )
        return 2

    source_metadata = _read_json(vqa_dir / "question_metadata.json")
    _write_selected_vqa_dir(
        output_dir,
        source_questions=questions,
        source_samples=samples,
        selected_records=selected_records,
        report=report,
        source_metadata=source_metadata,
        source_vqa_dir=vqa_dir,
    )
    _json_write(
        output_dir / "benchmark_state.json",
        {
            "signature": signature,
            "status": status,
            "selected_total": len(selected_records),
            "finished_at": _utc_now(),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
