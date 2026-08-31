#!/usr/bin/env python3
"""Filter SFT trajectories that contain information leakage.

This is the first pass of the trajectory post-processing pipeline.  It does
not rewrite a trajectory and does not change any tool call or tool result.  A
model-worker alias judges whether the assistant appears to use information
that was unavailable at that point in the trajectory.

The script accepts either a ShareGPT-style JSON array or JSONL.  In dry-run
mode it calls the judge and prints a summary, but never writes a filtered
dataset.  In normal mode it writes an output dataset containing only records
whose judge decision is ``keep``.

Example:

    python synthesis/post_process/filter_trajectory_information_leakage.py \
        --input data/sharegpt_dataset_test_1/trajectories_sharegpt.json \
        --output synthesis/ignore/leakage_filtered.json \
        --audit-jsonl synthesis/ignore/leakage_audit.jsonl \
        --model-alias qwen3-32b \
        --dry-run

``--dry-run`` still performs model calls; it only prevents the filtered
dataset from being written.  The audit file is optional and is written only
when explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest


_POST_PROCESS_FIXED_REQUEST_ID = "3200636808"


def _model_worker_metadata(trace_label: str) -> dict[str, str]:
    """Attach the routing/cache metadata used by the SFT API tools."""

    return {
        "trace_label": trace_label,
        "session_id": _POST_PROCESS_FIXED_REQUEST_ID,
        "prompt_cache_key": _POST_PROCESS_FIXED_REQUEST_ID,
        "user_id": _POST_PROCESS_FIXED_REQUEST_ID,
        "x_tt_logid": _POST_PROCESS_FIXED_REQUEST_ID,
    }


JUDGE_SYSTEM_PROMPT = """You are a strict auditing model for multimodal agent trajectories.

Only check whether there is information leakage in the trajectory. Do not judge whether the answer is factually correct, and do not judge whether the tool usage was efficient.

Information leakage means that an assistant action or reasoning step uses information that did not come from the user’s question/image or from tool results returned earlier in the trajectory, or that the assistant’s reasoning prematurely anchors on a certain intermediate object, fails to obtain satisfactory evidence after multiple rounds of search-based verification, and then quietly skips that verification process in the end. In particular, flag the following cases:

1. An entity, event, page, answer, or fact is suddenly used in a query or stated as known without any prior grounding source. If an entity is directly inferred from visible image content, or introduced by an earlier search result, it does not count as leakage.
2. References to hidden supervision or future information, such as “the answer says,” “according to the reference answer,” “the provided answer,” “the ground truth,” or other wording that explicitly indicates the answer comes from outside the user question and prior tool observations.
3. Reasoning that uses a later tool result before that result has appeared, or that uses information from a later turn while explaining an earlier action.

4. Beyond the obvious forms of information leakage above, you must also check for reasoning-logic defects caused by implicit leakage. The typical pattern: even though the response never explicitly reveals the target answer, once a particular intermediate object appears, the subsequent reasoning anchors onto it; then, although multiple rounds of seemingly reasonable searches all fail to verify that the object is correct, the final answer completely ignores this evidentiary defect and quietly bypasses the critical verification step without any legitimate basis (such as explicit reasoning or a clearly stated hypothesis).
Judge it as leakage only when ALL of the following hold simultaneously:
    (a) the defect directly bears on the main reasoning path — i.e., it is precisely the link that supports the final answer, such that if this unverified step were removed, the answer would lose its support;
    (b) the value that the final answer commits to has no other independent grounding source anywhere in the trajectory (no tool result, no visible image content, and no explicit common-sense/geographic identification);
    (c) the jump from "unverified" to "concluded" involves no explicit reasoning, no stated hypothesis, and no acknowledgment of the uncertainty.
Conversely, if ANY ONE of the following holds, do NOT flag it as leakage, even when some verification step is incomplete:
    the defect belongs to a side, indirect, or supplementary line of reasoning that bears little on the main path and does not affect whether the final answer holds;
    before drawing the conclusion, the assistant has explicitly acknowledged the gap and stated a hypothesis or lowered its confidence (honesty about a weak link is a quality signal, not leakage);
    the object was introduced by an earlier search result or is visible in the image, and the reasoning merely uses common sense to recognize or exclude candidates.
Do not flag normal hypotheses, queries reasonably derived from the image/question, or candidates introduced by earlier search results as leakage. Only flag leakage when unsupported information is materially used to choose an action or justify a conclusion.

Return exactly one JSON object and no Markdown:
{
  "decision": "keep" | "filter",
  "leakage": true | false,
  "confidence": 0.0,
  "leakage_types": ["unsupported_entity" | "hidden_answer" | "future_information" | "other"],
  "turns": [integer],
  "evidence": [
    {"turn": integer, "quote": "short exact quote", "reason": "why this uses unavailable information"}
  ],
  "reason": "concise explanation"
}

Use decision=filter if and only if leakage=true. If there is no material leakage, use decision=keep.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ShareGPT JSON array or JSONL file.")
    parser.add_argument("--output", type=Path, help="Filtered output dataset. Required unless --dry-run is used.")
    parser.add_argument("--audit-jsonl", type=Path, help="Optional JSONL file containing one judge record per input.")
    parser.add_argument(
        "--model-alias",
        default=os.environ.get("POST_PROCESS_MODEL_ALIAS") or os.environ.get("TEXT_PROCESS_MODEL") or "",
        help="Registered Model Worker alias. Defaults to POST_PROCESS_MODEL_ALIAS or TEXT_PROCESS_MODEL.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Optional completion-token cap. When omitted, use the selected "
            "Model Worker alias's own sampling configuration (or no explicit "
            "cap when the alias has none)."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Model Worker requests; keep low for QPM-limited aliases.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many records.")
    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--start",
        type=int,
        default=None,
        help="Skip records before this zero-based index (legacy name for --offset).",
    )
    offset_group.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Skip records before this zero-based offset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run leakage detection but do not write --output.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed keep/filter verdicts in the checkpoint state.")
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Directory for resumable JSONL checkpoint state (default: derived from --audit-jsonl or --output).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Publish checkpoint/audit updates after this many completed judgements (default: 20).",
    )
    return parser.parse_args()


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the file."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = handle.read(16 * 1024 * 1024)
        position = 0
        first = True
        while True:
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] in "[," ):
                position += 1
            if position >= len(buffer):
                more = handle.read(16 * 1024 * 1024)
                if not more:
                    return
                buffer = buffer[position:] + more
                position = 0
                continue
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = handle.read(16 * 1024 * 1024)
                if not more:
                    raise
                buffer = buffer[position:] + more
                position = 0
                continue
            position = end
            first = False
            if not isinstance(value, dict):
                raise ValueError("Each trajectory in a JSON array must be an object")
            yield value


def _iter_records(path: Path) -> tuple[Iterator[dict[str, Any]], bool]:
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while not first:
            first = handle.read(1)
            if not first:
                raise ValueError(f"Input is empty: {path}")
            if first.isspace():
                first = ""
    if first == "[":
        return _iter_json_array(path), True

    def jsonl_iterator() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                yield value

    return jsonl_iterator(), False


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("id", "question_id", "sample_id", "data_id", "case_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"record_{index}"


def _conversation_text(record: dict[str, Any]) -> str:
    """Render the original trajectory with explicit causal turn markers."""

    chunks: list[str] = []
    if record.get("system"):
        chunks.append("[SYSTEM]\n" + str(record["system"]))
    conversations = record.get("conversations") or record.get("messages") or record.get("turns")
    if isinstance(conversations, list):
        for turn_index, turn in enumerate(conversations):
            if isinstance(turn, dict):
                role = turn.get("from") or turn.get("role") or turn.get("speaker") or "unknown"
                content = turn.get("value") or turn.get("content") or turn.get("response_text") or turn
            else:
                role = "unknown"
                content = turn
            chunks.append(f"[TURN {turn_index}][{role}]\n{content}")
    else:
        chunks.append("[RECORD]\n" + json.dumps(record, ensure_ascii=False, default=str))
    return "\n\n".join(chunks)


def _judge_prompt(record: dict[str, Any], index: int) -> str:
    question = record.get("question") or record.get("query") or ""
    payload = _conversation_text(record)
    return (
        f"Record index: {index}\n"
        f"Record id: {_record_id(record, index)}\n"
        f"User question (if separately available): {question}\n\n"
        "Audit the following trajectory in causal order. Tool observations are "
        "available only after the tool call immediately preceding them.\n\n"
        "<trajectory>\n"
        f"{payload}\n"
        "</trajectory>"
    )


def _normalise_verdict(raw: dict[str, Any], record: dict[str, Any], index: int) -> dict[str, Any]:
    leakage = bool(raw.get("leakage", False))
    decision = str(raw.get("decision") or ("filter" if leakage else "keep")).strip().lower()
    if decision not in {"keep", "filter"}:
        decision = "filter" if leakage else "keep"
    if decision == "filter":
        leakage = True
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    types = raw.get("leakage_types")
    if not isinstance(types, list):
        types = []
    evidence = raw.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    return {
        "record_index": index,
        "record_id": _record_id(record, index),
        "decision": decision,
        "leakage": leakage,
        "confidence": confidence,
        "leakage_types": [str(item) for item in types],
        "turns": raw.get("turns") if isinstance(raw.get("turns"), list) else [],
        "evidence": evidence,
        "reason": str(raw.get("reason") or "").strip(),
    }


def _judge_one(
    record: dict[str, Any],
    index: int,
    model_alias: str,
    max_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = ModelRequest(
        model=model_alias,
        messages=[
            ModelMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ModelMessage(role="user", content=_judge_prompt(record, index)),
        ],
        response_format={"type": "json_object"},
        metadata=_model_worker_metadata("post_process_information_leakage"),
    )
    try:
        raw = LLM_WORKER.generate_json(request)
        verdict = _normalise_verdict(raw, record, index)
        return verdict, {"ok": True}
    except Exception as exc:
        return {
            "record_index": index,
            "record_id": _record_id(record, index),
            "decision": "error",
            "leakage": None,
            "confidence": 0.0,
            "leakage_types": [],
            "turns": [],
            "evidence": [],
            "reason": f"judge_error: {exc}",
        }, {"ok": False, "error": repr(exc)}


def _write_json_array(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for record in records:
            if count:
                handle.write(",\n")
            json.dump(record, handle, ensure_ascii=False)
            count += 1
        handle.write("\n]\n")
    return count


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A killed process may leave a partial final checkpoint line.
                # Earlier complete lines remain valid resume state.
                break
            if isinstance(value, dict):
                yield value


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.state_dir is not None:
        return args.state_dir
    anchor = args.audit_jsonl or args.output or args.input
    return anchor.parent / f".{anchor.stem}_filter_state"


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "model_alias": args.model_alias,
        "max_tokens": args.max_tokens,
        "offset": args.offset if args.offset is not None else (args.start or 0),
        "limit": args.limit,
        "dry_run": bool(args.dry_run),
    }


def _is_expandable_resume_config(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Allow a checkpointed prefix to be extended without recomputing it.

    A resume remains tied to the same source, alias, offset and dry-run mode.
    The selected range may only grow.  ``max_tokens=2048`` is accepted as a
    one-time compatibility migration from the former CLI default to the new
    "unspecified" default; arbitrary budget changes remain incompatible.
    """

    for key in ("schema_version", "input", "model_alias", "offset", "dry_run"):
        if actual.get(key) != expected.get(key):
            return False

    old_limit = actual.get("limit")
    new_limit = expected.get("limit")
    if old_limit is None:
        if new_limit is not None:
            return False  # A previously unbounded run cannot safely shrink.
    elif new_limit is not None:
        try:
            if int(new_limit) < int(old_limit):
                return False
        except (TypeError, ValueError):
            return False

    old_max_tokens = actual.get("max_tokens")
    new_max_tokens = expected.get("max_tokens")
    return old_max_tokens == new_max_tokens or (old_max_tokens == 2048 and new_max_tokens is None)


def _prepare_state(state_dir: Path, args: argparse.Namespace) -> Path:
    config_path = state_dir / "run_config.json"
    expected = _run_config(args)
    if args.resume:
        if not config_path.is_file():
            raise SystemExit(f"--resume requires existing checkpoint configuration: {config_path}")
        try:
            actual = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Unable to read checkpoint configuration {config_path}: {exc}") from exc
        if actual != expected and not _is_expandable_resume_config(actual, expected):
            raise SystemExit("--resume configuration mismatch; use a new --state-dir for a different run.")
        if actual != expected:
            # Persist the expanded range so a later --resume validates against
            # the current command instead of repeatedly treating it as a
            # migration.  Existing completed verdicts are keyed by source
            # record index and are reused below; only the new suffix is sent.
            config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                "Resuming checkpoint with an expanded selection; completed "
                "source records will be reused and only new records scheduled.",
                file=sys.stderr,
                flush=True,
            )
    else:
        if state_dir.exists() and any(state_dir.iterdir()):
            raise SystemExit(f"Checkpoint state already exists: {state_dir}; use --resume or a new --state-dir.")
        state_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state_dir / "verdicts.jsonl"


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def main() -> int:
    args = _parse_args()
    if not args.model_alias:
        raise SystemExit("--model-alias is required (or set POST_PROCESS_MODEL_ALIAS/TEXT_PROCESS_MODEL).")
    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is set.")
    offset = args.offset if args.offset is not None else (args.start or 0)
    if offset < 0 or args.limit is not None and args.limit < 0:
        raise SystemExit("--offset/--start and --limit must be non-negative.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1.")

    state_dir = _checkpoint_dir(args)
    checkpoint_path = _prepare_state(state_dir, args)

    iterator, is_array = _iter_records(args.input)
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(iterator):
        if index < offset:
            continue
        if args.limit is not None and len(selected) >= args.limit:
            break
        selected.append((index, record))

    prior_verdicts: dict[tuple[int, str], dict[str, Any]] = {}
    if args.resume:
        for row in _read_jsonl(checkpoint_path):
            index = row.get("record_index")
            record_id = row.get("record_id")
            verdict = row.get("verdict")
            if (
                isinstance(index, int)
                and isinstance(record_id, str)
                and isinstance(verdict, dict)
                and verdict.get("decision") in {"keep", "filter"}
            ):
                prior_verdicts[(index, record_id)] = verdict

    verdicts: dict[int, dict[str, Any]] = {}
    errors: dict[int, dict[str, Any]] = {}
    todo: list[tuple[int, dict[str, Any]]] = []
    resumed = 0
    for index, record in selected:
        prior = prior_verdicts.get((index, _record_id(record, index)))
        if prior is None:
            todo.append((index, record))
        else:
            verdicts[index] = prior
            resumed += 1

    pending_checkpoints: list[dict[str, Any]] = []
    pending_audit: list[dict[str, Any]] = []

    def commit(index: int, record: dict[str, Any], verdict: dict[str, Any], meta: dict[str, Any]) -> None:
        verdicts[index] = verdict
        if not meta.get("ok"):
            errors[index] = verdict
            return
        # Operational errors intentionally remain retryable and are not put in
        # resume state.  Only quality decisions are terminal checkpoints.
        if verdict.get("decision") not in {"keep", "filter"}:
            errors[index] = verdict
            return
        pending_checkpoints.append(
            {
                "record_index": index,
                "record_id": _record_id(record, index),
                "verdict": verdict,
            }
        )
        pending_audit.append(verdict)
        if len(pending_checkpoints) >= args.checkpoint_every:
            _append_jsonl(checkpoint_path, pending_checkpoints)
            if args.audit_jsonl:
                _append_jsonl(args.audit_jsonl, pending_audit)
            pending_checkpoints.clear()
            pending_audit.clear()

    progress = tqdm(total=len(selected), desc="Filtering trajectories", unit="trajectory")
    if resumed:
        progress.update(resumed)
        progress.set_postfix(keep=sum(v.get("decision") == "keep" for v in verdicts.values()), rejected=sum(v.get("decision") == "filter" for v in verdicts.values()), errors=0, resumed=resumed)
    if args.workers == 1:
        for index, record in todo:
            verdict, meta = _judge_one(record, index, args.model_alias, args.max_tokens)
            commit(index, record, verdict, meta)
            progress.update(1)
            progress.set_postfix(keep=sum(v.get("decision") == "keep" for v in verdicts.values()), rejected=sum(v.get("decision") == "filter" for v in verdicts.values()), errors=len(errors), resumed=resumed)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_judge_one, record, index, args.model_alias, args.max_tokens): index
                for index, record in todo
            }
            records_by_index = {index: record for index, record in todo}
            for future in as_completed(futures):
                index = futures[future]
                verdict, meta = future.result()
                commit(index, records_by_index[index], verdict, meta)
                progress.update(1)
                progress.set_postfix(keep=sum(v.get("decision") == "keep" for v in verdicts.values()), rejected=sum(v.get("decision") == "filter" for v in verdicts.values()), errors=len(errors), resumed=resumed)
    _append_jsonl(checkpoint_path, pending_checkpoints)
    if args.audit_jsonl:
        _append_jsonl(args.audit_jsonl, pending_audit)
    progress.close()

    ordered = [verdicts[index] for index, _ in selected]
    counts = Counter(verdict["decision"] for verdict in ordered)
    type_counts = Counter(
        leakage_type
        for verdict in ordered
        if verdict["decision"] == "filter"
        for leakage_type in verdict.get("leakage_types", [])
    )

    if args.audit_jsonl:
        # Rebuild a clean, source-ordered audit after a successful pass.  The
        # checkpoint is authoritative during interruption; this final rewrite
        # removes any harmless duplicate tail emitted before a crash.
        args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_jsonl.open("w", encoding="utf-8") as handle:
            for verdict in ordered:
                handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")

    written = 0
    if not args.dry_run and args.output:
        kept = [record for index, record in selected if verdicts[index]["decision"] == "keep"]
        if is_array:
            written = _write_json_array(args.output, kept)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for record in kept:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    summary = {
        "input": str(args.input),
        "model_alias": args.model_alias,
        "dry_run": bool(args.dry_run),
        "selected": len(selected),
        "decisions": dict(counts),
        "leakage_type_counts": dict(type_counts),
        "judge_errors": len(errors),
        "resumed": resumed,
        "checkpoint_state": str(state_dir),
        "written": written,
        "audit_jsonl": str(args.audit_jsonl) if args.audit_jsonl else None,
        "output": str(args.output) if args.output and not args.dry_run else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
