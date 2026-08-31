#!/usr/bin/env python3
"""Build a clean, rewritten ShareGPT training dataset in one causal pipeline.

The pipeline has two intentional, separately auditable stages:

1. Information-leakage / reasoning-integrity filtering.  Each ``keep``
   record is immediately handed to the rewrite stage through a bounded queue.
2. Left-to-right reasoning expansion, including optional distractor
   verification and integration, runs concurrently with remaining filtering.

The source dataset is never modified.  The output is always a new ShareGPT
JSONL training dataset plus its ``dataset_info.json``.  Existing source images
are referenced through paths relative to the new dataset directory; only
images newly downloaded during an integrated verification are copied into the
new dataset's ``images/verification`` directory.

The output directory contains:

* ``trajectories_train.jsonl``: final dataset accepted for training;
* ``dataset_info.json``: LLaMA-Factory/ShareGPT registration for that JSONL;
* ``filter_audit.jsonl`` and ``rewrite_audit.jsonl``: stage-level audits;
* ``pipeline_audit.jsonl``: final disposition of every selected source record;
* ``filtered_out.jsonl``: rejected source records with their filter verdict;
* ``intermediate/kept_after_filter.jsonl``: exact records passed to rewriting;
* ``images/verification``: isolated images newly introduced by verification.

Example:

    python synthesis/post_process/build_refined_training_dataset.py \\
        --input data/sharegpt_dataset_final/trajectories_sharegpt.json \\
        --output-dir data/sharegpt_dataset_refined \\
        --filter-model-alias gmmini35flash_internal_azure \\
        --rewrite-model-alias gpt55_internal_azure_chat \\
        --verification-model-alias gpt55_internal_azure \\
        --filter-workers 8 --rewrite-workers 4

``--dry-run`` still calls the filter/rewrite models and writes the audit plus
temporary two-pass review data required to run the pipeline, but does not
write the final training JSONL, dataset registration, or copied image assets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.post_process.expand_trajectory_reasoning import (
    _optimize_record,
    _print_compare_trajectories,
)
from synthesis.post_process.terminal_visual_evidence import VqaTerminalImageResolver
from synthesis.post_process.filter_trajectory_information_leakage import (
    _iter_records,
    _judge_one,
    _record_id,
)


T = TypeVar("T")
R = TypeVar("R")
_SAFE_FILE_STEM = re.compile(r"[^A-Za-z0-9._-]+")
_PUBLISH_FSYNC_WARNED: set[str] = set()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source ShareGPT JSON array or JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, empty directory for the refined dataset and all audit artifacts.",
    )
    parser.add_argument(
        "--filter-model-alias",
        default=os.environ.get("FILTER_MODEL_ALIAS") or os.environ.get("POST_PROCESS_FILTER_MODEL_ALIAS") or "",
        help="Model Worker alias used by the information-leakage filter.",
    )
    parser.add_argument(
        "--rewrite-model-alias",
        default=os.environ.get("POST_PROCESS_MODEL_ALIAS") or os.environ.get("TEXT_PROCESS_MODEL") or "",
        help="Model Worker alias used by the reasoning polisher and integration writer.",
    )
    parser.add_argument(
        "--verification-model-alias",
        default=os.environ.get("VERIFICATION_MODEL_ALIAS") or "",
        help="Optional dedicated Model Worker alias for distractor verification.",
    )
    parser.add_argument(
        "--visual-vqa-model-alias",
        default=os.environ.get("VISUAL_VQA_MODEL_ALIAS") or "",
        help="Dedicated Model Worker alias for terminal visual-evidence answers.",
    )
    parser.add_argument("--vqa-dir", type=Path, help="VQA run directory containing samples.jsonl/questions.jsonl.")
    parser.add_argument("--graph-dir", type=Path, help="Graph directory containing nodes.jsonl for terminal image-node lookup.")
    parser.add_argument("--filter-max-tokens", type=int, default=2048)
    parser.add_argument("--rewrite-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--history-window-turns",
        type=int,
        default=None,
        help=(
            "Forwarded to rewriting: retain only the most recent N prior "
            "assistant/tool rounds while preserving the original question and "
            "initial image(s). Default: full causal history."
        ),
    )
    parser.add_argument("--verification-max-tokens", type=int, default=4096)
    parser.add_argument("--verification-max-turns", type=int, default=5)
    parser.add_argument("--filter-workers", type=int, default=1)
    parser.add_argument("--rewrite-workers", type=int, default=1)
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help=(
            "Maximum kept trajectories buffered between filter and rewrite. "
            "Defaults to max(8, 2 * --rewrite-workers)."
        ),
    )
    parser.add_argument(
        "--debug-output",
        choices=("full", "verification_only", "silence", "slience"),
        default="silence",
        help="Forwarded to the rewrite stage. ``slience`` remains an accepted legacy spelling.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each rewritten assistant turn immediately.")
    parser.add_argument("--compare", action="store_true", help="Print complete before/after trajectories.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many selected records.")
    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--offset", type=int, default=None, help="Zero-based input offset.")
    offset_group.add_argument("--start", type=int, default=None, help="Legacy name for --offset.")
    parser.add_argument(
        "--sample-id",
        "--sample_id",
        dest="sample_ids",
        action="append",
        default=[],
        help="Only process these IDs. Repeat the option or give comma-separated IDs.",
    )
    parser.add_argument(
        "--dataset-name",
        default="opensearch_vl_sft_refined",
        help="Dataset key written into the new dataset_info.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write audit/review artifacts only; do not create the train JSONL or image assets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run in --output-dir. Completed terminal records are skipped; "
            "only unfinished records are filtered/rewritten again."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Flush all output JSONL streams after this many terminal sample dispositions (default: 20).",
    )
    return parser.parse_args()


def _auxiliary_debug_enabled(debug_output: str) -> bool:
    return debug_output not in {"silence", "slience"}


def _print_auxiliary_process_debug(record: dict[str, Any], audit: dict[str, Any], debug_output: str) -> None:
    """Print concise, plain-text outcomes for verifier/image-repair work.

    This runs after a trajectory's rewrite has completed, in the commit thread,
    so concurrent workers cannot interleave these reports.  Full trajectories
    remain controlled by the existing debug mode; this report deliberately
    contains only trigger, result, and failure reason.
    """

    if not _auxiliary_debug_enabled(debug_output):
        return
    changes = audit.get("changes") if isinstance(audit.get("changes"), list) else []
    relevant = [
        change for change in changes
        if isinstance(change, dict)
        and (
            "verification" in change
            or "terminal_visual_evidence" in change
            or str(change.get("editor_note") or "").startswith("verification_execution_failed")
        )
    ]
    if not relevant:
        return

    record_id = _record_id(record, int(audit.get("record_index", 0)))
    print(f"\n=== Auxiliary process summary | {record_id} ===", flush=True)
    for change in relevant:
        turn = change.get("message_index", "?")
        note = str(change.get("editor_note") or "unknown")
        if "terminal_visual_evidence" in change:
            visual = change.get("terminal_visual_evidence")
            visual = visual if isinstance(visual, dict) else {}
            if note == "terminal_visual_evidence_replaced_and_rewritten":
                result = "SUCCESS: terminal image replaced and final turn rewritten"
            elif note == "terminal_visual_evidence_matched_and_rewritten":
                result = "SUCCESS: existing terminal image supports the stored answer"
            elif "filter_recommended" in note or change.get("filter_recommended"):
                result = "FAILED: no usable terminal image supports the stored answer; record marked for filtering"
            else:
                result = "FAILED: terminal visual-evidence check could not be completed"
            print(f"[Terminal image check | turn {turn}]", flush=True)
            trigger_labels = {
                "trajectory_quality_complaint": "the original trajectory explicitly reported the current terminal image as insufficient",
                "writer_cannot_read": "the writer could not derive the final visual answer from the current terminal image",
            }
            trigger_reason = str(visual.get("trigger_reason") or "unknown")
            print(f"Trigger: {trigger_labels.get(trigger_reason, trigger_reason)}", flush=True)
            print(f"Result: {result}", flush=True)
            print(f"Question source: {visual.get('local_visual_question_source', 'unknown')}", flush=True)
            if visual.get("local_visual_question"):
                print(f"Visual question: {visual['local_visual_question']}", flush=True)
            if visual.get("downloaded_image_answer") is not None:
                print(f"Original terminal image answer: {visual['downloaded_image_answer']}", flush=True)
            if visual.get("terminal_node_image_answer") is not None:
                print(f"Image-node answer: {visual['terminal_node_image_answer']}", flush=True)
            if visual.get("error"):
                print(f"Reason: {visual['error']}", flush=True)
            elif "SUCCESS" not in result:
                print("Reason: visual answer did not semantically match the stored final answer.", flush=True)
            continue

        verification = change.get("verification")
        verification = verification if isinstance(verification, dict) else {}
        request = change.get("verification_request")
        request = request if isinstance(request, dict) else {}
        print(f"[Distractor verification | turn {turn}]", flush=True)
        if request:
            print(
                "Trigger: unresolved competing candidate "
                f"{request.get('object_to_verify', '?')} for referent {request.get('referent', '?')}",
                flush=True,
            )
            if request.get("verification_goal"):
                print(f"Verification goal: {request['verification_goal']}", flush=True)
        else:
            print("Trigger: writer requested external distractor verification.", flush=True)
        if note == "verification_integrated":
            result = "SUCCESS: verifier returned NO with usable evidence; verification was integrated"
        elif note == "verification_affirmed_candidate":
            result = "COMPLETED: verifier returned YES; source record marked for filtering"
        elif note == "verification_not_integrated":
            result = "FAILED: verifier result was incomplete or lacked usable evidence"
        elif note == "verification_execution_failed":
            result = "FAILED: verifier execution raised an error"
        else:
            result = f"NOT INTEGRATED: {note}"
        print(f"Result: {result}", flush=True)
        if verification:
            print(
                "Verifier details: "
                f"final={verification.get('normalized_final_text', '?')}, "
                f"complete={verification.get('generation_complete', '?')}, "
                f"usable_evidence={verification.get('usable_evidence_count', '?')}, "
                f"tool_calls={verification.get('tool_call_count', '?')}",
                flush=True,
            )
        if change.get("error"):
            print(f"Reason: {change['error']}", flush=True)
    print("=== End auxiliary process summary ===", flush=True)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise SystemExit(f"--input does not exist or is not a file: {args.input}")
    if not args.filter_model_alias:
        raise SystemExit("--filter-model-alias is required (or set FILTER_MODEL_ALIAS).")
    if not args.rewrite_model_alias:
        raise SystemExit("--rewrite-model-alias is required (or set POST_PROCESS_MODEL_ALIAS/TEXT_PROCESS_MODEL).")
    if bool(args.vqa_dir) != bool(args.visual_vqa_model_alias):
        raise SystemExit("--vqa-dir and --visual-vqa-model-alias must be provided together.")
    if args.vqa_dir is not None and not args.vqa_dir.is_dir():
        raise SystemExit(f"--vqa-dir does not exist or is not a directory: {args.vqa_dir}")
    if args.graph_dir is not None and not args.graph_dir.is_dir():
        raise SystemExit(f"--graph-dir does not exist or is not a directory: {args.graph_dir}")
    for name in ("filter_workers", "rewrite_workers", "verification_max_turns"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1.")
    if args.history_window_turns is not None and args.history_window_turns < 1:
        raise SystemExit("--history-window-turns must be at least 1 when provided.")
    if args.queue_size is not None and args.queue_size < 1:
        raise SystemExit("--queue-size must be at least 1.")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1.")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative.")
    offset = args.offset if args.offset is not None else (args.start or 0)
    if offset < 0:
        raise SystemExit("--offset/--start must be non-negative.")
    run_config = args.output_dir / "run_config.json"
    if args.resume:
        if not args.output_dir.is_dir():
            raise SystemExit(f"--resume requires an existing output directory: {args.output_dir}")
        if not run_config.is_file():
            raise SystemExit(
                f"--resume requires {run_config}; use a fresh --output-dir for a new run instead."
            )
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"--output-dir must be new or empty to prevent mixing runs: {args.output_dir}"
        )


def _requested_ids(raw_values: list[str]) -> set[str]:
    return {
        item.strip()
        for raw_value in raw_values
        for item in raw_value.split(",")
        if item.strip()
    }


def _iter_selected_records(
    input_path: Path,
    *,
    offset: int,
    limit: int | None,
    requested_ids: set[str],
    matched_ids: set[str],
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield selected source records without loading the full JSON array."""

    iterator, _is_array = _iter_records(input_path)
    selected_count = 0
    matching_offset = 0
    for index, record in enumerate(iterator):
        record_id = _record_id(record, index)
        if requested_ids:
            if record_id not in requested_ids:
                continue
            matched_ids.add(record_id)
            if matching_offset < offset:
                matching_offset += 1
                continue
        elif index < offset:
            continue
        if limit is not None and selected_count >= limit:
            break
        selected_count += 1
        yield index, record


def _count_selected_records(
    input_path: Path,
    *,
    offset: int,
    limit: int | None,
    requested_ids: set[str],
) -> int:
    """Make one inexpensive streaming pass so the main filter bar has a total.

    The source ShareGPT file is commonly one large JSON array, so its size on
    disk tells us nothing about the number of trajectories.  Counting it once
    is negligible compared with the model calls, and lets tqdm render a real
    percentage/ETA bar during the actual pipeline instead of only a raw count.
    """

    prepass_matches: set[str] = set()
    progress = tqdm(desc="Indexing selected trajectories", unit="trajectory")
    try:
        count = 0
        for _index, _record in _iter_selected_records(
            input_path,
            offset=offset,
            limit=limit,
            requested_ids=requested_ids,
            matched_ids=prepass_matches,
        ):
            count += 1
            progress.update(1)
        return count
    finally:
        progress.close()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


class _JsonlPublishBuffer:
    """Batch JSONL records, then publish each batch by closing its HDFS-FUSE file.

    HDFS-FUSE does not reliably expose growth from a long-lived append handle
    to other readers.  Mirroring the VQA/SFT runners, every checkpoint opens
    the file, appends the buffered rows, flushes/fsyncs, and closes it.
    """

    def __init__(self, path: Path, *, mode: str) -> None:
        self.path = path
        self._records: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create/truncate now, but never retain this handle across a batch.
        with self.path.open(mode, encoding="utf-8"):
            pass

    def __enter__(self) -> "_JsonlPublishBuffer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.flush()

    def append(self, record: dict[str, Any]) -> None:
        self._records.append(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if not self._records:
            return
        records = self._records
        self._records = []
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(records)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as exc:
                path_text = str(self.path)
                if path_text not in _PUBLISH_FSYNC_WARNED:
                    _PUBLISH_FSYNC_WARNED.add(path_text)
                    print(
                        f"[refined-output] fsync unavailable for {path_text}; "
                        f"file close will publish the batch: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    if isinstance(handle, _JsonlPublishBuffer):
        handle.append(record)
        return
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl_resume_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read complete JSONL objects from a prior run.

    A hard interruption can leave a partial final line.  Earlier complete
    records remain useful for resuming, while the malformed tail is ignored
    and will be regenerated if it was not committed in the pipeline audit.
    """

    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # JSONL writers only append. A malformed line is therefore a
                # crash tail, not a reason to discard all prior checkpoints.
                continue
            if isinstance(value, dict):
                yield value


def _repair_jsonl_append_tail(path: Path) -> str | None:
    """Make an interrupted append-only JSONL file safe to append to again.

    Only an invalid final record (or a missing final newline after a valid
    record) is repaired. If a malformed line is followed by later content, the
    file is not touched because that is data corruption rather than a normal
    interruption tail.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("r+b") as handle:
        last_good_offset = 0
        invalid_offset: int | None = None
        last_line_had_newline = True
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                if invalid_offset is None:
                    last_good_offset = handle.tell()
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if invalid_offset is None:
                    invalid_offset = line_start
                continue
            if not isinstance(value, dict):
                if invalid_offset is None:
                    invalid_offset = line_start
                continue
            if invalid_offset is not None:
                raise RuntimeError(
                    f"Refusing to resume because {path} has malformed JSONL before its final record."
                )
            last_good_offset = handle.tell()
            last_line_had_newline = raw_line.endswith(b"\n")
        if invalid_offset is not None:
            handle.truncate(last_good_offset)
            return "truncated_invalid_final_line"
        if last_good_offset and not last_line_had_newline:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            return "appended_missing_final_newline"
    return None


def _run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Persist the compatibility-critical identity of a resumable run."""

    return {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "dry_run": bool(args.dry_run),
        "dataset_name": args.dataset_name,
        "history_window_turns": args.history_window_turns,
    }


def _prepare_run_configuration(path: Path, args: argparse.Namespace) -> None:
    expected = _run_configuration(args)
    if not args.resume:
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Unable to read prior run configuration {path}: {exc}") from exc
    for key in ("input", "dry_run", "dataset_name", "history_window_turns"):
        if existing.get(key) != expected[key]:
            raise SystemExit(
                "--resume configuration mismatch for "
                f"{key}: prior={existing.get(key)!r} current={expected[key]!r}"
            )


def _load_resume_state(
    *,
    pipeline_audit_path: Path,
    final_path: Path,
    dry_run: bool,
) -> tuple[set[tuple[int, str]], set[str]]:
    """Return committed terminal records and IDs already present in train JSONL."""

    final_record_ids: set[str] = set()
    if not dry_run:
        for record in _load_jsonl_resume_records(final_path):
            final_record_ids.add(_record_id(record, -1))

    terminal_records: set[tuple[int, str]] = set()
    retryable_dispositions = {
        "filter_error_excluded",
        "rewrite_error_excluded",
        "new_verification_image_error_excluded",
        "image_placeholder_misalignment_excluded",
    }
    for entry in _load_jsonl_resume_records(pipeline_audit_path):
        index = entry.get("record_index")
        record_id = entry.get("record_id")
        if not isinstance(index, int) or not isinstance(record_id, str) or not record_id:
            continue
        disposition = str(entry.get("disposition") or "")
        # These are operational failures, not quality decisions. Retrying them
        # on resume is the useful behavior for transient model/tool/image
        # failures; deterministic quality exclusions remain terminal below.
        if disposition in retryable_dispositions:
            continue
        # ``accepted`` is committed only after its final JSONL row is present.
        # This avoids skipping a sample when a process died between an audit
        # append and a final-data append. All other dispositions are explicit
        # terminal exclusions and are safe to preserve on resume.
        if disposition == "accepted" and not dry_run and record_id not in final_record_ids:
            continue
        terminal_records.add((index, record_id))
    return terminal_records, final_record_ids


def _flush_handles(handles: Iterable[Any]) -> None:
    for handle in handles:
        if handle is not None:
            handle.flush()


def _ordered_parallel(
    items: Iterable[T],
    *,
    workers: int,
    fn: Callable[[T], R],
    max_pending: int | None = None,
) -> Iterator[tuple[T, R]]:
    """Run a bounded parallel map while yielding in source order.

    The source trajectory file is roughly gigabyte-scale.  Keeping only a
    small pending window avoids the all-records-in-memory behavior of the two
    standalone scripts while retaining deterministic output order.
    """

    if workers == 1:
        for item in items:
            yield item, fn(item)
        return

    pending_limit = max_pending or max(workers * 2, workers)
    source = iter(items)
    next_sequence = 0
    next_to_yield = 0
    futures: dict[Future[R], tuple[int, T]] = {}
    completed: dict[int, tuple[T, R]] = {}
    exhausted = False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while not exhausted or futures:
            while not exhausted and len(futures) < pending_limit:
                try:
                    item = next(source)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(fn, item)
                futures[future] = (next_sequence, item)
                next_sequence += 1
            if not futures:
                continue
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                sequence, item = futures.pop(future)
                completed[sequence] = (item, future.result())
            while next_to_yield in completed:
                yield completed.pop(next_to_yield)
                next_to_yield += 1


def _filter_one(
    item: tuple[int, dict[str, Any]],
    model_alias: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index, record = item
    return _judge_one(record, index, model_alias, max_tokens)


def _rewrite_one(
    item: tuple[dict[str, Any], dict[str, Any]],
    *,
    model_alias: str,
    max_tokens: int,
    verification_model_alias: str,
    verification_max_tokens: int,
    verification_max_turns: int,
    verification_workdir: Path,
    debug_output: str,
    verbose: bool,
    compare: bool,
    visual_vqa_model_alias: str = "",
    terminal_image_resolver: VqaTerminalImageResolver | None = None,
    visual_evidence_workdir: Path | None = None,
    history_window_turns: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, record = item
    output, audit = _optimize_record(
        record,
        int(manifest["record_index"]),
        model_alias,
        max_tokens,
        verification_model_alias,
        verification_max_tokens,
        verification_max_turns,
        debug_output,
        3,
        verbose,
        compare,
        verification_workdir,
        visual_vqa_model_alias,
        terminal_image_resolver,
        visual_evidence_workdir,
        history_window_turns,
    )
    return manifest, output, audit


def _has_rewrite_filter_recommendation(audit: dict[str, Any]) -> bool:
    return any(bool(change.get("filter_recommended")) for change in audit.get("changes", []))


def _source_image_reference_for_output(
    value: str,
    *,
    source_dataset_dir: Path,
    output_dir: Path,
) -> str:
    """Make an original asset path valid when resolved from the new dataset."""

    if value.startswith(("http://", "https://")):
        return value
    source = Path(value)
    if not source.is_absolute():
        source = source_dataset_dir / source
    return Path(os.path.relpath(source, output_dir)).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_verification_image(source: str, verification_dir: Path) -> str:
    """Copy one newly introduced local image into the isolated output library."""

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"new verification image is unavailable locally: {source}")
    digest = _sha256_file(path)
    suffix = path.suffix.casefold() or ".img"
    destination = verification_dir / f"{digest}{suffix}"
    verification_dir.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination.name


def _image_placeholder_count(record: dict[str, Any]) -> int:
    messages = record.get("conversations") or record.get("messages") or []
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            count += str(message).count("<image>")
            continue
        content = message.get("value")
        if content is None:
            content = message.get("content")
        if content is None:
            content = message.get("response_text")
        count += str(content or "").count("<image>")
    return count


def _materialize_record_images(
    rewritten_record: dict[str, Any],
    original_record: dict[str, Any],
    *,
    source_dataset_dir: Path,
    output_dir: Path,
    verification_dir: Path,
    copy_new_images: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate original assets and isolate verification assets for one record."""

    output = copy.deepcopy(rewritten_record)
    original_images = [str(value) for value in (original_record.get("images") or [])]
    original_image_set = set(original_images)
    translated_images: list[str] = []
    new_image_count = 0

    for raw_value in output.get("images") or []:
        value = str(raw_value)
        if value in original_image_set:
            translated_images.append(
                _source_image_reference_for_output(
                    value,
                    source_dataset_dir=source_dataset_dir,
                    output_dir=output_dir,
                )
            )
            continue
        new_image_count += 1
        if not copy_new_images:
            # Dry-run must not publish a new data reference to a volatile cache.
            translated_images.append(value)
            continue
        filename = _copy_verification_image(value, verification_dir)
        translated_images.append((Path("images") / "verification" / filename).as_posix())

    if "images" in output or translated_images:
        output["images"] = translated_images
    placeholder_count = _image_placeholder_count(output)
    return output, {
        "original_image_count": len(original_images),
        "new_verification_image_count": new_image_count,
        "output_image_count": len(translated_images),
        "image_placeholder_count": placeholder_count,
        "image_alignment_ok": placeholder_count == len(translated_images),
    }


def _write_dataset_info(path: Path, dataset_name: str) -> None:
    payload = {
        dataset_name: {
            "file_name": "trajectories_train.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pipeline_record(
    *,
    index: int,
    record: dict[str, Any],
    filter_audit: dict[str, Any],
    disposition: str,
    rewrite_audit: dict[str, Any] | None = None,
    image_audit: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_index": index,
        "record_id": _record_id(record, index),
        "disposition": disposition,
        "filter": filter_audit,
    }
    if rewrite_audit is not None:
        payload["rewrite"] = rewrite_audit
    if image_audit is not None:
        payload["images"] = image_audit
    if error:
        payload["error"] = error
    return payload


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    offset = args.offset if args.offset is not None else (args.start or 0)
    requested_ids = _requested_ids(args.sample_ids)
    matched_ids: set[str] = set()
    selected_total = _count_selected_records(
        args.input,
        offset=offset,
        limit=args.limit,
        requested_ids=requested_ids,
    )

    output_dir = args.output_dir.resolve()
    source_dataset_dir = args.input.resolve().parent
    intermediate_dir = output_dir / "intermediate"
    verification_workdir = intermediate_dir / "verification_runtime"
    visual_evidence_workdir = intermediate_dir / "terminal_visual_evidence_runtime"
    verification_image_dir = output_dir / "images" / "verification"
    kept_path = intermediate_dir / "kept_after_filter.jsonl"
    kept_manifest_path = intermediate_dir / "kept_after_filter_manifest.jsonl"
    filter_audit_path = output_dir / "filter_audit.jsonl"
    rewrite_audit_path = output_dir / "rewrite_audit.jsonl"
    pipeline_audit_path = output_dir / "pipeline_audit.jsonl"
    filtered_out_path = output_dir / "filtered_out.jsonl"
    final_path = output_dir / "trajectories_train.jsonl"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    terminal_image_resolver = (
        VqaTerminalImageResolver(
            args.vqa_dir,
            args.graph_dir,
            cache_dir=visual_evidence_workdir,
        )
        if args.vqa_dir is not None
        else None
    )
    _prepare_run_configuration(run_config_path, args)
    repaired_jsonl_tails: dict[str, str] = {}
    if args.resume:
        for path in (
            kept_path,
            kept_manifest_path,
            filter_audit_path,
            rewrite_audit_path,
            filtered_out_path,
            pipeline_audit_path,
            final_path,
        ):
            repair = _repair_jsonl_append_tail(path)
            if repair:
                repaired_jsonl_tails[str(path)] = repair
    terminal_records, final_record_ids = _load_resume_state(
        pipeline_audit_path=pipeline_audit_path,
        final_path=final_path,
        dry_run=bool(args.dry_run),
    )

    summary: dict[str, Any] = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "repaired_jsonl_tails": repaired_jsonl_tails,
        "checkpoint_every": args.checkpoint_every,
        "filter_model_alias": args.filter_model_alias,
        "rewrite_model_alias": args.rewrite_model_alias,
        "history_window_turns": args.history_window_turns,
        "verification_model_alias": args.verification_model_alias or None,
        "filter_workers": args.filter_workers,
        "rewrite_workers": args.rewrite_workers,
        "queue_size": args.queue_size or max(8, args.rewrite_workers * 2),
        "selected_total": selected_total,
        "selected": 0,
        "resumed_terminal_records": 0,
        "resumed_final_only_records": 0,
        "prior_training_records": len(final_record_ids),
        "filter_keep": 0,
        "filter_rejected": 0,
        "filter_errors": 0,
        "rewrite_ok": 0,
        "rewrite_errors": 0,
        "verification_requests": 0,
        "integrated_verifications": 0,
        "terminal_visual_evidence_checks": 0,
        "terminal_visual_evidence_trajectory_quality_complaint_checks": 0,
        "terminal_visual_evidence_writer_cannot_read_checks": 0,
        "terminal_visual_evidence_replacements": 0,
        "terminal_visual_evidence_filters": 0,
        "rewrite_filter_recommended": 0,
        "image_errors": 0,
        "accepted_records": 0,
        "training_records": len(final_record_ids),
        "new_verification_images": 0,
        "pipeline_errors": [],
    }

    queue_size = int(summary["queue_size"])
    rewrite_queue: queue.Queue[tuple[dict[str, Any], dict[str, Any]] | None] = queue.Queue(
        maxsize=queue_size
    )
    writer_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(8, queue_size * 2))

    def rewrite_item(
        item: tuple[dict[str, Any], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return _rewrite_one(
            item,
            model_alias=args.rewrite_model_alias,
            max_tokens=args.rewrite_max_tokens,
            verification_model_alias=args.verification_model_alias,
            verification_max_tokens=args.verification_max_tokens,
            verification_max_turns=args.verification_max_turns,
            verification_workdir=verification_workdir,
            debug_output=args.debug_output,
            verbose=args.verbose,
            compare=args.compare,
            visual_vqa_model_alias=args.visual_vqa_model_alias,
            terminal_image_resolver=terminal_image_resolver,
            visual_evidence_workdir=visual_evidence_workdir,
            history_window_turns=args.history_window_turns,
        )

    def rewrite_worker(worker_index: int) -> None:
        """Consume kept trajectories one at a time, preserving each trajectory's causality."""

        while True:
            item = rewrite_queue.get()
            try:
                if item is None:
                    return
                manifest, original_record = item
                try:
                    result_manifest, rewritten_record, rewrite_audit = rewrite_item(item)
                except Exception as exc:  # Keep a single bad trajectory from killing the pool.
                    result_manifest = manifest
                    rewritten_record = copy.deepcopy(original_record)
                    rewrite_audit = {
                        "record_index": int(manifest["record_index"]),
                        "record_id": str(manifest["record_id"]),
                        "status": "error",
                        "assistant_turns_attempted": 0,
                        "changed_turns": 0,
                        "error": repr(exc),
                    }
                writer_queue.put(
                    {
                        "type": "rewrite_result",
                        "manifest": result_manifest,
                        "original_record": original_record,
                        "rewritten_record": rewritten_record,
                        "rewrite_audit": rewrite_audit,
                        "worker_index": worker_index,
                    }
                )
            finally:
                rewrite_queue.task_done()

    rewrite_threads = [
        threading.Thread(
            target=rewrite_worker,
            args=(worker_index,),
            name=f"trajectory-rewriter-{worker_index}",
            daemon=True,
        )
        for worker_index in range(args.rewrite_workers)
    ]
    for thread in rewrite_threads:
        thread.start()

    def filter_dispatcher() -> None:
        """Filter in one pool and immediately feed kept records to the rewrite queue."""

        try:
            selected_records = _iter_selected_records(
                args.input,
                offset=offset,
                limit=args.limit,
                requested_ids=requested_ids,
                matched_ids=matched_ids,
            )

            def records_needing_work() -> Iterator[tuple[int, dict[str, Any]]]:
                for index, record in selected_records:
                    record_id = _record_id(record, index)
                    if args.resume and (index, record_id) in terminal_records:
                        writer_queue.put(
                            {
                                "type": "resume_terminal",
                                "record_index": index,
                                "record": record,
                            }
                        )
                        continue
                    if args.resume and not args.dry_run and record_id in final_record_ids:
                        # A final JSONL row is more authoritative than an audit
                        # line because the data row is what training consumes.
                        # Recover an audit marker without redoing the expensive
                        # rewrite if a process stopped just before its audit.
                        writer_queue.put(
                            {
                                "type": "resume_final_only",
                                "record_index": index,
                                "record": record,
                            }
                        )
                        continue
                    yield index, record

            def judge_item(item: tuple[int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
                return _filter_one(item, args.filter_model_alias, args.filter_max_tokens)

            for (index, record), (verdict, _meta) in _ordered_parallel(
                records_needing_work(),
                workers=args.filter_workers,
                fn=judge_item,
            ):
                writer_queue.put(
                    {
                        "type": "filter_result",
                        "record_index": index,
                        "record": record,
                        "verdict": verdict,
                    }
                )
                if verdict.get("decision") == "keep":
                    manifest = {
                        "record_index": index,
                        "record_id": _record_id(record, index),
                        "filter_audit": verdict,
                    }
                    # This bounded put provides backpressure: a quick filter
                    # cannot accumulate an unbounded in-memory rewrite backlog.
                    rewrite_queue.put((manifest, record))
        except Exception as exc:
            writer_queue.put(
                {
                    "type": "pipeline_error",
                    "stage": "filter_dispatcher",
                    "error": repr(exc),
                }
            )
        finally:
            for _ in rewrite_threads:
                rewrite_queue.put(None)
            for thread in rewrite_threads:
                thread.join()
            writer_queue.put(
                {
                    "type": "pipeline_complete",
                    "matched_sample_ids": sorted(matched_ids),
                }
            )

    filter_thread = threading.Thread(
        target=filter_dispatcher,
        name="trajectory-filter-dispatcher",
        daemon=True,
    )
    filter_thread.start()

    final_handle: _JsonlPublishBuffer | None = None
    try:
        file_mode = "a" if args.resume else "w"
        if not args.dry_run:
            final_handle = _JsonlPublishBuffer(final_path, mode=file_mode)
        with (
            _JsonlPublishBuffer(kept_path, mode=file_mode) as kept_handle,
            _JsonlPublishBuffer(kept_manifest_path, mode=file_mode) as manifest_handle,
            _JsonlPublishBuffer(filter_audit_path, mode=file_mode) as filter_audit_handle,
            _JsonlPublishBuffer(rewrite_audit_path, mode=file_mode) as rewrite_audit_handle,
            _JsonlPublishBuffer(filtered_out_path, mode=file_mode) as rejected_handle,
            _JsonlPublishBuffer(pipeline_audit_path, mode=file_mode) as pipeline_audit_handle,
        ):
            filter_progress = tqdm(
                total=selected_total,
                desc="Filtering trajectories",
                unit="trajectory",
            )
            rewrite_progress = tqdm(desc="Rewriting trajectories", unit="trajectory")
            keep_order: list[int] = []
            pending_rewrites: dict[int, dict[str, Any]] = {}
            next_keep_position = 0
            checkpoint_records = 0

            def refresh_filter_progress() -> None:
                """Show the live disposition counts alongside filter progress."""

                filter_progress.set_postfix(
                    keep=summary["filter_keep"],
                    rejected=summary["filter_rejected"],
                    errors=summary["filter_errors"],
                    refresh=False,
                )
                filter_progress.refresh()

            def refresh_rewrite_progress() -> None:
                """Show verifier activity and successful trajectory integration live."""

                rewrite_progress.set_postfix(
                    ok=summary["rewrite_ok"],
                    verification=summary["verification_requests"],
                    integrated=summary["integrated_verifications"],
                    visual_checks=summary["terminal_visual_evidence_checks"],
                    visual_trajectory_complaints=summary["terminal_visual_evidence_trajectory_quality_complaint_checks"],
                    visual_writer_cannot_read=summary["terminal_visual_evidence_writer_cannot_read_checks"],
                    final_image_replacements=summary["terminal_visual_evidence_replacements"],
                    visual_filters=summary["terminal_visual_evidence_filters"],
                    accepted=summary["accepted_records"],
                    errors=summary["rewrite_errors"],
                    refresh=False,
                )
                rewrite_progress.refresh()

            def checkpoint() -> None:
                _flush_handles(
                    (
                        kept_handle,
                        manifest_handle,
                        filter_audit_handle,
                        rewrite_audit_handle,
                        rejected_handle,
                        pipeline_audit_handle,
                        final_handle,
                    )
                )

            def mark_terminal_record() -> None:
                """Checkpoint after completed sample dispositions, not intermediate events."""

                nonlocal checkpoint_records
                checkpoint_records += 1
                if checkpoint_records % args.checkpoint_every == 0:
                    checkpoint()

            def commit_rewrite(event: dict[str, Any]) -> None:
                """Write one completed rewrite in stable source order."""

                manifest = event["manifest"]
                original_record = event["original_record"]
                rewritten_record = event["rewritten_record"]
                rewrite_audit = event["rewrite_audit"]
                index = int(manifest["record_index"])
                if int(rewrite_audit.get("record_index", index)) != index:
                    raise RuntimeError("rewrite audit no longer matches its source record")
                _write_jsonl_record(rewrite_audit_handle, rewrite_audit)
                rewrite_progress.update(1)
                filter_audit = manifest["filter_audit"]
                summary["verification_requests"] += int(
                    rewrite_audit.get("verification_requests", 0)
                )
                summary["integrated_verifications"] += int(
                    rewrite_audit.get("integrated_verifications", 0)
                )
                summary["terminal_visual_evidence_checks"] += int(
                    rewrite_audit.get("terminal_visual_evidence_checks", 0)
                )
                summary["terminal_visual_evidence_trajectory_quality_complaint_checks"] += int(
                    rewrite_audit.get("terminal_visual_evidence_trajectory_quality_complaint_checks", 0)
                )
                summary["terminal_visual_evidence_writer_cannot_read_checks"] += int(
                    rewrite_audit.get("terminal_visual_evidence_writer_cannot_read_checks", 0)
                )
                summary["terminal_visual_evidence_replacements"] += int(
                    rewrite_audit.get("terminal_visual_evidence_replacements", 0)
                )
                summary["terminal_visual_evidence_filters"] += int(
                    rewrite_audit.get("terminal_visual_evidence_filters", 0)
                )
                _print_auxiliary_process_debug(
                    original_record,
                    rewrite_audit,
                    args.debug_output,
                )
                # _optimize_record deliberately suppresses its worker-thread
                # printer when compare=True: the standalone expander prints
                # the pair in its ordered result collector.  Build has its
                # own ordered collector, so reproduce that final publication
                # here after the sample has been committed.
                if args.compare:
                    _print_compare_trajectories(
                        original_record,
                        rewritten_record,
                        _record_id(original_record, index),
                        args.debug_output,
                    )

                if rewrite_audit.get("status") != "ok":
                    summary["rewrite_errors"] += 1
                    _write_jsonl_record(
                        pipeline_audit_handle,
                        _pipeline_record(
                            index=index,
                            record=original_record,
                            filter_audit=filter_audit,
                            rewrite_audit=rewrite_audit,
                            disposition="rewrite_error_excluded",
                            error=str(rewrite_audit.get("error") or "unknown rewrite error"),
                        ),
                    )
                    refresh_rewrite_progress()
                    mark_terminal_record()
                    return

                summary["rewrite_ok"] += 1
                if _has_rewrite_filter_recommendation(rewrite_audit):
                    summary["rewrite_filter_recommended"] += 1
                    _write_jsonl_record(
                        pipeline_audit_handle,
                        _pipeline_record(
                            index=index,
                            record=original_record,
                            filter_audit=filter_audit,
                            rewrite_audit=rewrite_audit,
                            disposition="verification_affirmed_candidate_excluded",
                        ),
                    )
                    refresh_rewrite_progress()
                    mark_terminal_record()
                    return

                try:
                    train_record, image_audit = _materialize_record_images(
                        rewritten_record,
                        original_record,
                        source_dataset_dir=source_dataset_dir,
                        output_dir=output_dir,
                        verification_dir=verification_image_dir,
                        copy_new_images=not args.dry_run,
                    )
                except Exception as exc:
                    summary["image_errors"] += 1
                    _write_jsonl_record(
                        pipeline_audit_handle,
                        _pipeline_record(
                            index=index,
                            record=original_record,
                            filter_audit=filter_audit,
                            rewrite_audit=rewrite_audit,
                            disposition="new_verification_image_error_excluded",
                            error=repr(exc),
                        ),
                    )
                    refresh_rewrite_progress()
                    mark_terminal_record()
                    return

                summary["new_verification_images"] += int(image_audit["new_verification_image_count"])
                if not image_audit["image_alignment_ok"]:
                    summary["image_errors"] += 1
                    _write_jsonl_record(
                        pipeline_audit_handle,
                        _pipeline_record(
                            index=index,
                            record=original_record,
                            filter_audit=filter_audit,
                            rewrite_audit=rewrite_audit,
                            image_audit=image_audit,
                            disposition="image_placeholder_misalignment_excluded",
                        ),
                    )
                    refresh_rewrite_progress()
                    mark_terminal_record()
                    return

                summary["accepted_records"] += 1
                if final_handle is not None:
                    _write_jsonl_record(final_handle, train_record)
                    final_record_ids.add(_record_id(train_record, index))
                    summary["training_records"] = len(final_record_ids)
                _write_jsonl_record(
                    pipeline_audit_handle,
                    _pipeline_record(
                        index=index,
                        record=original_record,
                        filter_audit=filter_audit,
                        rewrite_audit=rewrite_audit,
                        image_audit=image_audit,
                        disposition="accepted" if not args.dry_run else "dry_run_accepted",
                    ),
                )
                refresh_rewrite_progress()
                mark_terminal_record()

            def flush_completed_rewrites() -> None:
                nonlocal next_keep_position
                while (
                    next_keep_position < len(keep_order)
                    and keep_order[next_keep_position] in pending_rewrites
                ):
                    index = keep_order[next_keep_position]
                    commit_rewrite(pending_rewrites.pop(index))
                    next_keep_position += 1

            while True:
                event = writer_queue.get()
                event_type = event.get("type")
                if event_type == "filter_result":
                    index = int(event["record_index"])
                    record = event["record"]
                    verdict = event["verdict"]
                    filter_progress.update(1)
                    summary["selected"] += 1
                    _write_jsonl_record(filter_audit_handle, verdict)
                    if verdict.get("decision") == "keep":
                        manifest = {
                            "record_index": index,
                            "record_id": _record_id(record, index),
                            "filter_audit": verdict,
                        }
                        _write_jsonl_record(kept_handle, record)
                        _write_jsonl_record(manifest_handle, manifest)
                        summary["filter_keep"] += 1
                        keep_order.append(index)
                        rewrite_progress.total = summary["filter_keep"]
                        refresh_filter_progress()
                        refresh_rewrite_progress()
                        continue

                    if verdict.get("decision") == "filter":
                        summary["filter_rejected"] += 1
                        disposition = "filtered_information_leakage"
                    else:
                        summary["filter_errors"] += 1
                        disposition = "filter_error_excluded"
                    _write_jsonl_record(
                        rejected_handle,
                        {
                            "record_index": index,
                            "record_id": _record_id(record, index),
                            "filter": verdict,
                            "record": record,
                        },
                    )
                    _write_jsonl_record(
                        pipeline_audit_handle,
                        _pipeline_record(
                            index=index,
                            record=record,
                            filter_audit=verdict,
                            disposition=disposition,
                        ),
                    )
                    refresh_filter_progress()
                    mark_terminal_record()
                    continue

                if event_type == "resume_terminal":
                    filter_progress.update(1)
                    summary["selected"] += 1
                    summary["resumed_terminal_records"] += 1
                    refresh_filter_progress()
                    mark_terminal_record()
                    continue

                if event_type == "resume_final_only":
                    index = int(event["record_index"])
                    record = event["record"]
                    filter_progress.update(1)
                    summary["selected"] += 1
                    summary["resumed_final_only_records"] += 1
                    _write_jsonl_record(
                        pipeline_audit_handle,
                        {
                            "record_index": index,
                            "record_id": _record_id(record, index),
                            "disposition": "accepted_recovered_from_train_jsonl",
                            "resume": True,
                        },
                    )
                    refresh_filter_progress()
                    mark_terminal_record()
                    continue

                if event_type == "rewrite_result":
                    index = int(event["manifest"]["record_index"])
                    pending_rewrites[index] = event
                    flush_completed_rewrites()
                    continue

                if event_type == "pipeline_error":
                    summary["pipeline_errors"].append(
                        {"stage": event.get("stage"), "error": event.get("error")}
                    )
                    continue

                if event_type == "pipeline_complete":
                    matched_ids.clear()
                    matched_ids.update(event.get("matched_sample_ids") or [])
                    flush_completed_rewrites()
                    if next_keep_position != len(keep_order) or pending_rewrites:
                        summary["pipeline_errors"].append(
                            {
                                "stage": "writer",
                                "error": "pipeline completed before every kept trajectory produced a rewrite result",
                            }
                        )
                    filter_progress.total = summary["selected"]
                    refresh_filter_progress()
                    refresh_rewrite_progress()
                    checkpoint()
                    break

                summary["pipeline_errors"].append(
                    {"stage": "writer", "error": f"unknown pipeline event: {event_type!r}"}
                )
            filter_progress.close()
            rewrite_progress.close()
    finally:
        if final_handle is not None:
            final_handle.flush()
        filter_thread.join(timeout=5.0)

    if not args.dry_run:
        _write_dataset_info(output_dir / "dataset_info.json", args.dataset_name)
    summary["requested_sample_ids"] = sorted(requested_ids) if requested_ids else None
    summary["matched_sample_ids"] = sorted(matched_ids) if requested_ids else None
    summary["missing_sample_ids"] = sorted(requested_ids - matched_ids) if requested_ids else []
    summary.update(
        {
            "final_training_jsonl": str(final_path) if not args.dry_run else None,
            "dataset_info": str(output_dir / "dataset_info.json") if not args.dry_run else None,
            "filter_audit_jsonl": str(filter_audit_path),
            "rewrite_audit_jsonl": str(rewrite_audit_path),
            "pipeline_audit_jsonl": str(pipeline_audit_path),
            "filtered_out_jsonl": str(filtered_out_path),
            "kept_after_filter_jsonl": str(kept_path),
            "verification_images_dir": str(verification_image_dir) if not args.dry_run else None,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not (
        summary["filter_errors"]
        or summary["rewrite_errors"]
        or summary["image_errors"]
        or summary["pipeline_errors"]
        or (requested_ids and not matched_ids)
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
