"""CLI: inspect offline, then run resumable augmentation into a new dataset."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4
from .timing import Timer
from .schema import Config, VERSION
from .storage import Cache, digest, file_digest, read_json, write_json
from .trajectory_index import index_record, resolve_path
from .parallel import completed_records
from .outcomes import disposition, should_process, exportable
from .resume_version import resume_manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument(
        "--llm-cache-dir",
        type=Path,
        help="Optional reusable LLM cache directory across workdirs",
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--mode", choices=["index", "augment"], default="index")
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--record-workers", type=int, help="Override concurrent record workers"
    )
    p.add_argument("--indices", help="Comma-separated zero-based record indices")
    p.add_argument("--selected-only", action="store_true")
    p.add_argument(
        "--drop-quarantined",
        action="store_true",
        help="Exclude quarantined records from the new export; originals remain untouched",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed checkpoints and retry execution failures; quality rejections remain excluded",
    )
    p.add_argument(
        "--resume-allow-code-change",
        action="store_true",
        help="With --resume, reuse old-code checkpoints and record a code migration; input/config must still match",
    )
    p.add_argument(
        "--retry-quarantined",
        action="store_true",
        help="Retry execution failures only (quality rejections are terminal)",
    )
    p.add_argument(
        "--drop-failure-kinds",
        nargs="*",
        default=[],
        choices=[
            "source_unavailable",
            "anchor_unresolved",
            "candidate_composition",
            "selection_exhausted",
            "retry_failed",
            "structure_rejected",
            "review_rejected",
            "model_or_tool_error",
            "index_error",
        ],
        help="Exclude only explicitly selected failure categories from the new export",
    )
    return p


def _main(argv, batch):
    args = parser().parse_args(argv)
    if args.resume_allow_code_change and not args.resume:
        raise ValueError("--resume-allow-code-change requires --resume")
    batch.path = args.workdir / "timing.jsonl"
    batch.emit({"event": "run_start", "started_at": batch.started_at})
    batch.record_runs = []
    config = Config(**(read_json(args.config) if args.config else {}))
    if args.record_workers is not None:
        config.record_workers = args.record_workers
    config.validate()
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.output and args.output.resolve() == args.input.resolve():
        raise ValueError("Output must not overwrite input")
    if args.mode == "augment" and not args.output:
        raise ValueError("--output required for augment")
    with batch.measure("input_load"):
        records = read_json(args.input)
    if not isinstance(records, list):
        raise ValueError("Input must be a ShareGPT JSON array")
    indices = (
        sorted(set(map(int, args.indices.split(","))))
        if args.indices
        else list(range(len(records)))
    )
    if args.limit is not None:
        indices = indices[: args.limit]
    if any(i < 0 or i >= len(records) for i in indices):
        raise ValueError("Record index out of range")
    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_digest": file_digest(args.input),
        "code_digest": digest(
            {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")}
        ),
        "config": asdict(config),
    }
    manifest_path = args.workdir / "manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    manifest = resume_manifest(
        previous,
        manifest,
        resume=args.resume,
        allow_code_change=args.resume_allow_code_change,
        run_id=batch.context["run_id"],
        started_at=batch.started_at,
    )
    write_json(manifest_path, manifest)
    legacy_code = manifest.get("legacy_code_digest", manifest["code_digest"])
    if previous and previous.get("code_digest") != manifest["code_digest"]:
        print(
            json.dumps(
                {
                    "event": "code_migration",
                    "from": previous["code_digest"],
                    "to": manifest["code_digest"],
                    "note": "Completed checkpoints reused without revalidation",
                }
            ),
            flush=True,
        )
    if args.mode == "index":
        rows, counts = [], Counter()
        for i in indices:
            try:
                windows = index_record(records[i])
                rows.append(
                    {
                        "index": i,
                        "id": records[i].get("id"),
                        "windows": [
                            {
                                **asdict(w),
                                "requires_progress_audit": True,
                            }
                            for w in windows
                        ],
                    }
                )
                counts["records"] += 1
                counts["with_i2i"] += bool(windows)
                counts["windows"] += len(windows)
                for w in windows:
                    counts[f"original_views_{len(w.reads)}"] += 1
                    counts["windows_pending_progress_audit"] += 1
            except Exception as exc:
                counts["index_errors"] += 1
                rows.append({"index": i, "error": str(exc)})
        write_json(args.workdir / "index.json", rows)
        write_json(args.workdir / "index_summary.json", dict(counts))
        batch.summary_path = args.workdir / "index_summary.json"
        batch.summary_payload = dict(counts)
        print(json.dumps(dict(counts), ensure_ascii=False))
        return 0

    from synthesis import model_worker
    from .llm import JsonModel
    from .replay_i2i import ToolBackend
    from .pipeline import process_record

    model_worker.LLM_RETRY_COUNT = config.llm_retries
    cache = Cache(args.workdir / "cache")
    llm_cache = Cache(
        (args.llm_cache_dir or (args.workdir / "llm_cache")) / manifest["code_digest"]
    )
    model = JsonModel(llm_cache, config.max_tokens, config.gpt_reasoning_effort)
    backend = ToolBackend(args.workdir / "resources", config, cache)
    audits, results = [], {}

    def process_one(i):
        record_timer = Timer(batch.path, run_id=batch.context["run_id"], index=i)
        checkpoint = args.workdir / "records" / f"{i}.json"
        with record_timer.measure("checkpoint_read"):
            saved = read_json(checkpoint) if checkpoint.exists() else None
        if saved is not None and saved.get("index") != i:
            raise ValueError("Invalid record checkpoint")
        process_needed = should_process(saved, args.resume, args.retry_quarantined)
        if process_needed:
            attempt = (saved.get("attempt", 1) if saved else 0) + 1
            # Keep successful search/download caches, but do not replay a cached
            # malformed or schema-invalid LLM response on every resume.
            record_model = (
                model
                if attempt == 1
                else JsonModel(
                    Cache(
                        args.workdir
                        / "retry_llm_cache"
                        / manifest["code_digest"]
                        / str(i)
                        / str(attempt)
                    ),
                    config.max_tokens,
                    config.gpt_reasoning_effort,
                )
            )
            try:
                with record_timer.activate(), record_timer.measure("process_record"):
                    record, audit = process_record(
                        records[i],
                        args.input.parent,
                        args.workdir / "records" / str(i),
                        config,
                        backend,
                        record_model,
                        cache,
                    )
            except Exception as exc:
                record, audit = (
                    records[i],
                    {
                        "status": "quarantined",
                        "failure_kind": "index_error",
                        "error": str(exc),
                        "windows": [],
                    },
                )
            audit["timing"] = record_timer.summary()
            audit["disposition"] = disposition(audit)
            saved = {
                "index": i,
                "record": record,
                "audit": audit,
                "attempt": attempt,
                "code_digest": manifest["code_digest"],
            }
            with record_timer.measure("checkpoint_write"):
                write_json(checkpoint, saved)
        elif "code_digest" not in saved:
            saved["code_digest"] = legacy_code
            with record_timer.measure("checkpoint_provenance"):
                write_json(checkpoint, saved)
        saved["audit"]["disposition"] = disposition(saved["audit"])
        current_timing = {
            **record_timer.summary(),
            "reused_checkpoint": not process_needed,
            "status": saved["audit"]["status"],
        }
        return saved, current_timing, process_needed

    for i, (saved, current_timing, process_needed) in completed_records(
        process_one, indices, config.record_workers
    ):
        results[i] = saved["record"]
        audits.append(
            {"index": i, **saved["audit"], "code_digest": saved["code_digest"]}
        )
        batch.record_runs.append(current_timing)
        batch.emit({"event": "record_complete", **current_timing})
        print(
            json.dumps(
                {
                    "index": i,
                    "status": saved["audit"]["status"],
                    "disposition": disposition(saved["audit"]),
                    "elapsed_s": current_timing["elapsed_s"],
                    "reused_checkpoint": not process_needed,
                }
            ),
            flush=True,
        )
    audits.sort(key=lambda item: item["index"])
    batch.record_runs.sort(key=lambda item: item["index"])
    if not args.selected_only:
        # Full export includes successful work from earlier selected batches.
        for checkpoint in sorted((args.workdir / "records").glob("*.json")):
            if not checkpoint.stem.isdigit():
                continue
            i = int(checkpoint.stem)
            if i in results:
                continue
            saved = read_json(checkpoint)
            if saved.get("index") != i or i >= len(records):
                raise ValueError("Invalid record checkpoint")
            results[i] = saved["record"]
            saved["audit"]["disposition"] = disposition(saved["audit"])
            audits.append(
                {
                    "index": i,
                    **saved["audit"],
                    "code_digest": saved.get("code_digest", legacy_code),
                }
            )
        audits.sort(key=lambda item: item["index"])
    quarantined = {a["index"] for a in audits if a["status"] == "quarantined"}
    excluded_by_kind = {
        a["index"]
        for a in audits
        if a["status"] == "quarantined"
        and ({a.get("failure_kind")} | {w.get("failure_kind") for w in a["windows"]})
        & set(args.drop_failure_kinds)
    }
    export = []
    discarded = {a["index"] for a in audits if a.get("discard")}
    by_index = {a["index"]: a for a in audits}
    quality_rejected = [a for a in audits if disposition(a) == "quality_rejected"]
    retry_pending = [a for a in audits if disposition(a) == "retry_pending"]
    pending_indices = []
    for i in indices if args.selected_only else range(len(records)):
        if i not in by_index:
            pending_indices.append(i)
            continue
        if (
            not exportable(by_index[i])
            or i in discarded
            or (args.drop_quarantined and i in quarantined)
            or i in excluded_by_kind
        ):
            continue
        record = dict(results.get(i, records[i]))
        # Export may live outside the original dataset root.
        record["images"] = [
            resolve_path(p, args.input.parent) for p in record.get("images", [])
        ]
        export.append(record)
    summary = {
        "dispositions": dict(Counter(disposition(a) for a in audits)),
        "record_code_digests": dict(Counter(a["code_digest"] for a in audits)),
        "current_code_digest": manifest["code_digest"],
        "code_history": manifest.get("code_history", []),
        "quality_rejected_indices": [a["index"] for a in quality_rejected],
        "retry_pending_indices": [a["index"] for a in retry_pending],
        "unprocessed_indices": pending_indices,
        "discarded_global_logic_indices": sorted(discarded),
        "records": dict(Counter(a["status"] for a in audits)),
        "branches": dict(
            Counter(w.get("branch", w["status"]) for a in audits for w in a["windows"])
        ),
        "wrong_counts": dict(
            Counter(
                str(w["wrong_count"])
                for a in audits
                for w in a["windows"]
                if "wrong_count" in w
            )
        ),
        "positive_ranks": dict(
            Counter(
                str(w["positive_rank"])
                for a in audits
                for w in a["windows"]
                if "positive_rank" in w
            )
        ),
        "export_records": len(export),
        "successful_error_exclusion_records": sum(
            a["status"] == "rewritten"
            and any(w.get("wrong_count", 0) > 0 for w in a["windows"])
            for a in audits
        ),
        "flow_version": "selector_writer_v1",
        "skip_reasons": dict(
            Counter(
                w["skip_reason"]
                for a in audits
                for w in a["windows"]
                if "skip_reason" in w
            )
        ),
        "failure_kinds": dict(
            Counter(
                w["failure_kind"]
                for a in audits
                for w in a["windows"]
                if "failure_kind" in w
            )
        ),
        "judge_uses_distinct_alias": config.judge_model
        not in {config.model, config.verifier_model},
        "note": "Only completed augmented/keep_original records are exported. Quality rejections, execution failures and unprocessed records are excluded. Resume retries execution failures only.",
    }
    with batch.measure("export"):
        write_json(args.workdir / "audit.json", audits)
        write_json(args.workdir / "quality_rejected.json", quality_rejected)
        write_json(args.workdir / "retry_pending.json", retry_pending)
        write_json(args.output, export)
    batch.summary_path = args.workdir / "summary.json"
    batch.summary_payload = summary
    return 0


def _run_with_batch(argv, batch):
    status = "error"
    try:
        result = _main(argv, batch)
        status = "ok"
        return result
    finally:
        if batch.path is not None:
            timing = {
                **batch.summary(),
                "status": status,
                "records": getattr(batch, "record_runs", []),
            }
            write_json(
                batch.path.parent / "timing_runs" / (batch.context["run_id"] + ".json"),
                timing,
            )
            batch.emit(
                {"event": "run_end", "elapsed_s": timing["elapsed_s"], "status": status}
            )
            batch.close()
            if status == "ok" and hasattr(batch, "summary_payload"):
                summary = {**batch.summary_payload, "timing": timing}
                write_json(batch.summary_path, summary)
                print(
                    json.dumps(
                        {
                            "run_id": batch.context["run_id"],
                            "elapsed_s": timing["elapsed_s"],
                            "status": status,
                            "timing_report": str(batch.summary_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


def main(argv=None):
    batch = Timer(run_id=uuid4().hex)
    try:
        return _run_with_batch(argv, batch)
    finally:
        # Also drain if computation or final report writing raised an exception.
        batch.close()


if __name__ == "__main__":
    raise SystemExit(main())
