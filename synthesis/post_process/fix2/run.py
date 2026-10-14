"""Command line entry point for the FVQA hard-case SFT pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .cases import load_cases, load_public_cases, prepare_cases_from_parquet, private_case_map, write_public_cases
from .base_router import watch_base_evaluations
from .collector import collect_cases
from .evaluation import judge_expert_records
from .filters import filter_records, quality_filter_details, summarize_filter_audit
from .io import append_jsonl, iter_json_records, write_json
from .livevqa import DEFAULT_DATASET_REPO, DEFAULT_IMAGE_BASE_URL, prepare_livevqa_from_parquet
from .mining import mine_base_failures
from .publisher import BaseTrajectoryCostIndex, publish_base_failures
from .probe import prepare_evaluation_probe_from_parquet
from .quality import review_trajectory_quality
from .schema import CollectorConfig, FilterConfig
from .style import build_legacy_style_examples_file, profile_sharegpt_file
from .task_queue import SqliteTaskQueue
from .visual import review_blind_visual_requests
from .worker import merge_atomic_teacher_results, merge_teacher_results, reconcile_teacher_results, run_teacher_workers


def _add_prepare(subparsers: Any) -> None:
    parser = subparsers.add_parser("prepare", help="Materialize FVQA images and write canonical case JSONL.")
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--allow-nontrain", action="store_true", help="Required for validation/test materialization.")
    parser.add_argument("--question-field")
    parser.add_argument("--answer-field")
    parser.add_argument("--image-field")
    parser.add_argument("--id-field")
    parser.add_argument("--start", type=int, default=0, help="Zero-based FVQA source-row offset.")
    parser.add_argument("--limit", type=int, default=0, help="Rows to materialize; 0 means all remaining rows.")
    parser.add_argument("--resume", action="store_true", help="Resume a partial private manifest by case ID.")
    parser.add_argument("--progress", action="store_true", help="Show a tqdm progress bar while images are materialized.")
    parser.add_argument(
        "--materialize-workers",
        type=int,
        default=1,
        help="Parallel image decode/hash/write workers; manifest appends remain coordinator-serial.",
    )
    parser.add_argument("--output-public-cases", help="Optional writer-safe manifest with no gold answers.")


def _add_prepare_livevqa(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare-livevqa",
        help="Sample LiveVQA train directly for teacher collection; no base-model filtering is used.",
    )
    parser.add_argument("--input-parquet", required=True, help="LiveVQA train parquet built from ONE-Lab/LiveVQA-new.")
    parser.add_argument("--output-cases", required=True, help="Private manifest retaining all valid reference answers.")
    parser.add_argument("--output-public-cases", required=True, help="Teacher-safe manifest with no labels/options.")
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--level1-count", type=int, required=True, help="Samples N Level 1 plus exactly 3N Level 2 cases.")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    parser.add_argument("--archive-cache-dir", help="Local directory for lazily downloaded LiveVQA source ZIP archives.")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO, help="Hub dataset repository containing LiveVQA parquet shards and image ZIPs.")
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--download-timeout-s", type=float, default=45.0)
    parser.add_argument("--max-image-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")


def _add_profile(subparsers: Any) -> None:
    parser = subparsers.add_parser("profile-style", help="Measure ShareGPT surface style without copying its content.")
    parser.add_argument("--input-sharegpt", required=True)
    parser.add_argument("--output-profile", required=True)
    parser.add_argument("--limit", type=int)


def _add_legacy_style_examples(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "build-legacy-style-examples",
        help="Extract a small set of non-final legacy ShareGPT assistant turns as style anchors.",
    )
    parser.add_argument("--input-sharegpt", required=True)
    parser.add_argument("--output-examples", required=True)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--scan-limit", type=int, default=2_000)


def _add_prepare_evaluation_probe(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare-evaluation-probe",
        help="Randomly sample already-judged test failures for evaluation only; never an SFT source.",
    )
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--base-eval-details", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-public-cases", required=True)
    parser.add_argument("--output-failure-manifest", required=True)
    parser.add_argument("--output-selection-report", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260918)


def _add_mine(subparsers: Any) -> None:
    parser = subparsers.add_parser("mine-failures", help="Join LLM-evaluated base failures back to canonical FVQA cases.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--base-eval-details", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-public-cases", help="Optional writer-safe manifest for the selected failures.")
    parser.add_argument("--limit", type=int)


def _add_collect(subparsers: Any) -> None:
    parser = subparsers.add_parser("collect", help="Generate hidden-answer expert trajectories with production tools.")
    parser.add_argument("--cases", required=True, help="Private canonical case manifest; converted to public view before collection.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", required=True, help="Registered SFT / Model Worker alias.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--api-mode", choices=["manual_react", "native_tools", "responses"], default="manual_react")
    parser.add_argument("--client-type", default="azure_openai")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--style-guide-file")
    parser.add_argument("--style-examples-file")
    parser.add_argument("--print-rounds", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)


def _add_judge(subparsers: Any) -> None:
    parser = subparsers.add_parser("judge-expert", help="Independently score expert final answers against labels.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--private-cases", required=True, help="Private canonical cases; never sent to the writer.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-style", choices=["local", "official"], default="local")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")


def _add_trajectory_quality_judge(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "judge-trajectory-quality",
        help="Label-blind evidence/logic review; nonvisual traces may pass when well grounded.",
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-alias", default="gemini38flash_internal_azure")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")


def _add_visual(subparsers: Any) -> None:
    parser = subparsers.add_parser("blind-visual-review", help="Review original/candidate images without search metadata or labels.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")


def _add_filter(subparsers: Any) -> None:
    parser = subparsers.add_parser("filter", help="Apply answer, provenance, visual and leakage gates.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--private-cases", required=True, help="Private canonical cases for the label-leakage audit only.")
    parser.add_argument("--answer-reviews", required=True)
    parser.add_argument("--trajectory-quality-reviews", required=True)
    parser.add_argument("--blind-reviews", required=True)
    parser.add_argument("--accepted-output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--require-visual-compare", action="store_true", help="Restrict this export to strong visual-comparison trajectories.")
    parser.add_argument("--allow-nonvisual", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-candidate-rejection", action="store_true", help="Require a grounded rejection only for an explicitly negative-candidate branch.")
    parser.add_argument("--allow-no-rejection", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-no-blind-review", action="store_true")
    parser.add_argument("--require-external-fact-evidence", action="store_true")
    parser.add_argument("--min-opened-visual-candidates", type=int, default=1)
    parser.add_argument("--max-assistant-turns", type=int, default=30)


def _add_filter_audit_summary(subparsers: Any) -> None:
    parser = subparsers.add_parser("summarize-filter-audit", help="Summarize final filter decisions and rejection reasons.")
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--show-quality-filters", action="store_true", help="Also print label-free details for quality-review rejections.")


def _add_queue_commands(subparsers: Any) -> None:
    base_eval = subparsers.add_parser(
        "watch-base-eval",
        help="Continuously judge new run_infer trajectory files while base workers are still running.",
    )
    base_eval.add_argument("--trajectory-dir", required=True)
    base_eval.add_argument("--answer-file", required=True)
    base_eval.add_argument("--output-path", required=True)
    base_eval.add_argument("--judge-model-alias", required=True)
    base_eval.add_argument("--judge-max-tokens", type=int, default=1024)
    base_eval.add_argument("--max-workers", type=int, default=8)
    base_eval.add_argument("--prompt-style", choices=["local", "official"], default="local")
    base_eval.add_argument("--poll-s", type=float, default=10.0)
    base_eval.add_argument("--once", action="store_true")
    base_eval.add_argument("--debug", action="store_true", help="Log base-trajectory discovery and each evaluator summary.")

    enqueue = subparsers.add_parser(
        "enqueue-failures",
        help="Publish independently judged base failures to the public teacher queue.",
    )
    enqueue.add_argument("--cases", required=True, help="Private canonical case manifest used only for base-error routing.")
    enqueue.add_argument("--base-eval-details", required=True)
    enqueue.add_argument(
        "--base-trajectory-dir",
        help="run_infer trajectory directory used only to route correct but tool-expensive base cases.",
    )
    enqueue.add_argument(
        "--min-correct-tool-calls",
        type=int,
        default=0,
        help="Also enqueue independently correct base cases with at least this many tool calls; 0 disables this bucket.",
    )
    enqueue.add_argument("--queue-db", required=True, help="Local coordinator SQLite DB; do not place on HDFS/NFS.")
    enqueue.add_argument("--queue-name", default="teacher_collect")
    enqueue.add_argument("--max-attempts", type=int, default=3)
    enqueue.add_argument(
        "--max-ready-tasks",
        type=int,
        default=0,
        help="Backpressure cap for pending+leased tasks; 0 means unlimited.",
    )
    enqueue.add_argument("--limit", type=int, help="Global maximum task count for this queue; 0 means no cap.")
    enqueue.add_argument("--follow", action="store_true", help="Poll the evaluator details file and publish new failures continuously.")
    enqueue.add_argument("--poll-s", type=float, default=10.0)

    enqueue_public = subparsers.add_parser(
        "enqueue-public-cases",
        help="Queue an already-public writer manifest; no labels/base judge data are accepted.",
    )
    enqueue_public.add_argument("--public-cases", required=True)
    enqueue_public.add_argument("--queue-db", required=True, help="Local coordinator SQLite DB; do not place on HDFS/NFS.")
    enqueue_public.add_argument("--queue-name", default="teacher_collect")
    enqueue_public.add_argument("--max-attempts", type=int, default=3)

    worker = subparsers.add_parser(
        "teacher-workers",
        help="Run a local teacher worker group; every worker leases one public sample at a time.",
    )
    worker.add_argument("--queue-db", required=True, help="Local coordinator SQLite DB; do not place on HDFS/NFS.")
    worker.add_argument("--results-dir", required=True, help="Atomic per-case raw results.")
    worker.add_argument("--model", required=True)
    worker.add_argument("--workdir", required=True)
    worker.add_argument("--workers", type=int, default=1)
    worker.add_argument("--queue-name", default="teacher_collect")
    worker.add_argument("--lease-s", type=float, default=3600.0)
    worker.add_argument("--poll-s", type=float, default=1.0)
    worker.add_argument("--idle-exit-s", type=float, default=30.0, help="0 waits indefinitely for producer tasks.")
    worker.add_argument("--max-tasks-per-worker", type=int)
    worker.add_argument("--api-mode", choices=["manual_react", "native_tools", "responses"], default="manual_react")
    worker.add_argument("--client-type", default="azure_openai")
    worker.add_argument("--max-tokens", type=int)
    worker.add_argument("--max-turns", type=int, default=8)
    worker.add_argument("--temperature", type=float)
    worker.add_argument("--timeout-s", type=float)
    worker.add_argument("--style-guide-file")
    worker.add_argument("--style-examples-file")
    worker.add_argument("--print-rounds", action="store_true")
    worker.add_argument("--debug", action="store_true", help="Log per-case queue lease, completion, and retry events.")

    merge = subparsers.add_parser("merge-teacher-results", help="Merge successful atomic per-case raw records deterministically.")
    merge.add_argument("--queue-db", required=True)
    merge.add_argument("--output-jsonl", required=True)
    merge.add_argument("--queue-name", default="teacher_collect")

    merge_atomic = subparsers.add_parser(
        "merge-atomic-teacher-results",
        help="Merge all immutable teacher_results/cases JSON records, independent of local queue state.",
    )
    merge_atomic.add_argument("--results-dir", required=True, help="Shared atomic teacher_results directory.")
    merge_atomic.add_argument("--output-jsonl", required=True)

    stats = subparsers.add_parser("queue-stats", help="Print durable task-state counts.")
    stats.add_argument("--queue-db", required=True)
    stats.add_argument("--queue-name", default="teacher_collect")

    requeue = subparsers.add_parser(
        "requeue-leased",
        help="Immediately return leased tasks after their whole worker group has stopped.",
    )
    requeue.add_argument("--queue-db", required=True, help="Local coordinator SQLite DB; do not place on HDFS/NFS.")
    requeue.add_argument("--queue-name", default="teacher_collect")
    requeue.add_argument("--reason", default="worker_group_restarted")

    reconcile = subparsers.add_parser(
        "reconcile-teacher-results",
        help="Recover queue successes from immutable teacher_results/cases JSON after local-state loss.",
    )
    reconcile.add_argument("--queue-db", required=True, help="Fresh coordinator-local SQLite queue DB; do not place on HDFS/NFS.")
    reconcile.add_argument("--results-dir", required=True, help="Shared immutable teacher_results directory to scan.")
    reconcile.add_argument("--queue-name", default="teacher_collect")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare(subparsers)
    _add_prepare_livevqa(subparsers)
    _add_profile(subparsers)
    _add_legacy_style_examples(subparsers)
    _add_prepare_evaluation_probe(subparsers)
    _add_mine(subparsers)
    _add_collect(subparsers)
    _add_judge(subparsers)
    _add_trajectory_quality_judge(subparsers)
    _add_visual(subparsers)
    _add_filter(subparsers)
    _add_filter_audit_summary(subparsers)
    _add_queue_commands(subparsers)
    return parser


def _write_cases(path: str | Path, cases: list[Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing case manifest: {target}")
    for case in cases:
        append_jsonl(target, case.to_dict())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if args.split != "train" and not args.allow_nontrain:
            raise ValueError("prepare refuses a non-train split without --allow-nontrain")
        result = prepare_cases_from_parquet(
            args.input_parquet,
            output_jsonl=args.output_cases,
            image_dir=args.image_dir,
            split=args.split,
            question_field=args.question_field,
            answer_field=args.answer_field,
            image_field=args.image_field,
            id_field=args.id_field,
            allow_nontrain=args.allow_nontrain,
            progress=args.progress,
            start=args.start,
            limit=args.limit or None,
            resume=args.resume,
            materialize_workers=args.materialize_workers,
        )
        if args.output_public_cases:
            result["public_cases"] = write_public_cases(load_cases(args.output_cases), args.output_public_cases, resume=args.resume)
    elif args.command == "prepare-livevqa":
        result = prepare_livevqa_from_parquet(
            args.input_parquet,
            output_cases=args.output_cases,
            output_public_cases=args.output_public_cases,
            selection_report=args.selection_report,
            image_dir=args.image_dir,
            level1_count=args.level1_count,
            seed=args.seed,
            image_base_url=args.image_base_url,
            download_workers=args.download_workers,
            download_timeout_s=args.download_timeout_s,
            max_image_bytes=args.max_image_bytes,
            archive_cache_dir=args.archive_cache_dir,
            dataset_repo=args.dataset_repo,
            resume=args.resume,
            progress=args.progress,
        )
    elif args.command == "profile-style":
        result = profile_sharegpt_file(args.input_sharegpt, args.output_profile, limit=args.limit)
    elif args.command == "prepare-evaluation-probe":
        result = prepare_evaluation_probe_from_parquet(
            args.input_parquet,
            iter_json_records(args.base_eval_details),
            output_cases=args.output_cases,
            output_public_cases=args.output_public_cases,
            output_failure_manifest=args.output_failure_manifest,
            output_selection_report=args.output_selection_report,
            image_dir=args.image_dir,
            limit=args.limit,
            seed=args.seed,
        )
    elif args.command == "mine-failures":
        selected, manifest, result = mine_base_failures(
            load_cases(args.cases), iter_json_records(args.base_eval_details), max_cases=args.limit
        )
        _write_cases(args.output_cases, selected)
        if Path(args.output_manifest).exists():
            raise FileExistsError(f"Refusing to overwrite existing failure manifest: {args.output_manifest}")
        for item in manifest:
            append_jsonl(args.output_manifest, item.to_dict())
        if args.output_public_cases:
            result["public_cases"] = write_public_cases(selected, args.output_public_cases)
    elif args.command == "collect":
        result = collect_cases(
            (case.public_case() for case in load_cases(args.cases)),
            config=CollectorConfig(
                model=args.model,
                workdir=args.workdir,
                api_mode=args.api_mode,
                client_type=args.client_type,
                max_tokens=args.max_tokens,
                max_turns=args.max_turns,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                style_guide_file=args.style_guide_file,
                style_examples_file=args.style_examples_file,
                print_rounds=args.print_rounds,
            ),
            output_jsonl=args.output_jsonl,
            resume=args.resume,
            limit=args.limit,
        )
    elif args.command == "build-legacy-style-examples":
        result = build_legacy_style_examples_file(
            args.input_sharegpt,
            args.output_examples,
            max_examples=args.max_examples,
            max_chars=args.max_chars,
            scan_limit=args.scan_limit,
        )
    elif args.command == "judge-expert":
        result = judge_expert_records(
            args.input_jsonl,
            args.output_jsonl,
            model_alias=args.model_alias,
            private_cases=private_case_map(args.private_cases),
            max_tokens=args.max_tokens,
            prompt_style=args.prompt_style,
            resume=args.resume,
            max_workers=args.max_workers,
        )
    elif args.command == "judge-trajectory-quality":
        result = review_trajectory_quality(
            args.input_jsonl,
            args.output_jsonl,
            model_alias=args.model_alias,
            max_tokens=args.max_tokens,
            resume=args.resume,
            max_workers=args.max_workers,
        )
    elif args.command == "blind-visual-review":
        result = review_blind_visual_requests(
            args.input_jsonl,
            args.output_jsonl,
            model_alias=args.model_alias,
            max_tokens=args.max_tokens,
            resume=args.resume,
            max_workers=args.max_workers,
        )
    elif args.command == "filter":
        result = filter_records(
            args.input_jsonl,
            args.accepted_output,
            args.rejected_output,
            args.audit_output,
            answer_reviews=iter_json_records(args.answer_reviews),
            trajectory_quality_reviews=iter_json_records(args.trajectory_quality_reviews),
            blind_reviews=iter_json_records(args.blind_reviews),
            config=FilterConfig(
                require_visual_compare=args.require_visual_compare and not args.allow_nonvisual,
                min_opened_visual_candidates=args.min_opened_visual_candidates,
                require_candidate_rejection=args.require_candidate_rejection and not args.allow_no_rejection,
                require_blind_visual_review=not args.allow_no_blind_review,
                require_external_fact_evidence=args.require_external_fact_evidence,
                max_assistant_turns=args.max_assistant_turns,
            ),
            private_cases=private_case_map(args.private_cases),
        )
    elif args.command == "summarize-filter-audit":
        result = summarize_filter_audit(args.audit_jsonl)
        if args.show_quality_filters:
            result["quality_filter_details"] = quality_filter_details(args.audit_jsonl)
    elif args.command == "enqueue-failures":
        if args.poll_s <= 0:
            raise ValueError("--poll-s must be positive")
        if args.min_correct_tool_calls < 0:
            raise ValueError("--min-correct-tool-calls must be nonnegative")
        if args.min_correct_tool_calls and not args.base_trajectory_dir:
            raise ValueError("--base-trajectory-dir is required when --min-correct-tool-calls is positive")
        queue = SqliteTaskQueue(args.queue_db)
        cases = load_cases(args.cases)
        cost_index = (
            BaseTrajectoryCostIndex(args.base_trajectory_dir)
            if args.min_correct_tool_calls and args.base_trajectory_dir
            else None
        )
        while True:
            try:
                cost_refresh = cost_index.refresh() if cost_index is not None else None
                result = publish_base_failures(
                    cases,
                    # ``watch-base-eval`` appends judge rows concurrently.
                    # Preserve its completed prefix if this poll catches an
                    # in-flight final JSONL row; the next poll retries it.
                    iter_json_records(
                        args.base_eval_details,
                        allow_incomplete_final_line=True,
                    ),
                    queue=queue,
                    queue_name=args.queue_name,
                    max_attempts=args.max_attempts,
                    max_cases=args.limit if args.limit and args.limit > 0 else None,
                    max_ready_tasks=args.max_ready_tasks,
                    base_tool_call_counts=(cost_index.tool_call_counts if cost_index is not None else None),
                    min_correct_tool_calls=args.min_correct_tool_calls,
                )
                if cost_refresh is not None:
                    result.update({f"base_cost_{key}": value for key, value in cost_refresh.items()})
                    result["base_cost_indexed_total"] = len(cost_index.tool_call_counts)
            except FileNotFoundError:
                if not args.follow:
                    raise
                result = {"status": "waiting_for_base_evaluation_details"}
            result.update({f"queue_{key}": value for key, value in queue.stats(args.queue_name).items()})
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if not args.follow:
                return 0
            time.sleep(args.poll_s)
    elif args.command == "enqueue-public-cases":
        queue = SqliteTaskQueue(args.queue_db)
        counts = {"seen": 0, "published": 0, "already_present": 0}
        for case in load_public_cases(args.public_cases):
            counts["seen"] += 1
            if queue.enqueue(args.queue_name, case.case_id, case.to_dict(), max_attempts=args.max_attempts):
                counts["published"] += 1
            else:
                counts["already_present"] += 1
        result = {**counts, **{f"queue_{key}": value for key, value in queue.stats(args.queue_name).items()}}
    elif args.command == "watch-base-eval":
        result = watch_base_evaluations(
            trajectory_dir=args.trajectory_dir,
            answer_file=args.answer_file,
            output_path=args.output_path,
            judge_model_alias=args.judge_model_alias,
            judge_max_tokens=args.judge_max_tokens,
            max_workers=args.max_workers,
            prompt_style=args.prompt_style,
            poll_s=args.poll_s,
            once=args.once,
            debug=args.debug,
        )
    elif args.command == "teacher-workers":
        result = run_teacher_workers(
            queue_path=args.queue_db,
            results_dir=args.results_dir,
            collector_config=CollectorConfig(
                model=args.model,
                workdir=args.workdir,
                api_mode=args.api_mode,
                client_type=args.client_type,
                max_tokens=args.max_tokens,
                max_turns=args.max_turns,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                style_guide_file=args.style_guide_file,
                style_examples_file=args.style_examples_file,
                print_rounds=args.print_rounds,
            ),
            workers=args.workers,
            queue_name=args.queue_name,
            lease_s=args.lease_s,
            poll_s=args.poll_s,
            idle_exit_s=args.idle_exit_s,
            max_tasks_per_worker=args.max_tasks_per_worker,
            debug=args.debug,
        )
    elif args.command == "merge-teacher-results":
        result = merge_teacher_results(
            queue_path=args.queue_db,
            output_jsonl=args.output_jsonl,
            queue_name=args.queue_name,
        )
    elif args.command == "merge-atomic-teacher-results":
        result = merge_atomic_teacher_results(
            results_dir=args.results_dir,
            output_jsonl=args.output_jsonl,
        )
    elif args.command == "queue-stats":
        result = SqliteTaskQueue(args.queue_db).stats(args.queue_name)
    elif args.command == "requeue-leased":
        queue = SqliteTaskQueue(args.queue_db)
        returned = queue.requeue_leased(args.queue_name, reason=args.reason)
        result = {"requeued": returned, **queue.stats(args.queue_name)}
    elif args.command == "reconcile-teacher-results":
        result = reconcile_teacher_results(
            queue_path=args.queue_db,
            results_dir=args.results_dir,
            queue_name=args.queue_name,
        )
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
