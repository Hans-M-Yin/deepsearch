"""Rewrite questions for paths stored in an existing VQA output directory.

Unlike :mod:`synthesis.vqa.run_batch`, this entrypoint never samples graph
paths. It reconstructs persisted ``PathCandidate`` records from a source VQA
directory and normally reruns question writing, difficulty enhancement, and
verification. With ``--shortcut-repair-only``, it instead reuses each saved
final question and hop chain, running only shortcut audit/repair and
verification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback
from typing import Any, Iterable

from synthesis.model_worker import LLM_WORKER
from synthesis.store import JsonlGraphStore

from .batch_runner import VqaBatchRunner
from .path_sampler import SamplerConfiguration
from .pipeline import VqaGenerationError, VqaGenerationPipeline
from .question_writer import QuestionWriter
from .schemas import (
    EvidenceBundle,
    PathCandidate,
    QuestionDraft,
    SampleProgress,
    SampleStatus,
    TrajectoryStats,
    VqaSample,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            yield payload


def _path_from_record(record: dict[str, Any], *, source_path: Path, line_no: int) -> PathCandidate:
    raw_path = record.get("path")
    if not isinstance(raw_path, dict):
        raise ValueError(f"{source_path}:{line_no} does not contain a path record")
    raw_trajectory = raw_path.get("trajectory")
    if not isinstance(raw_trajectory, dict):
        raise ValueError(f"{source_path}:{line_no} path is missing trajectory")

    try:
        trajectory = TrajectoryStats(**raw_trajectory)
        return PathCandidate(
            path_id=str(raw_path["path_id"]),
            node_ids=[str(value) for value in raw_path["node_ids"]],
            edge_ids=[str(value) for value in raw_path["edge_ids"]],
            node_types=[str(value) for value in raw_path["node_types"]],
            edge_types=[str(value) for value in raw_path["edge_types"]],
            relations=[str(value) for value in raw_path["relations"]],
            target_node_id=str(raw_path["target_node_id"]),
            start_node_id=str(raw_path["start_node_id"]),
            trajectory=trajectory,
            exact_signature=str(raw_path.get("exact_signature") or ""),
            skeleton_signature=str(raw_path.get("skeleton_signature") or ""),
            core_signature=str(raw_path.get("core_signature") or ""),
            score=float(raw_path.get("score") or 0.0),
            metadata=dict(raw_path.get("metadata") or {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not reconstruct path at {source_path}:{line_no}") from exc


def load_source_paths(source_vqa_dir: Path) -> list[tuple[PathCandidate, str | None]]:
    samples_path = source_vqa_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Source VQA samples do not exist: {samples_path}")

    paths: list[tuple[PathCandidate, str | None]] = []
    seen_path_ids: set[str] = set()
    for line_no, record in enumerate(_iter_jsonl(samples_path), start=1):
        path = _path_from_record(record, source_path=samples_path, line_no=line_no)
        if path.path_id in seen_path_ids:
            raise ValueError(f"Duplicate path_id in source samples: {path.path_id}")
        seen_path_ids.add(path.path_id)
        paths.append((path, str(record.get("sample_id") or "").strip() or None))
    return paths


def load_source_question_records(source_vqa_dir: Path) -> dict[str, dict[str, Any]]:
    """Load source records keyed by path ID for repair-only rewriting."""
    samples_path = source_vqa_dir / "samples.jsonl"
    records: dict[str, dict[str, Any]] = {}
    for line_no, record in enumerate(_iter_jsonl(samples_path), start=1):
        path = _path_from_record(record, source_path=samples_path, line_no=line_no)
        if path.path_id in records:
            raise ValueError(f"Duplicate path_id in source samples: {path.path_id}")
        records[path.path_id] = record
    return records


def _source_stage(record: dict[str, Any], *names: str) -> dict[str, Any]:
    writer_outputs = record.get("writer_outputs") or {}
    for name in names:
        value = record.get(name) or writer_outputs.get(name)
        if isinstance(value, dict):
            return value
    return {}


def source_repair_drafts(record: dict[str, Any]) -> tuple[QuestionDraft, QuestionDraft]:
    """Recover the pre-repair enhanced question and its original draft from a saved sample."""
    final_stage = _source_stage(record, "obfuscated", "final")
    enhanced_stage = _source_stage(record, "enhanced", "polished")
    drafted_stage = _source_stage(record, "drafted", "draft")
    enhanced_question = str(
        record.get("enhanced_question")
        or record.get("final_question")
        or final_stage.get("question")
        or record.get("question")
        or enhanced_stage.get("question")
        or ""
    ).strip()
    if not enhanced_question:
        raise ValueError("Source sample does not contain a final question to repair.")
    answer = str(
        record.get("answer")
        or final_stage.get("answer")
        or enhanced_stage.get("answer")
        or drafted_stage.get("answer")
        or ""
    ).strip()
    answer_type = str(
        final_stage.get("answer_type")
        or enhanced_stage.get("answer_type")
        or drafted_stage.get("answer_type")
        or "other"
    )
    hops = [
        item
        for item in (record.get("question_hop_chain") or record.get("hop_chain") or [])
        if isinstance(item, dict)
    ]
    drafted_question = str(
        record.get("drafted_question")
        or record.get("draft_question")
        or drafted_stage.get("question")
        or enhanced_question
    ).strip()
    draft = QuestionDraft(
        question=drafted_question,
        answer=answer,
        answer_type=answer_type,
        reasoning_steps=list(hops),
        used_evidence_ids=list(drafted_stage.get("used_evidence_ids") or []),
        metadata={"shortcut_repair_source": "drafted_question"},
    )
    enhanced = QuestionDraft(
        question=enhanced_question,
        answer=answer,
        answer_type=answer_type,
        reasoning_steps=list(hops),
        used_evidence_ids=list(final_stage.get("used_evidence_ids") or enhanced_stage.get("used_evidence_ids") or []),
        metadata={"shortcut_repair_source": "enhanced_question"},
    )
    return draft, enhanced


def resolve_graph_dir(*, source_vqa_dir: Path, graph_dir: Path | None) -> Path:
    if graph_dir is not None:
        return graph_dir.resolve()
    metadata_path = source_vqa_dir / "question_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "--graph-dir was not provided and source question_metadata.json is missing: "
            f"{metadata_path}"
        )
    metadata = _read_json(metadata_path)
    raw_graph_dir = ((metadata.get("paths") or {}).get("graph_dir"))
    if not raw_graph_dir:
        raise ValueError(f"Source metadata does not contain paths.graph_dir: {metadata_path}")
    return Path(str(raw_graph_dir)).resolve()


def _append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def _load_completed_path_ids(samples_path: Path) -> set[str]:
    if not samples_path.exists():
        return set()
    completed: set[str] = set()
    for record in _iter_jsonl(samples_path):
        path_id = str((record.get("path") or {}).get("path_id") or "").strip()
        if path_id:
            completed.add(path_id)
    return completed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vqa-dir", type=Path, required=True, help="Existing VQA directory containing samples.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for rewritten VQA samples.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Graph directory override; defaults to source question_metadata.json.")
    parser.add_argument("--samples", type=int, default=None, help="Number of source paths to rewrite (default: all).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-alias", default=None, help="Writer model alias; defaults to VQA_WRITER_MODEL.")
    parser.add_argument("--compress-hop-model-alias", default=None, help="Optional hop-compression model alias.")
    parser.add_argument("--image-bridge-model-alias", default=None, help="Optional image-bridge model alias.")
    parser.add_argument("--ask-target-verify-model-alias", default=None, help="Optional target-ask verifier model alias.")
    parser.add_argument(
        "--shortcut-audit-model-alias",
        default=None,
        help="Optional model alias that audits final wording for exposed objects.",
    )
    parser.add_argument(
        "--shortcut-repair-only",
        action="store_true",
        help="Reuse saved questions and hop chains, running only shortcut audit/repair followed by verification.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Replace rewrite output files instead of resuming by path_id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_to_parse = list(argv) if argv is not None else sys.argv[1:]
    args = build_arg_parser().parse_args(argv_to_parse)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.samples is not None and args.samples <= 0:
        raise ValueError("samples must be positive when provided")

    source_vqa_dir = args.source_vqa_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_vqa_dir == output_dir:
        raise ValueError("--output-dir must be different from --source-vqa-dir")
    graph_dir = resolve_graph_dir(source_vqa_dir=source_vqa_dir, graph_dir=args.graph_dir)
    source_paths = load_source_paths(source_vqa_dir)
    source_question_records = (
        load_source_question_records(source_vqa_dir) if args.shortcut_repair_only else {}
    )
    if args.samples is not None:
        source_paths = source_paths[: args.samples]

    model_alias = args.model_alias
    if model_alias is None:
        model_alias = os.environ.get("VQA_WRITER_MODEL")
    compress_hop_model_alias = args.compress_hop_model_alias or os.environ.get("VQA_COMPRESS_HOP_MODEL")
    image_bridge_model_alias = args.image_bridge_model_alias or os.environ.get("VQA_IMAGE_BRIDGE_MODEL")
    ask_target_verify_model_alias = args.ask_target_verify_model_alias or os.environ.get("ASK_TARGET_VERIFY_MODEL")
    shortcut_audit_model_alias = args.shortcut_audit_model_alias or os.environ.get("VQA_SHORTCUT_AUDIT_MODEL")
    if args.shortcut_repair_only and not model_alias:
        raise ValueError("--shortcut-repair-only requires --model-alias (or VQA_WRITER_MODEL).")
    if args.shortcut_repair_only and not shortcut_audit_model_alias:
        raise ValueError(
            "--shortcut-repair-only requires --shortcut-audit-model-alias "
            "(or VQA_SHORTCUT_AUDIT_MODEL)."
        )

    config = SamplerConfiguration(max_samples=len(source_paths))
    writer = QuestionWriter(
        model_client=LLM_WORKER if model_alias else None,
        model=model_alias,
        compress_hop_model_client=LLM_WORKER if compress_hop_model_alias else None,
        compress_hop_model=compress_hop_model_alias,
        image_bridge_model_client=LLM_WORKER if image_bridge_model_alias else None,
        image_bridge_model=image_bridge_model_alias,
        ask_target_verify_model_client=LLM_WORKER if ask_target_verify_model_alias else None,
        ask_target_verify_model=ask_target_verify_model_alias,
        shortcut_audit_model_client=LLM_WORKER if shortcut_audit_model_alias else None,
        shortcut_audit_model=shortcut_audit_model_alias,
    )
    pipeline = VqaGenerationPipeline(
        store=JsonlGraphStore(graph_dir),
        config=config,
        writer=writer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    questions_path = output_dir / "questions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    warnings_path = output_dir / "warnings.jsonl"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "question_metadata.json"
    if args.no_resume:
        completed_path_ids: set[str] = set()
        mode = "w"
    else:
        completed_path_ids = _load_completed_path_ids(samples_path)
        mode = "a"

    pending = [item for item in source_paths if item[0].path_id not in completed_path_ids]
    started_at = time.perf_counter()
    summary: dict[str, Any] = {
        "requested_total": len(source_paths),
        "source_path_count": len(source_paths),
        "existing_samples": len(source_paths) - len(pending),
        "rewritten": 0,
        "verified": 0,
        "rejected": 0,
        "failed": 0,
        "warnings": 0,
        "sampling_skipped": True,
        "shortcut_repair_only": args.shortcut_repair_only,
        "elapsed_seconds": 0.0,
    }
    metadata = {
        "entrypoint": "synthesis.vqa.rewrite_batch",
        "created_at": _utc_now(),
        "source_vqa_dir": str(source_vqa_dir),
        "source_samples_path": str((source_vqa_dir / "samples.jsonl")),
        "paths": {"graph_dir": str(graph_dir), "output_dir": str(output_dir)},
        "models": {
            "writer_model_alias": model_alias,
            "compress_hop_model_alias": compress_hop_model_alias,
            "image_bridge_model_alias": image_bridge_model_alias,
            "ask_target_verify_model_alias": ask_target_verify_model_alias,
            "shortcut_audit_model_alias": shortcut_audit_model_alias,
        },
        "invocation": {
            "argv": argv_to_parse,
            "replay_command": shlex.join([sys.executable, "-m", "synthesis.vqa.rewrite_batch", *argv_to_parse]),
            "cwd": str(Path.cwd().resolve()),
        },
        "run": {
            "status": "running",
            "resume": not args.no_resume,
            "workers": args.workers,
            "shortcut_repair_only": args.shortcut_repair_only,
        },
    }
    _write_json(metadata_path, metadata)

    def rewrite_one(path: PathCandidate, source_sample_id: str | None) -> VqaSample:
        path.metadata = {
            **path.metadata,
            "rewrite_source_vqa_dir": str(source_vqa_dir),
            "rewrite_source_sample_id": source_sample_id,
            "sampling_seconds": 0.0,
        }
        if args.shortcut_repair_only:
            source_record = source_question_records.get(path.path_id)
            if source_record is None:
                raise ValueError(f"No saved source question found for path_id={path.path_id}")
            drafted, enhanced = source_repair_drafts(source_record)
            started_at = time.perf_counter()
            repaired = writer.repair_shortcuts(draft=enhanced, path=path, graph=pipeline.graph)
            verification = pipeline.verifier.verify(question=repaired)
            status = SampleStatus.VERIFIED if verification.final_keep else SampleStatus.REJECTED
            return VqaSample(
                sample_id=f"sample_{path.path_id}",
                status=status,
                path=path,
                evidence=EvidenceBundle(
                    bundle_id=f"bundle_{path.path_id}",
                    path_id=path.path_id,
                    metadata={"source": "shortcut_repair_only"},
                ),
                draft=drafted,
                polished=enhanced,
                obfuscated=repaired,
                verification=verification,
                progress=SampleProgress(
                    drafted_at=_utc_now(),
                    polished_at=_utc_now(),
                    post_obfuscated_at=_utc_now(),
                    verified_at=_utc_now(),
                ),
                metadata={
                    "timings": {
                        "sampling_seconds": 0.0,
                        "shortcut_repair_only_seconds": time.perf_counter() - started_at,
                    }
                },
            )
        return pipeline.generate_path(path)

    question_number = summary["existing_samples"]
    progress = (
        tqdm(
            total=len(pending),
            desc="Rewriting VQA questions",
            unit="question",
            dynamic_ncols=True,
        )
        if pending and tqdm is not None
        else None
    )
    with (
        samples_path.open(mode, encoding="utf-8") as samples_file,
        questions_path.open(mode, encoding="utf-8") as questions_file,
        errors_path.open(mode, encoding="utf-8") as errors_file,
        warnings_path.open(mode, encoding="utf-8") as warnings_file,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures: dict[Future[VqaSample], tuple[PathCandidate, str | None]] = {
            executor.submit(rewrite_one, path, source_sample_id): (path, source_sample_id)
            for path, source_sample_id in pending
        }
        try:
            for future in as_completed(futures):
                path, source_sample_id = futures[future]
                try:
                    sample = future.result()
                except Exception as exc:
                    summary["failed"] += 1
                    stage = exc.stage if isinstance(exc, VqaGenerationError) else "unknown"
                    cause = exc.cause if isinstance(exc, VqaGenerationError) else exc
                    _append_jsonl(
                        errors_file,
                        {
                            "path_id": path.path_id,
                            "source_sample_id": source_sample_id,
                            "stage": stage,
                            "error_type": cause.__class__.__name__,
                            "error": str(cause),
                            "traceback": "".join(traceback.format_exception(exc)),
                            "path": path.to_dict(),
                            "created_at": _utc_now(),
                        },
                    )
                else:
                    sample_dict = sample.to_dict()
                    compact_sample = VqaBatchRunner._compact_sample_record(sample_dict)
                    _append_jsonl(samples_file, compact_sample)
                    question_number += 1
                    _append_jsonl(
                        questions_file,
                        VqaBatchRunner._compact_question_record(sample_dict, question_number=question_number),
                    )
                    summary["rewritten"] += 1
                    if sample.status.value == "verified":
                        summary["verified"] += 1
                    else:
                        summary["rejected"] += 1
                    for warning in sample.metadata.get("writer_warnings") or []:
                        summary["warnings"] += 1
                        _append_jsonl(
                            warnings_file,
                            {
                                "sample_id": sample.sample_id,
                                "path_id": path.path_id,
                                **warning,
                                "created_at": _utc_now(),
                            },
                        )
                finally:
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            verified=summary["verified"],
                            rejected=summary["rejected"],
                            failed=summary["failed"],
                        )
        finally:
            if progress is not None:
                progress.close()

    summary["elapsed_seconds"] = time.perf_counter() - started_at
    _write_json(summary_path, {**summary, "updated_at": _utc_now()})
    metadata["run"] = {
        "status": "completed",
        "resume": not args.no_resume,
        "workers": args.workers,
        "sampling_skipped": True,
        "shortcut_repair_only": args.shortcut_repair_only,
    }
    metadata["summary"] = summary
    metadata["output_files"] = {
        "samples_path": str(samples_path),
        "questions_path": str(questions_path),
        "errors_path": str(errors_path),
        "warnings_path": str(warnings_path),
        "summary_path": str(summary_path),
    }
    metadata["updated_at"] = _utc_now()
    _write_json(metadata_path, metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"samples: {samples_path}")
    print(f"questions: {questions_path}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
