"""Concurrent, resumable batch execution for VQA generation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Any, Iterator

from .pipeline import VqaGenerationError, VqaGenerationPipeline
from .schemas import PathCandidate, VqaSample


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class VqaBatchSummary:
    requested_total: int
    existing_samples: int
    sampled_paths: int
    completed: int = 0
    verified: int = 0
    rejected: int = 0
    failed: int = 0
    warnings: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VqaBatchRunner:
    pipeline: VqaGenerationPipeline
    output_dir: Path
    workers: int = 8
    resume: bool = True
    max_inflight: int | None = None

    @property
    def samples_path(self) -> Path:
        return self.output_dir / "samples.jsonl"

    @property
    def questions_path(self) -> Path:
        return self.output_dir / "questions.jsonl"

    @property
    def errors_path(self) -> Path:
        return self.output_dir / "errors.jsonl"

    @property
    def warnings_path(self) -> Path:
        return self.output_dir / "warnings.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    def run(self, *, limit: int) -> VqaBatchSummary:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.resume:
            self._reset_outputs()
        existing = self._load_existing_samples() if self.resume else {}
        self._rebuild_questions_file(existing.values())
        if self.resume and existing:
            self.pipeline.sampler.used_exact_signatures.update(
                record["path"]["exact_signature"]
                for record in existing.values()
                if (record.get("path") or {}).get("exact_signature")
            )

        remaining = max(0, limit - len(existing))
        paths = self.pipeline.sample_paths(limit=remaining) if remaining else []
        summary = VqaBatchSummary(
            requested_total=limit,
            existing_samples=len(existing),
            sampled_paths=len(paths),
        )
        started_at = time.perf_counter()
        if not paths:
            summary.elapsed_seconds = time.perf_counter() - started_at
            self._write_summary(summary)
            return summary

        mode = "a" if self.resume else "w"
        with (
            self.samples_path.open(mode, encoding="utf-8") as samples_file,
            self.questions_path.open("a", encoding="utf-8") as questions_file,
            self.errors_path.open(mode, encoding="utf-8") as errors_file,
            self.warnings_path.open(mode, encoding="utf-8") as warnings_file,
            ThreadPoolExecutor(max_workers=self.workers) as executor,
        ):
            path_iter = iter(paths)
            inflight_limit = max(
                self.workers,
                self.max_inflight or self.workers * 2,
            )
            futures: dict[Future[VqaSample], PathCandidate] = {}
            self._fill_inflight(
                executor=executor,
                path_iter=path_iter,
                futures=futures,
                limit=inflight_limit,
            )
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    path = futures.pop(future)
                    self._record_future(
                        future=future,
                        path=path,
                        summary=summary,
                        samples_file=samples_file,
                        questions_file=questions_file,
                        errors_file=errors_file,
                        warnings_file=warnings_file,
                    )
                self._fill_inflight(
                    executor=executor,
                    path_iter=path_iter,
                    futures=futures,
                    limit=inflight_limit,
                )
                self._write_summary(summary, elapsed=time.perf_counter() - started_at)

        summary.elapsed_seconds = time.perf_counter() - started_at
        self._write_summary(summary)
        return summary

    def _record_future(
        self,
        *,
        future: Future[VqaSample],
        path: PathCandidate,
        summary: VqaBatchSummary,
        samples_file,
        questions_file,
        errors_file,
        warnings_file,
    ) -> None:
        try:
            sample = future.result()
        except Exception as exc:
            summary.failed += 1
            stage = exc.stage if isinstance(exc, VqaGenerationError) else "unknown"
            cause = exc.cause if isinstance(exc, VqaGenerationError) else exc
            self._append_jsonl(
                errors_file,
                {
                    "path_id": path.path_id,
                    "stage": stage,
                    "error_type": cause.__class__.__name__,
                    "error": str(cause),
                    "traceback": "".join(traceback.format_exception(exc)),
                    "path": path.to_dict(),
                    "created_at": _utc_now(),
                },
            )
            return

        summary.completed += 1
        if sample.status.value == "verified":
            summary.verified += 1
        else:
            summary.rejected += 1
        sample_dict = sample.to_dict()
        self._append_jsonl(
            samples_file,
            self._compact_sample_record(sample_dict),
        )
        self._append_jsonl(
            questions_file,
            self._compact_question_record(
                sample_dict,
                question_number=summary.existing_samples + summary.completed,
            ),
        )

        for warning in sample.metadata.get("writer_warnings") or []:
            summary.warnings += 1
            self._append_jsonl(
                warnings_file,
                {
                    "sample_id": sample.sample_id,
                    "path_id": path.path_id,
                    **warning,
                    "created_at": _utc_now(),
                },
            )

    def _load_existing_samples(self) -> dict[str, dict[str, Any]]:
        if not self.samples_path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        with self.samples_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.samples_path}:{line_no} is not valid JSON") from exc
                path_id = (record.get("path") or {}).get("path_id")
                if path_id:
                    records[path_id] = record
        return records

    def _reset_outputs(self) -> None:
        for path in (
            self.samples_path,
            self.questions_path,
            self.errors_path,
            self.warnings_path,
        ):
            path.write_text("", encoding="utf-8")

    def _rebuild_questions_file(self, records) -> None:
        with self.questions_path.open("w", encoding="utf-8") as handle:
            for question_number, record in enumerate(records, start=1):
                self._append_jsonl(
                    handle,
                    self._compact_question_record(
                        record,
                        question_number=question_number,
                    ),
                )

    @staticmethod
    def _compact_question_record(
        sample: dict[str, Any],
        *,
        question_number: int,
    ) -> dict[str, Any]:
        writer_outputs = sample.get("writer_outputs") or {}
        draft_question = sample.get("draft") or writer_outputs.get("draft") or {}
        polished_question = sample.get("polished") or writer_outputs.get("polished") or {}
        final_question = (
            sample.get("obfuscated")
            or writer_outputs.get("obfuscated")
            or polished_question
            or draft_question
            or {}
        )
        path = sample.get("path") or {}
        return {
            "question_id": f"q_{question_number:06d}",
            "sample_id": sample.get("sample_id"),
            "path_id": path.get("path_id"),
            "status": sample.get("status"),
            "draft_question": draft_question.get("question"),
            "polished_question": polished_question.get("question"),
            "final_question": final_question.get("question"),
            "answer": final_question.get("answer"),
        }

    @staticmethod
    def _compact_sample_record(sample: dict[str, Any]) -> dict[str, Any]:
        path = sample.get("path") or {}
        final_question = (
            sample.get("obfuscated")
            or sample.get("polished")
            or sample.get("draft")
            or {}
        )
        return {
            "sample_id": sample.get("sample_id"),
            "status": sample.get("status"),
            "path": {
                "path_id": path.get("path_id"),
                "start_node_id": path.get("start_node_id"),
                "target_node_id": path.get("target_node_id"),
                "node_ids": path.get("node_ids") or [],
                "edge_ids": path.get("edge_ids") or [],
                "node_types": path.get("node_types") or [],
                "edge_types": path.get("edge_types") or [],
                "relations": path.get("relations") or [],
                "trajectory": path.get("trajectory") or {},
                "exact_signature": path.get("exact_signature"),
                "skeleton_signature": path.get("skeleton_signature"),
                "core_signature": path.get("core_signature"),
                "metadata": path.get("metadata") or {},
            },
            "hop_chain": [
                {
                    "hop_index": item.get("hop_index"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "statement": item.get("statement"),
                    "relation": item.get("relation"),
                    "retrieval_query": item.get("retrieval_query"),
                    "edge_id": item.get("edge_id"),
                    "src_node_id": item.get("src_node_id"),
                    "dst_node_id": item.get("dst_node_id"),
                }
                for item in (final_question.get("reasoning_steps") or [])
                if isinstance(item, dict)
            ],
            "writer_outputs": {
                "draft": VqaBatchRunner._compact_writer_stage(sample.get("draft") or {}),
                "polished": VqaBatchRunner._compact_writer_stage(sample.get("polished") or {}),
                "obfuscated": VqaBatchRunner._compact_writer_stage(sample.get("obfuscated") or {}),
            },
            "verification": sample.get("verification") or {},
            "progress": sample.get("progress") or {},
            "metadata": {
                "writer_warnings": list((sample.get("metadata") or {}).get("writer_warnings") or []),
            },
            "created_at": sample.get("created_at"),
            "updated_at": sample.get("updated_at"),
        }

    @staticmethod
    def _compact_writer_stage(stage: dict[str, Any]) -> dict[str, Any] | None:
        if not stage:
            return None
        return {
            "question": stage.get("question"),
            "answer": stage.get("answer"),
            "answer_type": stage.get("answer_type"),
            "used_evidence_ids": stage.get("used_evidence_ids") or [],
        }

    def _write_summary(self, summary: VqaBatchSummary, *, elapsed: float | None = None) -> None:
        payload = summary.to_dict()
        if elapsed is not None:
            payload["elapsed_seconds"] = elapsed
        payload["updated_at"] = _utc_now()
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")

    @staticmethod
    def _append_jsonl(handle, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    def _fill_inflight(
        self,
        *,
        executor: ThreadPoolExecutor,
        path_iter: Iterator[PathCandidate],
        futures: dict[Future[VqaSample], PathCandidate],
        limit: int,
    ) -> None:
        while len(futures) < limit:
            try:
                path = next(path_iter)
            except StopIteration:
                return
            future = executor.submit(self.pipeline.generate_path, path)
            futures[future] = path
