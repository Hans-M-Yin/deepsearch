"""Concurrent, resumable batch execution for VQA generation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
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
    sampler_state_input_path: Path | None = None
    question_metadata: dict[str, Any] = field(default_factory=dict)

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

    @property
    def sampler_state_path(self) -> Path:
        return self.output_dir / "sampler_state.json"

    @property
    def question_metadata_path(self) -> Path:
        return self.output_dir / "question_metadata.json"

    def run(self, *, limit: int) -> VqaBatchSummary:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = self._load_existing_samples() if self.resume else {}
        persisted_signatures, persisted_edge_usage_counts, sampler_state_info = self._restore_sampler_state(
            existing_records=existing.values(),
        )
        if not self.resume:
            self._reset_outputs(preserve_paths=self._reset_preserve_paths())
            existing = {}
        self._rebuild_questions_file(existing.values())

        self._write_question_metadata(
            limit=limit,
            existing_samples=len(existing),
            sampler_state_info=sampler_state_info,
            summary=None,
            status="running",
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
            self._write_sampler_state(
                used_exact_signatures=persisted_signatures,
                edge_usage_counts=persisted_edge_usage_counts,
            )
            self._refresh_sampler_state_info(
                sampler_state_info=sampler_state_info,
                used_exact_signatures=persisted_signatures,
                edge_usage_counts=persisted_edge_usage_counts,
            )
            self._write_question_metadata(
                limit=limit,
                existing_samples=len(existing),
                sampler_state_info=sampler_state_info,
                summary=summary,
                status="completed",
            )
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
                        persisted_signatures=persisted_signatures,
                        persisted_edge_usage_counts=persisted_edge_usage_counts,
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
        self._write_sampler_state(
            used_exact_signatures=persisted_signatures,
            edge_usage_counts=persisted_edge_usage_counts,
        )
        self._refresh_sampler_state_info(
            sampler_state_info=sampler_state_info,
            used_exact_signatures=persisted_signatures,
            edge_usage_counts=persisted_edge_usage_counts,
        )
        self._write_question_metadata(
            limit=limit,
            existing_samples=len(existing),
            sampler_state_info=sampler_state_info,
            summary=summary,
            status="completed",
        )
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
        persisted_signatures: set[str],
        persisted_edge_usage_counts: dict[str, int],
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
        compact_sample_record = self._compact_sample_record(sample_dict)
        self._append_jsonl(
            samples_file,
            compact_sample_record,
        )
        self._append_jsonl(
            questions_file,
            self._compact_question_record(
                sample_dict,
                question_number=summary.existing_samples + summary.completed,
            ),
        )
        self._merge_sample_record_into_state(
            compact_sample_record,
            used_exact_signatures=persisted_signatures,
            edge_usage_counts=persisted_edge_usage_counts,
        )
        self._print_sample_timing(sample)

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

    def _reset_outputs(self, *, preserve_paths: set[Path] | None = None) -> None:
        preserved = {path.resolve() for path in (preserve_paths or set())}
        for path in (
            self.samples_path,
            self.questions_path,
            self.errors_path,
            self.warnings_path,
            self.summary_path,
            self.question_metadata_path,
            self.sampler_state_path,
        ):
            if path.resolve() in preserved:
                continue
            if path.suffix == ".jsonl":
                path.write_text("", encoding="utf-8")
            elif path.exists():
                path.unlink()

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
        record = {
            "question_id": f"q_{question_number:06d}",
            "sample_id": sample.get("sample_id"),
            "path_id": path.get("path_id"),
            "status": sample.get("status"),
            "draft_question": draft_question.get("question"),
            "polished_question": polished_question.get("question"),
            "final_question": final_question.get("question"),
            "answer": final_question.get("answer"),
        }
        image_url = VqaBatchRunner._extract_input_image_url(sample)
        if image_url:
            record["image_url"] = image_url
        return record

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
                "timings": dict((sample.get("metadata") or {}).get("timings") or {}),
            },
            "input_image_url": VqaBatchRunner._extract_input_image_url(sample),
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

    @staticmethod
    def _extract_input_image_url(sample: dict[str, Any]) -> str | None:
        metadata = sample.get("metadata") or {}
        compact_image_url = str(sample.get("input_image_url") or metadata.get("input_image_url") or "").strip()
        if compact_image_url:
            return compact_image_url
        for stage_name in ("obfuscated", "polished", "draft"):
            stage = sample.get(stage_name) or {}
            stage_metadata = stage.get("metadata") or {}
            for key in ("starting_image_url", "polish_starting_image_url"):
                image_url = str(stage_metadata.get(key) or "").strip()
                if image_url:
                    return image_url
        return None

    def _write_summary(self, summary: VqaBatchSummary, *, elapsed: float | None = None) -> None:
        payload = summary.to_dict()
        if elapsed is not None:
            payload["elapsed_seconds"] = elapsed
        payload["updated_at"] = _utc_now()
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")

    def _write_sampler_state(
        self,
        *,
        used_exact_signatures: set[str],
        edge_usage_counts: dict[str, int],
    ) -> None:
        payload = self.pipeline.sampler.export_state(
            used_exact_signatures=used_exact_signatures,
            edge_usage_counts=edge_usage_counts,
        )
        payload["updated_at"] = _utc_now()
        payload["tracked_exact_signature_count"] = len(used_exact_signatures)
        payload["tracked_edge_count"] = len(edge_usage_counts)
        with self.sampler_state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")

    def _write_question_metadata(
        self,
        *,
        limit: int,
        existing_samples: int,
        sampler_state_info: dict[str, Any],
        summary: VqaBatchSummary | None,
        status: str,
    ) -> None:
        created_at = str(self.question_metadata.get("created_at") or _utc_now())
        self.question_metadata["created_at"] = created_at
        payload = dict(self.question_metadata)
        payload["output_files"] = {
            "samples_path": str(self.samples_path.resolve()),
            "questions_path": str(self.questions_path.resolve()),
            "question_metadata_path": str(self.question_metadata_path.resolve()),
            "errors_path": str(self.errors_path.resolve()),
            "warnings_path": str(self.warnings_path.resolve()),
            "summary_path": str(self.summary_path.resolve()),
            "sampler_state_path": str(self.sampler_state_path.resolve()),
        }
        payload["run"] = {
            "requested_total": limit,
            "existing_samples": existing_samples,
            "resume": self.resume,
            "workers": self.workers,
            "max_inflight": self.max_inflight,
            "status": status,
        }
        payload["sampler_state"] = {
            "mode": sampler_state_info.get("mode") or "fresh",
            "loaded_from": sampler_state_info.get("path"),
            "merged_existing_samples": sampler_state_info.get("merged_existing_samples", 0),
            "tracked_exact_signature_count": sampler_state_info.get("tracked_exact_signature_count", 0),
            "tracked_edge_count": sampler_state_info.get("tracked_edge_count", 0),
            "save_path": str(self.sampler_state_path.resolve()),
        }
        if summary is not None:
            payload["summary"] = summary.to_dict()
        payload["updated_at"] = _utc_now()
        payload["created_at"] = created_at
        with self.question_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")

    @staticmethod
    def _refresh_sampler_state_info(
        *,
        sampler_state_info: dict[str, Any],
        used_exact_signatures: set[str],
        edge_usage_counts: dict[str, int],
    ) -> None:
        sampler_state_info["tracked_exact_signature_count"] = len(used_exact_signatures)
        sampler_state_info["tracked_edge_count"] = len(edge_usage_counts)

    def _restore_sampler_state(
        self,
        *,
        existing_records,
    ) -> tuple[set[str], dict[str, int], dict[str, Any]]:
        loaded_state, source_info = self._load_sampler_state_file()
        used_exact_signatures, edge_usage_counts = self._state_accumulator_from_sampler_state(loaded_state)
        merged_existing = self._merge_sample_records_into_state(
            existing_records,
            used_exact_signatures=used_exact_signatures,
            edge_usage_counts=edge_usage_counts,
        )
        if source_info["mode"] == "fresh" and merged_existing:
            source_info = {
                "mode": "recovered_from_samples",
                "path": str(self.samples_path.resolve()),
            }
        source_info["merged_existing_samples"] = merged_existing
        source_info["tracked_exact_signature_count"] = len(used_exact_signatures)
        source_info["tracked_edge_count"] = len(edge_usage_counts)
        self.pipeline.sampler.load_state(
            self.pipeline.sampler.export_state(
                used_exact_signatures=used_exact_signatures,
                edge_usage_counts=edge_usage_counts,
            ),
            replace=True,
        )
        return used_exact_signatures, edge_usage_counts, source_info

    def _load_sampler_state_file(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        candidate = self._resolved_sampler_state_input_path()
        if candidate is None:
            return None, {"mode": "fresh", "path": None}
        if not candidate.exists():
            if self.sampler_state_input_path is not None:
                raise FileNotFoundError(f"sampler state does not exist: {candidate}")
            return None, {"mode": "fresh", "path": None}
        with candidate.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError(f"sampler state must be a JSON object: {candidate}")
        return state, {"mode": "file", "path": str(candidate)}

    def _resolved_sampler_state_input_path(self) -> Path | None:
        if self.sampler_state_input_path is not None:
            return self.sampler_state_input_path.resolve()
        auto_path = self.sampler_state_path.resolve()
        if self.resume and auto_path.exists():
            return auto_path
        return None

    def _reset_preserve_paths(self) -> set[Path]:
        preserved: set[Path] = set()
        resolved_input = self._resolved_sampler_state_input_path()
        if resolved_input is not None and resolved_input == self.sampler_state_path.resolve():
            preserved.add(resolved_input)
        return preserved

    @staticmethod
    def _state_accumulator_from_sampler_state(state: dict[str, Any] | None) -> tuple[set[str], dict[str, int]]:
        if not state:
            return set(), {}
        used_exact_signatures = {
            str(signature).strip()
            for signature in (state.get("used_exact_signatures") or [])
            if str(signature).strip()
        }
        edge_usage_counts: dict[str, int] = {}
        for edge_id, count in dict(state.get("edge_usage_counts") or {}).items():
            edge_text = str(edge_id).strip()
            if not edge_text:
                continue
            try:
                normalized_count = int(count)
            except (TypeError, ValueError):
                continue
            if normalized_count <= 0:
                continue
            edge_usage_counts[edge_text] = normalized_count
        return used_exact_signatures, edge_usage_counts

    @staticmethod
    def _merge_sample_records_into_state(
        records,
        *,
        used_exact_signatures: set[str],
        edge_usage_counts: dict[str, int],
    ) -> int:
        merged = 0
        for record in records:
            merged += int(
                VqaBatchRunner._merge_sample_record_into_state(
                    record,
                    used_exact_signatures=used_exact_signatures,
                    edge_usage_counts=edge_usage_counts,
                )
            )
        return merged

    @staticmethod
    def _merge_sample_record_into_state(
        record: dict[str, Any],
        *,
        used_exact_signatures: set[str],
        edge_usage_counts: dict[str, int],
    ) -> bool:
        path = record.get("path") or {}
        exact_signature = str(path.get("exact_signature") or "").strip()
        if not exact_signature or exact_signature in used_exact_signatures:
            return False
        used_exact_signatures.add(exact_signature)
        for edge_id in path.get("edge_ids") or []:
            edge_text = str(edge_id or "").strip()
            if not edge_text:
                continue
            edge_usage_counts[edge_text] = edge_usage_counts.get(edge_text, 0) + 1
        return True

    @staticmethod
    def _append_jsonl(handle, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    @staticmethod
    def _print_sample_timing(sample: VqaSample) -> None:
        timings = dict(sample.metadata.get("timings") or {})
        if not timings:
            return
        parts = [
            f"[vqa-timing] sample_id={sample.sample_id}",
            f"path_id={sample.path.path_id}",
            f"status={sample.status.value}",
        ]
        for key in (
            "sampling_seconds",
            "draft_seconds",
            "polish_seconds",
            "difficulty_enhancement_seconds",
            "verification_seconds",
            "total_generation_seconds",
        ):
            value = timings.get(key)
            if value is None:
                continue
            try:
                parts.append(f"{key}={float(value):.3f}s")
            except (TypeError, ValueError):
                continue
        print(" ".join(parts), flush=True)

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
