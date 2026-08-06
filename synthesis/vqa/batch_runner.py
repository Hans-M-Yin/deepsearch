"""Concurrent, resumable batch execution for VQA generation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from .path_sampler import RandomPathSampler, SamplerConfiguration
from .pipeline import VqaGenerationError, VqaGenerationPipeline
from .schemas import PathCandidate, VqaSample


_MAX_SAME_CORE_SIGNATURE = 2
_MAX_SAME_PREFIX_SIGNATURE = 2
_MAX_SAME_TARGET_NODE = 3
_MAX_EDGE_OVERLAP_RATIO = 0.75
_MAX_NODE_OVERLAP_RATIO = 0.80
_MIN_STALLED_PROPOSALS_PER_VERSION = 8
_STALLED_PROPOSALS_MULTIPLIER = 4
_JSONL_PUBLISH_BATCH_SIZE = 100
_FSYNC_WARNING_PATHS: set[str] = set()

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class VqaBatchSummary:
    requested_total: int
    existing_samples: int
    sampled_paths: int
    proposed_paths: int = 0
    acceptor_rejected: int = 0
    sampler_exhausted_proposals: int = 0
    completed: int = 0
    verified: int = 0
    rejected: int = 0
    failed: int = 0
    warnings: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _ProposalContext:
    proposal_id: int
    state_version: int
    sampler_seed: int


@dataclass(slots=True)
class _AcceptedPathFingerprint:
    path_id: str
    exact_signature: str
    core_signature: str
    prefix_signature: str
    target_node_id: str
    node_ids: tuple[str, ...]
    node_id_set: frozenset[str]
    edge_id_set: frozenset[str]
    edge_ids: tuple[str, ...]


@dataclass(slots=True)
class _AcceptorState:
    used_exact_signatures: set[str]
    edge_usage_counts: dict[str, int]
    version: int = 0
    core_counts: dict[str, int] = field(default_factory=dict)
    prefix_counts: dict[str, int] = field(default_factory=dict)
    target_counts: dict[str, int] = field(default_factory=dict)
    core_buckets: dict[str, list[_AcceptedPathFingerprint]] = field(default_factory=dict)
    prefix_buckets: dict[str, list[_AcceptedPathFingerprint]] = field(default_factory=dict)
    target_buckets: dict[str, list[_AcceptedPathFingerprint]] = field(default_factory=dict)


class _JsonlBatchWriter:
    """Publish JSONL records to HDFS-FUSE in bounded, closed-file batches."""

    def __init__(self, path: Path, *, batch_size: int = _JSONL_PUBLISH_BATCH_SIZE) -> None:
        self.path = path
        self.batch_size = max(1, int(batch_size))
        self._records: list[dict[str, Any]] = []

    def __enter__(self) -> "_JsonlBatchWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        del exc_type, exc_value, traceback_value
        self.flush()

    def append(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        if len(self._records) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        records = self._records
        self._records = []
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                VqaBatchRunner._append_jsonl(handle, record, flush=False)
            handle.flush()
            VqaBatchRunner._fsync_handle(handle)


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
        persisted_signatures, persisted_edge_usage_counts, persisted_fingerprints, acceptor_state, sampler_state_info = self._restore_sampler_state(
            existing_records=existing.values(),
        )
        if not self.resume:
            self._reset_outputs(preserve_paths=self._reset_preserve_paths())
            existing = {}
        self._rebuild_questions_file(existing.values())

        remaining = max(0, limit - len(existing))
        sampler_workers = self._sampler_worker_count(remaining)
        batch_parameters = dict(self.question_metadata.get("batch_parameters") or {})
        batch_parameters["pipeline_mode"] = "streaming_acceptor_writer"
        batch_parameters["sampler_workers"] = sampler_workers
        self.question_metadata["batch_parameters"] = batch_parameters

        self._write_question_metadata(
            limit=limit,
            existing_samples=len(existing),
            sampler_state_info=sampler_state_info,
            summary=None,
            status="running",
        )

        summary = VqaBatchSummary(
            requested_total=limit,
            existing_samples=len(existing),
            sampled_paths=0,
        )
        started_at = time.perf_counter()
        progress = None
        if remaining and tqdm is not None:
            progress = tqdm(
                total=limit,
                initial=len(existing),
                desc="VQA samples",
                unit="sample",
                dynamic_ncols=True,
            )
        if not remaining:
            summary.elapsed_seconds = time.perf_counter() - started_at
            self._write_summary(summary)
            self._write_sampler_state(
                used_exact_signatures=persisted_signatures,
                edge_usage_counts=persisted_edge_usage_counts,
                persisted_fingerprints=persisted_fingerprints,
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

        inflight_limit = max(
            self.workers,
            self.max_inflight or self.workers * 2,
        )
        stalled_threshold = max(
            _MIN_STALLED_PROPOSALS_PER_VERSION,
            sampler_workers * _STALLED_PROPOSALS_MULTIPLIER,
        )
        with (
            _JsonlBatchWriter(self.samples_path) as samples_file,
            _JsonlBatchWriter(self.questions_path) as questions_file,
            _JsonlBatchWriter(self.errors_path) as errors_file,
            _JsonlBatchWriter(self.warnings_path) as warnings_file,
            ThreadPoolExecutor(max_workers=sampler_workers) as sampler_executor,
            ThreadPoolExecutor(max_workers=self.workers) as writer_executor,
        ):
            proposal_futures: dict[Future[PathCandidate | None], _ProposalContext] = {}
            writer_futures: dict[Future[VqaSample], PathCandidate] = {}
            stalled_proposals_by_version: dict[int, int] = {}
            proposal_sequence = 0
            cached_snapshot_version: int | None = None
            cached_snapshot: dict[str, Any] | None = None

            def generation_goal_reached() -> bool:
                """Whether completed samples plus active writers fill the target.

                Path proposals are submitted before question generation.  A
                writer can now reject a path when target-question generation
                is unavailable, so proposal count must not consume the output
                quota permanently.
                """

                return summary.completed + len(writer_futures) >= remaining

            def current_version_is_stalled() -> bool:
                current_version = acceptor_state.version
                if generation_goal_reached():
                    return True
                if any(context.state_version == current_version for context in proposal_futures.values()):
                    return False
                return stalled_proposals_by_version.get(current_version, 0) >= stalled_threshold

            def current_snapshot() -> dict[str, Any]:
                nonlocal cached_snapshot, cached_snapshot_version
                if cached_snapshot is None or cached_snapshot_version != acceptor_state.version:
                    cached_snapshot = self._sampler_template().export_state(
                        used_exact_signatures=acceptor_state.used_exact_signatures,
                        edge_usage_counts=acceptor_state.edge_usage_counts,
                    )
                    cached_snapshot_version = acceptor_state.version
                return cached_snapshot

            def fill_sampler_futures() -> None:
                nonlocal proposal_sequence
                if current_version_is_stalled():
                    return
                room = max(0, inflight_limit - len(writer_futures))
                target = min(sampler_workers, room)
                while len(proposal_futures) < target and not generation_goal_reached():
                    proposal_sequence += 1
                    context = _ProposalContext(
                        proposal_id=proposal_sequence,
                        state_version=acceptor_state.version,
                        sampler_seed=self._proposal_seed(proposal_sequence),
                    )
                    future = sampler_executor.submit(
                        self._sample_path_proposal,
                        sampler_state=current_snapshot(),
                        proposal_id=context.proposal_id,
                        state_version=context.state_version,
                        sampler_seed=context.sampler_seed,
                    )
                    proposal_futures[future] = context

            fill_sampler_futures()
            while proposal_futures or writer_futures:
                done, _ = wait(tuple(proposal_futures) + tuple(writer_futures), return_when=FIRST_COMPLETED)
                for future in done:
                    if future in writer_futures:
                        path = writer_futures.pop(future)
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
                            progress=progress,
                            started_at=started_at,
                            target_total=limit,
                        )
                        continue

                    context = proposal_futures.pop(future)
                    try:
                        candidate = future.result()
                    except Exception as exc:
                        summary.failed += 1
                        stalled_proposals_by_version[context.state_version] = (
                            stalled_proposals_by_version.get(context.state_version, 0) + 1
                        )
                        self._record_sampler_failure(
                            exc=exc,
                            context=context,
                            errors_file=errors_file,
                        )
                        continue

                    if candidate is None:
                        summary.sampler_exhausted_proposals += 1
                        stalled_proposals_by_version[context.state_version] = (
                            stalled_proposals_by_version.get(context.state_version, 0) + 1
                        )
                        continue

                    summary.proposed_paths += 1
                    if generation_goal_reached():
                        continue

                    accepted, _reject_reason = self._accept_path_candidate(
                        path=candidate,
                        acceptor_state=acceptor_state,
                    )
                    if not accepted:
                        summary.acceptor_rejected += 1
                        stalled_proposals_by_version[context.state_version] = (
                            stalled_proposals_by_version.get(context.state_version, 0) + 1
                        )
                        continue

                    summary.sampled_paths += 1
                    writer_future = writer_executor.submit(self.pipeline.generate_path, candidate)
                    writer_futures[writer_future] = candidate

                fill_sampler_futures()
                self._write_summary(summary, elapsed=time.perf_counter() - started_at)

        if progress is not None:
            progress.close()
        summary.elapsed_seconds = time.perf_counter() - started_at
        self._write_summary(summary)
        self._write_sampler_state(
            used_exact_signatures=persisted_signatures,
            edge_usage_counts=persisted_edge_usage_counts,
            persisted_fingerprints=persisted_fingerprints,
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
        progress: Any | None = None,
        started_at: float | None = None,
        target_total: int | None = None,
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
                flush=True,
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
            flush=True,
        )
        self._append_jsonl(
            questions_file,
            self._compact_question_record(
                sample_dict,
                question_number=summary.existing_samples + summary.completed,
            ),
            flush=True,
        )
        self._merge_sample_record_into_state(
            compact_sample_record,
            used_exact_signatures=persisted_signatures,
            edge_usage_counts=persisted_edge_usage_counts,
        )
        elapsed_s = time.perf_counter() - started_at if started_at is not None else 0.0
        generated_total = summary.existing_samples + summary.completed
        remaining_total = max(0, (target_total or generated_total) - generated_total)
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                generated=generated_total,
                remaining=remaining_total,
                elapsed=f"{elapsed_s:.1f}s",
                verified=summary.verified,
                rejected=summary.rejected,
            )
        else:
            print(
                "[vqa-progress] "
                f"generated={generated_total} remaining={remaining_total} "
                f"elapsed_s={elapsed_s:.1f} verified={summary.verified} rejected={summary.rejected}",
                flush=True,
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
                flush=True,
            )

    def _record_sampler_failure(
        self,
        *,
        exc: Exception,
        context: _ProposalContext,
        errors_file,
    ) -> None:
        self._append_jsonl(
            errors_file,
            {
                "proposal_id": context.proposal_id,
                "stage": "sampling",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
                "sampler_state_version": context.state_version,
                "sampler_seed": context.sampler_seed,
                "created_at": _utc_now(),
            },
            flush=True,
        )

    def _sample_path_proposal(
        self,
        *,
        sampler_state: dict[str, Any],
        proposal_id: int,
        state_version: int,
        sampler_seed: int,
    ) -> PathCandidate | None:
        started_at = time.perf_counter()
        sampler = self._build_sampler_clone(
            sampler_state=sampler_state,
            sampler_seed=sampler_seed,
        )
        candidate = sampler.generate_one()
        if candidate is None:
            return None
        elapsed_s = time.perf_counter() - started_at
        candidate.metadata["sampling_seconds"] = elapsed_s
        candidate.metadata["proposal_id"] = proposal_id
        candidate.metadata["sampler_state_version"] = state_version
        candidate.metadata["sampler_seed"] = sampler_seed
        if sampler.last_generation_stats is not None:
            candidate.metadata["sampler_generation_stats"] = sampler.last_generation_stats.to_dict()
        return candidate

    def _build_sampler_clone(
        self,
        *,
        sampler_state: dict[str, Any],
        sampler_seed: int,
    ) -> RandomPathSampler:
        template = self._sampler_template()
        config_payload = dict(self.pipeline.config.to_dict())
        config_payload["random_seed"] = sampler_seed
        config = SamplerConfiguration(**config_payload)
        sampler = RandomPathSampler(
            graph=self.pipeline.graph,
            config=config,
            model_client=template.model_client,
            model=template.model,
            history_exposure_model_client=template.history_exposure_model_client,
            history_exposure_model=template.history_exposure_model,
            edge_quality_model_client=template.edge_quality_model_client,
            edge_quality_model=template.edge_quality_model,
            llm_max_tokens=template.llm_max_tokens,
        )
        sampler.load_state(sampler_state, replace=True)
        return sampler

    def _sampler_template(self) -> RandomPathSampler:
        sampler = self.pipeline.sampler
        if sampler is None:
            raise RuntimeError("pipeline sampler is not initialized")
        if not isinstance(sampler, RandomPathSampler):
            raise TypeError(f"VqaBatchRunner requires RandomPathSampler, got {type(sampler)!r}")
        return sampler

    def _accept_path_candidate(
        self,
        *,
        path: PathCandidate,
        acceptor_state: _AcceptorState,
    ) -> tuple[bool, str | None]:
        fingerprint = self._fingerprint_from_path_candidate(path)
        if fingerprint is None:
            return False, "invalid_path"
        if self.pipeline.config.dedup_by_exact_signature and fingerprint.exact_signature in acceptor_state.used_exact_signatures:
            return False, "duplicate_exact"
        if fingerprint.core_signature and acceptor_state.core_counts.get(fingerprint.core_signature, 0) >= _MAX_SAME_CORE_SIGNATURE:
            return False, "duplicate_core_signature"
        if fingerprint.prefix_signature and acceptor_state.prefix_counts.get(fingerprint.prefix_signature, 0) >= _MAX_SAME_PREFIX_SIGNATURE:
            return False, "duplicate_prefix_signature"
        if fingerprint.target_node_id and acceptor_state.target_counts.get(fingerprint.target_node_id, 0) >= _MAX_SAME_TARGET_NODE:
            return False, "duplicate_target_node"
        overlap_reason = self._overlap_reject_reason(
            fingerprint=fingerprint,
            acceptor_state=acceptor_state,
        )
        if overlap_reason is not None:
            return False, overlap_reason

        self._register_acceptor_fingerprint(
            acceptor_state=acceptor_state,
            fingerprint=fingerprint,
            increment_version=True,
            count_edge_usage=True,
        )
        path.metadata["acceptor"] = {
            "proposal_state_version": path.metadata.get("sampler_state_version"),
            "accepted_state_version": acceptor_state.version,
        }
        return True, None

    @staticmethod
    def _register_acceptor_fingerprint(
        *,
        acceptor_state: _AcceptorState,
        fingerprint: _AcceptedPathFingerprint,
        increment_version: bool,
        count_edge_usage: bool,
    ) -> None:
        acceptor_state.used_exact_signatures.add(fingerprint.exact_signature)
        if count_edge_usage:
            for edge_id in fingerprint.edge_ids:
                acceptor_state.edge_usage_counts[edge_id] = acceptor_state.edge_usage_counts.get(edge_id, 0) + 1
        if fingerprint.core_signature:
            acceptor_state.core_counts[fingerprint.core_signature] = (
                acceptor_state.core_counts.get(fingerprint.core_signature, 0) + 1
            )
            acceptor_state.core_buckets.setdefault(fingerprint.core_signature, []).append(fingerprint)
        if fingerprint.prefix_signature:
            acceptor_state.prefix_counts[fingerprint.prefix_signature] = (
                acceptor_state.prefix_counts.get(fingerprint.prefix_signature, 0) + 1
            )
            acceptor_state.prefix_buckets.setdefault(fingerprint.prefix_signature, []).append(fingerprint)
        if fingerprint.target_node_id:
            acceptor_state.target_counts[fingerprint.target_node_id] = (
                acceptor_state.target_counts.get(fingerprint.target_node_id, 0) + 1
            )
            acceptor_state.target_buckets.setdefault(fingerprint.target_node_id, []).append(fingerprint)
        if increment_version:
            acceptor_state.version += 1

    @staticmethod
    def _overlap_reject_reason(
        *,
        fingerprint: _AcceptedPathFingerprint,
        acceptor_state: _AcceptorState,
    ) -> str | None:
        seen_exact_signatures: set[str] = set()
        bucket_specs = [
            ("core", acceptor_state.core_buckets.get(fingerprint.core_signature, [])),
            ("prefix", acceptor_state.prefix_buckets.get(fingerprint.prefix_signature, [])),
            ("target", acceptor_state.target_buckets.get(fingerprint.target_node_id, [])),
        ]
        for bucket_name, bucket_records in bucket_specs:
            for existing in bucket_records:
                if existing.exact_signature in seen_exact_signatures:
                    continue
                seen_exact_signatures.add(existing.exact_signature)
                edge_overlap = VqaBatchRunner._overlap_ratio(fingerprint.edge_id_set, existing.edge_id_set)
                if edge_overlap >= _MAX_EDGE_OVERLAP_RATIO:
                    return f"high_{bucket_name}_edge_overlap"
                node_overlap = VqaBatchRunner._overlap_ratio(fingerprint.node_id_set, existing.node_id_set)
                if node_overlap >= _MAX_NODE_OVERLAP_RATIO:
                    return f"high_{bucket_name}_node_overlap"
        return None

    @staticmethod
    def _overlap_ratio(left: frozenset[str], right: frozenset[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / float(min(len(left), len(right)))

    @staticmethod
    def _fingerprint_from_path_candidate(path: PathCandidate) -> _AcceptedPathFingerprint | None:
        node_ids = tuple(str(node_id).strip() for node_id in path.node_ids if str(node_id).strip())
        edge_ids = tuple(str(edge_id).strip() for edge_id in path.edge_ids if str(edge_id).strip())
        if not node_ids:
            return None
        exact_signature = str(path.exact_signature or "|".join(node_ids)).strip()
        if not exact_signature:
            return None
        core_signature = str(
            path.core_signature or ("|".join(node_ids[-3:]) if len(node_ids) >= 3 else exact_signature)
        ).strip()
        start_node_id = str(path.start_node_id or node_ids[0]).strip()
        target_node_id = str(path.target_node_id or node_ids[-1]).strip()
        return _AcceptedPathFingerprint(
            path_id=str(path.path_id or exact_signature),
            exact_signature=exact_signature,
            core_signature=core_signature,
            prefix_signature=VqaBatchRunner._prefix_signature(start_node_id, edge_ids),
            target_node_id=target_node_id,
            node_ids=node_ids,
            node_id_set=frozenset(node_ids),
            edge_id_set=frozenset(edge_ids),
            edge_ids=edge_ids,
        )

    @staticmethod
    def _fingerprint_from_record(record: dict[str, Any]) -> _AcceptedPathFingerprint | None:
        path = record.get("path") or {}
        node_ids = tuple(str(node_id).strip() for node_id in (path.get("node_ids") or []) if str(node_id).strip())
        edge_ids = tuple(str(edge_id).strip() for edge_id in (path.get("edge_ids") or []) if str(edge_id).strip())
        if not node_ids:
            return None
        exact_signature = str(path.get("exact_signature") or "|".join(node_ids)).strip()
        if not exact_signature:
            return None
        core_signature = str(
            path.get("core_signature") or ("|".join(node_ids[-3:]) if len(node_ids) >= 3 else exact_signature)
        ).strip()
        start_node_id = str(path.get("start_node_id") or node_ids[0]).strip()
        target_node_id = str(path.get("target_node_id") or node_ids[-1]).strip()
        return _AcceptedPathFingerprint(
            path_id=str(path.get("path_id") or exact_signature),
            exact_signature=exact_signature,
            core_signature=core_signature,
            prefix_signature=VqaBatchRunner._prefix_signature(start_node_id, edge_ids),
            target_node_id=target_node_id,
            node_ids=node_ids,
            node_id_set=frozenset(node_ids),
            edge_id_set=frozenset(edge_ids),
            edge_ids=edge_ids,
        )

    @staticmethod
    def _fingerprint_from_state_record(record: dict[str, Any]) -> _AcceptedPathFingerprint | None:
        node_ids = tuple(str(node_id).strip() for node_id in (record.get("node_ids") or []) if str(node_id).strip())
        edge_ids = tuple(str(edge_id).strip() for edge_id in (record.get("edge_ids") or []) if str(edge_id).strip())
        exact_signature = str(record.get("exact_signature") or "|".join(node_ids)).strip()
        if not exact_signature:
            return None
        core_signature = str(record.get("core_signature") or exact_signature).strip()
        prefix_signature = str(record.get("prefix_signature") or "").strip()
        target_node_id = str(record.get("target_node_id") or "").strip()
        return _AcceptedPathFingerprint(
            path_id=str(record.get("path_id") or exact_signature),
            exact_signature=exact_signature,
            core_signature=core_signature,
            prefix_signature=prefix_signature,
            target_node_id=target_node_id,
            node_ids=node_ids,
            node_id_set=frozenset(node_ids),
            edge_id_set=frozenset(edge_ids),
            edge_ids=edge_ids,
        )

    @staticmethod
    def _prefix_signature(start_node_id: str, edge_ids: tuple[str, ...]) -> str:
        start = str(start_node_id or "").strip()
        if not start:
            return ""
        first_edge_id = edge_ids[0] if edge_ids else ""
        return f"{start}|{first_edge_id}" if first_edge_id else start

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
        drafted_question = (
            sample.get("draft")
            or writer_outputs.get("drafted")
            or writer_outputs.get("draft")
            or {}
        )
        enhanced_question = (
            sample.get("polished")
            or writer_outputs.get("enhanced")
            or writer_outputs.get("polished")
            or {}
        )
        final_question = (
            sample.get("obfuscated")
            or writer_outputs.get("obfuscated")
            or writer_outputs.get("final")
            or enhanced_question
            or drafted_question
            or {}
        )
        path = sample.get("path") or {}
        record = {
            "question_id": f"q_{question_number:06d}",
            "sample_id": sample.get("sample_id"),
            "path_id": path.get("path_id"),
            "status": sample.get("status"),
            "question": final_question.get("question"),
            "drafted_question": drafted_question.get("question"),
            "enhanced_question": enhanced_question.get("question"),
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
        raw_hop_summaries = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="raw_hop_summaries",
        )
        record = {
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
                for item in (raw_hop_summaries or final_question.get("reasoning_steps") or [])
                if isinstance(item, dict)
            ],
            "question_hop_chain": [
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
                "drafted": VqaBatchRunner._compact_writer_stage(sample.get("draft") or {}),
                "enhanced": VqaBatchRunner._compact_writer_stage(sample.get("polished") or {}),
                "final": VqaBatchRunner._compact_writer_stage(sample.get("obfuscated") or {}),
            },
            "entry_hop": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="entry_hop",
            ),
            "target_ask": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="target_ask",
            ),
            "question_target_ask": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="question_target_ask",
            ),
            "question_terminal_bridge": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="question_terminal_bridge",
            ),
            "image_bridge_normalization": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="image_bridge_normalization",
            ),
            "image_target_terminal_normalization": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="image_target_terminal_normalization",
            ),
            "compose": {
                "payload": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="compose_payload",
                ),
                "result": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="compose_result",
                ),
            },
            "difficulty_enhancement": {
                "payload": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="difficulty_enhancement_payload",
                ),
                "result": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="difficulty_enhancement_result",
                ),
            },
            "shortcut_repair": VqaBatchRunner._extract_stage_metadata_value(
                sample,
                field_name="shortcut_repair",
            ),
            "polish": {
                "payload": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="polish_payload",
                ),
                "subtasks": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="polish_subtasks",
                ),
                "rewrite_payload": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="polish_rewrite_payload",
                ),
                "result": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="polish_result",
                ),
                "rewrite_skipped": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="polish_rewrite_skipped",
                ),
                "rewrite_skip_reason": VqaBatchRunner._extract_stage_metadata_value(
                    sample,
                    field_name="polish_rewrite_skip_reason",
                ),
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
        image_target_candidates = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="image_target_candidates",
        )
        if image_target_candidates is not None:
            record["image_target_candidates"] = image_target_candidates
        image_target_candidate_verification = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="image_target_candidate_verification",
        )
        if image_target_candidate_verification is not None:
            record["image_target_candidate_verification"] = image_target_candidate_verification
        image_target_candidate_evaluation = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="image_target_candidate_evaluation",
        )
        if image_target_candidate_evaluation is not None:
            record["image_target_candidate_evaluation"] = image_target_candidate_evaluation
        text_target_candidates = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="text_target_candidates",
        )
        if text_target_candidates is not None:
            record["text_target_candidates"] = text_target_candidates
        text_target_candidate_verification = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="text_target_candidate_verification",
        )
        if text_target_candidate_verification is not None:
            record["text_target_candidate_verification"] = text_target_candidate_verification
        text_target_candidate_evaluation = VqaBatchRunner._extract_stage_metadata_value(
            sample,
            field_name="text_target_candidate_evaluation",
        )
        if text_target_candidate_evaluation is not None:
            record["text_target_candidate_evaluation"] = text_target_candidate_evaluation
        return record

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
    def _extract_stage_metadata_value(
        sample: dict[str, Any],
        *,
        field_name: str,
    ) -> dict[str, Any] | list[Any] | str | None:
        for stage_name in ("obfuscated", "polished", "draft"):
            stage = sample.get(stage_name) or {}
            stage_metadata = stage.get("metadata") or {}
            value = stage_metadata.get(field_name)
            if value is None:
                continue
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, list):
                return list(value)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
                continue
            return value
        return None

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
        persisted_fingerprints: list[_AcceptedPathFingerprint] | None = None,
    ) -> None:
        payload = self._sampler_template().export_state(
            used_exact_signatures=used_exact_signatures,
            edge_usage_counts=edge_usage_counts,
        )
        payload["accepted_path_fingerprints"] = [
            self._fingerprint_to_state_record(fingerprint)
            for fingerprint in self._persisted_path_fingerprints(base_fingerprints=persisted_fingerprints)
        ]
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
    ) -> tuple[set[str], dict[str, int], list[_AcceptedPathFingerprint], _AcceptorState, dict[str, Any]]:
        loaded_state, source_info = self._load_sampler_state_file()
        persisted_signatures, persisted_edge_usage_counts = self._state_accumulator_from_sampler_state(loaded_state)
        persisted_fingerprints = self._state_fingerprints_from_sampler_state(loaded_state)
        merged_existing = self._merge_sample_records_into_state(
            existing_records,
            used_exact_signatures=persisted_signatures,
            edge_usage_counts=persisted_edge_usage_counts,
        )
        if source_info["mode"] == "fresh" and merged_existing:
            source_info = {
                "mode": "recovered_from_samples",
                "path": str(self.samples_path.resolve()),
            }
        source_info["merged_existing_samples"] = merged_existing
        source_info["tracked_exact_signature_count"] = len(persisted_signatures)
        source_info["tracked_edge_count"] = len(persisted_edge_usage_counts)
        self._sampler_template().load_state(
            self._sampler_template().export_state(
                used_exact_signatures=persisted_signatures,
                edge_usage_counts=persisted_edge_usage_counts,
            ),
            replace=True,
        )
        acceptor_state = _AcceptorState(
            used_exact_signatures=set(persisted_signatures),
            edge_usage_counts=dict(persisted_edge_usage_counts),
        )
        registered_exact_signatures: set[str] = set()
        self._merge_fingerprints_into_acceptor_state(
            persisted_fingerprints,
            acceptor_state=acceptor_state,
            registered_exact_signatures=registered_exact_signatures,
        )
        self._merge_sample_records_into_acceptor_state(
            existing_records,
            acceptor_state=acceptor_state,
            registered_exact_signatures=registered_exact_signatures,
        )
        return persisted_signatures, persisted_edge_usage_counts, persisted_fingerprints, acceptor_state, source_info

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

    def _merge_sample_records_into_acceptor_state(
        self,
        records,
        *,
        acceptor_state: _AcceptorState,
        registered_exact_signatures: set[str] | None = None,
    ) -> None:
        registered = registered_exact_signatures if registered_exact_signatures is not None else set()
        for record in records:
            fingerprint = self._fingerprint_from_record(record)
            if fingerprint is None or fingerprint.exact_signature in registered:
                continue
            registered.add(fingerprint.exact_signature)
            self._register_acceptor_fingerprint(
                acceptor_state=acceptor_state,
                fingerprint=fingerprint,
                increment_version=True,
                count_edge_usage=False,
            )

    def _merge_fingerprints_into_acceptor_state(
        self,
        fingerprints: list[_AcceptedPathFingerprint],
        *,
        acceptor_state: _AcceptorState,
        registered_exact_signatures: set[str] | None = None,
    ) -> None:
        registered = registered_exact_signatures if registered_exact_signatures is not None else set()
        for fingerprint in fingerprints:
            if fingerprint.exact_signature in registered:
                continue
            registered.add(fingerprint.exact_signature)
            self._register_acceptor_fingerprint(
                acceptor_state=acceptor_state,
                fingerprint=fingerprint,
                increment_version=True,
                count_edge_usage=False,
            )

    @staticmethod
    def _fingerprint_to_state_record(fingerprint: _AcceptedPathFingerprint) -> dict[str, Any]:
        return {
            "path_id": fingerprint.path_id,
            "exact_signature": fingerprint.exact_signature,
            "core_signature": fingerprint.core_signature,
            "prefix_signature": fingerprint.prefix_signature,
            "target_node_id": fingerprint.target_node_id,
            "node_ids": list(fingerprint.node_ids),
            "edge_ids": list(fingerprint.edge_ids),
        }

    def _persisted_path_fingerprints(
        self,
        *,
        base_fingerprints: list[_AcceptedPathFingerprint] | None = None,
    ) -> list[_AcceptedPathFingerprint]:
        persisted_records = self._load_existing_samples().values()
        fingerprints: list[_AcceptedPathFingerprint] = []
        seen_exact_signatures: set[str] = set()
        for fingerprint in base_fingerprints or []:
            if fingerprint.exact_signature in seen_exact_signatures:
                continue
            seen_exact_signatures.add(fingerprint.exact_signature)
            fingerprints.append(fingerprint)
        for record in persisted_records:
            fingerprint = self._fingerprint_from_record(record)
            if fingerprint is None or fingerprint.exact_signature in seen_exact_signatures:
                continue
            seen_exact_signatures.add(fingerprint.exact_signature)
            fingerprints.append(fingerprint)
        return fingerprints

    @staticmethod
    def _state_fingerprints_from_sampler_state(state: dict[str, Any] | None) -> list[_AcceptedPathFingerprint]:
        if not state:
            return []
        raw_records = state.get("accepted_path_fingerprints") or []
        if not isinstance(raw_records, list):
            return []
        fingerprints: list[_AcceptedPathFingerprint] = []
        seen_exact_signatures: set[str] = set()
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                continue
            fingerprint = VqaBatchRunner._fingerprint_from_state_record(raw_record)
            if fingerprint is None or fingerprint.exact_signature in seen_exact_signatures:
                continue
            seen_exact_signatures.add(fingerprint.exact_signature)
            fingerprints.append(fingerprint)
        return fingerprints

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
    def _append_jsonl(handle, record: dict[str, Any], *, flush: bool = False) -> None:
        """Append one record and close path-backed writers to publish through HDFS-FUSE."""
        if isinstance(handle, _JsonlBatchWriter):
            handle.append(record)
            return
        if isinstance(handle, (str, Path)):
            path = Path(handle)
            with path.open("a", encoding="utf-8") as opened_handle:
                VqaBatchRunner._append_jsonl(opened_handle, record, flush=flush)
            return
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        if flush:
            handle.flush()
            VqaBatchRunner._fsync_handle(handle)

    @staticmethod
    def _fsync_handle(handle) -> None:
        try:
            os.fsync(handle.fileno())
        except OSError as exc:  # Some network/FUSE filesystems do not implement fsync.
            path = str(getattr(handle, "name", "<unknown>"))
            if path not in _FSYNC_WARNING_PATHS:
                _FSYNC_WARNING_PATHS.add(path)
                logger.warning(
                    "VQA JSONL fsync is unavailable for %s; readers may not see updates immediately: %s",
                    path,
                    exc,
                )

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

    def _sampler_worker_count(self, remaining: int) -> int:
        if remaining <= 0:
            return 0
        return max(1, min(self.workers, remaining))

    def _proposal_seed(self, proposal_id: int) -> int:
        return int(self.pipeline.config.random_seed) + proposal_id * 9973
