"""High-level graph-to-question pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os

from synthesis.model_worker import LLM_WORKER
from synthesis.store import JsonlGraphStore

from .graph_view import GraphView
from .path_sampler import RandomPathSampler, SamplerConfiguration
from .question_writer import QuestionWriter
from .schemas import EvidenceBundle, PathCandidate, SampleProgress, SampleStatus, VqaSample
from .verifier import SampleVerifier


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VqaGenerationError(RuntimeError):
    """A hard failure while processing one sampled path."""

    def __init__(self, *, path_id: str, stage: str, cause: Exception) -> None:
        self.path_id = path_id
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage} failed for {path_id}: {cause.__class__.__name__}: {cause}")


@dataclass(slots=True)
class VqaGenerationPipeline:
    """Orchestrate sampling, evidence construction, writing, and verification."""

    store: JsonlGraphStore
    config: SamplerConfiguration
    sampler: RandomPathSampler | None = None
    writer: QuestionWriter | None = None
    verifier: SampleVerifier | None = None
    graph: GraphView = field(init=False)

    def __post_init__(self) -> None:
        graph = GraphView(self.store, allowed_edge_types=set(self.config.allowed_edge_types))
        self.graph = graph
        sampler_model = os.environ.get("VQA_SAMPLER_MODEL")
        self.sampler = self.sampler or RandomPathSampler(
            graph=graph,
            config=self.config,
            model_client=LLM_WORKER if sampler_model and self.config.neighbor_selection_strategy == "llm_guided" else None,
            model=sampler_model,
        )
        self.sampler.graph = graph
        self.sampler.config = self.config
        writer_model = os.environ.get("VQA_WRITER_MODEL")
        self.writer = self.writer or QuestionWriter(
            model_client=LLM_WORKER if writer_model else None,
            model=writer_model,
        )
        self.verifier = self.verifier or SampleVerifier()

    def sample_paths(self, *, limit: int | None = None) -> list[PathCandidate]:
        """Sample paths serially so sampler diversity state remains consistent."""
        sample_limit = self.config.max_samples if limit is None else limit
        return self.sampler.generate(limit=sample_limit)

    def generate_path(self, path: PathCandidate) -> VqaSample:
        """Generate and verify one question from an already sampled path."""
        progress = SampleProgress()
        evidence = EvidenceBundle(
            bundle_id=f"bundle_{path.path_id}",
            path_id=path.path_id,
            metadata={"placeholder": True, "source": "pipeline_without_evidence_builder"},
        )

        draft = self._run_stage(
            path=path,
            stage="draft",
            operation=lambda: self.writer.draft(path=path, graph=self.graph),
        )
        progress.drafted_at = _utc_now()
        polished = self._run_stage(
            path=path,
            stage="polish",
            operation=lambda: self.writer.polish(draft=draft, path=path, graph=self.graph),
        )
        progress.polished_at = _utc_now()
        obfuscated = self._run_stage(
            path=path,
            stage="difficulty_enhancement",
            operation=lambda: self.writer.enhance_difficulty(draft=polished, path=path, graph=self.graph),
        )
        progress.post_obfuscated_at = _utc_now()
        verification = self._run_stage(
            path=path,
            stage="verification",
            operation=lambda: self.verifier.verify(question=obfuscated),
        )
        progress.verified_at = _utc_now()
        status = SampleStatus.VERIFIED if verification.final_keep else SampleStatus.REJECTED
        return VqaSample(
            sample_id=f"sample_{path.path_id}",
            status=status,
            path=path,
            evidence=evidence,
            draft=draft,
            polished=polished,
            obfuscated=obfuscated,
            verification=verification,
            progress=progress,
            metadata={
                "writer_warnings": list(obfuscated.metadata.get("writer_warnings") or []),
            },
        )

    def generate(self, *, limit: int | None = None) -> list[VqaSample]:
        return [self.generate_path(path) for path in self.sample_paths(limit=limit)]

    @staticmethod
    def _run_stage(*, path: PathCandidate, stage: str, operation):
        try:
            return operation()
        except Exception as exc:
            raise VqaGenerationError(path_id=path.path_id, stage=stage, cause=exc) from exc
