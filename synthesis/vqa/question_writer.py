"""Question writer interface for LLM-backed question construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schemas import EvidenceBundle, PathCandidate, QuestionDraft


class WriterBackend(Protocol):
    """Minimal backend contract for future LLM integration."""

    def generate_question(self, *, path: PathCandidate, evidence: EvidenceBundle) -> QuestionDraft:
        ...

    def polish_question(self, *, draft: QuestionDraft, path: PathCandidate, evidence: EvidenceBundle) -> QuestionDraft:
        ...


@dataclass(slots=True)
class QuestionWriter:
    """Thin wrapper around a future LLM-backed writing backend."""

    backend: WriterBackend | None = None

    def draft(self, *, path: PathCandidate, evidence: EvidenceBundle) -> QuestionDraft:
        if self.backend is None:
            return self._placeholder_draft(path=path, evidence=evidence)
        return self.backend.generate_question(path=path, evidence=evidence)

    def polish(self, *, draft: QuestionDraft, path: PathCandidate, evidence: EvidenceBundle) -> QuestionDraft:
        if self.backend is None:
            return draft
        return self.backend.polish_question(draft=draft, path=path, evidence=evidence)

    @staticmethod
    def _placeholder_draft(*, path: PathCandidate, evidence: EvidenceBundle) -> QuestionDraft:
        prompt_hint = evidence.writer_evidence[-1].transformed_content if evidence.writer_evidence else ""
        return QuestionDraft(
            question=f"Placeholder question for path {path.path_id}: {prompt_hint}",
            answer="TBD",
            answer_type="unknown",
            reasoning_steps=[],
            why_image_is_needed=None,
            used_evidence_ids=[item.evidence_id for item in evidence.writer_evidence],
            metadata={"placeholder": True},
        )
