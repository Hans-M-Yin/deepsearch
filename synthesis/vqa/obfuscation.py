"""Pre/post obfuscation hooks for anti-leakage processing."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .schemas import EvidenceBundle, QuestionDraft


@dataclass(slots=True)
class ObfuscationProcessor:
    """Apply lightweight anti-leakage transformations.

    The first version keeps the implementation intentionally conservative. We
    only expose structured hooks so the actual policy can be tightened later.
    """

    def pre_obfuscate(self, evidence: EvidenceBundle, *, target_title: str | None = None) -> EvidenceBundle:
        if not target_title:
            return evidence
        masked = self._mask_in_bundle(evidence, target_title)
        return masked

    def post_obfuscate(self, draft: QuestionDraft, *, target_title: str | None = None) -> QuestionDraft:
        if not target_title:
            return draft
        question = self._mask_text(draft.question, target_title)
        why_image = None if draft.why_image_is_needed is None else self._mask_text(draft.why_image_is_needed, target_title)
        return QuestionDraft(
            question=question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            why_image_is_needed=why_image,
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=dict(draft.metadata),
        )

    def _mask_in_bundle(self, evidence: EvidenceBundle, target_title: str) -> EvidenceBundle:
        def _rewrite(items: list):
            rewritten = []
            for item in items:
                rewritten.append(
                    type(item)(
                        evidence_id=item.evidence_id,
                        source_kind=item.source_kind,
                        source_node_id=item.source_node_id,
                        modality=item.modality,
                        title=self._mask_text(item.title, target_title) if item.title else item.title,
                        raw_content=self._mask_text(item.raw_content, target_title) if item.raw_content else item.raw_content,
                        transformed_content=self._mask_text(item.transformed_content, target_title)
                        if item.transformed_content
                        else item.transformed_content,
                        relation_hint=item.relation_hint,
                        leakage_flags=list(item.leakage_flags),
                        metadata=dict(item.metadata),
                    )
                )
            return rewritten

        return EvidenceBundle(
            bundle_id=evidence.bundle_id,
            path_id=evidence.path_id,
            oracle_evidence=_rewrite(evidence.oracle_evidence),
            writer_evidence=_rewrite(evidence.writer_evidence),
            verifier_evidence=_rewrite(evidence.verifier_evidence),
            metadata=dict(evidence.metadata),
        )

    @staticmethod
    def _mask_text(text: str, target_title: str) -> str:
        if not text or not target_title:
            return text
        pattern = re.compile(re.escape(target_title), flags=re.IGNORECASE)
        return pattern.sub("[MASKED_ENTITY]", text)
