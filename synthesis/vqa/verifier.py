"""Verification hooks for generated VQA samples."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import QuestionDraft, VerificationCheck, VerificationResult


@dataclass(slots=True)
class SampleVerifier:
    """First-pass verifier with structural placeholder checks."""

    def verify(self, *, question: QuestionDraft) -> VerificationResult:
        checks = [
            VerificationCheck(
                name="question_non_empty",
                passed=bool(question.question.strip()),
                detail="Question text must be non-empty.",
            ),
            VerificationCheck(
                name="answer_non_empty",
                passed=bool(question.answer.strip()),
                detail="Answer text must be non-empty.",
            ),
        ]
        final_keep = all(check.passed for check in checks)
        return VerificationResult(
            checks=checks,
            final_keep=final_keep,
            reject_reason=None if final_keep else "basic_validation_failed",
        )
