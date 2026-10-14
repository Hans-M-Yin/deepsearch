"""Stable data contracts for the FVQA hard-case pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VERSION = "fvqa-hard-trajectory-v2"


@dataclass(frozen=True)
class PublicCase:
    """The only case payload permitted in a writer task or task queue."""

    case_id: str
    question: str
    image_path: str
    split: str = "train"
    source_dataset: str = "fvqa"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PublicCase":
        forbidden = {
            "gold_answer",
            "answer",
            "answers",
            "acceptable_answers",
            "ground_truth_list",
            "hop_chain",
            "statement",
            "knowledge_graph",
        }
        exposed = forbidden.intersection(value)
        if exposed:
            raise ValueError(f"Public case contains private fields: {sorted(exposed)}")
        return cls(
            case_id=str(value["case_id"]),
            question=str(value["question"]),
            image_path=str(value["image_path"]),
            split=str(value.get("split") or "train"),
            source_dataset=str(value.get("source_dataset") or "fvqa"),
        )


@dataclass(frozen=True)
class CanonicalCase:
    """One FVQA example with supervision deliberately separated from prompt data."""

    case_id: str
    question: str
    gold_answer: str
    image_path: str
    split: str = "train"
    source_dataset: str = "fvqa"
    source_index: int | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    # Some sources, including LiveVQA, provide multiple semantically valid
    # references. This field is private and never reaches PublicCase/teacher.
    # Empty is omitted from serialization to preserve the existing FVQA
    # manifest format byte-for-byte at the field level.
    acceptable_answers: tuple[str, ...] = ()

    def public_case(self) -> PublicCase:
        """The complete case view the expert writer is permitted to see."""

        return PublicCase(
            case_id=self.case_id,
            question=self.question,
            image_path=self.image_path,
            split=self.split,
            source_dataset=self.source_dataset,
        )

    def prompt_view(self) -> dict[str, Any]:
        """Compatibility view for callers that need a JSON-safe public payload."""

        return self.public_case().to_dict()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if not value["acceptable_answers"]:
            value.pop("acceptable_answers")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalCase":
        raw_answers = value.get("acceptable_answers") or []
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        return cls(
            case_id=str(value["case_id"]),
            question=str(value["question"]),
            gold_answer=str(value["gold_answer"]),
            image_path=str(value["image_path"]),
            split=str(value.get("split") or "train"),
            source_dataset=str(value.get("source_dataset") or "fvqa"),
            source_index=value.get("source_index"),
            source_metadata=dict(value.get("source_metadata") or {}),
            acceptable_answers=tuple(
                str(answer).strip()
                for answer in raw_answers
                if str(answer).strip()
            ),
        )


@dataclass(frozen=True)
class FailureCandidate:
    """A base-model failure eligible for expert collection."""

    case_id: str
    reason: str
    base_answer: str = ""
    evaluator_confidence: float | None = None
    source_run: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollectorConfig:
    """Configuration needed to call the existing SFT tool runtime."""

    model: str
    workdir: str
    api_mode: str = "manual_react"
    client_type: str = "azure_openai"
    max_tokens: int | None = None
    max_turns: int = 8
    temperature: float | None = None
    timeout_s: float | None = None
    style_guide_file: str | None = None
    style_examples_file: str | None = None
    print_rounds: bool = False

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("CollectorConfig.model is required")
        if self.api_mode not in {"manual_react", "native_tools", "responses"}:
            raise ValueError(f"Unsupported api_mode: {self.api_mode!r}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")


@dataclass
class FilterConfig:
    """Deterministic gates; visual strength is tracked but not globally mandatory."""

    require_correct_answer: bool = True
    require_trajectory_quality_review: bool = True
    require_visual_compare: bool = False
    min_opened_visual_candidates: int = 1
    require_candidate_rejection: bool = False
    require_external_fact_evidence: bool = False
    reject_tool_errors_at_or_above: int = 2
    # The collector may use as many as 30 model turns to close an evidence
    # chain.  Keep the acceptance budget aligned with that generation budget:
    # answer correctness and the label-blind quality review, rather than trace
    # length alone, are the primary production quality gates.
    max_assistant_turns: int = 30
    require_blind_visual_review: bool = True

    def validate(self) -> None:
        if self.min_opened_visual_candidates < 1:
            raise ValueError("min_opened_visual_candidates must be positive")
        if self.reject_tool_errors_at_or_above < 1:
            raise ValueError("reject_tool_errors_at_or_above must be positive")
        if self.max_assistant_turns < 1:
            raise ValueError("max_assistant_turns must be positive")


@dataclass(frozen=True)
class BlindVisualRequest:
    """Payload for an independent reviewer; intentionally excludes gold/entity text."""

    case_id: str
    original_image: str
    candidate_images: list[str]
    public_target_description: str
    candidate_resource_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    decision: str
    reasons: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision == "keep"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
