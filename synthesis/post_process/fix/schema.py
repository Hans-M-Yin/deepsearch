"""Serializable contracts shared by independent pipeline stages."""

from dataclasses import dataclass, field, asdict
from typing import Any

VERSION = "i2i-fix-v4-selector-writer"


@dataclass
class Config:
    seed: int = 42
    replay_top_k: int = 20
    result_count: int = 5
    # Legacy planner compatibility fields; the v4 pipeline does not read them.
    branch_weights: dict[str, float] = field(
        default_factory=lambda: {
            "convergent": 0.20,
            "direct": 0.25,
            "contrast": 0.40,
            "recovery": 0.15,
        }
    )
    wrong_count_weights: dict[str, float] = field(
        default_factory=lambda: {"1": 0.6, "2": 0.4}
    )
    replacement_counts: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    positive_slots: list[int] = field(default_factory=lambda: [1, 2])
    convergence_min: int = 4
    similar_negative_probability: float = 0.8
    visual_check_workers: int = 16
    record_workers: int = 16
    writer_workers: int = 16
    global_visual_workers: int = 128
    gpt_reasoning_effort: str = "xhigh"
    context_patch_retries: int = 1
    progress_audit_retries: int = 1
    composition_retries: int = 1
    max_windows_per_record: int = 1
    max_original_views: int = 1
    max_selection_checks: int = 3
    retry_strategies: list[str] = field(
        default_factory=lambda: ["same", "full", "target_crop"]
    )
    require_distinct_judge: bool = False
    blind_visual_review: bool = False
    model: str = "gpt54_2_internal_azure"
    verifier_model: str = "gemini38flash_internal_azure"
    judge_model: str = "gpt54_2_internal_azure"
    max_tokens: int = 6000
    llm_retries: int = 2
    embedding_url: str = ""
    embedding_model: str = ""

    def validate(self):
        if type(self.composition_retries) is not int or self.composition_retries < 0:
            raise ValueError("composition_retries must be a nonnegative integer")
        if (
            type(self.progress_audit_retries) is not int
            or self.progress_audit_retries < 0
        ):
            raise ValueError("progress_audit_retries must be a nonnegative integer")
        for name in ("record_workers", "global_visual_workers", "writer_workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.context_patch_retries) is not int
            or self.context_patch_retries < 0
        ):
            raise ValueError("context_patch_retries must be a nonnegative integer")
        if self.gpt_reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("Invalid GPT reasoning effort")
        if type(self.visual_check_workers) is not int or self.visual_check_workers < 1:
            raise ValueError("visual_check_workers must be a positive integer")
        if self.max_original_views < 0 or not 1 <= self.max_selection_checks <= 3:
            raise ValueError("Invalid original-view or selection limit")
        if not self.retry_strategies or not set(self.retry_strategies) <= {
            "same",
            "full",
            "target_crop",
        }:
            raise ValueError("Invalid retry strategies")
        if self.require_distinct_judge and self.judge_model in {
            self.model,
            self.verifier_model,
        }:
            raise ValueError(
                "Configure a distinct judge model when require_distinct_judge is true"
            )
        if not 1 <= self.result_count <= 5 or self.replay_top_k < 5:
            raise ValueError("Visible result cap must be 1..5 and replay_top_k >= 5")
        if set(self.branch_weights) != {"convergent", "direct", "contrast", "recovery"}:
            raise ValueError("Invalid branch names")
        for weights in (self.branch_weights, self.wrong_count_weights):
            if any(v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError("Weights must be nonnegative with positive total")
        if not set(self.wrong_count_weights) <= {"1", "2"}:
            raise ValueError("Wrong candidate counts must be 1 or 2")
        if not self.replacement_counts or not set(self.replacement_counts) <= set(
            range(5)
        ):
            raise ValueError("Replacement counts must be in 0..4")
        if not self.positive_slots or not set(self.positive_slots) <= {1, 2}:
            raise ValueError("Positive slots must be 1 or 2")
        if not 1 <= self.convergence_min <= 5 or self.max_windows_per_record < 1:
            raise ValueError("Invalid convergence/window limits")
        if not 0 <= self.similar_negative_probability <= 1:
            raise ValueError("Invalid similar-negative probability")


@dataclass
class Window:
    call_index: int
    observation_index: int
    start: int
    end: int
    call: dict
    observation: dict
    candidates: list[dict]
    reads: dict[str, dict]
    question: str
    source_image: str
    protected_ids: list[str]


@dataclass
class Candidate:
    key: str
    public: dict
    resources: list[dict] = field(default_factory=list)
    historical: bool = False
    image_paths: list[str] = field(default_factory=list)
    read_observation: str = ""
    verdict: str = "UNCERTAIN"
    category: str = "UNRESOLVED"
    evidence: list[str] = field(default_factory=list)
    title_similarity: float = 0.0
    text_competitive: bool | None = None
    originally_viewed: bool = False
    text_reference_only: bool = False


@dataclass
class Plan:
    branch: str
    first: list[Candidate]
    attempts: list[Candidate]
    second: list[Candidate] = field(default_factory=list)
    retry_call: dict | None = None
    retry_after: int = 0
    positive_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
