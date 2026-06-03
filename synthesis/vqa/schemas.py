"""Core schemas for graph-to-question generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


class SampleStatus(str, Enum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(slots=True)
class TrajectoryStats:
    """Lightweight labels for later analysis of sampled path distributions."""

    start_modality: str
    end_modality: str
    modality_sequence: list[str]
    hop_count: int
    image_node_count: int
    text_node_count: int
    modality_switch_count: int
    starts_with_image: bool
    ends_with_image: bool
    image_only_at_start: bool = False
    image_only_at_end: bool = False
    has_mid_image: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class PathCandidate:
    """A sampled graph path before question writing."""

    path_id: str
    node_ids: list[str]
    edge_ids: list[str]
    node_types: list[str]
    edge_types: list[str]
    relations: list[str]
    target_node_id: str
    start_node_id: str
    trajectory: TrajectoryStats
    exact_signature: str
    skeleton_signature: str
    core_signature: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class EvidenceItem:
    """One normalized evidence unit extracted from a path node or edge."""

    evidence_id: str
    source_kind: str
    source_node_id: str | None
    modality: str
    title: str | None = None
    raw_content: str | None = None
    transformed_content: str | None = None
    relation_hint: str | None = None
    leakage_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class EvidenceBundle:
    """Multi-view evidence package for question writing and verification."""

    bundle_id: str
    path_id: str
    oracle_evidence: list[EvidenceItem] = field(default_factory=list)
    writer_evidence: list[EvidenceItem] = field(default_factory=list)
    verifier_evidence: list[EvidenceItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class QuestionDraft:
    """Draft or polished question produced by the writer stage."""

    question: str
    answer: str
    answer_type: str
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    used_evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class VerificationCheck:
    """One verification sub-result."""

    name: str
    passed: bool
    score: float | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class VerificationResult:
    """Full verification decision for one generated sample."""

    checks: list[VerificationCheck] = field(default_factory=list)
    final_keep: bool = False
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class SampleProgress:
    """Track intermediate artifacts for one sample candidate."""

    sampled_at: str = field(default_factory=_utc_now)
    pre_obfuscated_at: str | None = None
    drafted_at: str | None = None
    polished_at: str | None = None
    post_obfuscated_at: str | None = None
    verified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class VqaSample:
    """End-to-end sample record with full intermediate provenance."""

    sample_id: str
    status: SampleStatus
    path: PathCandidate
    evidence: EvidenceBundle
    draft: QuestionDraft | None = None
    polished: QuestionDraft | None = None
    verification: VerificationResult | None = None
    progress: SampleProgress = field(default_factory=SampleProgress)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class VqaGenerationConfig:
    """High-level configuration shared across pipeline stages."""

    min_hops: int = 3
    max_hops: int = 5
    max_samples: int = 100
    random_seed: int = 0
    allowed_edge_types: tuple[str, ...] = (
        "wiki_link",
        "wiki_attribute",
        "web_link",
        "search_retrieved",
        "image_source_page",
        "image_depicts",
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))
