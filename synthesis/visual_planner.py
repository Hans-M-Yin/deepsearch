"""Visual search planning objects and interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Protocol

from .evidence import Evidence
from .model_worker import ModelWorkerClient


def _stable_hash(*parts: object, length: int = 16) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:length]


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


class VisualTargetType(str, Enum):
    EVENT_PHOTO = "event_photo"
    POSTER = "poster"
    FIGURE = "figure"
    ARTWORK = "artwork"
    PRODUCT = "product"
    LOGO = "logo"
    MAP = "map"
    DOCUMENT = "document"
    GROUP_PHOTO = "group_photo"
    OBJECT_DETAIL = "object_detail"
    SCREENSHOT = "screenshot"
    OTHER = "other"


class DownstreamUse(str, Enum):
    ANSWER_EVIDENCE = "answer_evidence"
    ROUTING_CLUE = "routing_clue"
    GROUNDING = "grounding"
    DISTRACTOR = "distractor"


@dataclass(slots=True)
class SearchQuerySpec:
    """One text-to-image query proposed for a visual target."""

    query_id: str
    query: str
    target_evidence_id: str
    intent: str | None = None
    expected_visual: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def create(
        cls,
        query: str,
        target_evidence_id: str,
        *,
        intent: str | None = None,
        expected_visual: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SearchQuerySpec":
        return cls(
            query_id=f"query_{_stable_hash(target_evidence_id, query, intent)}",
            query=query,
            target_evidence_id=target_evidence_id,
            intent=intent,
            expected_visual=expected_visual,
            source=source,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class VisualSearchPlan:
    """MLLM-produced visual target plus the queries used to search for it."""

    plan_id: str
    target: Evidence
    queries: list[SearchQuerySpec] = field(default_factory=list)
    source_node_id: str | None = None
    source_evidence_ids: list[str] = field(default_factory=list)
    planner: str | None = None
    raw_model_asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = _jsonify(asdict(self))
        data["target"] = self.target.to_dict()
        data["queries"] = [query.to_dict() for query in self.queries]
        return data

    @classmethod
    def create(
        cls,
        target: Evidence,
        *,
        queries: list[SearchQuerySpec] | None = None,
        source_node_id: str | None = None,
        source_evidence_ids: list[str] | None = None,
        planner: str | None = None,
        raw_model_asset_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "VisualSearchPlan":
        return cls(
            plan_id=f"visual_plan_{_stable_hash(target.evidence_id)}",
            target=target,
            queries=queries or [],
            source_node_id=source_node_id,
            source_evidence_ids=source_evidence_ids or [],
            planner=planner,
            raw_model_asset_id=raw_model_asset_id,
            metadata=metadata or {},
        )


class VisualSearchPlanner(Protocol):
    """Plan visual targets and image-search queries from a text node."""

    model_client: ModelWorkerClient

    def plan(
        self,
        *,
        node: dict[str, Any],
        page_text: str,
        source_evidence_ids: list[str] | None = None,
        run_id: str | None = None,
    ) -> list[VisualSearchPlan]:
        """Return target evidences together with their image-search queries."""
