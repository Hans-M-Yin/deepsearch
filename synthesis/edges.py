"""Edge definitions for the multimodal data synthesis graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class EdgeType(str, Enum):
    WIKI_LINK = "wiki_link"
    WIKI_ATTRIBUTE = "wiki_attribute"
    WEB_LINK = "web_link"
    SEARCH_RETRIEVED = "search_retrieved"
    IMAGE_SOURCE_PAGE = "image_source_page"
    IMAGE_DEPICTS = "image_depicts"
    IMAGE_CONTAINS_REGION = "image_contains_region"
    REGION_IDENTIFIES = "region_identifies"
    REGION_HAS_TEXT = "region_has_text"
    VISUALLY_SIMILAR = "visually_similar"
    EVIDENCE_SUPPORTS = "evidence_supports"
    DERIVED = "derived"


class EdgeStatus(str, Enum):
    ACTIVE = "active"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(slots=True)
class EdgeSource:
    """Where an edge came from and which component produced it."""

    source_type: str
    url: str | None = None
    source_id: str | None = None
    run_id: str | None = None
    builder: str | None = None
    raw_ref: str | None = None


@dataclass(slots=True)
class EvidenceRef:
    """Reference to evidence supporting an edge."""

    evidence_id: str | None = None
    node_id: str | None = None
    asset_uri: str | None = None
    url: str | None = None
    quote: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    """Directed relationship between two graph nodes."""

    edge_id: str
    src_node_id: str
    dst_node_id: str
    edge_type: EdgeType
    relation: str
    src_node_type: str | None = None
    dst_node_type: str | None = None
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    source: EdgeSource | None = None
    confidence: float | None = None
    extractor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: EdgeStatus = EdgeStatus.ACTIVE
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def make_id(
        cls,
        src_node_id: str,
        dst_node_id: str,
        edge_type: EdgeType | str,
        relation: str,
        evidence_key: str | None = None,
    ) -> str:
        edge_type_value = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
        return f"edge_{_stable_hash(src_node_id, dst_node_id, edge_type_value, relation, evidence_key)}"

    @classmethod
    def create(
        cls,
        src_node_id: str,
        dst_node_id: str,
        *,
        edge_type: EdgeType,
        relation: str,
        src_node_type: str | None = None,
        dst_node_type: str | None = None,
        evidence_refs: list[EvidenceRef] | None = None,
        source: EdgeSource | None = None,
        confidence: float | None = None,
        extractor: str | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_key: str | None = None,
    ) -> "Edge":
        return cls(
            edge_id=cls.make_id(src_node_id, dst_node_id, edge_type, relation, evidence_key),
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            edge_type=edge_type,
            relation=relation,
            src_node_type=src_node_type,
            dst_node_type=dst_node_type,
            evidence_refs=evidence_refs or [],
            source=source,
            confidence=confidence,
            extractor=extractor,
            metadata=metadata or {},
        )


def allowed_edge_types(src_node_type: str, dst_node_type: str) -> set[EdgeType]:
    """Return common edge types for a node-type pair.

    This is intentionally advisory rather than enforced in Edge.create because
    new builders may introduce task-specific relations before the schema is
    finalized.
    """

    pair = (src_node_type, dst_node_type)
    mapping: dict[tuple[str, str], set[EdgeType]] = {
        ("text", "text"): {
            EdgeType.WIKI_LINK,
            EdgeType.WIKI_ATTRIBUTE,
            EdgeType.WEB_LINK,
            EdgeType.SEARCH_RETRIEVED,
            EdgeType.EVIDENCE_SUPPORTS,
            EdgeType.DERIVED,
        },
        ("text", "image"): {
            EdgeType.SEARCH_RETRIEVED,
            EdgeType.IMAGE_SOURCE_PAGE,
            EdgeType.EVIDENCE_SUPPORTS,
            EdgeType.DERIVED,
        },
        ("image", "text"): {
            EdgeType.IMAGE_SOURCE_PAGE,
            EdgeType.IMAGE_DEPICTS,
            EdgeType.SEARCH_RETRIEVED,
            EdgeType.EVIDENCE_SUPPORTS,
            EdgeType.DERIVED,
        },
        ("image", "image"): {
            EdgeType.VISUALLY_SIMILAR,
            EdgeType.SEARCH_RETRIEVED,
            EdgeType.DERIVED,
        },
        ("image", "region"): {
            EdgeType.IMAGE_CONTAINS_REGION,
        },
        ("region", "text"): {
            EdgeType.REGION_IDENTIFIES,
            EdgeType.REGION_HAS_TEXT,
            EdgeType.EVIDENCE_SUPPORTS,
            EdgeType.DERIVED,
        },
        ("region", "image"): {
            EdgeType.SEARCH_RETRIEVED,
            EdgeType.VISUALLY_SIMILAR,
            EdgeType.DERIVED,
        },
    }
    return mapping.get(pair, {EdgeType.DERIVED})
