"""Node definitions for the multimodal data synthesis graph.

The graph keeps only metadata and stable asset references. Large payloads such
as images, raw HTML, and embeddings should live in OSS or an external asset
store and be referenced from these objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, ClassVar


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


class NodeType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    REGION = "region"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(slots=True)
class NodeSource:
    """Where a node came from and how it can be traced back."""

    source_type: str
    url: str | None = None
    source_id: str | None = None
    run_id: str | None = None
    builder: str | None = None
    raw_ref: str | None = None


@dataclass(slots=True)
class AssetRef:
    """Reference to external payloads without storing the payload in the DB."""

    ref_type: str
    uri: str
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Node:
    """Base node shared by all graph node types."""

    node_type: ClassVar[NodeType]

    node_id: str
    title: str | None = None
    summary: str | None = None
    source: NodeSource | None = None
    asset_refs: list[AssetRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.ACTIVE
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = _jsonify(asdict(self))
        data["node_type"] = self.node_type.value
        return data

    @classmethod
    def make_id(cls, *parts: object) -> str:
        return f"{cls.node_type.value}_{_stable_hash(*parts)}"


@dataclass(slots=True)
class TextNode(Node):
    """Text-side node, either a grounded entity or a webpage/document."""

    node_type: ClassVar[NodeType] = NodeType.TEXT

    subtype: str = "webpage"
    canonical_id: str | None = None
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wiki_entity(
        cls,
        qid: str,
        title: str,
        *,
        summary: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        source_url: str | None = None,
        run_id: str | None = None,
    ) -> "TextNode":
        return cls(
            node_id=cls.make_id("wiki_entity", qid),
            subtype="wiki_entity",
            canonical_id=f"wikidata:{qid}" if not qid.startswith("wikidata:") else qid,
            title=title,
            summary=summary,
            description=description,
            aliases=aliases or [],
            attributes=attributes or {},
            source=NodeSource(
                source_type="wikidata",
                url=source_url,
                source_id=qid,
                run_id=run_id,
                builder="wiki_entity_builder",
            ),
        )

    @classmethod
    def from_webpage(
        cls,
        url: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        description: str | None = None,
        raw_ref: str | None = None,
        run_id: str | None = None,
    ) -> "TextNode":
        return cls(
            node_id=cls.make_id("webpage", url),
            subtype="webpage",
            title=title,
            summary=summary,
            description=description,
            source=NodeSource(
                source_type="webpage",
                url=url,
                run_id=run_id,
                builder="webpage_builder",
                raw_ref=raw_ref,
            ),
        )


@dataclass(slots=True)
class ImageNode(Node):
    """Image node with public and OSS references plus lightweight metadata."""

    node_type: ClassVar[NodeType] = NodeType.IMAGE

    image_url: str | None = None
    source_page_url: str | None = None
    oss_uri: str | None = None
    thumb_oss_uri: str | None = None
    caption: str | None = None
    width: int | None = None
    height: int | None = None
    content_type: str | None = None
    phash: str | None = None
    storage_status: str = "url_only"

    @classmethod
    def from_url(
        cls,
        image_url: str,
        *,
        source_page_url: str | None = None,
        caption: str | None = None,
        title: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ImageNode":
        return cls(
            node_id=cls.make_id("image", image_url),
            title=title,
            summary=caption,
            image_url=image_url,
            source_page_url=source_page_url,
            caption=caption,
            source=NodeSource(
                source_type="image_search",
                url=image_url,
                run_id=run_id,
                builder="image_discovery_builder",
            ),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class RegionNode(Node):
    """Region/crop node derived from an image node."""

    node_type: ClassVar[NodeType] = NodeType.REGION

    parent_image_id: str = ""
    bbox: tuple[float, float, float, float] | None = None
    crop_oss_uri: str | None = None
    caption: str | None = None
    visual_clues: list[str] = field(default_factory=list)

    @classmethod
    def from_bbox(
        cls,
        parent_image_id: str,
        bbox: tuple[float, float, float, float],
        *,
        caption: str | None = None,
        crop_oss_uri: str | None = None,
        visual_clues: list[str] | None = None,
        run_id: str | None = None,
    ) -> "RegionNode":
        return cls(
            node_id=cls.make_id("region", parent_image_id, bbox),
            title=caption,
            summary=caption,
            parent_image_id=parent_image_id,
            bbox=bbox,
            crop_oss_uri=crop_oss_uri,
            caption=caption,
            visual_clues=visual_clues or [],
            source=NodeSource(
                source_type="image_region",
                source_id=parent_image_id,
                run_id=run_id,
                builder="image_grounding_builder",
            ),
        )
