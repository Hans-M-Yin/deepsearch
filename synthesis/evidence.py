"""Evidence, asset, and search snapshot objects for graph construction."""

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


class EvidenceType(str, Enum):
    WEB_TEXT = "web_text"
    IMAGE = "image"
    IMAGE_REGION = "image_region"
    VISUAL_TARGET = "visual_target"
    CAPTION = "caption"
    OCR = "ocr"
    SEARCH_RESULT = "search_result"
    VLM_OUTPUT = "vlm_output"
    LLM_OUTPUT = "llm_output"


class AssetType(str, Enum):
    WEBPAGE_RAW = "webpage_raw"
    WEBPAGE_MARKDOWN = "webpage_markdown"
    IMAGE_ORIGINAL = "image_original"
    IMAGE_THUMBNAIL = "image_thumbnail"
    IMAGE_REGION = "image_region"
    SEARCH_RESPONSE = "search_response"
    MODEL_OUTPUT = "model_output"
    EMBEDDING = "embedding"


class SearchEngine(str, Enum):
    SERPER_TEXT = "serper_text"
    SERPER_IMAGE = "serper_image"
    JINA_READER = "jina_reader"
    WIKIDATA = "wikidata"
    WIKIPEDIA = "wikipedia"
    LOCAL_INDEX = "local_index"
    OTHER = "other"


class RecordStatus(str, Enum):
    ACTIVE = "active"
    REJECTED = "rejected"
    STALE = "stale"
    FAILED = "failed"


@dataclass(slots=True)
class Asset:
    """External payload reference, usually backed by OSS or local cache."""

    asset_id: str
    asset_type: AssetType
    uri: str
    original_url: str | None = None
    content_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.ACTIVE
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def make_id(cls, asset_type: AssetType | str, uri: str) -> str:
        asset_type_value = asset_type.value if isinstance(asset_type, AssetType) else asset_type
        return f"asset_{_stable_hash(asset_type_value, uri)}"

    @classmethod
    def create(
        cls,
        asset_type: AssetType,
        uri: str,
        *,
        original_url: str | None = None,
        content_type: str | None = None,
        sha256_value: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Asset":
        return cls(
            asset_id=cls.make_id(asset_type, uri),
            asset_type=asset_type,
            uri=uri,
            original_url=original_url,
            content_type=content_type,
            sha256=sha256_value,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class Evidence:
    """A compact, traceable piece of evidence supporting nodes or edges."""

    evidence_id: str
    evidence_type: EvidenceType
    content: str | None = None
    node_ids: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    url: str | None = None
    span: tuple[int, int] | None = None
    bbox: tuple[float, float, float, float] | None = None
    source_snapshot_id: str | None = None
    extractor: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.ACTIVE
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def make_id(
        cls,
        evidence_type: EvidenceType | str,
        *parts: object,
    ) -> str:
        evidence_type_value = evidence_type.value if isinstance(evidence_type, EvidenceType) else evidence_type
        return f"evidence_{_stable_hash(evidence_type_value, *parts)}"

    @classmethod
    def create(
        cls,
        evidence_type: EvidenceType,
        *,
        content: str | None = None,
        node_ids: list[str] | None = None,
        asset_ids: list[str] | None = None,
        url: str | None = None,
        span: tuple[int, int] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        source_snapshot_id: str | None = None,
        extractor: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_key: str | None = None,
    ) -> "Evidence":
        stable_key = evidence_key or content or url or source_snapshot_id or metadata
        return cls(
            evidence_id=cls.make_id(evidence_type, stable_key),
            evidence_type=evidence_type,
            content=content,
            node_ids=node_ids or [],
            asset_ids=asset_ids or [],
            url=url,
            span=span,
            bbox=bbox,
            source_snapshot_id=source_snapshot_id,
            extractor=extractor,
            confidence=confidence,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class SearchSnapshot:
    """Raw search/read request and response metadata for reproducibility."""

    snapshot_id: str
    engine: SearchEngine
    query: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    response_ref: str | None = None
    response_preview: str | None = None
    result_count: int | None = None
    status_code: int | None = None
    error: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.ACTIVE
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def make_id(
        cls,
        engine: SearchEngine | str,
        query: str | None,
        request: dict[str, Any] | None = None,
    ) -> str:
        engine_value = engine.value if isinstance(engine, SearchEngine) else engine
        request_key = repr(sorted((request or {}).items()))
        return f"snapshot_{_stable_hash(engine_value, query, request_key)}"

    @classmethod
    def create(
        cls,
        engine: SearchEngine,
        *,
        query: str | None = None,
        request: dict[str, Any] | None = None,
        response_ref: str | None = None,
        response_preview: str | None = None,
        result_count: int | None = None,
        status_code: int | None = None,
        error: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: RecordStatus = RecordStatus.ACTIVE,
    ) -> "SearchSnapshot":
        request_payload = request or {}
        return cls(
            snapshot_id=cls.make_id(engine, query, request_payload),
            engine=engine,
            query=query,
            request=request_payload,
            response_ref=response_ref,
            response_preview=response_preview,
            result_count=result_count,
            status_code=status_code,
            error=error,
            run_id=run_id,
            metadata=metadata or {},
            status=status,
        )
