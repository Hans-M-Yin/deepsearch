"""Image discovery strategy layer for visual search plans.

This module sits above the low-level image search clients. It runs one or more
text-to-image queries, records search traces, applies cheap candidate filters,
creates graph records, and leaves one image_check hook for future MLLM checks.
"""

from __future__ import annotations

import base64
import html
from io import BytesIO
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .edges import Edge, EdgeSource, EdgeType, EvidenceRef
from .evidence import (
    Asset,
    AssetType,
    Evidence,
    EvidenceType,
    RecordStatus,
    SearchEngine,
    SearchSnapshot,
)
from .model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelResponse, ModelWorkerClient
from .nodes import ImageNode, ImageVariant, NodeType, TextNode
from .search_client import ImageSearchResult, SearchClient, SearchResponse
from .store import JsonlGraphStore
from .visual_planner import SearchQuerySpec, VisualSearchPlan
from .wiki_text_builder import EnhancedReaderClient


def _trace_timing_enabled() -> bool:
    return os.environ.get("SYNTHESIS_TRACE_TIMING", "0") != "0"


def _trace_timing(message: str) -> None:
    if _trace_timing_enabled():
        print(f"[trace]{message}", file=sys.stderr, flush=True)
from .wiki_entity_resolver import WikiEntityResolver


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


class ImageCandidateStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


PROMPT_IMAGE_CHECK = """You are checking whether a candidate image is useful visual evidence for a multimodal deep-search target.

Judge primarily from the image content. Candidate metadata can help disambiguate, but it must not override what is visible in the image.

Accept if the image visibly matches the target or is a useful intermediate visual clue for the target.
Reject if the image is generic, unrelated, too ambiguous, only textually related, a placeholder, or an icon/logo when the target is not asking for one.

Output exactly one block:
<check>
decision: accept|reject
confidence: 0.0-1.0
reason: short reason
visual_fact: visible fact 1
visual_fact: visible fact 2
</check>
"""


PROMPT_IMAGE_GROUND = """You are analyzing an accepted image for multimodal graph construction.

Task:
Describe the image and ground only unique, searchable entities visible in or clearly represented by the image. Entity grounding is for linking this image to existing text/entity nodes.

Keep entities only if they are named or uniquely identifiable, such as a person, landmark, movie, book, album, artwork, product, brand, team, organization, event, document, map, or logo. Do not output generic objects such as person, woman, car, building, crowd, red shirt, tree.

You may receive webpage context associated with the image. Use that context only to disambiguate what is visible. Do not invent entities that are not visually supported by the image. If the image and context are insufficient to identify an entity confidently, omit it rather than guessing.

Important grounding rules:
1. Include indirect but clearly visible searchable entities when they are visually grounded.
   - Examples: an Adidas or Nike logo on clothing, a team crest on a jersey, a visible brand mark on an object. These marks point to a unique brand.
   
Output guidance:
1. `relation_to_image` is a visual locator, not an abstract semantic relation.
   It should help a user immediately point to the entity inside the image.
2. Describe the entity using visible position, local context, or distinctive appearance.
   Good examples:
   - second row, third person from the left
   - rightmost person on the album cover
   - gold trophy held in the man's arms
   - landmark behind the main character
   - logo on the front of the jersey
   - account name at the top of the screenshot
3. Prefer short, concrete, image-grounded locators:
   - relative position in a group
   - relative location in the frame
   - nearby object or nearby person
   - distinctive clothing, pose, or visible mark
4. Avoid abstract or non-localizable relations such as:
   - depicted in image
   - shown in image
   - associated with image
   - represented in image
   These are too generic and do not help locate the entity.
5. If the image contains multiple people or objects, `relation_to_image` must disambiguate the target.
6. `evidence` should be one short sentence explaining the visible cue that supports the grounding.
7. If two surface forms refer to the same entity, output only the canonical one and mention the alias/handle inside `evidence`.

Examples:
- For a 2025 G20 summit group photo:
  `entity: Emmanuel Macron | second row, third person from the left | visible as the suited male figure in that position`
- For the Queen II album cover:
  `entity: Roger Taylor | rightmost person on the album cover | visible as the face at the far right of the four-person composition`
- For a John Wick 4 poster:
  `entity: Eiffel Tower | landmark behind the main character | visible rising in the background behind John Wick`

Output exactly one block:
<ground>
caption: one concise image caption
entity: name | relation_to_image | evidence
entity: name | relation_to_image | evidence
</ground>
"""


PROMPT_IMAGE_QUERY_ENTITY_FILTER = """You are filtering grounded image entities for multi-hop graph expansion.

Goal:
We only want image-derived entities that add new information beyond the visual query itself.

Task:
Given:
- the source text node title
- the visual query text
- a list of grounded candidate entities from the image

Decide for each candidate whether it should be blocked because it is already explicitly mentioned in the query, or is just an alias / handle / surface form of an entity already mentioned in the query.

Block an entity if:
- it is the same entity as one already mentioned in the query
- it is only an alias, OCR handle, username, nickname, or alternate surface form of an entity already mentioned in the query

Keep an entity if:
- it is a new entity not already present in the query
- it is related to the query subject but still introduces a distinct new entity

Important:
- Be conservative. Only block when the overlap is clear.
- Do not block entities merely because they are associated with the query subject.
- Example: if the query mentions Lionel Messi, block "Messi" or "leomessi", but keep "Argentina national football team" unless the query already mentions it.

Output exactly one block:
<filter>
entity: candidate name | block|keep | short reason
entity: candidate name | block|keep | short reason
</filter>
"""


@dataclass(slots=True)
class ImageDiscoveryConfig:
    """Cheap gates and retrieval limits for image discovery."""

    per_query_limit: int = 10
    max_images_per_plan: int = 8
    persist_search_snapshots: bool = False
    min_width: int | None = 120
    min_height: int | None = 120
    allowed_content_types: set[str] | None = None
    rejected_extensions: set[str] = field(default_factory=lambda: {".svg"})
    store_rejected: bool = True
    force_accept_images: bool = False
    precheck_image_urls: bool = True
    precheck_timeout_s: float = 15.0
    precheck_max_bytes: int = 262144
    model_image_max_bytes: int | None = None
    model_image_max_edge: int | None = 1280
    precheck_retries: int = 3
    host_min_interval_s: float = 0.35
    wikimedia_host_min_interval_s: float = 1.25
    wikimedia_429_retry_after_s: float = 15.0
    user_agent: str | None = None
    cache_dir: str | None = None
    try_source_page_recovery: bool = True
    source_page_timeout_s: float = 20.0
    image_grounding_context_backend: str = "source_page_reader"
    image_grounding_reader_base_url: str = "http://127.0.0.1:8004"
    image_grounding_reader_timeout_s: float = 40.0
    image_grounding_max_context_chars: int = 6000
    expandable_entity_types: set[str] = field(
        default_factory=lambda: {
            "person",
            "team",
            "organization",
            "event",
            "movie",
            "book",
            "album",
            "brand",
            "product",
            "landmark",
            "document",
            "artwork",
        }
    )


@dataclass(slots=True)
class ImageValidationResult:
    """Result returned by the image_check function."""

    status: ImageCandidateStatus
    confidence: float | None = None
    reason: str | None = None
    drop_candidate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class ImageSearchCandidate:
    """One retrieved image candidate before/after validation."""

    candidate_id: str
    source_query: SearchQuerySpec
    source_snapshot: SearchSnapshot
    search_result: ImageSearchResult
    validation: ImageValidationResult
    used_fallback: bool = False
    is_primary: bool = False
    grounded_entities: list[dict[str, Any]] = field(default_factory=list)
    grounded_caption: str | None = None
    visual_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_query": self.source_query.to_dict(),
            "source_snapshot": self.source_snapshot.to_dict(),
            "search_result": self.search_result.to_dict(),
            "validation": self.validation.to_dict(),
            "used_fallback": self.used_fallback,
            "is_primary": self.is_primary,
            "grounded_entities": _jsonify(self.grounded_entities),
            "grounded_caption": self.grounded_caption,
            "visual_facts": list(self.visual_facts),
        }


@dataclass(slots=True)
class ResolvedImageAsset:
    cache_key: str
    original_url: str | None
    resolved_url: str | None
    source_page_url: str | None
    model_url: str
    asset_uri: str
    cache_path: str | None
    content_type: str | None
    width: int | None = None
    height: int | None = None
    strategy: str = "direct"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "original_url": self.original_url,
            "resolved_url": self.resolved_url,
            "source_page_url": self.source_page_url,
            "asset_uri": self.asset_uri,
            "cache_path": self.cache_path,
            "content_type": self.content_type,
            "width": self.width,
            "height": self.height,
            "strategy": self.strategy,
        }


@dataclass(slots=True)
class ImageDiscoveryResult:
    """All records produced for one visual search plan."""

    plan_id: str
    image_node: ImageNode | None = None
    edge: Edge | None = None
    image_evidence: Evidence | None = None
    search_evidence: Evidence | None = None
    grounded_edges: list[Edge] = field(default_factory=list)
    candidates: list[ImageSearchCandidate] = field(default_factory=list)
    queued_tasks: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[SearchSnapshot] = field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepted_images(self) -> list[ImageSearchCandidate]:
        return [
            image
            for image in self.candidates
            if image.validation.status == ImageCandidateStatus.ACCEPTED
        ]

    def usable_images(self) -> list[ImageSearchCandidate]:
        return [
            image
            for image in self.candidates
            if image.validation.status == ImageCandidateStatus.ACCEPTED
        ]

    def primary_image(self) -> ImageSearchCandidate | None:
        for image in self.candidates:
            if image.is_primary:
                return image
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "image_node": self.image_node.to_dict() if self.image_node else None,
            "edge": self.edge.to_dict() if self.edge else None,
            "image_evidence": self.image_evidence.to_dict() if self.image_evidence else None,
            "search_evidence": self.search_evidence.to_dict() if self.search_evidence else None,
            "grounded_edges": [edge.to_dict() for edge in self.grounded_edges],
            "candidates": [image.to_dict() for image in self.candidates],
            "queued_tasks": _jsonify(self.queued_tasks),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ImageGroundingContext:
    """Prompt-side context provider output for image grounding."""

    provider: str
    prompt_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class ImageDiscoveryBuilder:
    """Run image discovery for a visual target and persist graph records."""

    builder_name = "image_discovery_builder"

    def __init__(
        self,
        *,
        store: JsonlGraphStore | None = None,
        search_client: SearchClient,
        config: ImageDiscoveryConfig | None = None,
        model_client: ModelWorkerClient | None = None,
        image_check_model_alias: str | None = None,
        wiki_resolver: WikiEntityResolver | None = None,
    ) -> None:
        self.store = store
        self.search_client = search_client
        self.config = config or ImageDiscoveryConfig()
        self.model_client = model_client or LLM_WORKER
        self.image_check_model_alias = image_check_model_alias
        self.wiki_resolver = wiki_resolver or WikiEntityResolver()
        self.reader = EnhancedReaderClient(
            base_url=self.config.image_grounding_reader_base_url,
            timeout_s=self.config.image_grounding_reader_timeout_s,
        )
        self._resolved_image_cache: dict[str, ResolvedImageAsset] = {}
        self._grounding_context_cache: dict[str, ImageGroundingContext] = {}
        self._host_not_before: dict[str, float] = {}
        self._host_locks: dict[str, threading.Lock] = {}
        self._download_lock = threading.Lock()

    def discover_for_plan(
        self,
        plan: VisualSearchPlan,
        *,
        run_id: str | None = None,
        persist: bool = True,
    ) -> ImageDiscoveryResult:
        """Discover images for one visual plan."""

        total_started = time.perf_counter()
        result = ImageDiscoveryResult(plan_id=plan.plan_id)
        seen_keys: set[str] = set()
        decision_log: list[dict[str, Any]] = []
        _trace_timing(f"[image-discovery] phase=start plan_id={plan.plan_id} queries={len(plan.queries)}")

        started = time.perf_counter()
        result.candidates = self._discover_with_client(
            client=self.search_client,
            plan=plan,
            run_id=run_id,
            seen_keys=seen_keys,
            persist=persist,
            snapshots=result.snapshots,
            decision_log=decision_log,
        )
        _trace_timing(
            f"[image-discovery] stage=search_and_check plan_id={plan.plan_id} elapsed_s={time.perf_counter() - started:.3f} candidates={len(result.candidates)}"
        )
        result.candidates = result.candidates[: self.config.max_images_per_plan]
        result.fallback_used = any(candidate.used_fallback for candidate in result.candidates)
        primary_candidate = self._select_primary_candidate(result.candidates)
        if primary_candidate is not None:
            started = time.perf_counter()
            self._materialize_primary_candidate(
                result=result,
                plan=plan,
                candidate=primary_candidate,
                run_id=run_id,
                persist=persist,
            )
            _trace_timing(
                f"[image-discovery] stage=materialize_primary plan_id={plan.plan_id} elapsed_s={time.perf_counter() - started:.3f} primary_title={primary_candidate.search_result.title!r}"
            )
        result.metadata.update(
            {
                "query_count": len(plan.queries),
                "image_count": len(result.candidates),
                "usable_image_count": len(result.usable_images()),
                "accepted_image_count": len(result.accepted_images()),
                "queued_task_count": len(result.queued_tasks),
                "candidate_decisions": decision_log,
            }
        )
        if persist and self.store is not None:
            self.store.flush()
        _trace_timing(
            f"[image-discovery] phase=done plan_id={plan.plan_id} elapsed_s={time.perf_counter() - total_started:.3f} accepted={len(result.accepted_images())} kept={'yes' if result.image_node is not None else 'no'}"
        )
        return result

    def _discover_with_client(
        self,
        *,
        client: SearchClient,
        plan: VisualSearchPlan,
        run_id: str | None,
        seen_keys: set[str],
        persist: bool,
        snapshots: list[SearchSnapshot],
        decision_log: list[dict[str, Any]],
    ) -> list[ImageSearchCandidate]:
        discovered: list[ImageSearchCandidate] = []
        for query in plan.queries:
            try:
                started = time.perf_counter()
                response = client.search_image(query.query, limit=self.config.per_query_limit)
                _trace_timing(
                    f"[image-discovery] stage=search_query plan_id={plan.plan_id} query={query.query!r} elapsed_s={time.perf_counter() - started:.3f} returned={len(response.results)}"
                )
            except Exception as exc:
                snapshot = self._snapshot_from_error(
                    client=client,
                    query=query.query,
                    error=exc,
                    run_id=run_id,
                )
                snapshots.append(snapshot)
                if persist:
                    self._persist_snapshot(snapshot)
                decision_log.append(
                    {
                        "kind": "query_error",
                        "query": query.query,
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                continue

            snapshot = self._snapshot_from_response(response, run_id=run_id)
            snapshots.append(snapshot)
            if persist:
                self._persist_snapshot(snapshot)
            used_fallback = bool(response.metadata.get("fallback_used"))
            decision_log.append(
                {
                    "kind": "query_results",
                    "query": query.query,
                    "returned": len(response.results),
                    "fallback_used": used_fallback,
                }
            )

            for result_index, search_result in enumerate(response.results, start=1):
                if not isinstance(search_result, ImageSearchResult):
                    decision_log.append(
                        {
                            "kind": "candidate_skip",
                            "query": query.query,
                            "result_index": result_index,
                            "reason": "non_image_search_result",
                        }
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=None,
                        fate="skipped",
                        reason="non_image_search_result",
                    )
                    continue
                key = self._candidate_key(search_result)
                if not key or key in seen_keys:
                    decision_log.append(
                        self._candidate_decision_record(
                            kind="candidate_skip",
                            query=query.query,
                            search_result=search_result,
                            reason="missing_or_duplicate_candidate_key",
                            result_index=result_index,
                        )
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=search_result,
                        fate="skipped",
                        reason="missing_or_duplicate_candidate_key",
                    )
                    continue
                seen_keys.add(key)

                validation = self.image_check(
                    plan=plan,
                    query=query,
                    search_result=search_result,
                    run_id=run_id,
                )
                if validation.drop_candidate:
                    decision_log.append(
                        self._candidate_decision_record(
                            kind="candidate_drop",
                            query=query.query,
                            search_result=search_result,
                            reason=validation.reason or "drop_candidate",
                            result_index=result_index,
                            validation=validation,
                        )
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=search_result,
                        fate="dropped",
                        reason=validation.reason or "drop_candidate",
                        raw_model_output=(validation.metadata or {}).get("raw_model_output"),
                    )
                    continue
                if (
                    validation.status == ImageCandidateStatus.REJECTED
                    and not self.config.store_rejected
                ):
                    decision_log.append(
                        self._candidate_decision_record(
                            kind="candidate_skip",
                            query=query.query,
                            search_result=search_result,
                            reason=validation.reason or "rejected_not_stored",
                            result_index=result_index,
                            validation=validation,
                        )
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=search_result,
                        fate="skipped",
                        reason=validation.reason or "rejected_not_stored",
                        raw_model_output=(validation.metadata or {}).get("raw_model_output"),
                    )
                    continue

                discovered.append(
                    ImageSearchCandidate(
                        candidate_id=self._candidate_record_id(search_result),
                        source_query=query,
                        source_snapshot=snapshot,
                        search_result=search_result,
                        validation=validation,
                        used_fallback=used_fallback,
                    )
                )
                decision_log.append(
                    self._candidate_decision_record(
                        kind="candidate_kept",
                        query=query.query,
                        search_result=search_result,
                        reason=validation.reason or validation.status.value,
                        result_index=result_index,
                        status=validation.status.value,
                        bundle_count=len(discovered),
                        validation=validation,
                    )
                )
                self._log_image_result_fate(
                    plan_id=plan.plan_id,
                    query=query.query,
                    result_index=result_index,
                    search_result=search_result,
                    fate=(
                        "accepted"
                        if validation.status == ImageCandidateStatus.ACCEPTED
                        else "rejected"
                    ),
                    reason=validation.reason or validation.status.value,
                    raw_model_output=(validation.metadata or {}).get("raw_model_output"),
                )
                if len(discovered) >= self.config.max_images_per_plan:
                    decision_log.append(
                        {
                            "kind": "query_limit_reached",
                            "query": query.query,
                            "limit": self.config.max_images_per_plan,
                        }
                    )
                    return discovered
        return discovered

    @staticmethod
    def _candidate_decision_record(
        *,
        kind: str,
        query: str,
        search_result: ImageSearchResult,
        reason: str,
        result_index: int | None = None,
        status: str | None = None,
        bundle_count: int | None = None,
        validation: ImageValidationResult | None = None,
    ) -> dict[str, Any]:
        payload = {
            "kind": kind,
            "query": query,
            "rank": search_result.rank,
            "title": search_result.title,
            "url": search_result.image_url,
            "reason": reason,
        }
        if result_index is not None:
            payload["result_index"] = result_index
        if status is not None:
            payload["status"] = status
        if bundle_count is not None:
            payload["bundle_count"] = bundle_count
        if validation is not None:
            metadata = validation.metadata or {}
            if metadata.get("check") is not None:
                payload["check"] = metadata.get("check")
            if metadata.get("raw_model_output") is not None:
                payload["raw_model_output"] = metadata.get("raw_model_output")
            if metadata.get("visual_facts") is not None:
                payload["visual_facts"] = metadata.get("visual_facts")
        return payload

    @staticmethod
    def _candidate_record_id(search_result: ImageSearchResult) -> str:
        return ImageVariant.make_id(
            search_result.image_url,
            search_result.source_page_url,
            search_result.title,
        )

    @staticmethod
    def _select_primary_candidate(candidates: list[ImageSearchCandidate]) -> ImageSearchCandidate | None:
        accepted = [
            candidate
            for candidate in candidates
            if candidate.validation.status == ImageCandidateStatus.ACCEPTED
        ]
        if not accepted:
            return None
        accepted.sort(
            key=lambda candidate: (
                -(candidate.validation.confidence if candidate.validation.confidence is not None else 0.0),
                candidate.used_fallback,
                candidate.search_result.rank if candidate.search_result.rank is not None else 10**9,
            )
        )
        primary = accepted[0]
        primary.is_primary = True
        return primary

    def _materialize_primary_candidate(
        self,
        *,
        result: ImageDiscoveryResult,
        plan: VisualSearchPlan,
        candidate: ImageSearchCandidate,
        run_id: str | None,
        persist: bool,
    ) -> None:
        resolved_asset = self._resolved_image_from_validation(candidate.validation)
        provisional_node = self._image_node_from_result(
            candidate.search_result,
            run_id=run_id,
            resolved_asset=resolved_asset,
        )
        grounding = self.image_ground(
            plan=plan,
            search_result=candidate.search_result,
            image_node=provisional_node,
            validation=candidate.validation,
            run_id=run_id,
        )
        candidate.grounded_entities = list(grounding.get("grounded_entities", []))
        candidate.grounded_caption = grounding.get("caption")
        candidate.visual_facts = list(grounding.get("visual_facts", []))

        variants = [
            self._variant_from_candidate(
                item,
                is_primary=item.candidate_id == candidate.candidate_id,
            )
            for item in result.candidates
        ]
        source_node_title = self._source_node_title(plan.source_node_id) or plan.target.content
        primary_caption = candidate.grounded_caption or provisional_node.caption or candidate.search_result.snippet
        primary_image_uri = (
            resolved_asset.asset_uri
            if resolved_asset is not None
            else candidate.search_result.image_url or candidate.search_result.source_page_url or candidate.search_result.title or ""
        )
        image_node = ImageNode.from_bundle(
            primary_image_uri,
            primary_image_id=candidate.candidate_id,
            image_variants=variants,
            source_page_url=candidate.search_result.source_page_url,
            caption=primary_caption,
            title=candidate.search_result.title,
            width=resolved_asset.width if resolved_asset is not None and resolved_asset.width is not None else candidate.search_result.width,
            height=resolved_asset.height if resolved_asset is not None and resolved_asset.height is not None else candidate.search_result.height,
            content_type=resolved_asset.content_type if resolved_asset is not None else self._content_type(candidate.search_result),
            run_id=run_id,
            metadata={
                "search_query": candidate.source_query.query,
                "candidate_count": len(result.candidates),
                "visual_target": plan.target.content,
                "resolved_image": resolved_asset.to_metadata() if resolved_asset is not None else None,
            },
        )
        self._apply_grounding_to_image_node(image_node, grounding)

        original_asset = self._image_asset(
            candidate.search_result,
            image_node=image_node,
            resolved_asset=resolved_asset,
        )
        thumb_asset = self._thumbnail_asset(candidate.search_result)
        asset_ids = [original_asset.asset_id]
        if thumb_asset:
            asset_ids.append(thumb_asset.asset_id)

        search_evidence = Evidence.create(
            EvidenceType.SEARCH_RESULT,
            content=candidate.search_result.title or candidate.search_result.snippet,
            node_ids=[image_node.node_id],
            url=candidate.search_result.source_page_url or candidate.search_result.image_url,
            source_snapshot_id=candidate.source_snapshot.snapshot_id if self.config.persist_search_snapshots else None,
            extractor=self.builder_name,
            confidence=candidate.validation.confidence,
            metadata={
                "query_id": candidate.source_query.query_id,
                "query": candidate.source_query.query,
                "rank": candidate.search_result.rank,
                "engine": candidate.source_snapshot.engine.value,
                "snapshot_id": candidate.source_snapshot.snapshot_id,
                "used_fallback": candidate.used_fallback,
                "validation": candidate.validation.to_dict(),
            },
            evidence_key=f"{candidate.source_snapshot.snapshot_id}:{candidate.source_query.query_id}:{self._candidate_key(candidate.search_result)}",
        )
        image_evidence = Evidence.create(
            EvidenceType.IMAGE,
            content=primary_caption or candidate.search_result.title,
            node_ids=[image_node.node_id],
            asset_ids=asset_ids,
            url=candidate.search_result.image_url,
            source_snapshot_id=candidate.source_snapshot.snapshot_id if self.config.persist_search_snapshots else None,
            extractor=self.builder_name,
            confidence=candidate.validation.confidence,
            metadata={
                "source_page_url": candidate.search_result.source_page_url,
                "thumbnail_url": candidate.search_result.thumbnail_url,
                "snapshot_id": candidate.source_snapshot.snapshot_id,
                "query_id": candidate.source_query.query_id,
                "target_evidence_id": plan.target.evidence_id,
                "validation": candidate.validation.to_dict(),
                "primary_candidate_id": candidate.candidate_id,
            },
            evidence_key=f"image_bundle:{candidate.candidate_id}",
        )

        edge = self._edge_from_plan_to_image(
            plan=plan,
            query=candidate.source_query,
            image_node=image_node,
            search_evidence=search_evidence,
            image_evidence=image_evidence,
            search_result=candidate.search_result,
            run_id=run_id,
            used_fallback=candidate.used_fallback,
        )
        grounded_edges, queued_tasks = self._link_or_queue_grounded_entities(
            image_node=image_node,
            grounded_entities=candidate.grounded_entities,
            image_evidence=image_evidence,
            run_id=run_id,
            source_node_title=source_node_title,
            source_query_text=candidate.source_query.query,
        )

        if persist:
            self._persist_records(
                image_node=image_node,
                original_asset=original_asset,
                thumb_asset=thumb_asset,
                search_evidence=search_evidence,
                image_evidence=image_evidence,
                edge=edge,
                grounded_edges=grounded_edges,
            )

        result.image_node = image_node
        result.edge = edge
        result.image_evidence = image_evidence
        result.search_evidence = search_evidence
        result.grounded_edges = grounded_edges
        result.queued_tasks = queued_tasks

    @staticmethod
    def _variant_from_candidate(candidate: ImageSearchCandidate, *, is_primary: bool) -> ImageVariant:
        return ImageVariant(
            variant_id=candidate.candidate_id,
            image_url=candidate.search_result.image_url,
            source_page_url=candidate.search_result.source_page_url,
            thumbnail_url=candidate.search_result.thumbnail_url,
            title=candidate.search_result.title,
            search_caption=candidate.search_result.snippet,
            width=candidate.search_result.width,
            height=candidate.search_result.height,
            source=candidate.search_result.source,
            rank=candidate.search_result.rank,
            validation_status=candidate.validation.status.value,
            validation_confidence=candidate.validation.confidence,
            validation_reason=candidate.validation.reason,
            used_fallback=candidate.used_fallback,
            is_primary=is_primary,
            metadata={
                "query": candidate.source_query.query,
                "snapshot_id": candidate.source_snapshot.snapshot_id,
                "visual_facts": list(candidate.visual_facts),
                "resolved_image": (candidate.validation.metadata or {}).get("resolved_image"),
            },
        )

    def image_ground(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        image_node: ImageNode,
        validation: ImageValidationResult,
        run_id: str | None,
    ) -> dict[str, Any]:
        """Analyze an accepted image and ground unique visible entities."""

        model_alias = os.environ.get("IMAGE_GROUND_MODEL")
        if not model_alias:
            grounding = {
                "caption": image_node.caption,
                "grounded_entities": [],
                "check": "not_configured",
                "context": None,
            }
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

        resolved_asset = self._resolved_image_from_validation(validation)
        precheck_error: str | None = None
        if self.config.precheck_image_urls and resolved_asset is None:
            resolved_asset, precheck_error = self._resolve_image_asset(search_result)
        if self.config.precheck_image_urls and resolved_asset is None:
            precheck_error = precheck_error or "missing_resolved_image_asset"
            self._log_invalid_image_url(search_result.image_url, precheck_error, stage="image_ground")
            grounding = {
                "caption": image_node.caption,
                "grounded_entities": [],
                "check": "image_url_precheck_failed",
                "raw_model_output": None,
                "run_id": run_id,
                "context": None,
            }
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["image_ground_error"] = precheck_error
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

        if self.config.precheck_image_urls and resolved_asset is not None:
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["resolved_image"] = resolved_asset.to_metadata()

        grounding_context = self._build_image_grounding_context(search_result)
        image_node.metadata = dict(image_node.metadata or {})
        image_node.metadata["image_grounding_context"] = grounding_context.to_dict()
        image_node.metadata["image_grounding_prompt"] = {
            "system": PROMPT_IMAGE_GROUND,
            "user_text": grounding_context.prompt_text,
        }

        try:
            model_image_url = resolved_asset.model_url if resolved_asset is not None else search_result.image_url
            self._log_image_model_call(
                stage="image_ground",
                when="before",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=model_image_url,
            )
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_GROUND),
                        ModelMessage(
                            role="user",
                            content=[
                                {
                                    "type": "text",
                                    "text": grounding_context.prompt_text,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": model_image_url},
                                },
                            ],
                        ),
                    ],
                    temperature=0.0,
                    metadata={"trace_label": f"image_ground:{plan.plan_id}:{search_result.title or ''}"},
                )
            )
            self._log_image_model_call(
                stage="image_ground",
                when="after",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=model_image_url,
                model_output=response.content,
            )
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            self._log_invalid_image_url(search_result.image_url, error, stage="image_ground")
            grounding = {
                "caption": image_node.caption,
                "grounded_entities": [],
                "check": "mllm_grounding_failed",
                "raw_model_output": error,
                "run_id": run_id,
                "context": grounding_context.to_dict(),
                "debug_prompt_system": PROMPT_IMAGE_GROUND,
                "debug_prompt_user_text": grounding_context.prompt_text,
            }
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["image_ground_error"] = error
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

        grounding = self._parse_image_ground_response(
            response.content,
            run_id=run_id,
            model_alias=model_alias,
            usage=response.usage,
        )
        grounding["context"] = grounding_context.to_dict()
        grounding["debug_prompt_system"] = PROMPT_IMAGE_GROUND
        grounding["debug_prompt_user_text"] = grounding_context.prompt_text
        self._apply_grounding_to_image_node(image_node, grounding)
        return grounding

    def _build_image_grounding_context(self, search_result: ImageSearchResult) -> ImageGroundingContext:
        backend = (self.config.image_grounding_context_backend or "source_page_reader").strip().lower()
        cache_key = f"{backend}::{search_result.source_page_url or ''}::{search_result.title or ''}"
        cached = self._grounding_context_cache.get(cache_key)
        if cached is not None:
            return cached

        if backend == "source_page_reader":
            context = self._build_source_page_reader_grounding_context(search_result)
        elif backend == "title_only":
            context = self._build_title_only_grounding_context(search_result)
        else:
            context = self._build_title_only_grounding_context(
                search_result,
                fallback_reason=f"unsupported_backend:{backend}",
            )

        self._grounding_context_cache[cache_key] = context
        return context

    def _build_source_page_reader_grounding_context(self, search_result: ImageSearchResult) -> ImageGroundingContext:
        source_page_url = (search_result.source_page_url or "").strip()
        if not source_page_url:
            return self._build_title_only_grounding_context(search_result, fallback_reason="missing_source_page_url")

        try:
            document = self.reader.read(source_page_url)
        except Exception as exc:
            return self._build_title_only_grounding_context(
                search_result,
                fallback_reason=f"reader_error:{exc.__class__.__name__}",
            )

        page_title = (document.title or "").strip()
        page_content = self._trim_grounding_context_text(document.content)
        if not page_title and not page_content:
            return self._build_title_only_grounding_context(search_result, fallback_reason="reader_empty")

        prompt_parts = [
            "Webpage context for this image:",
            f"source_page_url: {source_page_url}",
            f"page_title: {page_title}",
            "",
            "Use this page context only to help identify entities that are actually visible in the image.",
            "If the page discusses entities not shown in the image, do not output them.",
            "",
            "Reader content:",
            page_content,
        ]
        return ImageGroundingContext(
            provider="source_page_reader",
            prompt_text="\n".join(part for part in prompt_parts if part is not None),
            metadata={
                "source_page_url": source_page_url,
                "page_title": page_title or None,
                "content_chars": len(page_content),
            },
        )

    def _build_title_only_grounding_context(
        self,
        search_result: ImageSearchResult,
        *,
        fallback_reason: str | None = None,
    ) -> ImageGroundingContext:
        title = (search_result.title or "").strip()
        if title:
            prompt_text = (
                "Fallback context for this image:\n"
                f"title: {title}\n\n"
                "Use the title only to disambiguate entities that are actually visible in the image.\n"
                "If the title is insufficient or conflicts with the image, trust the image and omit uncertain entities."
            )
        else:
            prompt_text = (
                "No external context is available for this image.\n"
                "Ground only entities that you can identify confidently from the image itself.\n"
                "If identity is uncertain, omit the entity."
            )
        return ImageGroundingContext(
            provider="title_only",
            prompt_text=prompt_text,
            metadata={
                "title": title or None,
                "fallback_reason": fallback_reason,
            },
        )

    def _trim_grounding_context_text(self, text: str | None) -> str:
        normalized = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", (text or "").strip()))
        if not normalized:
            return ""
        limit = max(256, int(self.config.image_grounding_max_context_chars))
        if len(normalized) <= limit:
            return normalized
        trimmed = normalized[:limit]
        last_break = max(trimmed.rfind("\n\n"), trimmed.rfind(". "))
        if last_break >= 256:
            trimmed = trimmed[: last_break + 1]
        return trimmed.rstrip()

    @staticmethod
    def _parse_image_ground_response(
        text: str,
        *,
        run_id: str | None,
        model_alias: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        match = re.search(r"<ground>(.*?)</ground>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        grounding: dict[str, Any] = {
            "caption": None,
            "grounded_entities": [],
            "raw_model_output": text,
            "run_id": run_id,
            "model_alias": model_alias,
            "usage": usage,
            "check": "mllm_grounding",
        }
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "caption":
                grounding["caption"] = value
            elif key == "entity":
                entity = ImageDiscoveryBuilder._parse_grounded_entity(value)
                if entity is not None:
                    grounding["grounded_entities"].append(entity)
        return grounding

    @staticmethod
    def _parse_grounded_entity(value: str) -> dict[str, Any] | None:
        parts = [part.strip() for part in value.split("|")]
        if not parts or not parts[0]:
            return None
        return {
            "name": parts[0],
            "relation_to_image": parts[1] if len(parts) > 1 and parts[1] else "depicted in image",
            "evidence": parts[2] if len(parts) > 2 else None,
        }

    @staticmethod
    def _apply_grounding_to_image_node(image_node: ImageNode, grounding: dict[str, Any]) -> None:
        caption = grounding.get("caption")
        if caption:
            image_node.caption = caption
            image_node.summary = caption
        image_node.metadata = dict(image_node.metadata or {})
        image_node.metadata["grounded_entities"] = grounding.get("grounded_entities", [])
        context = grounding.get("context") or image_node.metadata.get("image_grounding_context")
        if context is not None:
            image_node.metadata["image_grounding_context"] = context
        image_node.metadata["image_grounding"] = {
            "check": grounding.get("check"),
            "model_alias": grounding.get("model_alias"),
            "usage": grounding.get("usage"),
            "raw_model_output": grounding.get("raw_model_output"),
            "run_id": grounding.get("run_id"),
            "context": context,
            "debug_prompt_system": grounding.get("debug_prompt_system"),
            "debug_prompt_user_text": grounding.get("debug_prompt_user_text"),
        }

    def image_check(
        self,
        *,
        plan: VisualSearchPlan,
        query: SearchQuerySpec,
        search_result: ImageSearchResult,
        run_id: str | None,
    ) -> ImageValidationResult:
        """Check one candidate image.

        This single function owns both cheap deterministic gates and future MLLM
        semantic validation. Keeping them together makes the discovery flow only
        depend on one accept/reject decision.
        """

        if not search_result.image_url:
            return self._reject("missing_image_url")
        extension = self._extension(search_result.image_url)
        if extension and extension in self.config.rejected_extensions:
            return self._reject(f"rejected_extension:{extension}")
        if (
            self.config.min_width is not None
            and search_result.width is not None
            and search_result.width < self.config.min_width
        ):
            return self._reject(f"width_below_min:{search_result.width}")
        if (
            self.config.min_height is not None
            and search_result.height is not None
            and search_result.height < self.config.min_height
        ):
            return self._reject(f"height_below_min:{search_result.height}")
        content_type = self._content_type(search_result)
        if self.config.allowed_content_types and content_type:
            if content_type not in self.config.allowed_content_types:
                return self._reject(f"content_type_not_allowed:{content_type}")

        model_alias = self.image_check_model_alias or os.environ.get("IMAGE_CHECK_MODEL")
        resolved_asset: ResolvedImageAsset | None = None
        if self.config.precheck_image_urls:
            resolved_asset, precheck_error = self._resolve_image_asset(search_result)
            if precheck_error is not None or resolved_asset is None:
                self._log_invalid_image_url(search_result.image_url, precheck_error, stage="image_check")
                return self._reject(
                    f"image_url_precheck_failed:{precheck_error}",
                    drop_candidate=True,
                )

        if self.config.force_accept_images:
            metadata: dict[str, Any] = {
                "check": "force_accept_images",
                "debug_force_accept_images": True,
            }
            if resolved_asset is not None:
                metadata["resolved_image_key"] = resolved_asset.cache_key
                metadata["resolved_image"] = resolved_asset.to_metadata()
            return ImageValidationResult(
                status=ImageCandidateStatus.ACCEPTED,
                confidence=1.0,
                reason="force_accept_images",
                metadata=metadata,
            )

        if model_alias:
            try:
                result = self._image_check_with_mllm(
                    plan=plan,
                    search_result=search_result,
                    model_alias=model_alias,
                    run_id=run_id,
                    resolved_asset=resolved_asset,
                )
                if resolved_asset is not None:
                    result.metadata = dict(result.metadata or {})
                    result.metadata["resolved_image_key"] = resolved_asset.cache_key
                    result.metadata["resolved_image"] = resolved_asset.to_metadata()
                return result
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                self._log_invalid_image_url(search_result.image_url, error, stage="image_check")
                return self._reject(
                    f"image_check_model_error:{error}",
                    drop_candidate=True,
                )

        del query, run_id
        return ImageValidationResult(
            status=ImageCandidateStatus.ACCEPTED,
            confidence=None,
            metadata={"check": "basic_url_format_size"},
        )

    def _resolve_image_asset(
        self,
        search_result: ImageSearchResult,
    ) -> tuple[ResolvedImageAsset | None, str | None]:
        image_url = search_result.image_url
        source_page_url = search_result.source_page_url
        cache_key = self._resolved_image_cache_key(image_url, source_page_url)
        cached = self._resolved_image_cache.get(cache_key)
        if cached is not None:
            return cached, None

        attempted_errors: list[str] = []
        direct_asset, direct_error = self._download_and_prepare_image_asset(
            image_url,
            source_page_url=source_page_url,
            strategy="direct",
            cache_key=cache_key,
        )
        if direct_asset is not None:
            self._resolved_image_cache[cache_key] = direct_asset
            return direct_asset, None
        if direct_error:
            attempted_errors.append(direct_error)

        if self.config.try_source_page_recovery and source_page_url:
            for recovered_url in self._recover_candidate_image_urls(search_result):
                recovered_asset, recovered_error = self._download_and_prepare_image_asset(
                    recovered_url,
                    source_page_url=source_page_url,
                    strategy="source_page_recovery",
                    cache_key=cache_key,
                )
                if recovered_asset is not None:
                    self._resolved_image_cache[cache_key] = recovered_asset
                    self._log_recovered_image_url(
                        original_url=image_url,
                        recovered_url=recovered_url,
                        source_page_url=source_page_url,
                    )
                    return recovered_asset, None
                if recovered_error:
                    attempted_errors.append(recovered_error)

        return None, " | ".join(attempted_errors) if attempted_errors else "unresolved_image_asset"

    def _download_and_prepare_image_asset(
        self,
        image_url: str | None,
        *,
        source_page_url: str | None,
        strategy: str,
        cache_key: str,
    ) -> tuple[ResolvedImageAsset | None, str | None]:
        if not image_url:
            return None, "missing_image_url"

        download_result = self._download_image_payload(
            image_url,
            max_bytes=self.config.model_image_max_bytes,
        )
        if isinstance(download_result, str):
            return None, f"{image_url} -> {download_result}"
        payload, content_type = download_result
        if not payload:
            return None, f"{image_url} -> empty_response_body"

        normalized_content_type = (content_type or "").lower()
        sniffed_content_type = self._sniff_content_type(payload)
        if normalized_content_type and not normalized_content_type.startswith("image/"):
            if not sniffed_content_type:
                return None, f"{image_url} -> non_image_content_type:{content_type}"

        try:
            from PIL import Image

            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                image.verify()
        except ImportError:
            width = None
            height = None
        except Exception as exc:
            return None, f"{image_url} -> decode_error:{exc.__class__.__name__}:{exc}"

        content_type = (
            sniffed_content_type
            if normalized_content_type == "application/octet-stream" and sniffed_content_type
            else content_type or sniffed_content_type or "image/jpeg"
        )
        cache_path = self._write_image_cache_file(cache_key, payload, content_type)
        asset_uri = self._maybe_upload_cached_image(cache_path, cache_key) or cache_path
        model_content_type, model_payload = self._prepare_model_payload(
            payload=payload,
            content_type=content_type,
            max_edge=self.config.model_image_max_edge,
        )
        model_url = self._data_url(model_content_type, model_payload)
        return (
            ResolvedImageAsset(
                cache_key=cache_key,
                original_url=image_url,
                resolved_url=image_url,
                source_page_url=source_page_url,
                model_url=model_url,
                asset_uri=asset_uri,
                cache_path=cache_path,
                content_type=content_type,
                width=width,
                height=height,
                strategy=strategy,
            ),
            None,
        )

    def _download_image_payload(
        self,
        image_url: str,
        *,
        max_bytes: int | None,
    ) -> tuple[bytes, str | None] | str:
        host = urlparse(image_url).netloc.lower()
        last_error: str | None = None
        for attempt in range(1, max(1, self.config.precheck_retries) + 1):
            host_lock = self._host_lock(host)
            with host_lock:
                self._wait_for_host_slot(host)
                request = Request(
                    image_url,
                    headers={
                        "Accept": "image/*,*/*;q=0.8",
                        "User-Agent": self._user_agent(),
                    },
                )
                started_at = time.perf_counter()
                try:
                    with urlopen(request, timeout=self.config.precheck_timeout_s) as response:
                        content_type = response.headers.get("Content-Type", "")
                        payload = response.read() if not max_bytes or max_bytes <= 0 else response.read(max_bytes)
                    elapsed_s = time.perf_counter() - started_at
                    self._mark_host_slot(host, success=True)
                    self._log_image_download(
                        image_url=image_url,
                        byte_count=len(payload),
                        elapsed_s=elapsed_s,
                        content_type=content_type,
                        attempt=attempt,
                        max_bytes=max_bytes,
                    )
                    return payload, content_type
                except HTTPError as exc:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"http_{exc.code}",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    retry_after = self._retry_after_seconds(exc)
                    if exc.code == 429 and attempt < self.config.precheck_retries:
                        backoff_s = retry_after or self._default_retry_after_seconds(host, attempt)
                        self._mark_host_slot(host, retry_after=backoff_s)
                        last_error = "http_429"
                        continue
                    return f"http_{exc.code}"
                except URLError as exc:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"url_error:{exc.reason}",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    last_error = f"url_error:{exc.reason}"
                except TimeoutError:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"timeout_after_{self.config.precheck_timeout_s}s",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    last_error = f"timeout_after_{self.config.precheck_timeout_s}s"
                except Exception as exc:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"download_error:{exc.__class__.__name__}:{exc}",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    last_error = f"download_error:{exc.__class__.__name__}:{exc}"
            if attempt < self.config.precheck_retries:
                time.sleep(min(6.0, attempt * 1.5))
        return last_error or "download_failed"

    def _recover_candidate_image_urls(self, search_result: ImageSearchResult) -> list[str]:
        source_page_url = search_result.source_page_url
        if not source_page_url:
            return []
        html_text = self._fetch_source_page_html(source_page_url)
        if html_text is None:
            return []

        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'"display_url"\s*:\s*"([^"]+)"',
            r'"image_url"\s*:\s*"([^"]+)"',
        ]
        candidates: list[str] = []
        seen = set()
        for pattern in patterns:
            for match in re.findall(pattern, html_text, flags=re.IGNORECASE):
                candidate = html.unescape(match).replace("\\u0026", "&").replace("\\/", "/").strip()
                if not candidate.startswith(("http://", "https://")) or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
        return candidates

    def _fetch_source_page_html(self, source_page_url: str) -> str | None:
        request = Request(
            source_page_url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "User-Agent": self._user_agent(),
            },
        )
        try:
            with urlopen(request, timeout=self.config.source_page_timeout_s) as response:
                payload = response.read(min(self.config.precheck_max_bytes * 4, 1048576))
            return payload.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _resolved_image_from_validation(
        self,
        validation: ImageValidationResult,
    ) -> ResolvedImageAsset | None:
        key = (validation.metadata or {}).get("resolved_image_key")
        if not key:
            return None
        return self._resolved_image_cache.get(key)

    @staticmethod
    def _resolved_image_cache_key(image_url: str | None, source_page_url: str | None) -> str:
        payload = f"{image_url or ''}||{source_page_url or ''}"
        return sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _write_image_cache_file(self, cache_key: str, payload: bytes, content_type: str) -> str:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{cache_key}{self._suffix_for_content_type(content_type)}"
        if not path.exists():
            path.write_bytes(payload)
        return str(path.resolve())

    def _cache_dir(self) -> Path:
        configured = self.config.cache_dir
        if configured:
            return Path(configured)
        return Path(__file__).resolve().parent / ".image_cache"

    @staticmethod
    def _suffix_for_content_type(content_type: str | None) -> str:
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/avif": ".avif",
        }
        return mapping.get((content_type or "").lower(), ".img")

    @staticmethod
    def _data_url(content_type: str, payload: bytes) -> str:
        return f"data:{content_type};base64,{base64.b64encode(payload).decode('ascii')}"

    @staticmethod
    def _sniff_content_type(payload: bytes) -> str | None:
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if payload.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if payload[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
            return "image/webp"
        return None

    @staticmethod
    def _prepare_model_payload(
        *,
        payload: bytes,
        content_type: str,
        max_edge: int | None,
    ) -> tuple[str, bytes]:
        if not max_edge or max_edge <= 0:
            return content_type, payload
        try:
            from PIL import Image
        except ImportError:
            return content_type, payload
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                if max(width, height) <= max_edge:
                    return content_type, payload
                resized = image.copy()
                resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                has_alpha = resized.mode in ("RGBA", "LA") or (
                    resized.mode == "P" and "transparency" in resized.info
                )
                output = BytesIO()
                if has_alpha:
                    resized.save(output, format="PNG", optimize=True)
                    return "image/png", output.getvalue()
                resized = resized.convert("RGB")
                resized.save(output, format="JPEG", quality=90, optimize=True)
                return "image/jpeg", output.getvalue()
        except Exception:
            return content_type, payload

    def _maybe_upload_cached_image(self, cache_path: str, cache_key: str) -> str | None:
        try:
            from PIL import Image
            from opensearch_vl.opensearch_infer import cos_upload
        except Exception:
            return None
        if not cos_upload.upload_available():
            return None
        try:
            with Image.open(cache_path) as pil_image:
                pil_copy = pil_image.copy()
        except Exception:
            return None
        return cos_upload.upload_pil_image(
            pil_copy,
            filename_prefix="synthesis",
            case_idx=0,
            turn_num=0,
            tool_name=f"image_cache_{cache_key}",
        )

    def _wait_for_host_slot(self, host: str) -> None:
        if not host:
            return
        with self._download_lock:
            not_before = self._host_not_before.get(host, 0.0)
        now = time.time()
        if not_before > now:
            time.sleep(not_before - now)

    def _mark_host_slot(self, host: str, retry_after: float | None = None, *, success: bool = False) -> None:
        if not host:
            return
        with self._download_lock:
            min_interval_s = self._host_min_interval_seconds(host)
            delay = min_interval_s if success else max(
                min_interval_s,
                retry_after or min_interval_s,
            )
            self._host_not_before[host] = time.time() + delay

    def _host_lock(self, host: str) -> threading.Lock:
        with self._download_lock:
            lock = self._host_locks.get(host)
            if lock is None:
                lock = threading.Lock()
                self._host_locks[host] = lock
            return lock

    def _host_min_interval_seconds(self, host: str) -> float:
        if self._is_wikimedia_host(host):
            return self.config.wikimedia_host_min_interval_s
        return self.config.host_min_interval_s

    def _default_retry_after_seconds(self, host: str, attempt: int) -> float:
        if self._is_wikimedia_host(host):
            return max(self.config.wikimedia_429_retry_after_s, float(attempt) * self.config.wikimedia_429_retry_after_s)
        return float(attempt) * 2.0

    @staticmethod
    def _is_wikimedia_host(host: str) -> bool:
        normalized = (host or "").lower()
        return normalized.endswith((".wikimedia.org", ".wikipedia.org", ".mediawiki.org"))

    def _user_agent(self) -> str:
        configured = (
            self.config.user_agent
            or os.environ.get("SYNTHESIS_USER_AGENT")
            or os.environ.get("WIKIMEDIA_USER_AGENT")
        )
        if configured:
            return configured
        return "DeepSearchBot/0.1 (https://github.com/shawn0728/OpenSearch-VL; automated research image fetcher)"

    @staticmethod
    def _retry_after_seconds(error: HTTPError) -> float | None:
        value = error.headers.get("Retry-After") if error.headers else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    @staticmethod
    def _log_invalid_image_url(image_url: str | None, reason: str, *, stage: str) -> None:
        return

    @staticmethod
    def _log_image_download(
        *,
        image_url: str,
        byte_count: int,
        elapsed_s: float,
        content_type: str | None,
        attempt: int,
        max_bytes: int | None,
    ) -> None:
        return

    @staticmethod
    def _log_image_download_failure(
        *,
        image_url: str,
        reason: str,
        elapsed_s: float,
        attempt: int,
    ) -> None:
        return

    @staticmethod
    def _format_byte_count(size: int) -> str:
        value = float(max(0, size))
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0 or unit == "GB":
                return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
            value /= 1024.0
        return f"{int(value)}B"

    @staticmethod
    def _log_recovered_image_url(
        *,
        original_url: str | None,
        recovered_url: str,
        source_page_url: str | None,
    ) -> None:
        return

    @staticmethod
    def _log_image_result_fate(
        *,
        plan_id: str,
        query: str,
        result_index: int | None,
        search_result: ImageSearchResult | None,
        fate: str,
        reason: str,
        raw_model_output: str | None = None,
    ) -> None:
        return

    def _image_check_with_mllm(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        model_alias: str,
        run_id: str | None,
        resolved_asset: ResolvedImageAsset | None = None,
    ) -> ImageValidationResult:
        if not search_result.image_url:
            return self._reject("missing_image_url_for_mllm_check")
        image_for_model = resolved_asset.model_url if resolved_asset is not None else search_result.image_url
        self._log_image_model_call(
            stage="image_check",
            when="before",
            model_alias=model_alias,
            plan_id=plan.plan_id,
            search_result=search_result,
            model_image_url=image_for_model,
        )
        response = self.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_IMAGE_CHECK),
                    ModelMessage(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": self._image_check_prompt_input(
                                    plan=plan,
                                    search_result=search_result,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_for_model},
                            },
                        ],
                    ),
                ],
                temperature=0.0,
                metadata={"trace_label": f"image_check:{plan.plan_id}:{search_result.title or ''}"},
            )
        )
        self._log_image_model_call(
            stage="image_check",
            when="after",
            model_alias=model_alias,
            plan_id=plan.plan_id,
            search_result=search_result,
            model_image_url=image_for_model,
            model_output=response.content,
        )
        return self._parse_image_check_response(
            response.content,
            run_id=run_id,
            model_alias=model_alias,
            usage=response.usage,
        )

    @staticmethod
    def _log_image_model_call(
        *,
        stage: str,
        when: str,
        model_alias: str,
        plan_id: str,
        search_result: ImageSearchResult,
        model_image_url: str | None,
        model_output: str | None = None,
    ) -> None:
        return

    @staticmethod
    def _image_check_prompt_input(
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
    ) -> str:
        return (
            f"Target:\n{plan.target.content or ''}\n\n"
            "Candidate metadata:\n"
            f"title: {search_result.title or ''}\n"
            f"caption/snippet: {search_result.snippet or ''}\n"
            f"source_page_url: {search_result.source_page_url or ''}\n"
        )

    @staticmethod
    def _parse_image_check_response(
        text: str,
        *,
        run_id: str | None,
        model_alias: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> ImageValidationResult:
        match = re.search(r"<check>(.*?)</check>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        fields: dict[str, Any] = {"visual_facts": []}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "visual_fact":
                fields["visual_facts"].append(value)
            else:
                fields[key] = value

        decision = str(fields.get("decision", "")).lower()
        status = (
            ImageCandidateStatus.ACCEPTED
            if decision == "accept"
            else ImageCandidateStatus.REJECTED
        )
        confidence = ImageDiscoveryBuilder._parse_confidence(fields.get("confidence"))
        return ImageValidationResult(
            status=status,
            confidence=confidence,
            reason=fields.get("reason"),
            metadata={
                "check": "mllm_semantic",
                "model_alias": model_alias,
                "usage": usage,
                "visual_facts": fields.get("visual_facts", []),
                "raw_model_output": text,
                "run_id": run_id,
            },
        )

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _reject(reason: str, *, drop_candidate: bool = False) -> ImageValidationResult:
        return ImageValidationResult(
            status=ImageCandidateStatus.REJECTED,
            confidence=0.0,
            reason=reason,
            drop_candidate=drop_candidate,
        )

    @staticmethod
    def _candidate_key(result: ImageSearchResult) -> str | None:
        return result.image_url or result.source_page_url or result.title

    @staticmethod
    def _extension(url: str | None) -> str | None:
        if not url:
            return None
        path = url.split("?", 1)[0].split("#", 1)[0].lower()
        if "." not in path:
            return None
        return "." + path.rsplit(".", 1)[-1]

    @staticmethod
    def _content_type(result: ImageSearchResult) -> str | None:
        imageinfo = result.raw.get("imageinfo") if result.raw else None
        if not isinstance(imageinfo, list) or not imageinfo:
            return None
        first = imageinfo[0]
        if not isinstance(first, dict):
            return None
        mime = first.get("mime")
        return mime if isinstance(mime, str) else None

    @staticmethod
    def _snapshot_engine(response: SearchResponse) -> SearchEngine:
        engine = response.engine.lower()
        if "commons" in engine:
            return SearchEngine.WIKIMEDIA_COMMONS
        if "serpapi" in engine and "image" in engine:
            return SearchEngine.SERPAPI_IMAGE
        if "serpapi" in engine:
            return SearchEngine.SERPAPI_TEXT
        if "serper" in engine or "image" in engine:
            return SearchEngine.SERPER_IMAGE
        return SearchEngine.OTHER

    def _snapshot_from_response(
        self,
        response: SearchResponse,
        *,
        run_id: str | None,
    ) -> SearchSnapshot:
        return SearchSnapshot.create(
            self._snapshot_engine(response),
            query=response.query,
            request={"query": response.query, "engine": response.engine},
            response_preview=self._response_preview(response),
            result_count=len(response.results),
            status_code=response.status_code,
            run_id=run_id,
            metadata={
                "raw_engine": response.engine,
                "response_metadata": response.metadata,
            },
        )

    def _snapshot_from_error(
        self,
        *,
        client: SearchClient,
        query: str,
        error: Exception,
        run_id: str | None,
    ) -> SearchSnapshot:
        return SearchSnapshot.create(
            self._engine_from_client(client),
            query=query,
            request={
                "query": query,
                "client": client.__class__.__name__,
                "limit": self.config.per_query_limit,
            },
            result_count=0,
            error=f"{error.__class__.__name__}: {error}",
            run_id=run_id,
            status=RecordStatus.FAILED,
        )

    @staticmethod
    def _engine_from_client(client: SearchClient) -> SearchEngine:
        name = client.__class__.__name__.lower()
        if "commons" in name:
            return SearchEngine.WIKIMEDIA_COMMONS
        if "serpapi" in name:
            return SearchEngine.SERPAPI_IMAGE
        if "serper" in name:
            return SearchEngine.SERPER_IMAGE
        return SearchEngine.OTHER

    @staticmethod
    def _response_preview(response: SearchResponse, *, limit: int = 5) -> str:
        preview = [item.to_dict() for item in response.results[:limit]]
        return repr(preview)

    @staticmethod
    def _image_node_from_result(
        result: ImageSearchResult,
        *,
        run_id: str | None,
        resolved_asset: ResolvedImageAsset | None = None,
    ) -> ImageNode:
        metadata = {
            "search_source": result.source,
            "thumbnail_url": result.thumbnail_url,
            "rank": result.rank,
            "raw": result.raw,
        }
        if resolved_asset is not None:
            metadata["resolved_image"] = resolved_asset.to_metadata()
        return ImageNode.from_url(
            (
                resolved_asset.asset_uri
                if resolved_asset is not None
                else result.image_url or result.source_page_url or result.title or ""
            ),
            source_page_url=result.source_page_url,
            caption=result.snippet,
            title=result.title,
            run_id=run_id,
            metadata=metadata,
        )

    @staticmethod
    def _image_asset(
        result: ImageSearchResult,
        *,
        image_node: ImageNode,
        resolved_asset: ResolvedImageAsset | None = None,
    ) -> Asset:
        uri = (
            resolved_asset.asset_uri
            if resolved_asset is not None
            else result.image_url or image_node.image_url or image_node.node_id
        )
        return Asset.create(
            AssetType.IMAGE_ORIGINAL,
            uri,
            original_url=result.image_url,
            content_type=resolved_asset.content_type if resolved_asset is not None else ImageDiscoveryBuilder._content_type(result),
            metadata={
                "source_page_url": result.source_page_url,
                "width": result.width,
                "height": result.height,
                "storage_status": image_node.storage_status,
                "resolved_url": resolved_asset.resolved_url if resolved_asset is not None else None,
                "cache_path": resolved_asset.cache_path if resolved_asset is not None else None,
                "resolution_strategy": resolved_asset.strategy if resolved_asset is not None else None,
            },
        )

    @staticmethod
    def _thumbnail_asset(result: ImageSearchResult) -> Asset | None:
        if not result.thumbnail_url:
            return None
        return Asset.create(
            AssetType.IMAGE_THUMBNAIL,
            result.thumbnail_url,
            original_url=result.thumbnail_url,
            metadata={
                "source_page_url": result.source_page_url,
                "original_image_url": result.image_url,
            },
        )

    def _link_or_queue_grounded_entities(
        self,
        *,
        image_node: ImageNode,
        grounded_entities: list[dict[str, Any]],
        image_evidence: Evidence,
        run_id: str | None,
        source_node_title: str | None,
        source_query_text: str | None,
    ) -> tuple[list[Edge], list[dict[str, Any]]]:
        if self.store is None or not grounded_entities:
            return [], []

        edges: list[Edge] = []
        unresolved: list[dict[str, Any]] = []
        queued_tasks: list[dict[str, Any]] = []
        blocked_query_entities = self._query_implied_entity_labels(
            source_query_text,
            source_node_title=source_node_title,
            grounded_entities=grounded_entities,
        )
        for entity in grounded_entities:
            if not self._should_expand_entity(entity):
                unresolved.append({**entity, "status": "filtered_out"})
                continue
            if self._is_query_implied_entity(entity, blocked_query_entities):
                unresolved.append({**entity, "status": "filtered_by_query_entity_overlap"})
                continue
            matched_node = self._match_text_node(entity.get("name"))
            if matched_node is None:
                resolved_target = self._resolve_grounded_entity(
                    entity,
                    source_node_title=source_node_title,
                    image_caption=image_node.caption,
                )
                if resolved_target is None:
                    unresolved.append({**entity, "status": "unresolved"})
                    continue
                existing_by_url = self._find_text_node_by_url(resolved_target["url"])
                if existing_by_url is not None:
                    matched_node = existing_by_url
                else:
                    queued_tasks.append(
                        {
                            "url": resolved_target["url"],
                            "title": resolved_target.get("title") or entity.get("name"),
                            "pending_link": {
                                "link_type": "image_entity",
                                "parent_node_id": image_node.node_id,
                                "source_evidence_id": image_evidence.evidence_id,
                                "entity": entity,
                                "resolved_target": resolved_target,
                            },
                        }
                    )
                    continue
            relation = entity.get("relation_to_image") or "depicts"
            edge = Edge.create(
                image_node.node_id,
                matched_node["node_id"],
                edge_type=EdgeType.IMAGE_DEPICTS,
                relation=relation,
                src_node_type=NodeType.IMAGE.value,
                dst_node_type=NodeType.TEXT.value,
                evidence_refs=[
                    EvidenceRef(
                        evidence_id=image_evidence.evidence_id,
                        quote=entity.get("evidence"),
                        metadata={
                            "grounded_entity": entity,
                            "matched_title": matched_node.get("title"),
                        },
                    )
                ],
                source=EdgeSource(
                    source_type="image_grounding",
                    url=image_node.image_url,
                    run_id=run_id,
                    builder=self.builder_name,
                ),
                extractor=self.builder_name,
                metadata={
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                    "match_method": matched_node.get("_match_method"),
                },
                evidence_key=f"{image_evidence.evidence_id}:{entity.get('name')}:{matched_node['node_id']}",
            )
            edges.append(edge)

        if unresolved:
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["unresolved_grounded_entities"] = unresolved
        return edges, queued_tasks

    def _query_implied_entity_labels(
        self,
        query_text: str | None,
        *,
        source_node_title: str | None = None,
        grounded_entities: list[dict[str, Any]] | None = None,
    ) -> set[str]:
        if not query_text:
            return set()
        blocked_from_llm = self._query_implied_entity_labels_with_llm(
            query_text,
            source_node_title=source_node_title,
            grounded_entities=grounded_entities or [],
        )
        if blocked_from_llm:
            return blocked_from_llm
        if self.store is None:
            return set()
        normalized_query = self._normalize_entity_label(query_text)
        if not normalized_query:
            return set()

        blocked: set[str] = set()
        query_tokens = set(normalized_query.split())
        source_title_label = self._normalize_entity_label(source_node_title or "")
        if source_title_label and (
            source_title_label == normalized_query
            or source_title_label in normalized_query
            or set(source_title_label.split()).issubset(query_tokens)
        ):
            blocked.add(source_title_label)
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            labels = [node.get("title") or "", *(node.get("aliases") or [])]
            for label in labels:
                normalized_label = self._normalize_entity_label(label)
                if not normalized_label or len(normalized_label) < 4:
                    continue
                label_tokens = set(normalized_label.split())
                if not label_tokens:
                    continue
                if (
                    normalized_label == normalized_query
                    or normalized_label in normalized_query
                    or label_tokens.issubset(query_tokens)
                    or (len(label_tokens) == 1 and next(iter(label_tokens)) in query_tokens)
                ):
                    blocked.add(normalized_label)
        return blocked

    def _query_implied_entity_labels_with_llm(
        self,
        query_text: str,
        *,
        source_node_title: str | None,
        grounded_entities: list[dict[str, Any]],
    ) -> set[str]:
        if not grounded_entities:
            return set()
        model_alias = (
            os.environ.get("IMAGE_QUERY_ENTITY_FILTER_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
            or self.image_check_model_alias
        )
        if not model_alias:
            return set()
        candidate_names = []
        seen: set[str] = set()
        for entity in grounded_entities:
            label = (entity.get("name") or "").strip()
            normalized = self._normalize_entity_label(label)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidate_names.append(label)
        if not candidate_names:
            return set()
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_QUERY_ENTITY_FILTER),
                        ModelMessage(
                            role="user",
                            content=(
                                f"Source text node title:\n{source_node_title or ''}\n\n"
                                f"Visual query text:\n{query_text}\n\n"
                                "Grounded candidate entities:\n"
                                + "\n".join(f"- {name}" for name in candidate_names)
                            ),
                        ),
                    ],
                    temperature=0.0,
                    metadata={"trace_label": f"image_query_entity_filter:{source_node_title or ''}:{query_text[:80]}"},
                )
            )
        except Exception:
            return set()
        return self._parse_query_entity_filter_response(response.content)

    def _parse_query_entity_filter_response(self, text: str) -> set[str]:
        match = re.search(r"<filter>(.*?)</filter>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        blocked: set[str] = set()
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line.lower().startswith("entity:"):
                continue
            value = line.split(":", 1)[1].strip()
            parts = [part.strip() for part in value.split("|")]
            if len(parts) < 2:
                continue
            name = parts[0]
            decision = parts[1].lower()
            if decision == "block":
                normalized = self._normalize_entity_label(name)
                if normalized:
                    blocked.add(normalized)
        return blocked

    def _is_query_implied_entity(self, entity: dict[str, Any], blocked_query_entities: set[str]) -> bool:
        label = self._normalize_entity_label(entity.get("name") or "")
        if not label:
            return False
        return label in blocked_query_entities

    def _resolve_grounded_entity(
        self,
        entity: dict[str, Any],
        *,
        source_node_title: str | None,
        image_caption: str | None,
    ) -> dict[str, Any] | None:
        label = (entity.get("name") or "").strip()
        if not label:
            return None
        context_parts = [part for part in (entity.get("evidence"), image_caption, source_node_title) if part]
        resolved = self.wiki_resolver.resolve(
            label,
            entity_type=entity.get("type"),
            source_title=source_node_title,
            context=" ".join(context_parts),
        )
        if resolved is None:
            return None
        return resolved.to_dict()

    def _find_text_node_by_url(self, url: str | None) -> dict[str, Any] | None:
        if self.store is None or not url:
            return None
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            source = node.get("source") or {}
            if isinstance(source, dict) and source.get("url") == url:
                return dict(node)
        return None

    def _source_node_title(self, node_id: str | None) -> str | None:
        if self.store is None or not node_id:
            return None
        record = self.store.get_node(node_id)
        if record is None:
            return None
        return record.get("title") or record.get("canonical_id")

    def _should_expand_entity(self, entity: dict[str, Any]) -> bool:
        label = (entity.get("name") or "").strip()
        if len(label) < 2:
            return False
        entity_type = self._normalize_entity_type(entity.get("type"))
        if entity_type and entity_type not in self.config.expandable_entity_types:
            return False
        return True

    def _match_text_node(self, label: str | None) -> dict[str, Any] | None:
        if self.store is None or not label:
            return None
        needle = self._normalize_entity_label(label)
        if not needle:
            return None

        exact_matches: list[tuple[dict[str, Any], str]] = []
        contains_matches: list[tuple[dict[str, Any], str]] = []
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            title = node.get("title") or ""
            aliases = node.get("aliases") or []
            labels = [title, *aliases]
            normalized_labels = [self._normalize_entity_label(item) for item in labels if item]
            if needle in normalized_labels:
                exact_matches.append((node, "exact_or_alias"))
                continue
            for normalized_label in normalized_labels:
                if self._is_unique_contains_match(needle, normalized_label):
                    contains_matches.append((node, "unique_contains"))
                    break

        if len(exact_matches) == 1:
            node, method = exact_matches[0]
            matched = dict(node)
            matched["_match_method"] = method
            return matched
        if len(contains_matches) == 1:
            node, method = contains_matches[0]
            matched = dict(node)
            matched["_match_method"] = method
            return matched
        return None

    @staticmethod
    def _normalize_entity_label(label: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", label).lower()).strip()

    @staticmethod
    def _is_unique_contains_match(needle: str, candidate: str) -> bool:
        if not needle or not candidate or needle == candidate:
            return False
        if len(needle) < 4:
            return False
        needle_tokens = set(needle.split())
        candidate_tokens = set(candidate.split())
        return needle_tokens.issubset(candidate_tokens)

    @staticmethod
    def _normalize_entity_type(entity_type: str | None) -> str | None:
        normalized = re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", (entity_type or "").lower())).strip()
        return normalized or None

    def _edge_from_plan_to_image(
        self,
        *,
        plan: VisualSearchPlan,
        query: SearchQuerySpec,
        image_node: ImageNode,
        search_evidence: Evidence,
        image_evidence: Evidence,
        search_result: ImageSearchResult,
        run_id: str | None,
        used_fallback: bool,
    ) -> Edge | None:
        if not plan.source_node_id:
            return None
        return Edge.create(
            plan.source_node_id,
            image_node.node_id,
            edge_type=EdgeType.SEARCH_RETRIEVED,
            relation=query.query or "retrieved_image_for_visual_target",
            src_node_type=NodeType.TEXT.value,
            dst_node_type=NodeType.IMAGE.value,
            evidence_refs=[
                EvidenceRef(evidence_id=plan.target.evidence_id),
                EvidenceRef(evidence_id=search_evidence.evidence_id),
                EvidenceRef(evidence_id=image_evidence.evidence_id),
            ],
            source=EdgeSource(
                source_type="image_search",
                url=search_result.source_page_url or search_result.image_url,
                run_id=run_id,
                builder=self.builder_name,
            ),
            extractor=self.builder_name,
            metadata={
                "query_id": query.query_id,
                "query": query.query,
                "used_fallback": used_fallback,
            },
            evidence_key=f"{query.query_id}:{image_node.node_id}",
        )

    def _persist_snapshot(self, snapshot: SearchSnapshot) -> None:
        if self.store is not None and self.config.persist_search_snapshots:
            self.store.upsert_search_snapshot(snapshot)

    def _persist_records(
        self,
        *,
        image_node: ImageNode,
        original_asset: Asset,
        thumb_asset: Asset | None,
        search_evidence: Evidence,
        image_evidence: Evidence,
        edge: Edge | None,
        grounded_edges: list[Edge] | None = None,
    ) -> None:
        if self.store is None:
            return
        self.store.upsert_node(image_node)
        self.store.upsert_asset(original_asset)
        if thumb_asset is not None:
            self.store.upsert_asset(thumb_asset)
        self.store.upsert_evidence(search_evidence)
        self.store.upsert_evidence(image_evidence)
        if edge is not None:
            self.store.upsert_edge(edge)
        for grounded_edge in grounded_edges or []:
            self.store.upsert_edge(grounded_edge)


def _smoke_test() -> None:
    import os
    import tempfile

    class MockImageSearchClient:
        def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
            del limit, kwargs
            return SearchResponse(query=query, engine="mock:text", results=[])

        def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
            del limit, kwargs
            return SearchResponse(
                query=query,
                engine="mock:image",
                results=[
                    ImageSearchResult(
                        title="Kobe Bryant final game",
                        image_url="https://example.com/kobe-final-game.jpg",
                        source_page_url="https://example.com/kobe",
                        snippet="Kobe Bryant in final game uniform",
                        width=640,
                        height=480,
                    )
                ],
            )

    class MockModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            system = request.messages[0].content
            if "checking whether a candidate image" in system:
                return ModelResponse(
                    content="""<check>
decision: accept
confidence: 0.9
reason: visible player in uniform
visual_fact: Kobe Bryant is visible
</check>"""
                )
            return ModelResponse(
                content="""<ground>
caption: Kobe Bryant in his final game
visual_fact: basketball uniform
entity: Los Angeles Lakers | jersey logo | visible team branding on the uniform
</ground>"""
            )

    old_check = os.environ.get("IMAGE_CHECK_MODEL")
    old_ground = os.environ.get("IMAGE_GROUND_MODEL")
    os.environ["IMAGE_CHECK_MODEL"] = "mock_image"
    os.environ["IMAGE_GROUND_MODEL"] = "mock_image"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir)
            text_node = TextNode.from_wiki_entity(
                "Q25369",
                "Kobe Bryant",
                aliases=["Kobe"],
                source_url="https://en.wikipedia.org/wiki/Kobe_Bryant",
            )
            lakers_node = TextNode.from_wiki_entity(
                "Q121783",
                "Los Angeles Lakers",
                aliases=["Lakers"],
                source_url="https://en.wikipedia.org/wiki/Los_Angeles_Lakers",
            )
            store.upsert_node(text_node)
            store.upsert_node(lakers_node)
            target = Evidence.create(
                EvidenceType.VISUAL_TARGET,
                content="Kobe Bryant final game uniform",
                node_ids=[text_node.node_id],
                metadata={"expected_visual": "Kobe Bryant in a Lakers uniform"},
            )
            query = SearchQuerySpec.create(
                "Kobe Bryant final game uniform photo",
                target.evidence_id,
                expected_visual="Kobe Bryant in a Lakers uniform",
            )
            plan = VisualSearchPlan.create(
                target,
                queries=[query],
                source_node_id=text_node.node_id,
                source_evidence_ids=["evidence_text"],
            )
            builder = ImageDiscoveryBuilder(
                store=store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    precheck_image_urls=False,
                ),
                model_client=MockModel(),
            )
            result = builder.discover_for_plan(plan, run_id="run_smoke")
            assert len(result.accepted_images()) == 1
            image = result.primary_image()
            assert image is not None
            assert result.image_node is not None
            assert result.image_node.caption == "Kobe Bryant in his final game"
            assert result.edge is not None
            assert result.grounded_edges
            assert result.image_node.metadata.get("image_grounding", {}).get("context") is not None
            assert store.stats()["nodes"] == 3
    finally:
        if old_check is None:
            os.environ.pop("IMAGE_CHECK_MODEL", None)
        else:
            os.environ["IMAGE_CHECK_MODEL"] = old_check
        if old_ground is None:
            os.environ.pop("IMAGE_GROUND_MODEL", None)
        else:
            os.environ["IMAGE_GROUND_MODEL"] = old_ground
    print("image_discovery smoke test passed")


if __name__ == "__main__":
    _smoke_test()
