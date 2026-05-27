"""Image discovery strategy layer for visual search plans.

This module sits above the low-level image search clients. It runs one or more
text-to-image queries, records search traces, applies cheap candidate filters,
creates graph records, and leaves one image_check hook for future MLLM checks.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import Any

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
from .nodes import ImageNode, NodeType, TextNode
from .search_client import ImageSearchResult, SearchClient, SearchResponse
from .store import JsonlGraphStore
from .visual_planner import SearchQuerySpec, VisualSearchPlan


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

Use candidate metadata only to disambiguate what is visible. Do not invent entities that are not visually supported.

Output exactly one block:
<ground>
caption: one concise image caption
visual_fact: visible fact 1
visual_fact: visible fact 2
ocr_text: visible text if any
entity: name | type | relation_to_image | evidence
entity: name | type | relation_to_image | evidence
</ground>
"""


@dataclass(slots=True)
class ImageDiscoveryConfig:
    """Cheap gates and retrieval limits for image discovery."""

    per_query_limit: int = 10
    max_images_per_plan: int = 8
    min_width: int | None = 120
    min_height: int | None = 120
    allowed_content_types: set[str] | None = None
    rejected_extensions: set[str] = field(default_factory=lambda: {".svg"})
    fallback_if_no_validated: bool = True
    store_rejected: bool = False


@dataclass(slots=True)
class ImageValidationResult:
    """Result returned by the image_check function."""

    status: ImageCandidateStatus
    confidence: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class DiscoveredImage:
    """Graph records created from a single retrieved image."""

    image_node: ImageNode
    image_evidence: Evidence
    search_evidence: Evidence
    edge: Edge | None
    source_query: SearchQuerySpec
    source_snapshot: SearchSnapshot
    search_result: ImageSearchResult
    validation: ImageValidationResult
    grounded_edges: list[Edge] = field(default_factory=list)
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_node": self.image_node.to_dict(),
            "image_evidence": self.image_evidence.to_dict(),
            "search_evidence": self.search_evidence.to_dict(),
            "edge": self.edge.to_dict() if self.edge else None,
            "grounded_edges": [edge.to_dict() for edge in self.grounded_edges],
            "source_query": self.source_query.to_dict(),
            "source_snapshot": self.source_snapshot.to_dict(),
            "search_result": self.search_result.to_dict(),
            "validation": self.validation.to_dict(),
            "used_fallback": self.used_fallback,
        }


@dataclass(slots=True)
class ImageDiscoveryResult:
    """All records produced for one visual search plan."""

    plan_id: str
    images: list[DiscoveredImage] = field(default_factory=list)
    snapshots: list[SearchSnapshot] = field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepted_images(self) -> list[DiscoveredImage]:
        return [
            image
            for image in self.images
            if image.validation.status == ImageCandidateStatus.ACCEPTED
        ]

    def usable_images(self) -> list[DiscoveredImage]:
        return [
            image
            for image in self.images
            if image.validation.status == ImageCandidateStatus.ACCEPTED
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "images": [image.to_dict() for image in self.images],
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
        }


class ImageDiscoveryBuilder:
    """Run image discovery for a visual target and persist graph records."""

    builder_name = "image_discovery_builder"

    def __init__(
        self,
        *,
        store: JsonlGraphStore | None = None,
        commons_client: SearchClient,
        fallback_client: SearchClient | None = None,
        config: ImageDiscoveryConfig | None = None,
        model_client: ModelWorkerClient | None = None,
        image_check_model_alias: str | None = None,
    ) -> None:
        self.store = store
        self.commons_client = commons_client
        self.fallback_client = fallback_client
        self.config = config or ImageDiscoveryConfig()
        self.model_client = model_client or LLM_WORKER
        self.image_check_model_alias = image_check_model_alias

    def discover_for_plan(
        self,
        plan: VisualSearchPlan,
        *,
        run_id: str | None = None,
        persist: bool = True,
    ) -> ImageDiscoveryResult:
        """Discover images for one visual plan."""

        result = ImageDiscoveryResult(plan_id=plan.plan_id)
        seen_keys: set[str] = set()

        primary_images = self._discover_with_client(
            client=self.commons_client,
            plan=plan,
            run_id=run_id,
            used_fallback=False,
            seen_keys=seen_keys,
            persist=persist,
            snapshots=result.snapshots,
        )
        result.images.extend(primary_images)

        should_fallback = self._should_fallback(primary_images)
        if should_fallback and self.fallback_client is not None:
            fallback_images = self._discover_with_client(
                client=self.fallback_client,
                plan=plan,
                run_id=run_id,
                used_fallback=True,
                seen_keys=seen_keys,
                persist=persist,
                snapshots=result.snapshots,
            )
            result.images.extend(fallback_images)
            result.fallback_used = bool(fallback_images)

        result.images = result.images[: self.config.max_images_per_plan]
        result.metadata.update(
            {
                "query_count": len(plan.queries),
                "image_count": len(result.images),
                "usable_image_count": len(result.usable_images()),
                "accepted_image_count": len(result.accepted_images()),
            }
        )
        if persist and self.store is not None:
            self.store.flush()
        return result

    def _discover_with_client(
        self,
        *,
        client: SearchClient,
        plan: VisualSearchPlan,
        run_id: str | None,
        used_fallback: bool,
        seen_keys: set[str],
        persist: bool,
        snapshots: list[SearchSnapshot],
    ) -> list[DiscoveredImage]:
        discovered: list[DiscoveredImage] = []
        for query in plan.queries:
            try:
                response = client.search_image(query.query, limit=self.config.per_query_limit)
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
                continue

            snapshot = self._snapshot_from_response(response, run_id=run_id)
            snapshots.append(snapshot)
            if persist:
                self._persist_snapshot(snapshot)

            for search_result in response.results:
                if not isinstance(search_result, ImageSearchResult):
                    continue
                key = self._candidate_key(search_result)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)

                validation = self.image_check(
                    plan=plan,
                    query=query,
                    search_result=search_result,
                    run_id=run_id,
                )
                if (
                    validation.status == ImageCandidateStatus.REJECTED
                    and not self.config.store_rejected
                ):
                    continue

                discovered.append(
                    self._build_discovered_image(
                        plan=plan,
                        query=query,
                        search_result=search_result,
                        snapshot=snapshot,
                        validation=validation,
                        run_id=run_id,
                        used_fallback=used_fallback,
                        persist=persist,
                    )
                )
                if len(discovered) >= self.config.max_images_per_plan:
                    return discovered
        return discovered

    def _build_discovered_image(
        self,
        *,
        plan: VisualSearchPlan,
        query: SearchQuerySpec,
        search_result: ImageSearchResult,
        snapshot: SearchSnapshot,
        validation: ImageValidationResult,
        run_id: str | None,
        used_fallback: bool,
        persist: bool,
    ) -> DiscoveredImage:
        image_node = self._image_node_from_result(search_result, run_id=run_id)
        grounding = self.image_ground(
            plan=plan,
            search_result=search_result,
            image_node=image_node,
            validation=validation,
            run_id=run_id,
        )
        original_asset = self._image_asset(search_result, image_node=image_node)
        thumb_asset = self._thumbnail_asset(search_result)
        asset_ids = [original_asset.asset_id]
        if thumb_asset:
            asset_ids.append(thumb_asset.asset_id)

        search_evidence = Evidence.create(
            EvidenceType.SEARCH_RESULT,
            content=search_result.title or search_result.snippet,
            node_ids=[image_node.node_id],
            url=search_result.source_page_url or search_result.image_url,
            source_snapshot_id=snapshot.snapshot_id,
            extractor=self.builder_name,
            confidence=validation.confidence,
            metadata={
                "query_id": query.query_id,
                "query": query.query,
                "rank": search_result.rank,
                "engine": snapshot.engine.value,
                "used_fallback": used_fallback,
                "validation": validation.to_dict(),
            },
            evidence_key=f"{snapshot.snapshot_id}:{query.query_id}:{self._candidate_key(search_result)}",
        )
        image_evidence = Evidence.create(
            EvidenceType.IMAGE,
            content=search_result.snippet or search_result.title,
            node_ids=[image_node.node_id],
            asset_ids=asset_ids,
            url=search_result.image_url,
            source_snapshot_id=snapshot.snapshot_id,
            extractor=self.builder_name,
            confidence=validation.confidence,
            metadata={
                "source_page_url": search_result.source_page_url,
                "thumbnail_url": search_result.thumbnail_url,
                "query_id": query.query_id,
                "target_evidence_id": plan.target.evidence_id,
                "validation": validation.to_dict(),
            },
            evidence_key=f"image:{self._candidate_key(search_result)}",
        )

        edge = self._edge_from_plan_to_image(
            plan=plan,
            query=query,
            image_node=image_node,
            search_evidence=search_evidence,
            image_evidence=image_evidence,
            search_result=search_result,
            run_id=run_id,
            used_fallback=used_fallback,
        )
        grounded_edges = self._link_grounded_entities(
            image_node=image_node,
            grounded_entities=grounding.get("grounded_entities", []),
            image_evidence=image_evidence,
            run_id=run_id,
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

        return DiscoveredImage(
            image_node=image_node,
            image_evidence=image_evidence,
            search_evidence=search_evidence,
            edge=edge,
            source_query=query,
            source_snapshot=snapshot,
            search_result=search_result,
            validation=validation,
            grounded_edges=grounded_edges,
            used_fallback=used_fallback,
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
                "visual_facts": validation.metadata.get("visual_facts", []),
                "ocr_texts": [],
                "grounded_entities": [],
                "check": "not_configured",
            }
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

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
                                "text": self._image_ground_prompt_input(
                                    plan=plan,
                                    search_result=search_result,
                                    validation=validation,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": search_result.image_url},
                            },
                        ],
                    ),
                ],
                temperature=0.0,
            )
        )
        grounding = self._parse_image_ground_response(response.content, run_id=run_id)
        self._apply_grounding_to_image_node(image_node, grounding)
        return grounding

    @staticmethod
    def _image_ground_prompt_input(
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        validation: ImageValidationResult,
    ) -> str:
        return (
            f"Target:\n{plan.target.content or ''}\n\n"
            "Candidate metadata:\n"
            f"title: {search_result.title or ''}\n"
            f"caption/snippet: {search_result.snippet or ''}\n"
            f"source_page_url: {search_result.source_page_url or ''}\n\n"
            "Prior image_check:\n"
            f"reason: {validation.reason or ''}\n"
            f"visual_facts: {validation.metadata.get('visual_facts', [])}\n"
        )

    @staticmethod
    def _parse_image_ground_response(text: str, *, run_id: str | None) -> dict[str, Any]:
        match = re.search(r"<ground>(.*?)</ground>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        grounding: dict[str, Any] = {
            "caption": None,
            "visual_facts": [],
            "ocr_texts": [],
            "grounded_entities": [],
            "raw_model_output": text,
            "run_id": run_id,
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
            elif key == "visual_fact":
                grounding["visual_facts"].append(value)
            elif key == "ocr_text":
                grounding["ocr_texts"].append(value)
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
            "type": parts[1] if len(parts) > 1 else None,
            "relation_to_image": parts[2] if len(parts) > 2 else "depicts",
            "evidence": parts[3] if len(parts) > 3 else None,
        }

    @staticmethod
    def _apply_grounding_to_image_node(image_node: ImageNode, grounding: dict[str, Any]) -> None:
        caption = grounding.get("caption")
        if caption:
            image_node.caption = caption
            image_node.summary = caption
        image_node.metadata = dict(image_node.metadata or {})
        image_node.metadata["visual_facts"] = grounding.get("visual_facts", [])
        image_node.metadata["ocr_texts"] = grounding.get("ocr_texts", [])
        image_node.metadata["grounded_entities"] = grounding.get("grounded_entities", [])
        image_node.metadata["image_grounding"] = {
            "check": grounding.get("check"),
            "raw_model_output": grounding.get("raw_model_output"),
            "run_id": grounding.get("run_id"),
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
        if model_alias:
            return self._image_check_with_mllm(
                plan=plan,
                search_result=search_result,
                model_alias=model_alias,
                run_id=run_id,
            )

        del query, run_id
        return ImageValidationResult(
            status=ImageCandidateStatus.ACCEPTED,
            confidence=None,
            metadata={"check": "basic_url_format_size"},
        )

    def _image_check_with_mllm(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        model_alias: str,
        run_id: str | None,
    ) -> ImageValidationResult:
        if not search_result.image_url:
            return self._reject("missing_image_url_for_mllm_check")
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
                                "image_url": {"url": search_result.image_url},
                            },
                        ],
                    ),
                ],
                temperature=0.0,
            )
        )
        return self._parse_image_check_response(response.content, run_id=run_id)

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
    def _parse_image_check_response(text: str, *, run_id: str | None) -> ImageValidationResult:
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

    def _should_fallback(self, images: list[DiscoveredImage]) -> bool:
        if not self.config.fallback_if_no_validated:
            return False
        if not images:
            return True
        return not any(
            image.validation.status == ImageCandidateStatus.ACCEPTED
            for image in images
        )

    @staticmethod
    def _reject(reason: str) -> ImageValidationResult:
        return ImageValidationResult(
            status=ImageCandidateStatus.REJECTED,
            confidence=0.0,
            reason=reason,
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
    ) -> ImageNode:
        metadata = {
            "search_source": result.source,
            "thumbnail_url": result.thumbnail_url,
            "rank": result.rank,
            "raw": result.raw,
        }
        return ImageNode.from_url(
            result.image_url or result.source_page_url or result.title or "",
            source_page_url=result.source_page_url,
            caption=result.snippet,
            title=result.title,
            run_id=run_id,
            metadata=metadata,
        )

    @staticmethod
    def _image_asset(result: ImageSearchResult, *, image_node: ImageNode) -> Asset:
        uri = result.image_url or image_node.image_url or image_node.node_id
        return Asset.create(
            AssetType.IMAGE_ORIGINAL,
            uri,
            original_url=result.image_url,
            content_type=ImageDiscoveryBuilder._content_type(result),
            metadata={
                "source_page_url": result.source_page_url,
                "width": result.width,
                "height": result.height,
                "storage_status": image_node.storage_status,
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

    def _link_grounded_entities(
        self,
        *,
        image_node: ImageNode,
        grounded_entities: list[dict[str, Any]],
        image_evidence: Evidence,
        run_id: str | None,
    ) -> list[Edge]:
        if self.store is None or not grounded_entities:
            return []

        edges: list[Edge] = []
        unresolved: list[dict[str, Any]] = []
        for entity in grounded_entities:
            matched_node = self._match_text_node(entity.get("name"))
            if matched_node is None:
                unresolved.append(entity)
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
        return edges

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
            relation="retrieved_image_for_visual_target",
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
        if self.store is not None:
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
entity: Kobe | person | depicts | visible player
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
            store.upsert_node(text_node)
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
                commons_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(per_query_limit=1, max_images_per_plan=1),
                model_client=MockModel(),
            )
            result = builder.discover_for_plan(plan, run_id="run_smoke")
            assert len(result.accepted_images()) == 1
            image = result.accepted_images()[0]
            assert image.image_node.caption == "Kobe Bryant in his final game"
            assert image.edge is not None
            assert image.grounded_edges
            assert store.stats()["nodes"] == 2
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
