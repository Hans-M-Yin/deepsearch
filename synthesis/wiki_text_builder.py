"""Wikipedia text-node construction and neighbor extraction."""

from __future__ import annotations

import re
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .edges import Edge, EdgeSource, EdgeType, EvidenceRef
from .evidence import Evidence, EvidenceType, RecordStatus, SearchEngine, SearchSnapshot
from .model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelResponse, ModelWorkerClient
from .nodes import AssetRef, NodeSource, NodeStatus, NodeType, TextNode
from .store import JsonlGraphStore


WIKI_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
WIKI_PAGE_PREFIXES_TO_SKIP = (
    "File:",
    "Category:",
    "Help:",
    "Special:",
    "Template:",
    "Talk:",
    "Wikipedia:",
    "Portal:",
)
GENERIC_WIKI_TITLES = {
    "United States",
    "English language",
    "IMDb",
    "ISBN",
    "Wayback Machine",
    "WorldCat",
}
LOW_VALUE_ANCHORS = {
    "he",
    "she",
    "it",
    "they",
    "this",
    "that",
    "here",
    "there",
    "website",
    "official website",
    "archive",
    "source",
}
REFERENCE_CONTEXT_MARKERS = (
    "references",
    "external links",
    "further reading",
    "isbn",
    "doi:",
    "retrieved",
    "archived",
)
RELATION_CONTEXT_HINTS = (
    "played for",
    "member of",
    "born in",
    "died in",
    "located in",
    "based in",
    "founded",
    "directed",
    "written by",
    "starring",
    "produced by",
    "created by",
    "part of",
    "known for",
    "won",
    "awarded",
    "married",
)


PROMPT_EXTRACT_ATTRIBUTE = """You are extracting attributes for a Wikipedia text node.

Task:
Extract clear, explicit attributes about the subject. Attributes can describe
internal facts, external/visual characteristics, roles, identity, achievements,
relationships, dates, locations, aliases, distinctive features, and major
events connected to the subject. Do not miss signature or iconic traits.

Rules:
- Use only information supported by the provided node content.
- Prefer attributes that are useful for multi-hop search/question construction.
- Include major events and event-specific facts when present.
- Use concise keys.
- Values should be short but specific.
- Do not output explanations, markdown, JSON, bullet lists, or extra text.

Output format:
<attr>key: value</attr><attr>key: value</attr><attr>key: value</attr>

Example:
<attr>Job: basketball player</attr><attr>Team: Los Angeles Lakers</attr><attr>Score in the last game of his life: 60 points</attr>
"""


PROMPT_EXTRACT_RELATION = """You are extracting a semantic relation between two Wikipedia text nodes.

Given a source entity, target entity, anchor text, and local hyperlink context,
infer the short predicate connecting the source to the target.

Rules:
- Use only the local context.
- Predicate must be concise snake_case.
- If the relation is unclear, use related_to.
- Keep direction as source_to_target unless the local context clearly says the target acts on the source.
- Do not output explanations or markdown.

Output exactly:
<relation>
predicate: played_for
direction: source_to_target
confidence: 0.0
evidence: short quote from context
</relation>
"""


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(slots=True)
class ReaderDocument:
    """Content returned by a webpage reader."""

    url: str
    title: str | None = None
    content: str = ""
    raw_markdown: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class ReaderClient(Protocol):
    """Minimal webpage reader interface."""

    def read(self, url: str, **kwargs: Any) -> ReaderDocument:
        """Read a webpage into main-content text or markdown."""


class EnhancedReaderClient:
    """HTTP client for the local enhanced reader service."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8004",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def read(self, url: str, **kwargs: Any) -> ReaderDocument:
        del kwargs
        target = url if url.startswith(("http://", "https://")) else f"https://{url}"
        request = Request(
            f"{self.base_url}/{target}",
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        document = data.get("data") or {}
        return ReaderDocument(
            url=document.get("url") or target,
            title=document.get("title") or None,
            content=document.get("content") or "",
            raw_markdown=document.get("raw_markdown") or None,
            raw=data,
        )


@dataclass(slots=True)
class WikiLinkCandidate:
    """A neighboring Wikipedia page discovered from the current page."""

    title: str
    url: str
    anchor_text: str
    source_url: str
    context: str | None = None
    rank: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    window_id: int | None = None
    score: float = 0.0
    quality_reasons: list[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return TextNode.make_id("wikipedia_page", self.url)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class WikiTextBuildResult:
    """Text node plus outgoing Wikipedia neighbors extracted from one page."""

    node: TextNode
    text_evidence: Evidence
    snapshot: SearchSnapshot
    linked_entities: list[WikiLinkCandidate] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "text_evidence": self.text_evidence.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "linked_entities": [entity.to_dict() for entity in self.linked_entities],
            "edges": [edge.to_dict() for edge in self.edges],
            "from_cache": self.from_cache,
        }


class WikiTextBuilder:
    """Build a text node from a Wikipedia page and extract hyperlink neighbors."""

    builder_name = "wiki_text_builder"

    def __init__(
        self,
        *,
        reader: ReaderClient,
        store: JsonlGraphStore | None = None,
        model_client: ModelWorkerClient | None = None,
        max_links: int = 80,
        max_raw_links: int | None = None,
        persist_snapshots: bool = False,
        diversity_window_size: int = 1200,
        max_links_per_window: int = 2,
        min_link_char_distance: int = 500,
        lead_chars: int = 3000,
        lead_max_links_per_window: int = 4,
    ) -> None:
        self.reader = reader
        self.store = store
        self.model_client = model_client or LLM_WORKER
        self.max_links = max_links
        self.max_raw_links = max_raw_links or max(max_links * 5, max_links)
        self.persist_snapshots = persist_snapshots
        self.diversity_window_size = diversity_window_size
        self.max_links_per_window = max_links_per_window
        self.min_link_char_distance = min_link_char_distance
        self.lead_chars = lead_chars
        self.lead_max_links_per_window = lead_max_links_per_window

    def build_from_url(
        self,
        url: str,
        *,
        title: str | None = None,
        run_id: str | None = None,
        persist: bool = True,
        force: bool = False,
    ) -> WikiTextBuildResult:
        input_url = self._normalize_wikipedia_url(url)
        cached = None if force else self._cached_build_result(input_url)
        if cached is not None:
            return cached

        document = self.reader.read(url)
        page_url = self._normalize_wikipedia_url(document.url or url)
        cached = None if force else self._cached_build_result(page_url)
        if cached is not None:
            return cached
        page_title = title or document.title or self._title_from_url(page_url)
        link_markdown = document.raw_markdown or document.content

        snapshot = SearchSnapshot.create(
            SearchEngine.JINA_READER,
            query=page_url,
            request={"url": page_url, "reader": self.reader.__class__.__name__},
            response_preview=document.content[:2000],
            result_count=1 if document.content else 0,
            status_code=200,
            run_id=run_id,
            metadata={"raw": document.raw},
        )
        node = TextNode(
            # TODO: alise is not implemented yet.
            node_id=TextNode.make_id("wikipedia_page", page_url),
            subtype="wiki_page",
            canonical_id=f"wikipedia:{page_title}" if page_title else None,
            title=page_title,
            summary=self._first_paragraph(document.content),
            description=document.content,
            source=NodeSource(
                source_type="wikipedia",
                url=page_url,
                source_id=page_title,
                run_id=run_id,
                builder=self.builder_name,
            ),
            metadata={
                "source_url": page_url,
                "reader": self.reader.__class__.__name__,
            },
        )
        text_evidence = Evidence.create(
            EvidenceType.WEB_TEXT,
            content=document.content,
            node_ids=[node.node_id],
            url=page_url,
            source_snapshot_id=snapshot.snapshot_id if self.persist_snapshots else None,
            extractor=self.builder_name,
            evidence_key=f"wiki_text:{page_url}",
        )

        linked_entities = self.extract_wiki_links(link_markdown, source_url=page_url)
        edges = [
            self._edge_to_linked_entity(node, candidate, text_evidence, run_id=run_id)
            for candidate in linked_entities
        ]

        result = WikiTextBuildResult(
            node=node,
            text_evidence=text_evidence,
            snapshot=snapshot,
            linked_entities=linked_entities,
            edges=edges,
        )
        if persist:
            self._persist_result(result)
        return result

    def _cached_build_result(self, page_url: str) -> WikiTextBuildResult | None:
        if self.store is None:
            return None
        node_id = TextNode.make_id("wikipedia_page", page_url)
        node_record = self.store.get_node(node_id)
        if node_record is None:
            return None

        evidence_record = self._find_text_evidence(node_id, page_url)
        snapshot_record = None
        if evidence_record and evidence_record.get("source_snapshot_id"):
            snapshot_record = self.store.get_search_snapshot(evidence_record["source_snapshot_id"])
        if evidence_record is None:
            evidence_record = self._placeholder_text_evidence(node_record, page_url)
        if snapshot_record is None:
            snapshot_record = self._placeholder_snapshot(page_url)

        return WikiTextBuildResult(
            node=self._text_node_from_record(node_record),
            text_evidence=self._evidence_from_record(evidence_record),
            snapshot=self._snapshot_from_record(snapshot_record),
            edges=[],
            linked_entities=[],
            from_cache=True,
        )

    def _find_text_evidence(self, node_id: str, page_url: str) -> dict[str, Any] | None:
        if self.store is None:
            return None
        for evidence in self.store.list_evidence():
            if evidence.get("evidence_type") != EvidenceType.WEB_TEXT.value:
                continue
            if node_id in evidence.get("node_ids", []):
                return evidence
            if evidence.get("url") == page_url:
                return evidence
        return None

    @staticmethod
    def _text_node_from_record(record: dict[str, Any]) -> TextNode:
        source = record.get("source")
        asset_refs = record.get("asset_refs") or []
        return TextNode(
            node_id=record["node_id"],
            title=record.get("title"),
            summary=record.get("summary"),
            source=NodeSource(**source) if isinstance(source, dict) else None,
            asset_refs=[
                AssetRef(**asset_ref)
                for asset_ref in asset_refs
                if isinstance(asset_ref, dict)
            ],
            metadata=dict(record.get("metadata") or {}),
            status=NodeStatus(record.get("status", NodeStatus.ACTIVE.value)),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
            subtype=record.get("subtype", "wiki_page"),
            canonical_id=record.get("canonical_id"),
            description=record.get("description"),
            aliases=list(record.get("aliases") or []),
            attributes=dict(record.get("attributes") or {}),
        )

    @staticmethod
    def _evidence_from_record(record: dict[str, Any]) -> Evidence:
        return Evidence(
            evidence_id=record["evidence_id"],
            evidence_type=EvidenceType(record["evidence_type"]),
            content=record.get("content"),
            node_ids=list(record.get("node_ids") or []),
            asset_ids=list(record.get("asset_ids") or []),
            url=record.get("url"),
            span=tuple(record["span"]) if record.get("span") else None,
            bbox=tuple(record["bbox"]) if record.get("bbox") else None,
            source_snapshot_id=record.get("source_snapshot_id"),
            extractor=record.get("extractor"),
            confidence=record.get("confidence"),
            metadata=dict(record.get("metadata") or {}),
            status=RecordStatus(record.get("status", RecordStatus.ACTIVE.value)),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
        )

    @staticmethod
    def _snapshot_from_record(record: dict[str, Any]) -> SearchSnapshot:
        return SearchSnapshot(
            snapshot_id=record["snapshot_id"],
            engine=SearchEngine(record["engine"]),
            query=record.get("query"),
            request=dict(record.get("request") or {}),
            response_ref=record.get("response_ref"),
            response_preview=record.get("response_preview"),
            result_count=record.get("result_count"),
            status_code=record.get("status_code"),
            error=record.get("error"),
            run_id=record.get("run_id"),
            metadata=dict(record.get("metadata") or {}),
            status=RecordStatus(record.get("status", RecordStatus.ACTIVE.value)),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
        )

    @staticmethod
    def _placeholder_text_evidence(node_record: dict[str, Any], page_url: str) -> dict[str, Any]:
        return Evidence.create(
            EvidenceType.WEB_TEXT,
            content=node_record.get("description"),
            node_ids=[node_record["node_id"]],
            url=page_url,
            extractor=WikiTextBuilder.builder_name,
            evidence_key=f"wiki_text:{page_url}:cached_placeholder",
        ).to_dict()

    @staticmethod
    def _placeholder_snapshot(page_url: str) -> dict[str, Any]:
        return SearchSnapshot.create(
            SearchEngine.JINA_READER,
            query=page_url,
            request={"url": page_url, "cache": True},
            result_count=1,
            metadata={"cache_placeholder": True},
        ).to_dict()

    def extract_attributes(
        self,
        node: TextNode,
        *,
        source_evidence_ids: list[str] | None = None,
        run_id: str | None = None,
        persist: bool = True,
    ) -> Evidence:
        """Extract and attach structured attributes to a built TextNode.

        TODO: implement the Router/LLM call inside _extract_attributes_with_llm.
        Once that call returns attributes, this method writes them into
        node.attributes and records one evidence item for traceability.
        """

        extracted_attributes = self._extract_attributes_with_llm(node)
        return self._attach_attributes(
            node,
            extracted_attributes,
            source_evidence_ids=source_evidence_ids,
            run_id=run_id,
            persist=persist,
        )

    def _attach_attributes(
        self,
        node: TextNode,
        extracted_attributes: dict[str, Any] | list[dict[str, Any]],
        *,
        source_evidence_ids: list[str] | None = None,
        run_id: str | None = None,
        persist: bool = True,
    ) -> Evidence:
        normalized = self._normalize_attributes(extracted_attributes)
        node.attributes = dict(node.attributes or {})
        node.attributes.update(normalized)
        node.metadata = dict(node.metadata or {})
        node.metadata["attribute_extractor"] = self.builder_name

        evidence = Evidence.create(
            EvidenceType.LLM_OUTPUT,
            content=json.dumps(
                {
                    "node_id": node.node_id,
                    "attributes": normalized,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            node_ids=[node.node_id],
            url=node.source.url if node.source else None,
            extractor=self.builder_name,
            metadata={
                "run_id": run_id,
                "prompt_name": "PROMPT_EXTRACT_ATTRIBUTE",
                "source_evidence_ids": source_evidence_ids or [],
            },
            evidence_key=f"attributes:{node.node_id}",
        )
        if persist and self.store is not None:
            self.store.upsert_node(node)
            self.store.upsert_evidence(evidence)
            self.store.flush()
        return evidence

    def _extract_attributes_with_llm(self, node: TextNode) -> dict[str, Any] | list[dict[str, Any]]:
        model_alias = os.environ.get("TEXT_PROCESS_MODEL")
        if not model_alias:
            raise ValueError("TEXT_PROCESS_MODEL is required for attribute extraction.")

        content = node.description or node.summary or ""
        if not content.strip():
            raise ValueError(f"TextNode has no description/summary content: {node.node_id}")

        response = self.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_EXTRACT_ATTRIBUTE),
                    ModelMessage(role="user", content=self._attribute_prompt_input(node, content)),
                ],
                temperature=0.0,
            )
        )
        return self._parse_attribute_tags(response.content)

    @staticmethod
    def _attribute_prompt_input(node: TextNode, content: str) -> str:
        title = node.title or ""
        return f"Title:\n{title}\n\nContent:\n{content}"

    @staticmethod
    def _parse_attribute_tags(text: str) -> list[dict[str, Any]]:
        attributes: list[dict[str, Any]] = []
        for raw_attr in re.findall(r"<attr>(.*?)</attr>", text, flags=re.DOTALL | re.IGNORECASE):
            item = re.sub(r"\s+", " ", raw_attr).strip()
            if not item:
                continue
            if ":" in item:
                key, value = item.split(":", 1)
            elif "：" in item:
                key, value = item.split("：", 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            if key and value:
                attributes.append({"key": key, "value": value})
        if not attributes:
            raise ValueError(f"No <attr>key: value</attr> attributes found in model output: {text[:500]}")
        return attributes

    @staticmethod
    def _normalize_attributes(
        extracted_attributes: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(extracted_attributes, dict):
            payload = extracted_attributes.get("attributes")
            if isinstance(payload, list):
                return WikiTextBuilder._normalize_attributes(payload)
            return dict(extracted_attributes)

        normalized: dict[str, Any] = {}
        for item in extracted_attributes:
            key = item.get("key") or item.get("name")
            if not key:
                raise ValueError(f"Attribute item is missing key/name: {item!r}")
            value = item.get("value")
            metadata = {
                field_name: item[field_name]
                for field_name in ("source_text", "confidence", "description", "metadata")
                if field_name in item and item[field_name] is not None
            }
            normalized[str(key)] = {"value": value, **metadata} if metadata else value
        return normalized

    def extract_wiki_links(
        self,
        markdown: str,
        *,
        source_url: str,
    ) -> list[WikiLinkCandidate]:
        seen_urls: set[str] = set()
        candidates: list[WikiLinkCandidate] = []
        for rank, match in enumerate(WIKI_LINK_RE.finditer(markdown), start=1):
            anchor_text, href = match.groups()
            url = self._wiki_url_from_href(href, source_url=source_url)
            if not url or url in seen_urls:
                continue
            title = self._title_from_url(url)
            if not title or title.startswith(WIKI_PAGE_PREFIXES_TO_SKIP):
                continue
            context = self._context(markdown, match.start(), match.end())
            score, reasons = self._score_link_candidate(
                title=title,
                anchor_text=anchor_text,
                context=context,
                rank=rank,
            )
            if score <= 0:
                continue
            seen_urls.add(url)
            candidates.append(
                WikiLinkCandidate(
                    title=title,
                    url=url,
                    anchor_text=anchor_text.strip(),
                    source_url=source_url,
                    context=context,
                    rank=rank,
                    start_char=match.start(),
                    end_char=match.end(),
                    window_id=self._window_id(match.start()),
                    score=score,
                    quality_reasons=reasons,
                )
            )
            if len(candidates) >= self.max_raw_links:
                break
        return self._select_position_diverse_links(candidates)

    def _select_position_diverse_links(
        self,
        candidates: list[WikiLinkCandidate],
    ) -> list[WikiLinkCandidate]:
        sorted_candidates = sorted(candidates, key=lambda item: (-item.score, item.rank or 10**9))
        selected: list[WikiLinkCandidate] = []
        window_counts: dict[int, int] = {}

        for candidate in sorted_candidates:
            if len(selected) >= self.max_links:
                break
            if not self._passes_position_diversity(candidate, selected, window_counts):
                continue
            selected.append(candidate)
            if candidate.window_id is not None:
                window_counts[candidate.window_id] = window_counts.get(candidate.window_id, 0) + 1
            candidate.quality_reasons.append("position_diverse")

        return selected

    def _passes_position_diversity(
        self,
        candidate: WikiLinkCandidate,
        selected: list[WikiLinkCandidate],
        window_counts: dict[int, int],
    ) -> bool:
        start = candidate.start_char
        if start is None:
            return True

        window_id = candidate.window_id
        if window_id is not None:
            window_limit = self._window_limit(start)
            if window_counts.get(window_id, 0) >= window_limit:
                return False

        if self.min_link_char_distance <= 0:
            return True
        for picked in selected:
            picked_start = picked.start_char
            if picked_start is None:
                continue
            if abs(start - picked_start) < self.min_link_char_distance:
                return False
        return True

    def _window_id(self, start_char: int) -> int | None:
        if self.diversity_window_size <= 0:
            return None
        return start_char // self.diversity_window_size

    def _window_limit(self, start_char: int) -> int:
        if self.lead_chars > 0 and start_char < self.lead_chars:
            return max(1, self.lead_max_links_per_window)
        return max(1, self.max_links_per_window)

    def _edge_to_linked_entity(
        self,
        source_node: TextNode,
        candidate: WikiLinkCandidate,
        evidence: Evidence,
        *,
        run_id: str | None,
    ) -> Edge:
        relation_info = self._extract_relation_for_link(source_node, candidate)
        relation = relation_info.get("predicate") or candidate.anchor_text
        return Edge.create(
            source_node.node_id,
            candidate.node_id,
            edge_type=EdgeType.WIKI_LINK,
            relation=relation,
            src_node_type=NodeType.TEXT.value,
            dst_node_type=NodeType.TEXT.value,
            evidence_refs=[
                EvidenceRef(
                    evidence_id=evidence.evidence_id,
                    url=candidate.url,
                    quote=candidate.context,
                    metadata={
                        "anchor_text": candidate.anchor_text,
                        "relation_info": relation_info,
                    },
                )
            ],
            source=EdgeSource(
                source_type="wikipedia_hyperlink",
                url=candidate.url,
                run_id=run_id,
                builder=self.builder_name,
            ),
            extractor=self.builder_name,
            metadata={
                "target_title": candidate.title,
                "target_url": candidate.url,
                "rank": candidate.rank,
                "start_char": candidate.start_char,
                "end_char": candidate.end_char,
                "window_id": candidate.window_id,
                "anchor_text": candidate.anchor_text,
                "link_score": candidate.score,
                "quality_reasons": candidate.quality_reasons,
                "position_diversity": {
                    "window_size": self.diversity_window_size,
                    "max_links_per_window": self.max_links_per_window,
                    "min_char_distance": self.min_link_char_distance,
                    "lead_chars": self.lead_chars,
                    "lead_max_links_per_window": self.lead_max_links_per_window,
                },
                "relation_info": relation_info,
            },
            evidence_key=f"{evidence.evidence_id}:{candidate.url}",
        )

    def _extract_relation_for_link(
        self,
        source_node: TextNode,
        candidate: WikiLinkCandidate,
    ) -> dict[str, Any]:
        model_alias = os.environ.get("WIKI_RELATION_MODEL")
        if not model_alias:
            return {
                "predicate": candidate.anchor_text,
                "direction": "source_to_target",
                "confidence": None,
                "evidence": candidate.context,
                "method": "anchor_fallback",
            }

        response = self.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_EXTRACT_RELATION),
                    ModelMessage(
                        role="user",
                        content=self._relation_prompt_input(source_node, candidate),
                    ),
                ],
                temperature=0.0,
            )
        )
        try:
            parsed = self._parse_relation_response(response.content)
        except ValueError:
            return {
                "predicate": candidate.anchor_text,
                "direction": "source_to_target",
                "confidence": None,
                "evidence": candidate.context,
                "method": "anchor_fallback_parse_failed",
                "raw_model_output": response.content,
            }
        parsed["method"] = "llm_relation_extraction"
        parsed["raw_model_output"] = response.content
        return parsed

    @staticmethod
    def _relation_prompt_input(source_node: TextNode, candidate: WikiLinkCandidate) -> str:
        return (
            f"Source entity:\n{source_node.title or source_node.node_id}\n\n"
            f"Target entity:\n{candidate.title}\n\n"
            f"Anchor text:\n{candidate.anchor_text}\n\n"
            f"Local context:\n{candidate.context or ''}\n"
        )

    @staticmethod
    def _parse_relation_response(text: str) -> dict[str, Any]:
        match = re.search(r"<relation>(.*?)</relation>", text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError("No <relation> block found.")
        block = match.group(1)
        fields: dict[str, Any] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        predicate = WikiTextBuilder._normalize_relation_predicate(fields.get("predicate") or "related_to")
        return {
            "predicate": predicate,
            "direction": fields.get("direction") or "source_to_target",
            "confidence": WikiTextBuilder._parse_confidence(fields.get("confidence")),
            "evidence": fields.get("evidence"),
        }

    @staticmethod
    def _normalize_relation_predicate(predicate: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", predicate.strip().lower()).strip("_")
        return normalized or "related_to"

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _score_link_candidate(
        *,
        title: str,
        anchor_text: str,
        context: str,
        rank: int,
    ) -> tuple[float, list[str]]:
        title_clean = title.strip()
        anchor_clean = re.sub(r"\s+", " ", anchor_text.strip())
        title_lower = title_clean.lower()
        anchor_lower = anchor_clean.lower()
        context_lower = context.lower()

        if title_clean in GENERIC_WIKI_TITLES:
            return 0.0, ["generic_title"]
        if WikiTextBuilder._is_year_or_date_title(title_clean):
            return 0.0, ["year_or_date_title"]
        if anchor_lower in LOW_VALUE_ANCHORS or len(anchor_clean) < 2:
            return 0.0, ["low_value_anchor"]
        if any(marker in context_lower for marker in REFERENCE_CONTEXT_MARKERS):
            return 0.0, ["reference_like_context"]

        score = 1.0
        reasons = ["base"]
        if rank <= 30:
            score += 1.0
            reasons.append("early_link")
        elif rank <= 100:
            score += 0.4
            reasons.append("middle_link")

        if WikiTextBuilder._looks_like_named_entity(title_clean):
            score += 1.2
            reasons.append("named_entity_title")
        if anchor_clean != title_clean and WikiTextBuilder._looks_like_named_entity(anchor_clean):
            score += 0.4
            reasons.append("named_entity_anchor")
        if any(hint in context_lower for hint in RELATION_CONTEXT_HINTS):
            score += 1.0
            reasons.append("relation_context_hint")
        if len(title_clean.split()) >= 2:
            score += 0.3
            reasons.append("specific_multiword_title")
        return score, reasons

    @staticmethod
    def _is_year_or_date_title(title: str) -> bool:
        stripped = title.strip()
        if re.fullmatch(r"\d{3,4}", stripped):
            return True
        return bool(re.fullmatch(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}", stripped))

    @staticmethod
    def _looks_like_named_entity(text: str) -> bool:
        words = [word for word in re.split(r"\s+", text.strip()) if word]
        if not words:
            return False
        uppercase_like = sum(1 for word in words if word[:1].isupper() or word.isupper())
        return uppercase_like >= max(1, len(words) // 2)

    def _persist_result(self, result: WikiTextBuildResult) -> None:
        if self.store is None:
            return
        if self.persist_snapshots:
            self.store.upsert_search_snapshot(result.snapshot)
        self.store.upsert_node(result.node)
        self.store.upsert_evidence(result.text_evidence)
        for edge in result.edges:
            self.store.upsert_edge(edge)
        self.store.flush()

    @staticmethod
    def _wiki_url_from_href(href: str, *, source_url: str) -> str | None:
        href = href.strip()
        if href.startswith("#"):
            return None
        parsed_source = urlparse(source_url)
        host = parsed_source.netloc or "en.wikipedia.org"

        if href.startswith("/wiki/"):
            title = href.removeprefix("/wiki/").split("#", 1)[0]
            return f"https://{host}/wiki/{quote(unquote(title), safe=':/()_,')}"
        if href.startswith(("http://", "https://")):
            parsed = urlparse(href)
            if not parsed.netloc.endswith("wikipedia.org") or not parsed.path.startswith("/wiki/"):
                return None
            title = parsed.path.removeprefix("/wiki/").split("#", 1)[0]
            return f"https://{parsed.netloc}/wiki/{quote(unquote(title), safe=':/()_,')}"
        return None

    @staticmethod
    def _normalize_wikipedia_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("wikipedia.org") and parsed.path.startswith("/wiki/"):
            title = parsed.path.removeprefix("/wiki/").split("#", 1)[0]
            return f"https://{parsed.netloc}/wiki/{quote(unquote(title), safe=':/()_,')}"
        return url

    @staticmethod
    def _title_from_url(url: str) -> str | None:
        parsed = urlparse(url)
        if not parsed.path.startswith("/wiki/"):
            return None
        return unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")

    @staticmethod
    def _first_paragraph(text: str, *, max_chars: int = 1000) -> str | None:
        for block in re.split(r"\n\s*\n", text):
            cleaned = block.strip()
            if cleaned:
                return cleaned[:max_chars]
        return None

    @staticmethod
    def _context(text: str, start: int, end: int, *, window: int = 180) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return re.sub(r"\s+", " ", text[left:right]).strip()


def _smoke_test() -> None:
    import tempfile

    class MockReader:
        def read(self, url: str, **kwargs: Any) -> ReaderDocument:
            del kwargs
            return ReaderDocument(
                url=url,
                title="Kobe Bryant",
                content=(
                    "Kobe Bryant was an American basketball player. "
                    "He played for the Los Angeles Lakers."
                ),
                raw_markdown=(
                    "Kobe Bryant played for the [Los Angeles Lakers]"
                    "(https://en.wikipedia.org/wiki/Los_Angeles_Lakers). "
                    "See also [official website](https://example.com). "
                    "Ignore [File:Photo.jpg](https://en.wikipedia.org/wiki/File:Photo.jpg)."
                ),
                raw={"mock": True},
            )

    class MockModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            assert request.model == "mock_text"
            return ModelResponse(
                content="<attr>occupation: basketball player</attr><attr>team: Los Angeles Lakers</attr>"
            )

    old_model = os.environ.get("TEXT_PROCESS_MODEL")
    os.environ["TEXT_PROCESS_MODEL"] = "mock_text"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir)
            builder = WikiTextBuilder(
                reader=MockReader(),
                store=store,
                model_client=MockModel(),
                max_links=5,
                diversity_window_size=120,
                max_links_per_window=1,
                min_link_char_distance=80,
                lead_chars=0,
            )
            result = builder.build_from_url("https://en.wikipedia.org/wiki/Kobe_Bryant")
            assert result.node.title == "Kobe Bryant"
            assert len(result.linked_entities) == 1
            assert result.linked_entities[0].title == "Los Angeles Lakers"
            nearby_markdown = (
                "[1950](https://en.wikipedia.org/wiki/1950_NBA_Finals) "
                "[1951](https://en.wikipedia.org/wiki/1951_NBA_Finals) "
                "[1952](https://en.wikipedia.org/wiki/1952_NBA_Finals) "
                + ("x" * 160)
                + " [Jerry West](https://en.wikipedia.org/wiki/Jerry_West)"
            )
            diverse_links = builder.extract_wiki_links(
                nearby_markdown,
                source_url="https://en.wikipedia.org/wiki/NBA_Finals",
            )
            assert len(diverse_links) == 2
            assert any(link.title == "Jerry West" for link in diverse_links)
            assert all(link.window_id is not None for link in diverse_links)
            evidence = builder.extract_attributes(
                result.node,
                source_evidence_ids=[result.text_evidence.evidence_id],
            )
            assert result.node.attributes["occupation"] == "basketball player"
            assert evidence.evidence_type == EvidenceType.LLM_OUTPUT
    finally:
        if old_model is None:
            os.environ.pop("TEXT_PROCESS_MODEL", None)
        else:
            os.environ["TEXT_PROCESS_MODEL"] = old_model
    print("wiki_text_builder smoke test passed")


if __name__ == "__main__":
    _smoke_test()
