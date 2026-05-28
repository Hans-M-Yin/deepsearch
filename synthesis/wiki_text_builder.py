"""Wikipedia text-node construction and neighbor extraction."""

from __future__ import annotations

import re
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import sys
import time
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
MARKDOWN_TRUNCATION_MARKER = "\n\n<!-- wiki_text_builder_truncated -->"
WIKI_MISSING_PAGE_PATTERNS = (
    "wikipedia does not have an article with this exact name",
    "there is currently no text in this page",
    "you may create the page",
    "维基百科没有与该名称完全匹配的条目",
    "没有与该名称完全匹配的条目",
    "创建新条目",
)
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
- The predicate is for multi-hop graph reasoning, so it should be strongly target-identifying from the source side.
- Prefer a relation that would point to this target uniquely, or as close to uniquely as the local context allows.
- If a generic predicate could apply to multiple targets for the same source, add distinguishing qualifiers directly into the predicate.
- Good qualifiers include time period, role, outcome, ordinal/superlative, event, award, location, work, team, or other locally explicit constraints.
- Do not use an underspecified generic predicate such as played_for, lived_in, won, member_of, or born_in if that would leave multiple plausible targets for the same source.
- Example: use club_where_he_won_multiple_champions_league_titles instead of played_for when the generic relation would match several clubs.
- Example: use won_gold_with_2008_olympic_team instead of member_of if the local context supports that stronger description.
- If the local context does not support a unique relation, still make the predicate as specific as possible rather than falling back to a broad label.
- Use related_to only as a last resort when the relation is truly unclear.
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


PROMPT_FILTER_WIKI_NEIGHBORS = """You are selecting useful neighboring Wikipedia entities for building a multi-hop research graph.

Given one source entity and candidate outgoing Wikipedia links, decide which candidates should be kept as expansion targets.

The goal is not to keep the closest or most obvious links. Prefer candidates that can become useful intermediate hops for diverse, natural multi-hop questions.
The graph will later rely on source_node + relation -> target_node reasoning, so prefer neighbors whose relation from the source can be phrased in a strong, target-identifying way.

Keep candidates that:
- are unique, concrete entities or events, not broad classes, generic concepts, dates, list pages, maintenance pages, or ambiguous labels;
- have a meaningful but not trivial relation to the source entity;
- admit a relation description that can distinguish this target from other likely neighbors of the same source, possibly by adding explicit qualifiers from context;
- can act as a useful bridge for multi-hop questions;
- are supported by the local hyperlink context.

Reject candidates that:
- are too close to the source, such as the same entity, aliases, purely self-descriptive links, or repeated administrative editions;
- are too far or only appear in references/navigation/template noise;
- are broad categories or common concepts rather than identifiable entities/events;
- would only support a vague relation that is likely to map from the source to many different targets unless the local context clearly provides a stronger distinguishing qualifier;
- are unlikely to have a stable Wikipedia page representing one specific target.

Relation is open-ended. Use a concise snake_case predicate that best describes the local context.
Do not force the relation into a fixed taxonomy, and do not overuse a small set of generic predicates.
The relation should be written from the source to this candidate in a way that is as uniquely target-identifying as possible.
If a broad predicate would fit multiple candidates for the same source, add qualifiers so the relation becomes more discriminative.
If rejecting a candidate, relation can be a short rejection label such as too_generic, too_close, too_far, ambiguous_entity, list_page, reference_noise, or templatic_edition.

Positive examples:
- Source: Kobe Bryant
  Candidate: Los Angeles Lakers
  keep: yes
  relation: played_for
  reason: specific team central to career, useful bridge

- Source: Kobe Bryant
  Candidate: 2008 Summer Olympics
  keep: yes
  relation: won_gold_at
  reason: specific event connected to the source, not merely a broad sports concept

- Source: Parasite
  Candidate: Bong Joon-ho
  keep: yes
  relation: directed_by
  reason: unique person strongly connected to the work

- Source: South Korea
  Candidate: Hallasan
  keep: yes
  relation: highest_point
  reason: concrete landmark attribute useful for geographic or visual hops

- Source: Boston Celtics
  Candidate: TD Garden
  keep: yes
  relation: home_arena
  reason: concrete place connected to the team

- Source: Lionel Messi
  Candidate: FC Barcelona
  keep: yes
  relation: club_where_he_won_multiple_champions_league_titles
  reason: more target-identifying than generic played_for because the source has multiple clubs

- Source: Lionel Messi
  Candidate: Paris Saint-Germain
  keep: yes
  relation: post_barcelona_club
  reason: local timeline qualifier makes the relation less ambiguous than generic played_for

Negative examples:
- Source: Kobe Bryant
  Candidate: basketball
  keep: no
  relation: too_generic
  reason: broad category, not a unique entity

- Source: NBA Finals
  Candidate: 1951 NBA Finals
  keep: no
  relation: templatic_edition
  reason: yearly edition pattern is repetitive and too narrow unless local context makes it special

- Source: South Korea
  Candidate: country
  keep: no
  relation: too_generic
  reason: class label, not a unique entity

- Source: Some company
  Candidate: References
  keep: no
  relation: reference_noise
  reason: navigation/reference artifact rather than a content entity

- Source: Kobe Bryant
  Candidate: Kobe Bryant career achievements
  keep: no
  relation: too_close
  reason: likely a self-descriptive or duplicate topic rather than a useful new entity

- Source: Lionel Messi
  Candidate: Inter Miami CF
  keep: no
  relation: played_for
  reason: generic relation is under-specified because the source has multiple clubs; keep only if local context supports a stronger distinguishing predicate

Return one XML-like item per candidate. Copy the candidate title exactly into
the title attribute so debugging output is readable. Do not output markdown or
explanations outside the tags.

Output format:
<neighbor id="1" title="Los Angeles Lakers" keep="yes" score="4.2" relation="played_for" reason="Specific team linked by source career context"/>
<neighbor id="2" title="basketball" keep="no" score="1.0" relation="too_generic" reason="Broad class, not a unique entity"/>

Scores are 0.0 to 5.0. Use keep="yes" only for candidates with score >= 3.0.
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
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "text_evidence": self.text_evidence.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "linked_entities": [entity.to_dict() for entity in self.linked_entities],
            "edges": [edge.to_dict() for edge in self.edges],
            "from_cache": self.from_cache,
            "timing": self.timing,
        }


class InvalidWikiPageError(ValueError):
    """Raised when a fetched Wikipedia URL is not an actual article page."""


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
        max_content_chars: int | None = 50000,
        max_link_markdown_chars: int | None = 80000,
        max_llm_neighbor_candidates: int = 40,
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
        self.max_content_chars = max_content_chars
        self.max_link_markdown_chars = max_link_markdown_chars
        self.max_llm_neighbor_candidates = max_llm_neighbor_candidates

    def build_from_url(
        self,
        url: str,
        *,
        title: str | None = None,
        run_id: str | None = None,
        persist: bool = True,
        force: bool = False,
    ) -> WikiTextBuildResult:
        total_started = time.perf_counter()
        timing: dict[str, float] = {}
        input_url = self._normalize_wikipedia_url(url)
        started = time.perf_counter()
        cached = None if force else self._cached_build_result(input_url)
        timing["cache_lookup_input_s"] = time.perf_counter() - started
        if cached is not None:
            cached.timing = {**cached.timing, **timing, "total_s": time.perf_counter() - total_started}
            return cached

        started = time.perf_counter()
        document = self.reader.read(url)
        timing["reader_read_s"] = time.perf_counter() - started
        page_url = self._normalize_wikipedia_url(document.url or url)
        started = time.perf_counter()
        cached = None if force else self._cached_build_result(page_url)
        timing["cache_lookup_page_s"] = time.perf_counter() - started
        if cached is not None:
            cached.timing = {**cached.timing, **timing, "total_s": time.perf_counter() - total_started}
            return cached

        started = time.perf_counter()
        page_title = title or document.title or self._title_from_url(page_url)
        self._validate_article_page(page_url=page_url, page_title=page_title, content=document.content)
        content = self._safe_truncate_markdown(document.content, self.max_content_chars)
        link_markdown = self._safe_truncate_markdown(
            document.raw_markdown or document.content,
            self.max_link_markdown_chars,
        )
        content_truncated = content != document.content
        link_markdown_original = document.raw_markdown or document.content
        link_markdown_truncated = link_markdown != link_markdown_original

        snapshot = SearchSnapshot.create(
            SearchEngine.JINA_READER,
            query=page_url,
            request={"url": page_url, "reader": self.reader.__class__.__name__},
            response_preview=content[:2000],
            result_count=1 if content else 0,
            status_code=200,
            run_id=run_id,
            metadata={
                "raw": document.raw,
                "content_original_chars": len(document.content),
                "content_stored_chars": len(content),
                "content_truncated": content_truncated,
                "link_markdown_original_chars": len(link_markdown_original),
                "link_markdown_used_chars": len(link_markdown),
                "link_markdown_truncated": link_markdown_truncated,
            },
        )
        node = TextNode(
            # TODO: alise is not implemented yet.
            node_id=TextNode.make_id("wikipedia_page", page_url),
            subtype="wiki_page",
            canonical_id=f"wikipedia:{page_title}" if page_title else None,
            title=page_title,
            summary=self._first_paragraph(content),
            description=content,
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
                "content_original_chars": len(document.content),
                "content_stored_chars": len(content),
                "content_truncated": content_truncated,
                "link_markdown_used_chars": len(link_markdown),
                "link_markdown_truncated": link_markdown_truncated,
            },
        )
        text_evidence = Evidence.create(
            EvidenceType.WEB_TEXT,
            content=content,
            node_ids=[node.node_id],
            url=page_url,
            source_snapshot_id=snapshot.snapshot_id if self.persist_snapshots else None,
            extractor=self.builder_name,
            evidence_key=f"wiki_text:{page_url}",
        )
        timing["validate_and_truncate_s"] = time.perf_counter() - started
        timing["node_evidence_create_s"] = time.perf_counter() - started

        started = time.perf_counter()
        linked_entities = self.extract_wiki_links(link_markdown, source_url=page_url)
        timing["link_extract_s"] = time.perf_counter() - started

        result = WikiTextBuildResult(
            node=node,
            text_evidence=text_evidence,
            snapshot=snapshot,
            linked_entities=linked_entities,
            edges=[],
        )
        if persist:
            started = time.perf_counter()
            self._persist_result(result)
            timing["persist_s"] = time.perf_counter() - started
        else:
            timing["persist_s"] = 0.0
        timing["total_s"] = time.perf_counter() - total_started
        result.timing = timing
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
        for rank, (anchor_text, href, start, end) in enumerate(self._iter_markdown_links(markdown), start=1):
            url = self._wiki_url_from_href(href, source_url=source_url)
            if not url or url in seen_urls:
                continue
            title = self._title_from_url(url)
            if not title or title.startswith(WIKI_PAGE_PREFIXES_TO_SKIP):
                continue
            context = self._context(markdown, start, end)
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
                    start_char=start,
                    end_char=end,
                    window_id=self._window_id(start),
                    score=score,
                    quality_reasons=reasons,
                )
            )
            if len(candidates) >= self.max_raw_links:
                break
        candidates = self._filter_links_with_llm(source_url=source_url, candidates=candidates)
        return self._select_position_diverse_links(candidates)

    def _filter_links_with_llm(
        self,
        *,
        source_url: str,
        candidates: list[WikiLinkCandidate],
    ) -> list[WikiLinkCandidate]:
        model_alias = os.environ.get("WIKI_NEIGHBOR_MODEL")
        if not model_alias or not candidates:
            return candidates

        source_title = self._title_from_url(source_url) or source_url
        debug_enabled = os.environ.get("WIKI_NEIGHBOR_DEBUG", "1") != "0"
        ranked_candidates = sorted(candidates, key=lambda item: (-item.score, item.rank or 10**9))
        prompt_candidates = ranked_candidates[: max(1, self.max_llm_neighbor_candidates)]
        rule_scores = {candidate.url: candidate.score for candidate in prompt_candidates}

        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_FILTER_WIKI_NEIGHBORS),
                        ModelMessage(
                            role="user",
                            content=self._neighbor_filter_prompt_input(source_title, source_url, prompt_candidates),
                        ),
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
            )
            decisions = self._parse_neighbor_filter_response(response.content)
            if debug_enabled:
                self._debug_print_neighbor_filter_raw(
                    source_title=source_title,
                    source_url=source_url,
                    raw_output=response.content,
                )
        except Exception as exc:
            for candidate in candidates:
                candidate.quality_reasons.append(f"llm_neighbor_filter_failed:{exc.__class__.__name__}")
            if debug_enabled:
                self._debug_print_neighbor_filter_failure(
                    source_title=source_title,
                    source_url=source_url,
                    candidates=prompt_candidates,
                    error=exc,
                )
            return candidates

        kept: list[WikiLinkCandidate] = []
        debug_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(prompt_candidates, start=1):
            decision = decisions.get(index)
            if decision is None:
                candidate.quality_reasons.append("llm_neighbor_missing_decision")
                debug_rows.append(
                    self._neighbor_debug_row(
                        index=index,
                        candidate=candidate,
                        rule_score=rule_scores.get(candidate.url, candidate.score),
                        decision=None,
                        final_score=candidate.score,
                    )
                )
                continue
            keep = decision.get("keep") == "yes"
            llm_score = self._parse_neighbor_score(decision.get("score"))
            relation = decision.get("relation")
            reason = decision.get("reason")
            if llm_score is not None:
                candidate.score = llm_score
            candidate.quality_reasons.append("llm_neighbor_keep" if keep else "llm_neighbor_reject")
            if relation:
                candidate.quality_reasons.append(f"llm_relation:{relation}")
            if reason:
                candidate.quality_reasons.append(f"llm_reason:{reason[:120]}")
            if keep and candidate.score >= 3.0:
                kept.append(candidate)
            debug_rows.append(
                self._neighbor_debug_row(
                    index=index,
                    candidate=candidate,
                    rule_score=rule_scores.get(candidate.url, candidate.score),
                    decision=decision,
                    final_score=candidate.score,
                )
            )

        if debug_enabled:
            self._debug_print_neighbor_filter(
                source_title=source_title,
                source_url=source_url,
                model_alias=model_alias,
                rows=debug_rows,
                kept_urls={candidate.url for candidate in kept},
            )

        if not kept:
            for candidate in candidates:
                candidate.quality_reasons.append("llm_neighbor_filter_empty_fallback")
            return candidates

        return kept

    @staticmethod
    def _neighbor_debug_row(
        *,
        index: int,
        candidate: WikiLinkCandidate,
        rule_score: float,
        decision: dict[str, str] | None,
        final_score: float,
    ) -> dict[str, Any]:
        context = re.sub(r"\s+", " ", candidate.context or "").strip()
        return {
            "index": index,
            "title": candidate.title,
            "url": candidate.url,
            "anchor": candidate.anchor_text,
            "rank": candidate.rank,
            "rule_score": rule_score,
            "keep": decision.get("keep") if decision else "missing",
            "llm_score": decision.get("score") if decision else "",
            "final_score": final_score,
            "relation": decision.get("relation") if decision else "",
            "reason": decision.get("reason") if decision else "missing LLM decision",
            "context": context[:220],
        }

    @staticmethod
    def _debug_print_neighbor_filter(
        *,
        source_title: str,
        source_url: str,
        model_alias: str,
        rows: list[dict[str, Any]],
        kept_urls: set[str],
    ) -> None:
        print(
            f"[wiki_neighbor_filter] source={source_title!r} url={source_url} "
            f"model={model_alias} candidates={len(rows)} kept={len(kept_urls)}",
            file=sys.stderr,
            flush=True,
        )
        for row in rows:
            status = "KEEP" if row["url"] in kept_urls else "DROP"
            print(
                "[wiki_neighbor_filter] "
                f"{status} #{row['index']} title={row['title']!r} anchor={row['anchor']!r} "
                f"rank={row['rank']} rule={row['rule_score']:.2f} "
                f"llm_keep={row['keep']} llm_score={row['llm_score']} final={row['final_score']:.2f} "
                f"relation={row['relation']!r} reason={row['reason']!r} url={row['url']}",
                file=sys.stderr,
                flush=True,
            )
            if row["context"]:
                print(
                    f"[wiki_neighbor_filter]      context={row['context']!r}",
                    file=sys.stderr,
                    flush=True,
                )

    @staticmethod
    def _debug_print_neighbor_filter_raw(
        *,
        source_title: str,
        source_url: str,
        raw_output: str,
    ) -> None:
        print(
            f"[wiki_neighbor_filter_raw] source={source_title!r} url={source_url}",
            file=sys.stderr,
            flush=True,
        )
        for line in raw_output.strip().splitlines():
            if line.strip():
                print(f"[wiki_neighbor_filter_raw] {line}", file=sys.stderr, flush=True)

    @staticmethod
    def _debug_print_neighbor_filter_failure(
        *,
        source_title: str,
        source_url: str,
        candidates: list[WikiLinkCandidate],
        error: Exception,
    ) -> None:
        print(
            f"[wiki_neighbor_filter] FAILED source={source_title!r} url={source_url} "
            f"error={error.__class__.__name__}: {error} candidates={len(candidates)}",
            file=sys.stderr,
            flush=True,
        )
        for index, candidate in enumerate(candidates, start=1):
            print(
                "[wiki_neighbor_filter] "
                f"FALLBACK #{index} title={candidate.title!r} anchor={candidate.anchor_text!r} "
                f"rank={candidate.rank} rule={candidate.score:.2f} url={candidate.url}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _neighbor_filter_prompt_input(
        source_title: str,
        source_url: str,
        candidates: list[WikiLinkCandidate],
    ) -> str:
        lines = [
            f"Source entity: {source_title}",
            f"Source URL: {source_url}",
            "",
            "Candidates:",
        ]
        for index, candidate in enumerate(candidates, start=1):
            context = re.sub(r"\s+", " ", candidate.context or "").strip()
            lines.extend(
                [
                    f"[{index}]",
                    f"Title: {candidate.title}",
                    f"URL: {candidate.url}",
                    f"Anchor: {candidate.anchor_text}",
                    f"Rule score: {candidate.score:.2f}",
                    f"Local context: {context[:500]}",
                    "",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_neighbor_filter_response(text: str) -> dict[int, dict[str, str]]:
        decisions: dict[int, dict[str, str]] = {}
        for match in re.finditer(r"<neighbor\s+([^>]*)/?>", text, flags=re.IGNORECASE | re.DOTALL):
            attrs: dict[str, str] = {}
            for key, double_value, single_value in re.findall(r"""(\w+)=(?:"([^"]*)"|'([^']*)')""", match.group(1)):
                attrs[key.lower()] = double_value or single_value
            try:
                candidate_id = int(attrs.get("id", ""))
            except ValueError:
                continue
            keep = attrs.get("keep", "").strip().lower()
            if keep not in {"yes", "no"}:
                keep = "no"
            decisions[candidate_id] = {
                "title": attrs.get("title", ""),
                "keep": keep,
                "score": attrs.get("score", "0"),
                "relation": attrs.get("relation", ""),
                "reason": attrs.get("reason", ""),
            }
        return decisions

    @staticmethod
    def _parse_neighbor_score(value: Any) -> float | None:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(5.0, score))

    @staticmethod
    def _iter_markdown_links(markdown: str):
        """Yield markdown links while tolerating parentheses and optional titles in URLs."""

        pos = 0
        length = len(markdown)
        while pos < length:
            start = markdown.find("[", pos)
            if start < 0:
                break
            if start > 0 and markdown[start - 1] == "!":
                pos = start + 1
                continue
            close = markdown.find("]", start + 1)
            if close < 0 or close + 1 >= length or markdown[close + 1] != "(":
                pos = start + 1
                continue

            href_start = close + 2
            depth = 1
            idx = href_start
            while idx < length:
                char = markdown[idx]
                if char == "\\":
                    idx += 2
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        anchor = markdown[start + 1 : close]
                        href = markdown[href_start:idx]
                        yield anchor, href, start, idx + 1
                        pos = idx + 1
                        break
                idx += 1
            else:
                break

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
        self.store.flush()

    @staticmethod
    def _validate_article_page(
        *,
        page_url: str,
        page_title: str | None,
        content: str,
    ) -> None:
        normalized = re.sub(r"\s+", " ", content or "").strip().lower()
        if not normalized:
            raise InvalidWikiPageError(f"Empty Wikipedia reader content: {page_url}")

        if any(pattern in normalized for pattern in WIKI_MISSING_PAGE_PATTERNS):
            raise InvalidWikiPageError(f"Wikipedia page does not exist as an article: {page_url}")

        title = (page_title or "").strip()
        if title.startswith(WIKI_PAGE_PREFIXES_TO_SKIP):
            raise InvalidWikiPageError(f"Non-article Wikipedia namespace is not allowed: {page_url}")

    @staticmethod
    def _safe_truncate_markdown(text: str, max_chars: int | None) -> str:
        if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
            return text

        preferred_breaks = ("\n\n", "\n# ", "\n## ", "\n### ", "\n- ", ". ", "。", " ")
        min_cut = int(max_chars * 0.65)
        cut_at = -1
        for needle in preferred_breaks:
            pos = text.rfind(needle, 0, max_chars)
            if pos >= min_cut:
                cut_at = pos + len(needle)
                break
        if cut_at < min_cut:
            cut_at = max_chars

        candidate = text[:cut_at].rstrip()
        open_bracket = candidate.rfind("[")
        close_bracket = candidate.rfind("]")
        open_paren = candidate.rfind("(")
        close_paren = candidate.rfind(")")
        if open_bracket > close_bracket or open_paren > close_paren:
            rollback = max(candidate.rfind("\n\n", 0, open_bracket), candidate.rfind("\n", 0, open_bracket))
            if rollback >= min_cut:
                candidate = candidate[:rollback].rstrip()
        return candidate + MARKDOWN_TRUNCATION_MARKER

    @staticmethod
    def _clean_markdown_href(href: str) -> str:
        href = href.strip()
        if href.startswith("<") and ">" in href:
            return href[1 : href.find(">")].strip()
        match = re.match(r"""^(\S+)(?:\s+['"].*['"])?\s*$""", href, flags=re.DOTALL)
        return (match.group(1) if match else href).strip()

    @staticmethod
    def _wiki_url_from_href(href: str, *, source_url: str) -> str | None:
        href = WikiTextBuilder._clean_markdown_href(href)
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
                    "(https://en.wikipedia.org/wiki/Los_Angeles_Lakers \"Los Angeles Lakers\"). "
                    "He was selected for the [Dream Team]"
                    "(https://en.wikipedia.org/wiki/Dream_Team_(basketball)). "
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
                max_links_per_window=2,
                min_link_char_distance=0,
                lead_chars=0,
                max_content_chars=30,
                max_link_markdown_chars=500,
            )
            result = builder.build_from_url("https://en.wikipedia.org/wiki/Kobe_Bryant")
            assert result.node.title == "Kobe Bryant"
            assert result.node.metadata["content_truncated"] is True
            titles = {link.title for link in result.linked_entities}
            assert "Los Angeles Lakers" in titles
            assert "Dream Team (basketball)" in titles
            assert all('"' not in link.url for link in result.linked_entities)
            parsed_neighbors = WikiTextBuilder._parse_neighbor_filter_response(
                '<neighbor id="1" title="Los Angeles Lakers" keep="yes" score="4.2" relation="played_for" reason="Specific team"/>'
                '<neighbor id="2" title="basketball" keep="no" score="1.0" relation="too_generic" reason="Broad class"/>'
            )
            assert parsed_neighbors[1]["keep"] == "yes"
            assert parsed_neighbors[1]["title"] == "Los Angeles Lakers"
            assert WikiTextBuilder._parse_neighbor_score(parsed_neighbors[1]["score"]) == 4.2
            nearby_markdown = (
                "[1950](https://en.wikipedia.org/wiki/1950_NBA_Finals) "
                "[1951](https://en.wikipedia.org/wiki/1951_NBA_Finals) "
                "[1952](https://en.wikipedia.org/wiki/1952_NBA_Finals) "
                + ("x" * 160)
                + " [Jerry West](https://en.wikipedia.org/wiki/Jerry_West)"
            )
            diversity_builder = WikiTextBuilder(
                reader=MockReader(),
                max_links=5,
                diversity_window_size=120,
                max_links_per_window=1,
                min_link_char_distance=80,
                lead_chars=0,
            )
            diverse_links = diversity_builder.extract_wiki_links(
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

            missing_page = "维基百科没有与该名称完全匹配的条目。请在维基百科中搜索。"
            try:
                builder._validate_article_page(
                    page_url="https://en.wikipedia.org/wiki/United_Nations_%22United_Nations%22",
                    page_title='United Nations "United Nations"',
                    content=missing_page,
                )
            except InvalidWikiPageError:
                pass
            else:
                raise AssertionError("missing Wikipedia pages should be rejected")
    finally:
        if old_model is None:
            os.environ.pop("TEXT_PROCESS_MODEL", None)
        else:
            os.environ["TEXT_PROCESS_MODEL"] = old_model
    print("wiki_text_builder smoke test passed")


if __name__ == "__main__":
    _smoke_test()
