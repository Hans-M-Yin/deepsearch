"""Core graph expansion strategy for synthesis data construction."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import sys
from threading import RLock
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .evidence import Evidence
from .edges import Edge
from .image_discovery import ImageDiscoveryBuilder, ImageDiscoveryResult
from .store import JsonlGraphStore
from .visual_planner import VisualSearchPlan, VisualSearchPlanner
from .wiki_text_builder import InvalidWikiPageError, WikiLinkCandidate, WikiTextBuilder, WikiTextBuildResult


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_jsonify(item) for item in value]
    return value


class ExpansionTaskStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class ExpansionTask:
    """A queued text node/page waiting to be built."""

    url: str
    depth: int = 0
    title: str | None = None
    parent_node_id: str | None = None
    parent_edge_id: str | None = None
    priority: float = 0.0
    status: ExpansionTaskStatus = ExpansionTaskStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def from_wiki_link(
        cls,
        candidate: WikiLinkCandidate,
        *,
        depth: int,
        parent_node_id: str | None,
        source_evidence_id: str | None = None,
        parent_edge_id: str | None = None,
    ) -> "ExpansionTask":
        pending_parent_link = {
            "parent_node_id": parent_node_id,
            "source_evidence_id": source_evidence_id,
            "candidate": candidate.to_dict(),
        }
        return cls(
            url=candidate.url,
            depth=depth,
            title=candidate.title,
            parent_node_id=parent_node_id,
            parent_edge_id=parent_edge_id,
            priority=0.0 if candidate.rank is None else -float(candidate.rank),
            metadata={
                "anchor_text": candidate.anchor_text,
                "source_url": candidate.source_url,
                "context": candidate.context,
                "rank": candidate.rank,
                "pending_parent_links": [pending_parent_link],
            },
        )


@dataclass(slots=True)
class NodeExpansionResult:
    """Result of expanding one text page."""

    task: ExpansionTask
    text_result: WikiTextBuildResult | None = None
    attribute_evidence: Evidence | None = None
    attribute_error: str | None = None
    visual_plans: list[VisualSearchPlan] = field(default_factory=list)
    image_results: list[ImageDiscoveryResult] = field(default_factory=list)
    materialized_edges: list[Edge] = field(default_factory=list)
    queued_tasks: list[ExpansionTask] = field(default_factory=list)
    error: str | None = None
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "text_result": self.text_result.to_dict() if self.text_result else None,
            "attribute_evidence": self.attribute_evidence.to_dict() if self.attribute_evidence else None,
            "attribute_error": self.attribute_error,
            "visual_plans": [plan.to_dict() for plan in self.visual_plans],
            "image_results": [result.to_dict() for result in self.image_results],
            "materialized_edges": [edge.to_dict() for edge in self.materialized_edges],
            "queued_tasks": [task.to_dict() for task in self.queued_tasks],
            "error": self.error,
            "timing": self.timing,
        }


@dataclass(slots=True)
class GraphExpansionConfig:
    """Traversal limits for online graph expansion."""

    max_depth: int = 2
    max_new_text_neighbors: int = 30
    extract_attributes: bool = True
    attribute_errors_fatal: bool = False
    enable_image_expansion: bool = True
    persist: bool = True


class GraphExpansionStrategy:
    """Orchestrate text-node construction, neighbor queuing, and image expansion."""

    def __init__(
        self,
        *,
        store: JsonlGraphStore,
        wiki_builder: WikiTextBuilder,
        visual_planner: VisualSearchPlanner | None = None,
        image_builder: ImageDiscoveryBuilder | None = None,
        config: GraphExpansionConfig | None = None,
    ) -> None:
        self.store = store
        self.wiki_builder = wiki_builder
        self.visual_planner = visual_planner
        self.image_builder = image_builder
        self.config = config or GraphExpansionConfig()
        self._queue: deque[ExpansionTask] = deque()
        self._seen_urls: set[str] = set()
        self._pending_parent_links_by_url: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()

    def add_seed(
        self,
        url: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExpansionTask:
        task = ExpansionTask(url=url, title=title, metadata=metadata or {})
        self.enqueue(task)
        return task

    def enqueue(self, task: ExpansionTask) -> bool:
        with self._lock:
            if task.url in self._seen_urls:
                return False
            self._seen_urls.add(task.url)
            pending_links = self._pending_parent_links_by_url.pop(task.url, [])
            if pending_links:
                links = list(task.metadata.get("pending_parent_links") or [])
                links.extend(pending_links)
                task.metadata["pending_parent_links"] = links
            self._queue.append(task)
            return True

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def expand_next(self, *, run_id: str | None = None) -> NodeExpansionResult | None:
        task = self.pop_next_task()
        if task is None:
            return None
        return self.expand_task(task, run_id=run_id)

    def pop_next_task(self) -> ExpansionTask | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def pop_next_batch(self, batch_size: int) -> list[ExpansionTask]:
        with self._lock:
            tasks: list[ExpansionTask] = []
            limit = max(1, int(batch_size))
            while self._queue and len(tasks) < limit:
                tasks.append(self._queue.popleft())
            return tasks

    def queue_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.to_dict() for task in self._queue]

    def seen_urls(self) -> list[str]:
        with self._lock:
            return sorted(self._seen_urls)

    def add_seen_urls(self, urls: list[str]) -> None:
        with self._lock:
            self._seen_urls.update(urls)

    def expand_task(
        self,
        task: ExpansionTask,
        *,
        run_id: str | None = None,
    ) -> NodeExpansionResult:
        total_started = time.perf_counter()
        timing: dict[str, float] = {}
        try:
            started = time.perf_counter()
            text_result = self.wiki_builder.build_from_url(
                task.url,
                title=task.title,
                run_id=run_id,
                persist=self.config.persist,
            )
            timing["text_build_s"] = time.perf_counter() - started
            for key, value in text_result.timing.items():
                timing[f"text_{key}"] = value

            started = time.perf_counter()
            materialized_edges = self._materialize_pending_parent_links(
                task,
                target_result=text_result,
                run_id=run_id,
            )
            timing["materialize_parent_edges_s"] = time.perf_counter() - started

            started = time.perf_counter()
            attribute_evidence, attribute_error = self._extract_attributes(
                text_result,
                run_id=run_id,
            )
            timing["attribute_s"] = time.perf_counter() - started

            started = time.perf_counter()
            queued_tasks, existing_target_edges = self._process_text_neighbors(
                text_result,
                depth=task.depth + 1,
                run_id=run_id,
            )
            materialized_edges.extend(existing_target_edges)
            timing["queue_neighbors_s"] = time.perf_counter() - started

            started = time.perf_counter()
            visual_plans, image_results = self._expand_images(text_result, run_id=run_id)
            timing["image_expansion_s"] = time.perf_counter() - started
            timing["total_s"] = time.perf_counter() - total_started
            task.status = ExpansionTaskStatus.DONE
            return NodeExpansionResult(
                task=task,
                text_result=text_result,
                attribute_evidence=attribute_evidence,
                attribute_error=attribute_error,
                visual_plans=visual_plans,
                image_results=image_results,
                materialized_edges=materialized_edges,
                queued_tasks=queued_tasks,
                timing=timing,
            )
        except InvalidWikiPageError as exc:
            task.status = ExpansionTaskStatus.SKIPPED
            timing["total_s"] = time.perf_counter() - total_started
            return NodeExpansionResult(
                task=task,
                error=None,
                timing=timing,
                attribute_error=f"{exc.__class__.__name__}: {exc}",
            )
        except Exception as exc:
            task.status = ExpansionTaskStatus.FAILED
            timing["total_s"] = time.perf_counter() - total_started
            return NodeExpansionResult(
                task=task,
                error=f"{exc.__class__.__name__}: {exc}",
                timing=timing,
            )

    def _extract_attributes(
        self,
        text_result: WikiTextBuildResult,
        *,
        run_id: str | None,
    ) -> tuple[Evidence | None, str | None]:
        if not self.config.extract_attributes:
            return None, None
        try:
            evidence = self.wiki_builder.extract_attributes(
                text_result.node,
                source_evidence_ids=[text_result.text_evidence.evidence_id],
                run_id=run_id,
                persist=self.config.persist,
            )
            return evidence, None
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            if self.config.attribute_errors_fatal:
                raise
            text_result.node.metadata = dict(text_result.node.metadata or {})
            text_result.node.metadata["attribute_error"] = error
            if self.config.persist:
                self.store.upsert_node(text_result.node)
            return None, error

    def _process_text_neighbors(
        self,
        text_result: WikiTextBuildResult,
        *,
        depth: int,
        run_id: str | None,
    ) -> tuple[list[ExpansionTask], list[Edge]]:
        if depth > self.config.max_depth:
            return [], []

        queued: list[ExpansionTask] = []
        materialized_edges: list[Edge] = []
        for candidate in text_result.linked_entities[: self.config.max_new_text_neighbors]:
            if self.store.get_node(candidate.node_id) is not None:
                edge = self._materialize_edge_to_existing_node(
                    source_result=text_result,
                    candidate=candidate,
                    run_id=run_id,
                )
                if edge is not None:
                    materialized_edges.append(edge)
                continue

            task = ExpansionTask.from_wiki_link(
                candidate,
                depth=depth,
                parent_node_id=text_result.node.node_id,
                source_evidence_id=text_result.text_evidence.evidence_id,
            )
            if self.enqueue(task):
                queued.append(task)
            else:
                self._append_pending_link_to_queued_task(
                    candidate.url,
                    {
                        "parent_node_id": text_result.node.node_id,
                        "source_evidence_id": text_result.text_evidence.evidence_id,
                        "candidate": candidate.to_dict(),
                    },
                )
        return queued, materialized_edges

    def _materialize_pending_parent_links(
        self,
        task: ExpansionTask,
        *,
        target_result: WikiTextBuildResult,
        run_id: str | None,
    ) -> list[Edge]:
        pending_links = list(task.metadata.get("pending_parent_links") or [])
        with self._lock:
            pending_links.extend(self._pending_parent_links_by_url.pop(task.url, []))
        materialized: list[Edge] = []
        for pending in pending_links:
            edge = self._materialize_pending_parent_link(
                pending,
                target_result=target_result,
                run_id=run_id,
            )
            if edge is not None:
                materialized.append(edge)
        return materialized

    def _materialize_pending_parent_link(
        self,
        pending: dict[str, Any],
        *,
        target_result: WikiTextBuildResult,
        run_id: str | None,
    ) -> Edge | None:
        parent_node_id = pending.get("parent_node_id")
        source_evidence_id = pending.get("source_evidence_id")
        candidate_record = pending.get("candidate")
        if not parent_node_id or not source_evidence_id or not isinstance(candidate_record, dict):
            return None

        source_node_record = self.store.get_node(parent_node_id)
        source_evidence_record = self.store.get_evidence(source_evidence_id)
        if source_node_record is None or source_evidence_record is None:
            return None

        candidate = self._candidate_from_record(candidate_record)
        if candidate.node_id != target_result.node.node_id:
            candidate = WikiLinkCandidate(
                title=target_result.node.title or candidate.title,
                url=target_result.node.source.url if target_result.node.source and target_result.node.source.url else candidate.url,
                anchor_text=candidate.anchor_text,
                source_url=candidate.source_url,
                context=candidate.context,
                rank=candidate.rank,
                start_char=candidate.start_char,
                end_char=candidate.end_char,
                window_id=candidate.window_id,
                score=candidate.score,
                quality_reasons=list(candidate.quality_reasons),
            )

        source_node = WikiTextBuilder._text_node_from_record(source_node_record)
        source_evidence = WikiTextBuilder._evidence_from_record(source_evidence_record)
        edge = self.wiki_builder._edge_to_linked_entity(
            source_node,
            candidate,
            source_evidence,
            run_id=run_id,
        )
        if self.config.persist:
            self.store.upsert_edge(edge)
        return edge

    def _materialize_edge_to_existing_node(
        self,
        *,
        source_result: WikiTextBuildResult,
        candidate: WikiLinkCandidate,
        run_id: str | None,
    ) -> Edge | None:
        edge = self.wiki_builder._edge_to_linked_entity(
            source_result.node,
            candidate,
            source_result.text_evidence,
            run_id=run_id,
        )
        if self.config.persist:
            self.store.upsert_edge(edge)
        return edge

    def _append_pending_link_to_queued_task(
        self,
        url: str,
        pending_link: dict[str, Any],
    ) -> bool:
        with self._lock:
            for task in self._queue:
                if task.url != url:
                    continue
                links = list(task.metadata.get("pending_parent_links") or [])
                key = (
                    pending_link.get("parent_node_id"),
                    pending_link.get("source_evidence_id"),
                    (pending_link.get("candidate") or {}).get("url"),
                )
                for existing in links:
                    existing_key = (
                        existing.get("parent_node_id"),
                        existing.get("source_evidence_id"),
                        (existing.get("candidate") or {}).get("url"),
                    )
                    if existing_key == key:
                        return False
                links.append(pending_link)
                task.metadata["pending_parent_links"] = links
                return True
            links = self._pending_parent_links_by_url.setdefault(url, [])
            key = (
                pending_link.get("parent_node_id"),
                pending_link.get("source_evidence_id"),
                (pending_link.get("candidate") or {}).get("url"),
            )
            for existing in links:
                existing_key = (
                    existing.get("parent_node_id"),
                    existing.get("source_evidence_id"),
                    (existing.get("candidate") or {}).get("url"),
                )
                if existing_key == key:
                    return False
            links.append(pending_link)
            return True

    @staticmethod
    def _candidate_from_record(record: dict[str, Any]) -> WikiLinkCandidate:
        return WikiLinkCandidate(
            title=record["title"],
            url=record["url"],
            anchor_text=record.get("anchor_text") or record.get("title") or "",
            source_url=record.get("source_url") or "",
            context=record.get("context"),
            rank=record.get("rank"),
            start_char=record.get("start_char"),
            end_char=record.get("end_char"),
            window_id=record.get("window_id"),
            score=float(record.get("score") or 0.0),
            quality_reasons=list(record.get("quality_reasons") or []),
        )

    def _expand_images(
        self,
        text_result: WikiTextBuildResult,
        *,
        run_id: str | None,
    ) -> tuple[list[VisualSearchPlan], list[ImageDiscoveryResult]]:
        if not self.config.enable_image_expansion:
            return [], []
        if self.visual_planner is None or self.image_builder is None:
            return [], []

        plans = self.visual_planner.plan(
            node=text_result.node.to_dict(),
            page_text=text_result.node.description or "",
            source_evidence_ids=[text_result.text_evidence.evidence_id],
            run_id=run_id,
        )
        image_results = [
            self.image_builder.discover_for_plan(
                plan,
                run_id=run_id,
                persist=self.config.persist,
            )
            for plan in plans
        ]
        return plans, image_results


def _smoke_test() -> None:
    import tempfile

    from .edges import Edge, EdgeType
    from .evidence import EvidenceType, SearchEngine, SearchSnapshot
    from .nodes import TextNode

    class MockWikiBuilder:
        def __init__(self, store: JsonlGraphStore) -> None:
            self.store = store

        def build_from_url(
            self,
            url: str,
            *,
            title: str | None = None,
            run_id: str | None = None,
            persist: bool = True,
        ) -> WikiTextBuildResult:
            page_title = title or ("Neighbor" if url.endswith("/Neighbor") else "Seed")
            node = TextNode.from_webpage(url, title=page_title, description=f"{page_title} page")
            evidence = Evidence.create(
                EvidenceType.WEB_TEXT,
                content=f"{page_title} page",
                node_ids=[node.node_id],
                url=url,
                evidence_key=f"text:{url}",
            )
            snapshot = SearchSnapshot.create(
                SearchEngine.JINA_READER,
                query=url,
                request={"url": url},
                run_id=run_id,
            )
            linked_entities = []
            if not url.endswith("/Neighbor"):
                linked_entities = [
                    WikiLinkCandidate(
                        title="Neighbor",
                        url="https://en.wikipedia.org/wiki/Neighbor",
                        anchor_text="Neighbor",
                        source_url=url,
                        rank=1,
                    )
                ]
            if persist:
                self.store.upsert_node(node)
                self.store.upsert_evidence(evidence)
                self.store.upsert_search_snapshot(snapshot)
            return WikiTextBuildResult(
                node=node,
                text_evidence=evidence,
                snapshot=snapshot,
                linked_entities=linked_entities,
                edges=[],
            )

        def _edge_to_linked_entity(
            self,
            source_node: TextNode,
            candidate: WikiLinkCandidate,
            evidence: Evidence,
            *,
            run_id: str | None = None,
        ) -> Edge:
            del evidence, run_id
            return Edge.create(
                source_node.node_id,
                candidate.node_id,
                edge_type=EdgeType.WIKI_LINK,
                relation=candidate.anchor_text,
            )

        def extract_attributes(
            self,
            node: TextNode,
            *,
            source_evidence_ids: list[str] | None = None,
            run_id: str | None = None,
            persist: bool = True,
        ) -> Evidence:
            del source_evidence_ids, run_id
            node.attributes["mock"] = "yes"
            evidence = Evidence.create(EvidenceType.LLM_OUTPUT, content="{}", node_ids=[node.node_id])
            if persist:
                self.store.upsert_node(node)
                self.store.upsert_evidence(evidence)
            return evidence

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlGraphStore(tmpdir)
        strategy = GraphExpansionStrategy(
            store=store,
            wiki_builder=MockWikiBuilder(store),
            config=GraphExpansionConfig(max_depth=1, max_new_text_neighbors=1, enable_image_expansion=False),
        )
        strategy.add_seed("https://en.wikipedia.org/wiki/Seed")
        result = strategy.expand_next(run_id="run_smoke")
        assert result is not None
        assert result.attribute_evidence is not None
        assert result.queued_tasks[0].title == "Neighbor"
        assert strategy.queue_size() == 1
        assert store.stats()["edges"] == 0
        child_result = strategy.expand_next(run_id="run_smoke")
        assert child_result is not None
        assert len(child_result.materialized_edges) == 1
        assert store.stats()["edges"] == 1
    print("graph_expansion smoke test passed")


if __name__ == "__main__":
    _smoke_test()
