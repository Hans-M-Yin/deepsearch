"""Core graph expansion strategy for synthesis data construction."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .evidence import Evidence
from .image_discovery import ImageDiscoveryBuilder, ImageDiscoveryResult
from .store import JsonlGraphStore
from .visual_planner import VisualSearchPlan, VisualSearchPlanner
from .wiki_text_builder import WikiLinkCandidate, WikiTextBuilder, WikiTextBuildResult


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
        parent_edge_id: str | None = None,
    ) -> "ExpansionTask":
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
    queued_tasks: list[ExpansionTask] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "text_result": self.text_result.to_dict() if self.text_result else None,
            "attribute_evidence": self.attribute_evidence.to_dict() if self.attribute_evidence else None,
            "attribute_error": self.attribute_error,
            "visual_plans": [plan.to_dict() for plan in self.visual_plans],
            "image_results": [result.to_dict() for result in self.image_results],
            "queued_tasks": [task.to_dict() for task in self.queued_tasks],
            "error": self.error,
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
        if task.url in self._seen_urls:
            return False
        self._seen_urls.add(task.url)
        self._queue.append(task)
        return True

    def queue_size(self) -> int:
        return len(self._queue)

    def expand_next(self, *, run_id: str | None = None) -> NodeExpansionResult | None:
        if not self._queue:
            return None
        task = self._queue.popleft()
        return self.expand_task(task, run_id=run_id)

    def expand_task(
        self,
        task: ExpansionTask,
        *,
        run_id: str | None = None,
    ) -> NodeExpansionResult:
        try:
            text_result = self.wiki_builder.build_from_url(
                task.url,
                title=task.title,
                run_id=run_id,
                persist=self.config.persist,
            )
            attribute_evidence, attribute_error = self._extract_attributes(
                text_result,
                run_id=run_id,
            )
            queued_tasks = self._queue_text_neighbors(text_result, depth=task.depth + 1)
            visual_plans, image_results = self._expand_images(text_result, run_id=run_id)
            task.status = ExpansionTaskStatus.DONE
            return NodeExpansionResult(
                task=task,
                text_result=text_result,
                attribute_evidence=attribute_evidence,
                attribute_error=attribute_error,
                visual_plans=visual_plans,
                image_results=image_results,
                queued_tasks=queued_tasks,
            )
        except Exception as exc:
            task.status = ExpansionTaskStatus.FAILED
            return NodeExpansionResult(task=task, error=f"{exc.__class__.__name__}: {exc}")

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

    def _queue_text_neighbors(
        self,
        text_result: WikiTextBuildResult,
        *,
        depth: int,
    ) -> list[ExpansionTask]:
        if depth > self.config.max_depth:
            return []

        queued: list[ExpansionTask] = []
        edge_by_dst = {edge.dst_node_id: edge.edge_id for edge in text_result.edges}
        for candidate in text_result.linked_entities[: self.config.max_new_text_neighbors]:
            task = ExpansionTask.from_wiki_link(
                candidate,
                depth=depth,
                parent_node_id=text_result.node.node_id,
                parent_edge_id=edge_by_dst.get(candidate.node_id),
            )
            if self.enqueue(task):
                queued.append(task)
        return queued

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
            node = TextNode.from_webpage(url, title=title or "Seed", description="Seed page")
            evidence = Evidence.create(
                EvidenceType.WEB_TEXT,
                content="Seed page",
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
            candidate = WikiLinkCandidate(
                title="Neighbor",
                url="https://en.wikipedia.org/wiki/Neighbor",
                anchor_text="Neighbor",
                source_url=url,
                rank=1,
            )
            edge = Edge.create(
                node.node_id,
                candidate.node_id,
                edge_type=EdgeType.WIKI_LINK,
                relation="Neighbor",
            )
            if persist:
                self.store.upsert_node(node)
                self.store.upsert_evidence(evidence)
                self.store.upsert_search_snapshot(snapshot)
                self.store.upsert_edge(edge)
            return WikiTextBuildResult(
                node=node,
                text_evidence=evidence,
                snapshot=snapshot,
                linked_entities=[candidate],
                edges=[edge],
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
    print("graph_expansion smoke test passed")


if __name__ == "__main__":
    _smoke_test()
