"""Core graph expansion strategy for synthesis data construction."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import os
from pathlib import Path
import random
import sys
from threading import RLock
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .evidence import Evidence, EvidenceType
from .edges import Edge, EdgeSource, EdgeType, EvidenceRef
from .image_discovery import ImageDiscoveryBuilder, ImageDiscoveryResult
from .store import JsonlGraphStore
from .visual_planner import SearchQuerySpec, VisualSearchPlan, VisualSearchPlanner
from .wiki_text_builder import (
    InvalidWikiPageError,
    RawMarkdownReaderClient,
    WikiInlineImageCandidate,
    WikiLinkCandidate,
    WikiTextBuildResult,
    WikiTextBuilder,
)


def _trace_timing_enabled() -> bool:
    return os.environ.get("SYNTHESIS_TRACE_TIMING", "0") != "0"


def _trace_timing(message: str) -> None:
    if _trace_timing_enabled():
        print(f"[trace]{message}", file=sys.stderr, flush=True)


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


class ExpansionTaskType(str, Enum):
    TEXT_EXPAND = "text_expand"
    IMAGE_EXPAND = "image_expand"


@dataclass(slots=True)
class ExpansionTask:
    """A queued text node/page waiting to be built."""

    url: str
    task_type: ExpansionTaskType = ExpansionTaskType.TEXT_EXPAND
    depth: int = 0
    title: str | None = None
    parent_node_id: str | None = None
    parent_edge_id: str | None = None
    priority: float = 0.0
    status: ExpansionTaskStatus = ExpansionTaskStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    def dedupe_key(self) -> str:
        if self.task_type == ExpansionTaskType.IMAGE_EXPAND:
            source_node_id = (self.metadata or {}).get("source_text_node_id")
            if source_node_id:
                return f"{self.task_type.value}:{source_node_id}"
            return f"{self.task_type.value}:{self.url}"
        return f"{self.task_type.value}:{self.url}"

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
            task_type=ExpansionTaskType.TEXT_EXPAND,
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

    @classmethod
    def from_image_entity(
        cls,
        *,
        url: str,
        title: str | None,
        parent_image_node_id: str,
        source_evidence_id: str,
        entity: dict[str, Any],
    ) -> "ExpansionTask":
        pending_parent_link = {
            "link_type": "image_entity",
            "parent_node_id": parent_image_node_id,
            "source_evidence_id": source_evidence_id,
            "entity": entity,
            "resolved_target": {"url": url, "title": title},
        }
        return cls(
            url=url,
            task_type=ExpansionTaskType.TEXT_EXPAND,
            depth=0,
            title=title,
            parent_node_id=parent_image_node_id,
            priority=-1.0,
            metadata={
                "pending_parent_links": [pending_parent_link],
                "task_origin": "image_entity",
                "entity_name": entity.get("name"),
                "entity_type": entity.get("type"),
            },
        )

    @classmethod
    def from_image_expansion(
        cls,
        *,
        url: str,
        title: str | None,
        depth: int,
        source_text_node_id: str,
        source_evidence_id: str,
    ) -> "ExpansionTask":
        return cls(
            url=url,
            task_type=ExpansionTaskType.IMAGE_EXPAND,
            depth=depth,
            title=title,
            parent_node_id=source_text_node_id,
            priority=1.0,
            metadata={
                "task_origin": "image_expand",
                "source_text_node_id": source_text_node_id,
                "source_evidence_id": source_evidence_id,
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
    visual_plan_trace: dict[str, Any] = field(default_factory=dict)
    image_results: list[ImageDiscoveryResult] = field(default_factory=list)
    materialized_edges: list[Edge] = field(default_factory=list)
    parent_link_failures: list[dict[str, Any]] = field(default_factory=list)
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
            "visual_plan_trace": _jsonify(self.visual_plan_trace),
            "image_results": [result.to_dict() for result in self.image_results],
            "materialized_edges": [edge.to_dict() for edge in self.materialized_edges],
            "parent_link_failures": [dict(item) for item in self.parent_link_failures],
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
    max_wiki_inline_images_per_page: int = 3
    wiki_inline_random_seed: str = "wiki_inline_page_cap_v1"
    queue_pop_strategy: str = "fifo"
    queue_pop_random_seed: str = "graph_expansion_queue_v1"


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
        self._seen_task_keys: set[str] = set()
        self._pending_parent_links_by_url: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()
        self._raw_markdown_reader = RawMarkdownReaderClient(
            base_url=os.environ.get("WIKI_INLINE_IMAGE_READER_BASE_URL", "http://127.0.0.1:8003"),
            timeout_s=float(os.environ.get("WIKI_INLINE_IMAGE_READER_TIMEOUT_S") or 180.0),
        )

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
            task_key = task.dedupe_key()
            if task_key in self._seen_task_keys:
                return False
            self._seen_task_keys.add(task_key)
            pending_links = self._pending_parent_links_by_url.pop(task.url, [])
            if pending_links:
                links = list(task.metadata.get("pending_parent_links") or [])
                links.extend(pending_links)
                task.metadata["pending_parent_links"] = links
            self._queue.append(task)
            return True

    def queue_size(self, task_type: ExpansionTaskType | None = None) -> int:
        with self._lock:
            if task_type is not None:
                return sum(1 for task in self._queue if task.task_type == task_type)
            return len(self._queue)

    def expand_next(
        self,
        *,
        run_id: str | None = None,
        allowed_task_types: set[ExpansionTaskType] | None = None,
    ) -> NodeExpansionResult | None:
        task = self.pop_next_task(allowed_task_types=allowed_task_types)
        if task is None:
            return None
        return self.expand_task(task, run_id=run_id)

    def pop_next_task(
        self,
        *,
        allowed_task_types: set[ExpansionTaskType] | None = None,
        text_task_origin: str | None = None,
        exclude_text_task_origin: str | None = None,
        root_text_only: bool = False,
    ) -> ExpansionTask | None:
        with self._lock:
            if not self._queue:
                return None
            if allowed_task_types is None:
                allowed_task_types = {ExpansionTaskType.TEXT_EXPAND, ExpansionTaskType.IMAGE_EXPAND}
            eligible_indices: list[int] = []
            for index, task in enumerate(self._queue):
                if task.task_type in allowed_task_types:
                    metadata = task.metadata or {}
                    task_origin = metadata.get("task_origin")
                    if text_task_origin is not None:
                        if task.task_type != ExpansionTaskType.TEXT_EXPAND or task_origin != text_task_origin:
                            continue
                    if exclude_text_task_origin is not None:
                        if task.task_type == ExpansionTaskType.TEXT_EXPAND and task_origin == exclude_text_task_origin:
                            continue
                    if root_text_only:
                        if (
                            task.task_type != ExpansionTaskType.TEXT_EXPAND
                            or task.parent_node_id is not None
                            or task_origin is not None
                        ):
                            continue
                    eligible_indices.append(index)
            if not eligible_indices:
                return None
            selected_index = self._select_queue_index(eligible_indices)
            task = self._queue[selected_index]
            del self._queue[selected_index]
            return task

    def pop_next_batch(
        self,
        batch_size: int,
        *,
        allowed_task_types: set[ExpansionTaskType] | None = None,
    ) -> list[ExpansionTask]:
        with self._lock:
            tasks: list[ExpansionTask] = []
            limit = max(1, int(batch_size))
            if allowed_task_types is None:
                while self._queue and len(tasks) < limit:
                    if self.config.queue_pop_strategy == "random":
                        selected_index = self._select_queue_index(list(range(len(self._queue))))
                        task = self._queue[selected_index]
                        del self._queue[selected_index]
                        tasks.append(task)
                    else:
                        tasks.append(self._queue.popleft())
                return tasks
            index = 0
            while index < len(self._queue) and len(tasks) < limit:
                task = self._queue[index]
                if task.task_type in allowed_task_types:
                    tasks.append(task)
                    del self._queue[index]
                    continue
                index += 1
            return tasks

    def _select_queue_index(self, eligible_indices: list[int]) -> int:
        if (self.config.queue_pop_strategy or "fifo").lower() != "random":
            return eligible_indices[0]
        seed = str(self.config.queue_pop_random_seed or "graph_expansion_queue_v1")
        best_index = eligible_indices[0]
        best_key: str | None = None
        for index in eligible_indices:
            task = self._queue[index]
            digest = sha256(f"{seed}||{task.dedupe_key()}||{index}".encode("utf-8")).hexdigest()
            if best_key is None or digest < best_key:
                best_key = digest
                best_index = index
        return best_index

    def queue_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.to_dict() for task in self._queue]

    def seen_task_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._seen_task_keys)

    def add_seen_task_keys(self, keys: list[str]) -> None:
        with self._lock:
            self._seen_task_keys.update(keys)

    def expand_task(
        self,
        task: ExpansionTask,
        *,
        run_id: str | None = None,
    ) -> NodeExpansionResult:
        if task.task_type == ExpansionTaskType.IMAGE_EXPAND:
            return self._expand_image_task(task, run_id=run_id)

        total_started = time.perf_counter()
        timing: dict[str, float] = {}
        _trace_timing(f"[expand-task] phase=start url={task.url!r} title={task.title!r} depth={task.depth}")
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
            _trace_timing(
                f"[expand-task] stage=text_build url={task.url!r} elapsed_s={timing['text_build_s']:.3f} node_id={text_result.node.node_id!r}"
            )

            started = time.perf_counter()
            materialized_edges, parent_link_failures = self._materialize_pending_parent_links(
                task,
                target_result=text_result,
                run_id=run_id,
            )
            timing["materialize_parent_edges_s"] = time.perf_counter() - started
            _trace_timing(
                f"[expand-task] stage=materialize_parent_edges url={task.url!r} elapsed_s={timing['materialize_parent_edges_s']:.3f} edges={len(materialized_edges)} failures={len(parent_link_failures)}"
            )

            started = time.perf_counter()
            attribute_evidence, attribute_error = self._extract_attributes(
                text_result,
                run_id=run_id,
            )
            timing["attribute_s"] = time.perf_counter() - started
            _trace_timing(
                f"[expand-task] stage=attribute_extract url={task.url!r} elapsed_s={timing['attribute_s']:.3f} status={'error' if attribute_error else 'ok'}"
            )

            started = time.perf_counter()
            queued_tasks, existing_target_edges = self._process_text_neighbors(
                text_result,
                depth=task.depth + 1,
                run_id=run_id,
            )
            materialized_edges.extend(existing_target_edges)
            timing["queue_neighbors_s"] = time.perf_counter() - started
            _trace_timing(
                f"[expand-task] stage=neighbor_expand url={task.url!r} elapsed_s={timing['queue_neighbors_s']:.3f} queued={len(queued_tasks)} existing_edges={len(existing_target_edges)}"
            )

            started = time.perf_counter()
            image_task = self._enqueue_image_expansion_task(text_result, depth=task.depth)
            if image_task is not None:
                queued_tasks.append(image_task)
            timing["image_enqueue_s"] = time.perf_counter() - started
            _trace_timing(
                f"[expand-task] stage=image_enqueue url={task.url!r} elapsed_s={timing['image_enqueue_s']:.3f} queued={'yes' if image_task is not None else 'no'}"
            )
            timing["total_s"] = time.perf_counter() - total_started
            _trace_timing(f"[expand-task] phase=done url={task.url!r} elapsed_s={timing['total_s']:.3f}")
            task.status = ExpansionTaskStatus.DONE
            return NodeExpansionResult(
                task=task,
                text_result=text_result,
                attribute_evidence=attribute_evidence,
                attribute_error=attribute_error,
                visual_plans=[],
                image_results=[],
                materialized_edges=materialized_edges,
                parent_link_failures=parent_link_failures,
                queued_tasks=queued_tasks,
                timing=timing,
            )
        except InvalidWikiPageError as exc:
            task.status = ExpansionTaskStatus.SKIPPED
            timing["total_s"] = time.perf_counter() - total_started
            _trace_timing(f"[expand-task] phase=skipped url={task.url!r} elapsed_s={timing['total_s']:.3f} error={exc}")
            return NodeExpansionResult(
                task=task,
                error=None,
                timing=timing,
                attribute_error=f"{exc.__class__.__name__}: {exc}",
            )
        except Exception as exc:
            task.status = ExpansionTaskStatus.FAILED
            timing["total_s"] = time.perf_counter() - total_started
            _trace_timing(f"[expand-task] phase=failed url={task.url!r} elapsed_s={timing['total_s']:.3f} error={exc.__class__.__name__}: {exc}")
            return NodeExpansionResult(
                task=task,
                error=f"{exc.__class__.__name__}: {exc}",
                timing=timing,
            )

    def _expand_image_task(
        self,
        task: ExpansionTask,
        *,
        run_id: str | None = None,
    ) -> NodeExpansionResult:
        total_started = time.perf_counter()
        timing: dict[str, float] = {}
        _trace_timing(f"[expand-image-task] phase=start url={task.url!r} title={task.title!r} depth={task.depth}")
        try:
            started = time.perf_counter()
            text_result = self.wiki_builder.build_from_url(
                task.url,
                title=task.title,
                run_id=run_id,
                persist=self.config.persist,
            )
            timing["image_source_load_s"] = time.perf_counter() - started
            for key, value in text_result.timing.items():
                timing[f"image_source_{key}"] = value
            _trace_timing(
                f"[expand-image-task] stage=source_load url={task.url!r} elapsed_s={timing['image_source_load_s']:.3f} node_id={text_result.node.node_id!r} cache={'yes' if text_result.from_cache else 'no'}"
            )

            started = time.perf_counter()
            visual_plans, image_results, queued_tasks, visual_plan_trace = self._expand_images(text_result, run_id=run_id)
            timing["image_expansion_s"] = time.perf_counter() - started
            _trace_timing(
                f"[expand-image-task] stage=image_expand url={task.url!r} elapsed_s={timing['image_expansion_s']:.3f} plans={len(visual_plans)} image_nodes={sum(1 for item in image_results if item.image_node is not None)}"
            )
            timing["total_s"] = time.perf_counter() - total_started
            _trace_timing(f"[expand-image-task] phase=done url={task.url!r} elapsed_s={timing['total_s']:.3f}")
            task.status = ExpansionTaskStatus.DONE
            return NodeExpansionResult(
                task=task,
                text_result=text_result,
                visual_plans=visual_plans,
                visual_plan_trace=visual_plan_trace,
                image_results=image_results,
                materialized_edges=[],
                parent_link_failures=[],
                queued_tasks=queued_tasks,
                timing=timing,
            )
        except Exception as exc:
            task.status = ExpansionTaskStatus.FAILED
            timing["total_s"] = time.perf_counter() - total_started
            _trace_timing(
                f"[expand-image-task] phase=failed url={task.url!r} elapsed_s={timing['total_s']:.3f} error={exc.__class__.__name__}: {exc}"
            )
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
    ) -> tuple[list[Edge], list[dict[str, Any]]]:
        pending_links = list(task.metadata.get("pending_parent_links") or [])
        with self._lock:
            pending_links.extend(self._pending_parent_links_by_url.pop(task.url, []))
        materialized: list[Edge] = []
        failures: list[dict[str, Any]] = []
        backlink_candidates = self._find_pending_parent_backlinks(
            pending_links,
            source_result=target_result,
        )
        existing_backlink_targets = (
            {
                edge.get("dst_node_id")
                for edge in self.store.edges_from(target_result.node.node_id)
            }
            if backlink_candidates
            else set()
        )
        for pending in pending_links:
            edge = self._materialize_pending_parent_link(
                pending,
                target_result=target_result,
                run_id=run_id,
            )
            if edge is not None:
                materialized.append(edge)
            else:
                diagnostic = self._diagnose_pending_parent_link_failure(
                    pending,
                    target_result=target_result,
                )
                if diagnostic is not None:
                    failures.append(diagnostic)

            parent_node_id = pending.get("parent_node_id")
            candidate = backlink_candidates.get(parent_node_id)
            if candidate is None:
                continue
            if candidate.node_id in existing_backlink_targets:
                continue
            backlink = self.wiki_builder._edge_to_linked_entity(
                target_result.node,
                candidate,
                target_result.text_evidence,
                run_id=run_id,
            )
            if self.config.persist:
                self.store.upsert_edge(backlink)
            materialized.append(backlink)
            existing_backlink_targets.add(candidate.node_id)
        return materialized, failures

    def _find_pending_parent_backlinks(
        self,
        pending_links: list[dict[str, Any]],
        *,
        source_result: WikiTextBuildResult,
    ) -> dict[str, WikiLinkCandidate]:
        markdown = source_result.link_markdown or ""
        source_url = source_result.node.source.url if source_result.node.source else None
        if not markdown or not source_url:
            return {}

        parent_by_url: dict[str, tuple[str, dict[str, Any]]] = {}
        for pending in pending_links:
            if (pending.get("link_type") or "wiki_link") != "wiki_link":
                continue
            parent_node_id = pending.get("parent_node_id")
            if not parent_node_id:
                continue
            parent_node = self.store.get_node(parent_node_id)
            parent_source = parent_node.get("source") if parent_node else None
            parent_url = parent_source.get("url") if isinstance(parent_source, dict) else None
            normalized_url = WikiTextBuilder._normalize_wikipedia_url(parent_url) if parent_url else None
            if normalized_url and parent_node is not None:
                parent_by_url[normalized_url] = (parent_node_id, parent_node)
        if not parent_by_url:
            return {}

        matches: dict[str, WikiLinkCandidate] = {}
        for rank, (anchor_text, href, start, end) in enumerate(
            WikiTextBuilder._iter_markdown_links(markdown),
            start=1,
        ):
            target_url = WikiTextBuilder._wiki_url_from_href(href, source_url=source_url)
            if not target_url:
                continue
            normalized_url = WikiTextBuilder._normalize_wikipedia_url(target_url)
            parent = parent_by_url.get(normalized_url)
            if parent is None:
                continue
            parent_node_id, parent_node = parent
            matches[parent_node_id] = WikiLinkCandidate(
                title=parent_node.get("title") or WikiTextBuilder._title_from_url(normalized_url) or "",
                url=normalized_url,
                anchor_text=anchor_text.strip(),
                source_url=source_url,
                context=WikiTextBuilder._context(markdown, start, end),
                rank=rank,
                start_char=start,
                end_char=end,
            )
            if len(matches) == len(parent_by_url):
                break
        return matches

    def _diagnose_pending_parent_link_failure(
        self,
        pending: dict[str, Any],
        *,
        target_result: WikiTextBuildResult,
    ) -> dict[str, Any] | None:
        link_type = pending.get("link_type") or "wiki_link"
        target_node = target_result.node
        base = {
            "link_type": link_type,
            "target_node_id": target_node.node_id,
            "target_title": target_node.title,
            "target_url": target_node.source.url if target_node.source else None,
        }
        if link_type == "image_entity":
            parent_node_id = pending.get("parent_node_id")
            source_evidence_id = pending.get("source_evidence_id")
            entity = pending.get("entity")
            if not parent_node_id:
                return {
                    **base,
                    "reason": "missing_parent_image_node_id",
                    "source_evidence_id": source_evidence_id,
                }
            if not source_evidence_id:
                return {
                    **base,
                    "reason": "missing_source_image_evidence_id",
                    "parent_node_id": parent_node_id,
                }
            if not isinstance(entity, dict):
                return {
                    **base,
                    "reason": "invalid_grounded_entity_payload",
                    "parent_node_id": parent_node_id,
                    "source_evidence_id": source_evidence_id,
                }
            if self.store.get_node(parent_node_id) is None:
                return {
                    **base,
                    "reason": "missing_parent_image_node",
                    "parent_node_id": parent_node_id,
                    "source_evidence_id": source_evidence_id,
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                }
            if self.store.get_evidence(source_evidence_id) is None:
                return {
                    **base,
                    "reason": "missing_source_image_evidence",
                    "parent_node_id": parent_node_id,
                    "source_evidence_id": source_evidence_id,
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                }
            return {
                **base,
                "reason": "unknown_image_entity_parent_link_failure",
                "parent_node_id": parent_node_id,
                "source_evidence_id": source_evidence_id,
                "entity_name": entity.get("name"),
                "entity_type": entity.get("type"),
            }

        parent_node_id = pending.get("parent_node_id")
        source_evidence_id = pending.get("source_evidence_id")
        candidate_record = pending.get("candidate")
        if not parent_node_id:
            return {**base, "reason": "missing_parent_node_id", "source_evidence_id": source_evidence_id}
        if not source_evidence_id:
            return {**base, "reason": "missing_source_evidence_id", "parent_node_id": parent_node_id}
        if not isinstance(candidate_record, dict):
            return {
                **base,
                "reason": "invalid_candidate_payload",
                "parent_node_id": parent_node_id,
                "source_evidence_id": source_evidence_id,
            }
        if self.store.get_node(parent_node_id) is None:
            return {
                **base,
                "reason": "missing_parent_text_node",
                "parent_node_id": parent_node_id,
                "source_evidence_id": source_evidence_id,
                "candidate_title": candidate_record.get("title"),
            }
        if self.store.get_evidence(source_evidence_id) is None:
            return {
                **base,
                "reason": "missing_source_text_evidence",
                "parent_node_id": parent_node_id,
                "source_evidence_id": source_evidence_id,
                "candidate_title": candidate_record.get("title"),
            }
        return {
            **base,
            "reason": "unknown_wiki_parent_link_failure",
            "parent_node_id": parent_node_id,
            "source_evidence_id": source_evidence_id,
            "candidate_title": candidate_record.get("title"),
        }

    def _materialize_pending_parent_link(
        self,
        pending: dict[str, Any],
        *,
        target_result: WikiTextBuildResult,
        run_id: str | None,
    ) -> Edge | None:
        link_type = pending.get("link_type") or "wiki_link"
        if link_type == "image_entity":
            edge = self._materialize_pending_image_entity_link(
                pending,
                target_node_record=target_result.node.to_dict(),
            )
            if edge is not None and self.config.persist:
                self.store.upsert_edge(edge)
            return edge

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

    def _materialize_pending_image_entity_link(
        self,
        pending: dict[str, Any],
        *,
        target_node_record: dict[str, Any],
    ) -> Edge | None:
        parent_node_id = pending.get("parent_node_id")
        source_evidence_id = pending.get("source_evidence_id")
        entity = pending.get("entity")
        if not parent_node_id or not source_evidence_id or not isinstance(entity, dict):
            return None
        source_node_record = self.store.get_node(parent_node_id)
        source_evidence_record = self.store.get_evidence(source_evidence_id)
        if source_node_record is None or source_evidence_record is None:
            return None
        relation = entity.get("relation_to_image") or "depicts"
        query_overlap_entity = bool(pending.get("query_overlap_entity"))
        return Edge.create(
            parent_node_id,
            target_node_record["node_id"],
            edge_type=EdgeType.IMAGE_DEPICTS,
            relation=relation,
            src_node_type=source_node_record.get("node_type"),
            dst_node_type=target_node_record.get("node_type"),
            evidence_refs=[
                EvidenceRef(
                    evidence_id=source_evidence_id,
                    quote=entity.get("evidence"),
                    metadata={
                        "grounded_entity": entity,
                        "resolved_target": pending.get("resolved_target"),
                        "query_overlap_entity": query_overlap_entity,
                    },
                )
            ],
            source=EdgeSource(
                source_type="image_grounding_delayed",
                url=(source_node_record.get("source") or {}).get("url") if isinstance(source_node_record.get("source"), dict) else None,
                builder="graph_expansion_strategy",
            ),
            extractor="graph_expansion_strategy",
            metadata={
                "entity_name": entity.get("name"),
                "entity_type": entity.get("type"),
                "link_type": "image_entity",
                "query_overlap_entity": query_overlap_entity,
            },
            evidence_key=f"{source_evidence_id}:{entity.get('name')}:{target_node_record['node_id']}",
        )

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
        def _pending_key(record: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
            candidate = record.get("candidate") or {}
            resolved_target = record.get("resolved_target") or {}
            entity = record.get("entity") or {}
            return (
                record.get("link_type") or "wiki_link",
                record.get("parent_node_id"),
                record.get("source_evidence_id"),
                candidate.get("url") or resolved_target.get("url") or entity.get("name"),
            )

        with self._lock:
            for task in self._queue:
                if task.url != url:
                    continue
                links = list(task.metadata.get("pending_parent_links") or [])
                key = _pending_key(pending_link)
                for existing in links:
                    existing_key = _pending_key(existing)
                    if existing_key == key:
                        return False
                links.append(pending_link)
                task.metadata["pending_parent_links"] = links
                return True
            links = self._pending_parent_links_by_url.setdefault(url, [])
            key = _pending_key(pending_link)
            for existing in links:
                existing_key = _pending_key(existing)
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
            relation=record.get("relation"),
            relation_info=dict(record.get("relation_info") or {}),
        )

    def _enqueue_image_expansion_task(
        self,
        text_result: WikiTextBuildResult,
        *,
        depth: int,
    ) -> ExpansionTask | None:
        if not self.config.enable_image_expansion:
            return None
        if self.visual_planner is None or self.image_builder is None:
            return None
        source_url = text_result.node.source.url if text_result.node.source else None
        if not source_url:
            return None
        task = ExpansionTask.from_image_expansion(
            url=source_url,
            title=text_result.node.title,
            depth=depth,
            source_text_node_id=text_result.node.node_id,
            source_evidence_id=text_result.text_evidence.evidence_id,
        )
        if self.enqueue(task):
            return task
        return None

    def _expand_images(
        self,
        text_result: WikiTextBuildResult,
        *,
        run_id: str | None,
    ) -> tuple[list[VisualSearchPlan], list[ImageDiscoveryResult], list[ExpansionTask], dict[str, Any]]:
        if not self.config.enable_image_expansion:
            return [], [], [], {"status": "image_expansion_disabled"}
        if self.visual_planner is None or self.image_builder is None:
            return [], [], [], {"status": "image_expansion_unavailable"}

        visual_plans = self.visual_planner.plan(
            node=text_result.node.to_dict(),
            page_text=text_result.node.description or "",
            source_evidence_ids=[text_result.text_evidence.evidence_id],
            run_id=run_id,
        )
        planner_trace = getattr(self.visual_planner, "last_plan_trace", {})
        visual_plan_trace = dict(planner_trace) if isinstance(planner_trace, dict) else {}
        wiki_plans = self._build_wiki_inline_image_plans(text_result, run_id=run_id)
        plans = list(visual_plans) + list(wiki_plans)
        self._log_image_plan_start(text_result, plans)
        image_results: list[ImageDiscoveryResult] = []
        for index, plan in enumerate(visual_plans, start=1):
            self._log_image_plan_execute_start(text_result, plan_index=index, plan=plan)
            try:
                image_result = self.image_builder.discover_for_plan(
                    plan,
                    run_id=run_id,
                    persist=self.config.persist,
                )
            except Exception as exc:
                self._log_image_plan_execute_failure(
                    text_result,
                    plan_index=index,
                    plan=plan,
                    error=exc,
                )
                raise
            image_results.append(image_result)
            self._log_image_plan_execute_done(
                text_result,
                plan_index=index,
                result=image_result,
            )
        image_results.extend(
            self._execute_wiki_inline_plans(
                text_result,
                wiki_plans,
                run_id=run_id,
                start_index=len(visual_plans) + 1,
            )
        )
        self._log_image_plan_results(text_result, plans, image_results)
        queued_tasks: list[ExpansionTask] = []
        for image_result in image_results:
            for pending in image_result.queued_tasks:
                task = self._enqueue_image_entity_task(pending)
                if task is not None:
                    queued_tasks.append(task)
        return plans, image_results, queued_tasks, visual_plan_trace

    def _execute_wiki_inline_plans(
        self,
        text_result: WikiTextBuildResult,
        wiki_plans: list[VisualSearchPlan],
        *,
        run_id: str | None,
        start_index: int,
    ) -> list[ImageDiscoveryResult]:
        if not wiki_plans:
            return []

        page_cap = int(self.config.max_wiki_inline_images_per_page)
        needs_two_stage = page_cap > 0 and len(wiki_plans) > page_cap
        provisional_results: list[tuple[int, VisualSearchPlan, Any, ImageDiscoveryResult]] = []

        for offset, plan in enumerate(wiki_plans, start=start_index):
            self._log_image_plan_execute_start(text_result, plan_index=offset, plan=plan)
            search_result = self._wiki_inline_image_search_result(plan)
            if search_result is None:
                continue
            try:
                image_result = self.image_builder.discover_for_wiki_inline_image(
                    plan,
                    search_result=search_result,
                    run_id=run_id,
                    persist=self.config.persist and not needs_two_stage,
                )
            except Exception as exc:
                self._log_image_plan_execute_failure(
                    text_result,
                    plan_index=offset,
                    plan=plan,
                    error=exc,
                )
                raise
            provisional_results.append((offset, plan, search_result, image_result))

        if not needs_two_stage:
            final_results = [result for _, _, _, result in provisional_results]
            for plan_index, _, _, result in provisional_results:
                self._log_image_plan_execute_done(
                    text_result,
                    plan_index=plan_index,
                    result=result,
                )
            return final_results

        accepted_positions = [
            position
            for position, (_, _, _, result) in enumerate(provisional_results)
            if result.image_node is not None
        ]
        kept_positions = self._sample_kept_wiki_inline_positions(
            text_result,
            accepted_positions=accepted_positions,
            page_cap=page_cap,
        )
        seed_text = self._wiki_inline_page_seed(text_result)
        final_results: list[ImageDiscoveryResult] = []

        for position, (plan_index, plan, search_result, result) in enumerate(provisional_results):
            final_result = result
            if position in kept_positions:
                if self.config.persist:
                    final_result = self.image_builder.discover_for_wiki_inline_image(
                        plan,
                        search_result=search_result,
                        run_id=run_id,
                        persist=True,
                    )
                self._annotate_wiki_inline_result(
                    final_result,
                    random_cap_applied=True,
                    selected_for_page_cap=True,
                    page_cap=page_cap,
                    accepted_count=len(accepted_positions),
                    selected_count=len(kept_positions),
                    seed_text=seed_text,
                    reason="selected_after_random_page_cap",
                )
            else:
                was_accepted = final_result.image_node is not None
                self._annotate_wiki_inline_result(
                    final_result,
                    random_cap_applied=True,
                    selected_for_page_cap=False,
                    page_cap=page_cap,
                    accepted_count=len(accepted_positions),
                    selected_count=len(kept_positions),
                    seed_text=seed_text,
                    reason=(
                        "dropped_after_random_page_cap"
                        if was_accepted
                        else "not_accepted_before_random_page_cap"
                    ),
                )
                if was_accepted:
                    final_result.image_node = None
                    final_result.edge = None
                    final_result.image_evidence = None
                    final_result.search_evidence = None
                    final_result.grounded_edges = []
                    final_result.queued_tasks = []
            final_results.append(final_result)
            self._log_image_plan_execute_done(
                text_result,
                plan_index=plan_index,
                result=final_result,
            )

        return final_results

    def _wiki_inline_page_seed(self, text_result: WikiTextBuildResult) -> str:
        return f"{self.config.wiki_inline_random_seed}:{text_result.node.node_id}"

    def _sample_kept_wiki_inline_positions(
        self,
        text_result: WikiTextBuildResult,
        *,
        accepted_positions: list[int],
        page_cap: int,
    ) -> set[int]:
        if page_cap <= 0 or len(accepted_positions) <= page_cap:
            return set(accepted_positions)
        rng = random.Random(self._wiki_inline_page_seed(text_result))
        sampled = rng.sample(accepted_positions, k=page_cap)
        return set(sampled)

    @staticmethod
    def _annotate_wiki_inline_result(
        result: ImageDiscoveryResult,
        *,
        random_cap_applied: bool,
        selected_for_page_cap: bool,
        page_cap: int,
        accepted_count: int,
        selected_count: int,
        seed_text: str,
        reason: str,
    ) -> None:
        result.metadata = dict(result.metadata or {})
        result.metadata.update(
            {
                "wiki_inline_random_cap_applied": random_cap_applied,
                "wiki_inline_random_cap_selected": selected_for_page_cap,
                "wiki_inline_random_cap_limit": page_cap,
                "wiki_inline_random_cap_accepted_count": accepted_count,
                "wiki_inline_random_cap_selected_count": selected_count,
                "wiki_inline_random_cap_seed": seed_text,
                "wiki_inline_random_cap_reason": reason,
            }
        )

    def _build_wiki_inline_image_plans(
        self,
        text_result: WikiTextBuildResult,
        *,
        run_id: str | None,
    ) -> list[VisualSearchPlan]:
        source_url = text_result.node.source.url if text_result.node.source else None
        if not source_url or "wikipedia.org" not in source_url:
            return []
        try:
            document = self._raw_markdown_reader.read(source_url)
        except Exception:
            return []
        markdown = document.raw_markdown or document.content or ""
        candidates = WikiTextBuilder.extract_wiki_inline_images(markdown, source_url=source_url)
        plans: list[VisualSearchPlan] = []
        for candidate in candidates:
            plan = self._wiki_inline_image_plan(text_result, candidate, run_id=run_id)
            if plan is not None:
                plans.append(plan)
        return plans

    @staticmethod
    def _wiki_inline_image_plan(
        text_result: WikiTextBuildResult,
        candidate: WikiInlineImageCandidate,
        *,
        run_id: str | None,
    ) -> VisualSearchPlan | None:
        caption = str(candidate.caption or "").strip()
        if not caption:
            return None
        target = Evidence.create(
            EvidenceType.VISUAL_TARGET,
            content=caption,
            node_ids=[text_result.node.node_id],
            url=candidate.image_url,
            extractor="wikipedia_inline_image_planner",
            metadata={
                "source_evidence_ids": [text_result.text_evidence.evidence_id],
                "run_id": run_id,
                "source_page_url": candidate.source_page_url,
                "file_page_url": candidate.file_page_url,
                "image_url": candidate.image_url,
                "thumbnail_url": candidate.thumbnail_url,
                "caption": caption,
                "rank": candidate.rank,
            },
            evidence_key=f"{text_result.node.node_id}:wiki_inline:{candidate.rank}:{candidate.image_url}",
        )
        query = SearchQuerySpec.create(
            caption,
            target.evidence_id,
            expected_visual=caption,
            source="wikipedia_inline_image",
            metadata={
                "source_page_url": candidate.source_page_url,
                "file_page_url": candidate.file_page_url,
                "image_url": candidate.image_url,
                "thumbnail_url": candidate.thumbnail_url,
                "rank": candidate.rank,
            },
        )
        return VisualSearchPlan.create(
            target,
            queries=[query],
            source_node_id=text_result.node.node_id,
            source_evidence_ids=[text_result.text_evidence.evidence_id],
            planner="wikipedia_inline_image_planner",
            metadata={
                "plan_source": "wikipedia_inline_image",
                "image_url": candidate.image_url,
                "thumbnail_url": candidate.thumbnail_url,
                "source_page_url": candidate.source_page_url,
                "file_page_url": candidate.file_page_url,
                "caption": caption,
                "raw_caption": candidate.raw_caption,
                "alt_text": candidate.alt_text,
                "rank": candidate.rank,
            },
        )

    @staticmethod
    def _is_wiki_inline_image_plan(plan: VisualSearchPlan) -> bool:
        return (plan.metadata or {}).get("plan_source") == "wikipedia_inline_image"

    @staticmethod
    def _wiki_inline_image_search_result(plan: VisualSearchPlan):
        from .search_client import ImageSearchResult

        metadata = plan.metadata or {}
        image_url = metadata.get("image_url")
        if not image_url:
            return None
        caption = metadata.get("caption") or plan.target.content
        source_page_url = metadata.get("source_page_url")
        file_page_url = metadata.get("file_page_url")
        title = metadata.get("alt_text") or caption or file_page_url or image_url
        return ImageSearchResult(
            title=title,
            image_url=image_url,
            source_page_url=source_page_url,
            thumbnail_url=metadata.get("thumbnail_url"),
            snippet=caption,
            source="wikipedia_inline",
            rank=metadata.get("rank"),
            raw={
                "file_page_url": file_page_url,
                "thumbnail_url": metadata.get("thumbnail_url"),
                "raw_caption": metadata.get("raw_caption"),
                "alt_text": metadata.get("alt_text"),
                "plan_source": "wikipedia_inline_image",
            },
        )

    @staticmethod
    def _log_image_plan_start(
        text_result: WikiTextBuildResult,
        plans: list[VisualSearchPlan],
    ) -> None:
        return

    @staticmethod
    def _log_image_plan_execute_start(
        text_result: WikiTextBuildResult,
        *,
        plan_index: int,
        plan: VisualSearchPlan,
    ) -> None:
        return

    @staticmethod
    def _log_image_plan_execute_done(
        text_result: WikiTextBuildResult,
        *,
        plan_index: int,
        result: ImageDiscoveryResult,
    ) -> None:
        return

    @staticmethod
    def _log_image_plan_execute_failure(
        text_result: WikiTextBuildResult,
        *,
        plan_index: int,
        plan: VisualSearchPlan,
        error: Exception,
    ) -> None:
        return

    @staticmethod
    def _log_image_plan_results(
        text_result: WikiTextBuildResult,
        plans: list[VisualSearchPlan],
        image_results: list[ImageDiscoveryResult],
    ) -> None:
        return

    def _enqueue_image_entity_task(self, pending: dict[str, Any]) -> ExpansionTask | None:
        url = pending.get("url")
        title = pending.get("title")
        pending_link = pending.get("pending_link")
        if not url or not isinstance(pending_link, dict):
            return None
        parent_image_node_id = pending_link.get("parent_node_id")
        source_evidence_id = pending_link.get("source_evidence_id")
        entity = pending_link.get("entity")
        if not parent_image_node_id or not source_evidence_id or not isinstance(entity, dict):
            return None

        existing_node = self._find_text_node_by_source_url(url)
        if existing_node is not None:
            edge = self._materialize_pending_image_entity_link(
                pending_link,
                target_node_record=existing_node,
            )
            if edge is not None and self.config.persist:
                self.store.upsert_edge(edge)
            return None

        task = ExpansionTask.from_image_entity(
            url=url,
            title=title,
            parent_image_node_id=parent_image_node_id,
            source_evidence_id=source_evidence_id,
            entity=entity,
        )
        if self.enqueue(task):
            return task
        self._append_pending_link_to_queued_task(url, pending_link)
        return None

    def _find_text_node_by_source_url(self, url: str) -> dict[str, Any] | None:
        return self.store.find_node_by_source_url(url, node_type="text")


def _smoke_test() -> None:
    import tempfile

    from .edges import Edge, EdgeType
    from .evidence import EvidenceType, SearchEngine, SearchSnapshot
    from .image_discovery import ImageDiscoveryResult
    from .nodes import ImageNode, NodeSource, TextNode
    from .visual_planner import SearchQuerySpec, VisualSearchPlan

    class MockWikiBuilder:
        def __init__(self, store: JsonlGraphStore) -> None:
            self.store = store
            self.read_calls = 0

        def build_from_url(
            self,
            url: str,
            *,
            title: str | None = None,
            run_id: str | None = None,
            persist: bool = True,
        ) -> WikiTextBuildResult:
            cached = self._cached_build_result(url)
            if cached is not None:
                return cached
            self.read_calls += 1
            page_title = title or ("Neighbor" if url.endswith("/Neighbor") else "Seed")
            node = TextNode(
                node_id=TextNode.make_id("wikipedia_page", url),
                subtype="wiki_page",
                title=page_title,
                description=f"{page_title} page",
                source=NodeSource(source_type="wikipedia", url=url),
            )
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
                link_markdown=(
                    "Neighbor has a documented connection to [the original seed](https://en.wikipedia.org/wiki/Seed)."
                    if url.endswith("/Neighbor")
                    else None
                ),
            )

        def _cached_build_result(self, page_url: str) -> WikiTextBuildResult | None:
            node_id = TextNode.make_id("wikipedia_page", page_url)
            node_record = self.store.get_node(node_id)
            if node_record is None:
                return None
            evidence_record = None
            for evidence in self.store.list_evidence():
                if evidence.get("evidence_type") != EvidenceType.WEB_TEXT.value:
                    continue
                if node_id in evidence.get("node_ids", []):
                    evidence_record = evidence
                    break
            if evidence_record is None:
                return None
            snapshot = SearchSnapshot.create(SearchEngine.JINA_READER, query=page_url)
            result = WikiTextBuildResult(
                node=WikiTextBuilder._text_node_from_record(node_record),
                text_evidence=WikiTextBuilder._evidence_from_record(evidence_record),
                snapshot=snapshot,
                linked_entities=[],
                edges=[],
                from_cache=True,
            )
            result.timing = {"total_s": 0.0}
            return result

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

    class MockVisualPlanner:
        def plan(
            self,
            *,
            node: dict[str, Any],
            page_text: str,
            source_evidence_ids: list[str] | None = None,
            run_id: str | None = None,
        ) -> list[VisualSearchPlan]:
            del page_text, run_id
            target = Evidence.create(
                EvidenceType.VISUAL_TARGET,
                content=f"visual target for {node.get('title')}",
                node_ids=[node.get("node_id")] if node.get("node_id") else [],
            )
            return [
                VisualSearchPlan.create(
                    target,
                    queries=[
                        SearchQuerySpec.create(
                            f"{node.get('title')} image",
                            target.evidence_id,
                        )
                    ],
                    source_node_id=node.get("node_id"),
                    source_evidence_ids=source_evidence_ids or [],
                )
            ]

    class MockImageBuilder:
        def discover_for_plan(
            self,
            plan: VisualSearchPlan,
            *,
            run_id: str | None = None,
            persist: bool = True,
        ) -> ImageDiscoveryResult:
            del run_id, persist
            return ImageDiscoveryResult(plan_id=plan.plan_id)

    class MockInlineImageBuilder(MockImageBuilder):
        def __init__(self) -> None:
            self.inline_calls: list[tuple[str, bool]] = []

        def discover_for_wiki_inline_image(
            self,
            plan: VisualSearchPlan,
            *,
            search_result,
            run_id: str | None = None,
            persist: bool = True,
        ) -> ImageDiscoveryResult:
            del run_id
            self.inline_calls.append((plan.plan_id, persist))
            result = ImageDiscoveryResult(plan_id=plan.plan_id)
            result.image_node = ImageNode.from_url(
                search_result.image_url or "https://example.com/inline.jpg",
                source_page_url=search_result.source_page_url,
                title=search_result.title,
                caption=search_result.snippet,
                metadata={"mock_inline": True},
            )
            result.metadata = {"mock_inline": True}
            return result

    class MockNoVisualPlanner:
        def plan(
            self,
            *,
            node: dict[str, Any],
            page_text: str,
            source_evidence_ids: list[str] | None = None,
            run_id: str | None = None,
        ) -> list[VisualSearchPlan]:
            del node, page_text, source_evidence_ids, run_id
            return []

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlGraphStore(tmpdir)
        wiki_builder = MockWikiBuilder(store)
        strategy = GraphExpansionStrategy(
            store=store,
            wiki_builder=wiki_builder,
            visual_planner=MockVisualPlanner(),
            image_builder=MockImageBuilder(),
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
        assert len(child_result.materialized_edges) == 2
        stored_edges = store.list_edges()
        assert len(stored_edges) == 2
        seed_node_id = TextNode.make_id("wikipedia_page", "https://en.wikipedia.org/wiki/Seed")
        neighbor_node_id = TextNode.make_id("wikipedia_page", "https://en.wikipedia.org/wiki/Neighbor")
        assert {
            (edge["src_node_id"], edge["dst_node_id"], edge["relation"])
            for edge in stored_edges
        } == {
            (seed_node_id, neighbor_node_id, "Neighbor"),
            (neighbor_node_id, seed_node_id, "the original seed"),
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlGraphStore(tmpdir)
        wiki_builder = MockWikiBuilder(store)
        strategy = GraphExpansionStrategy(
            store=store,
            wiki_builder=wiki_builder,
            visual_planner=MockVisualPlanner(),
            image_builder=MockImageBuilder(),
            config=GraphExpansionConfig(max_depth=0, max_new_text_neighbors=0, enable_image_expansion=True),
        )
        strategy.add_seed("https://en.wikipedia.org/wiki/Seed")
        result = strategy.expand_next(run_id="run_smoke")
        assert result is not None
        assert len(result.queued_tasks) == 1
        assert result.queued_tasks[0].task_type == ExpansionTaskType.IMAGE_EXPAND
        assert strategy.queue_size() == 1
        assert wiki_builder.read_calls == 1
        image_result = strategy.expand_next(run_id="run_smoke")
        assert image_result is not None
        assert image_result.task.task_type == ExpansionTaskType.IMAGE_EXPAND
        assert wiki_builder.read_calls == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlGraphStore(tmpdir)
        wiki_builder = MockWikiBuilder(store)
        inline_builder = MockInlineImageBuilder()
        strategy = GraphExpansionStrategy(
            store=store,
            wiki_builder=wiki_builder,
            visual_planner=MockNoVisualPlanner(),
            image_builder=inline_builder,
            config=GraphExpansionConfig(
                max_depth=0,
                max_new_text_neighbors=0,
                enable_image_expansion=True,
                max_wiki_inline_images_per_page=2,
                wiki_inline_random_seed="smoke_seed",
            ),
        )
        text_result = wiki_builder.build_from_url(
            "https://en.wikipedia.org/wiki/Seed",
            run_id="run_smoke",
            persist=True,
        )

        def make_inline_plan(image_url: str, rank: int) -> VisualSearchPlan:
            target = Evidence.create(
                EvidenceType.VISUAL_TARGET,
                content=f"caption for {rank}",
                node_ids=[text_result.node.node_id],
                url=image_url,
            )
            return VisualSearchPlan.create(
                target,
                queries=[SearchQuerySpec.create(f"caption for {rank}", target.evidence_id)],
                source_node_id=text_result.node.node_id,
                source_evidence_ids=[text_result.text_evidence.evidence_id],
                planner="wikipedia_inline_image_planner",
                metadata={
                    "plan_source": "wikipedia_inline_image",
                    "image_url": image_url,
                    "source_page_url": "https://en.wikipedia.org/wiki/Seed",
                    "caption": f"caption for {rank}",
                    "rank": rank,
                },
            )

        inline_results = strategy._execute_wiki_inline_plans(
            text_result,
            [
                make_inline_plan("https://example.com/1.jpg", 1),
                make_inline_plan("https://example.com/2.jpg", 2),
                make_inline_plan("https://example.com/3.jpg", 3),
            ],
            run_id="run_smoke",
            start_index=1,
        )
        assert len(inline_results) == 3
        assert len([item for item in inline_builder.inline_calls if item[1] is False]) == 3
        assert len([item for item in inline_builder.inline_calls if item[1] is True]) == 2
        selected = [
            item for item in inline_results if (item.metadata or {}).get("wiki_inline_random_cap_selected")
        ]
        dropped = [
            item for item in inline_results if (item.metadata or {}).get("wiki_inline_random_cap_reason") == "dropped_after_random_page_cap"
        ]
        assert len(selected) == 2
        assert len(dropped) == 1
        assert dropped[0].image_node is None
    print("graph_expansion smoke test passed")


if __name__ == "__main__":
    _smoke_test()
