"""Single-threaded graph expansion runner with checkpointed state."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .graph_expansion import (
    ExpansionTask,
    ExpansionTaskStatus,
    GraphExpansionStrategy,
    NodeExpansionResult,
)
from .store import JsonlGraphStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(slots=True)
class GraphRunnerConfig:
    """Limits for one runner execution."""

    max_steps: int = 100
    max_nodes: int | None = None
    checkpoint_every: int = 1
    stop_on_error: bool = False
    state_file_name: str = "graph_runner_state.json"

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class GraphRunnerState:
    """Serializable runner state for restart/resume."""

    run_id: str
    status: str = "initialized"
    step: int = 0
    completed_tasks: list[dict[str, Any]] = field(default_factory=list)
    failed_tasks: list[dict[str, Any]] = field(default_factory=list)
    skipped_tasks: list[dict[str, Any]] = field(default_factory=list)
    queue: list[dict[str, Any]] = field(default_factory=list)
    seen_urls: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphRunnerState":
        return cls(
            run_id=payload["run_id"],
            status=payload.get("status", "initialized"),
            step=int(payload.get("step", 0)),
            completed_tasks=list(payload.get("completed_tasks") or []),
            failed_tasks=list(payload.get("failed_tasks") or []),
            skipped_tasks=list(payload.get("skipped_tasks") or []),
            queue=list(payload.get("queue") or []),
            seen_urls=list(payload.get("seen_urls") or []),
            stats=dict(payload.get("stats") or {}),
            created_at=payload.get("created_at", _utc_now()),
            updated_at=payload.get("updated_at", _utc_now()),
        )


@dataclass(slots=True)
class GraphRunnerResult:
    """Summary returned after a run loop exits."""

    run_id: str
    status: str
    steps: int
    queue_size: int
    completed_count: int
    failed_count: int
    store_stats: dict[str, int]
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class GraphRunner:
    """Run graph expansion repeatedly and checkpoint progress."""

    def __init__(
        self,
        *,
        strategy: GraphExpansionStrategy,
        store: JsonlGraphStore,
        config: GraphRunnerConfig | None = None,
        run_id: str | None = None,
        state_path: str | Path | None = None,
        resume: bool = True,
    ) -> None:
        self.strategy = strategy
        self.store = store
        self.config = config or GraphRunnerConfig()
        self.state_path = Path(state_path) if state_path else store.root_dir / self.config.state_file_name
        self.state = self._load_or_create_state(run_id=run_id, resume=resume)
        self._restore_strategy_state()

    def add_seed(
        self,
        url: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExpansionTask:
        task = ExpansionTask(url=url, title=title, metadata=metadata or {})
        if self.strategy.enqueue(task):
            self._sync_state_from_strategy()
            self.save_state()
        return task

    def add_seeds(self, urls: list[str]) -> list[ExpansionTask]:
        return [self.add_seed(url) for url in urls]

    def run(self) -> GraphRunnerResult:
        self.state.status = "running"
        self.save_state()
        last_error: str | None = None

        while self._should_continue():
            result = self.strategy.expand_next(run_id=self.state.run_id)
            if result is None:
                self.state.status = "completed"
                break

            self.state.step += 1
            self._record_result(result)
            if result.error:
                last_error = result.error
                if self.config.stop_on_error:
                    self.state.status = "failed"
                    self._sync_state_from_strategy()
                    self.save_state()
                    break

            if self.config.checkpoint_every <= 1 or self.state.step % self.config.checkpoint_every == 0:
                self._sync_state_from_strategy()
                self.save_state()

        if self.state.status == "running":
            self.state.status = "completed" if self.strategy.queue_size() == 0 else "paused"

        self._sync_state_from_strategy()
        self.save_state()
        self.store.flush()
        return GraphRunnerResult(
            run_id=self.state.run_id,
            status=self.state.status,
            steps=self.state.step,
            queue_size=self.strategy.queue_size(),
            completed_count=len(self.state.completed_tasks),
            failed_count=len(self.state.failed_tasks),
            store_stats=self.store.stats(),
            last_error=last_error,
        )

    def save_state(self) -> None:
        self.state.updated_at = _utc_now()
        self.state.stats = {
            "queue_size": self.strategy.queue_size(),
            "store": self.store.stats(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.state.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(tmp_path, self.state_path)

    def _load_or_create_state(self, *, run_id: str | None, resume: bool) -> GraphRunnerState:
        if resume and self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                return GraphRunnerState.from_dict(json.load(handle))
        return GraphRunnerState(run_id=run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

    def _restore_strategy_state(self) -> None:
        for task_record in self.state.queue:
            self.strategy.enqueue(self._task_from_record(task_record))
        self.strategy._seen_urls.update(self.state.seen_urls)

    def _sync_state_from_strategy(self) -> None:
        self.state.queue = [task.to_dict() for task in self.strategy._queue]
        self.state.seen_urls = sorted(self.strategy._seen_urls)

    def _record_result(self, result: NodeExpansionResult) -> None:
        record = {
            "step": self.state.step,
            "task": result.task.to_dict(),
            "error": result.error,
            "text_node_id": result.text_result.node.node_id if result.text_result else None,
            "attribute_evidence_id": result.attribute_evidence.evidence_id if result.attribute_evidence else None,
            "attribute_error": result.attribute_error,
            "queued_count": len(result.queued_tasks),
            "visual_plan_count": len(result.visual_plans),
            "image_result_count": len(result.image_results),
        }
        if result.error:
            self.state.failed_tasks.append(record)
        elif result.task.status == ExpansionTaskStatus.SKIPPED:
            self.state.skipped_tasks.append(record)
        else:
            self.state.completed_tasks.append(record)

    def _should_continue(self) -> bool:
        if self.strategy.queue_size() <= 0:
            return False
        if self.state.step >= self.config.max_steps:
            return False
        if self.config.max_nodes is not None and self.store.stats().get("nodes", 0) >= self.config.max_nodes:
            return False
        return True

    @staticmethod
    def _task_from_record(record: dict[str, Any]) -> ExpansionTask:
        return ExpansionTask(
            url=record["url"],
            depth=int(record.get("depth", 0)),
            title=record.get("title"),
            parent_node_id=record.get("parent_node_id"),
            parent_edge_id=record.get("parent_edge_id"),
            priority=float(record.get("priority", 0.0)),
            status=ExpansionTaskStatus(record.get("status", ExpansionTaskStatus.PENDING.value)),
            metadata=dict(record.get("metadata") or {}),
        )


def _smoke_test() -> None:
    import tempfile

    from .evidence import Evidence, EvidenceType, SearchEngine, SearchSnapshot
    from .graph_expansion import GraphExpansionConfig, GraphExpansionStrategy, WikiTextBuildResult
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
            node = TextNode.from_webpage(url, title=title or "Runner Seed", description="Runner page")
            evidence = Evidence.create(EvidenceType.WEB_TEXT, content="Runner page", node_ids=[node.node_id])
            snapshot = SearchSnapshot.create(SearchEngine.JINA_READER, query=url, run_id=run_id)
            if persist:
                self.store.upsert_node(node)
                self.store.upsert_evidence(evidence)
                self.store.upsert_search_snapshot(snapshot)
            return WikiTextBuildResult(node=node, text_evidence=evidence, snapshot=snapshot)

        def extract_attributes(
            self,
            node: TextNode,
            *,
            source_evidence_ids: list[str] | None = None,
            run_id: str | None = None,
            persist: bool = True,
        ) -> Evidence:
            del source_evidence_ids, run_id
            node.attributes["runner"] = "yes"
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
            config=GraphExpansionConfig(max_depth=0, enable_image_expansion=False),
        )
        runner = GraphRunner(
            strategy=strategy,
            store=store,
            config=GraphRunnerConfig(max_steps=1, max_nodes=5),
            run_id="run_smoke",
            resume=False,
        )
        runner.add_seed("https://en.wikipedia.org/wiki/Runner_Seed")
        result = runner.run()
        assert result.status == "completed"
        assert result.steps == 1
        assert result.store_stats["nodes"] == 1
        assert (Path(tmpdir) / "graph_runner_state.json").exists()
    print("graph_runner smoke test passed")


if __name__ == "__main__":
    _smoke_test()
