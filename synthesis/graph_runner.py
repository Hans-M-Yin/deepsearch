"""Single-threaded graph expansion runner with checkpointed state."""

from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    ExpansionTaskType,
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
    parallel_workers: int = 1
    batch_size: int | None = None
    show_progress: bool = True
    persist_visual_plans: bool = True
    visual_plans_file_name: str = "visual_plans.jsonl"

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
    seen_task_keys: list[str] = field(default_factory=list)
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
            seen_task_keys=list(payload.get("seen_task_keys") or payload.get("seen_urls") or []),
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
    skipped_count: int = 0
    timing_summary: dict[str, Any] = field(default_factory=dict)
    image_summary: dict[str, int] = field(default_factory=dict)
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
        self.visual_plans_path = store.root_dir / self.config.visual_plans_file_name
        self.state = self._load_or_create_state(run_id=run_id, resume=resume)
        self._restore_strategy_state()
        self._saved_visual_plan_ids = self._load_saved_visual_plan_ids()
        self._progress_width = 0

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
            if self.config.parallel_workers <= 1:
                results = self._run_one()
            else:
                results = self._run_parallel_batch()
            if not results:
                self.state.status = "completed"
                break

            for result in results:
                self.state.step += 1
                self._record_result(result)
                self._emit_created_node_events(result)
                self._emit_node_status(result)
                self._emit_progress()
                self._emit_warning(result)
                if result.error:
                    last_error = result.error
                    if self.config.stop_on_error:
                        self.state.status = "failed"
                        break

            if self.state.status == "failed":
                self._sync_state_from_strategy()
                self.save_state()
                break

            if self.config.checkpoint_every <= 1 or self.state.step % self.config.checkpoint_every == 0:
                self._sync_state_from_strategy()
                self.save_state()

        if self.state.status == "running":
            self.state.status = "completed" if self.strategy.queue_size() == 0 else "paused"

        self._finish_progress()
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
            skipped_count=len(self.state.skipped_tasks),
            store_stats=self.store.stats(),
            timing_summary=self._timing_summary(),
            image_summary=self._image_summary(),
            last_error=last_error,
        )

    def _run_one(self) -> list[NodeExpansionResult]:
        task = self._pop_next_schedulable_task(in_flight_text_count=0)
        if task is None:
            return []
        result = self.strategy.expand_task(task, run_id=self.state.run_id)
        return [result] if result is not None else []

    def _run_parallel_batch(self) -> list[NodeExpansionResult]:
        remaining_steps = self.config.max_steps - self.state.step
        if remaining_steps <= 0:
            return []
        max_inflight = self.config.batch_size or self.config.parallel_workers
        max_inflight = max(1, min(int(max_inflight), int(self.config.parallel_workers), remaining_steps))

        initial_tasks: list[ExpansionTask] = []
        initial_text_count = 0
        while len(initial_tasks) < max_inflight:
            task = self._pop_next_schedulable_task(in_flight_text_count=initial_text_count)
            if task is None:
                break
            initial_tasks.append(task)
            if task.task_type == ExpansionTaskType.TEXT_EXPAND:
                initial_text_count += 1
        if not initial_tasks:
            return []

        results: list[NodeExpansionResult] = []
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_task = {}
            for task in initial_tasks:
                future_to_task[executor.submit(self.strategy.expand_task, task, run_id=self.state.run_id)] = task
            in_flight_text_count = initial_text_count

            while future_to_task:
                done, _ = wait(tuple(future_to_task.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    task = future_to_task.pop(future)
                    if task.task_type == ExpansionTaskType.TEXT_EXPAND:
                        in_flight_text_count -= 1
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        task.status = ExpansionTaskStatus.FAILED
                        results.append(
                            NodeExpansionResult(
                                task=task,
                                error=f"{exc.__class__.__name__}: {exc}",
                            )
                        )

                    while (
                        len(results) + len(future_to_task) < remaining_steps
                        and len(future_to_task) < max_inflight
                    ):
                        next_task = self._pop_next_schedulable_task(
                            in_flight_text_count=in_flight_text_count,
                        )
                        if next_task is None:
                            break
                        next_future = executor.submit(
                            self.strategy.expand_task,
                            next_task,
                            run_id=self.state.run_id,
                        )
                        future_to_task[next_future] = next_task
                        if next_task.task_type == ExpansionTaskType.TEXT_EXPAND:
                            in_flight_text_count += 1
        return results

    def _text_node_count(self) -> int:
        return sum(1 for node in self.store.list_nodes() if node.get("node_type") == "text")

    def _queue_breakdown(self) -> dict[str, int]:
        text_neighbor_queue = 0
        image_entity_queue = 0
        text_queue_size = 0
        image_queue_size = 0
        for record in self.strategy.queue_records():
            task_type = record.get("task_type")
            if task_type == ExpansionTaskType.IMAGE_EXPAND.value:
                image_queue_size += 1
                continue
            if task_type != ExpansionTaskType.TEXT_EXPAND.value:
                continue
            text_queue_size += 1
            metadata = record.get("metadata") or {}
            if metadata.get("task_origin") == "image_entity":
                image_entity_queue += 1
            else:
                text_neighbor_queue += 1
        return {
            "text_queue": text_queue_size,
            "image_queue": image_queue_size,
            "text_neighbor_queue": text_neighbor_queue,
            "image_entity_queue": image_entity_queue,
        }

    def _remaining_text_slots(self, *, in_flight_text_count: int = 0) -> int | None:
        if self.config.max_nodes is None:
            return None
        return max(0, self.config.max_nodes - self._text_node_count() - in_flight_text_count)

    def _pop_next_schedulable_task(
        self,
        *,
        in_flight_text_count: int,
    ) -> ExpansionTask | None:
        remaining_text_slots = self._remaining_text_slots(
            in_flight_text_count=in_flight_text_count,
        )
        allowed_types = {ExpansionTaskType.IMAGE_EXPAND}
        if remaining_text_slots is None or remaining_text_slots > 0:
            allowed_types.add(ExpansionTaskType.TEXT_EXPAND)
        return self.strategy.pop_next_task(allowed_task_types=allowed_types)

    def _has_schedulable_tasks(self) -> bool:
        if self.strategy.queue_size(ExpansionTaskType.IMAGE_EXPAND) > 0:
            return True
        remaining_text_slots = self._remaining_text_slots()
        return (
            (remaining_text_slots is None or remaining_text_slots > 0)
            and self.strategy.queue_size(ExpansionTaskType.TEXT_EXPAND) > 0
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
        self.strategy.add_seen_task_keys(self.state.seen_task_keys)

    def _sync_state_from_strategy(self) -> None:
        self.state.queue = self.strategy.queue_records()
        self.state.seen_task_keys = self.strategy.seen_task_keys()

    def _record_result(self, result: NodeExpansionResult) -> None:
        record = {
            "step": self.state.step,
            "task": result.task.to_dict(),
            "error": result.error,
            "text_node_id": result.text_result.node.node_id if result.text_result else None,
            "attribute_evidence_id": result.attribute_evidence.evidence_id if result.attribute_evidence else None,
            "attribute_error": result.attribute_error,
            "queued_count": len(result.queued_tasks),
            "materialized_edge_count": len(result.materialized_edges),
            "parent_link_failure_count": len(result.parent_link_failures),
            "parent_link_failures": [dict(item) for item in result.parent_link_failures],
            "visual_plan_count": len(result.visual_plans),
            "image_result_count": len(result.image_results),
            "image_summary": self._summarize_image_results(result.image_results),
            "timing": result.timing,
        }
        if result.error:
            self.state.failed_tasks.append(record)
        elif result.task.status == ExpansionTaskStatus.SKIPPED:
            self.state.skipped_tasks.append(record)
        else:
            self.state.completed_tasks.append(record)
        self._persist_visual_plans(result)

    def _persist_visual_plans(self, result: NodeExpansionResult) -> None:
        if not self.config.persist_visual_plans or not result.visual_plans:
            return
        node = result.text_result.node.to_dict() if result.text_result is not None else None
        records: list[dict[str, Any]] = []
        for plan in result.visual_plans:
            if plan.plan_id in self._saved_visual_plan_ids:
                continue
            records.append(self._visual_plan_record(result=result, node=node, plan=plan))
            self._saved_visual_plan_ids.add(plan.plan_id)
        if not records:
            return
        self.visual_plans_path.parent.mkdir(parents=True, exist_ok=True)
        with self.visual_plans_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    def _load_saved_visual_plan_ids(self) -> set[str]:
        if not self.config.persist_visual_plans or not self.visual_plans_path.exists():
            return set()
        seen: set[str] = set()
        try:
            with self.visual_plans_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    plan_id = record.get("plan_id")
                    if isinstance(plan_id, str) and plan_id:
                        seen.add(plan_id)
        except Exception:
            return set()
        return seen

    def _visual_plan_record(
        self,
        *,
        result: NodeExpansionResult,
        node: dict[str, Any] | None,
        plan: Any,
    ) -> dict[str, Any]:
        target = plan.target
        source = (node or {}).get("source") or {}
        return {
            "run_id": self.state.run_id,
            "step": self.state.step,
            "task_url": result.task.url,
            "task_title": result.task.title,
            "task_depth": result.task.depth,
            "node_id": (node or {}).get("node_id"),
            "node_title": (node or {}).get("title") or (node or {}).get("canonical_id"),
            "node_source_url": source.get("url") if isinstance(source, dict) else None,
            "plan_id": plan.plan_id,
            "target_evidence_id": target.evidence_id,
            "target_description": target.content,
            "target_type": target.metadata.get("target_type"),
            "downstream_use": target.metadata.get("downstream_use"),
            "source_passage": target.metadata.get("source_passage"),
            "source_quote": target.metadata.get("source_quote"),
            "uniqueness": target.metadata.get("uniqueness"),
            "reason": target.metadata.get("reason"),
            "expected_visual": target.metadata.get("expected_visual") or target.metadata.get("query"),
            "queries": [query.query for query in plan.queries],
            "query_specs": [query.to_dict() for query in plan.queries],
            "target": target.to_dict(),
            "planner": plan.planner,
            "metadata": plan.metadata,
        }

    def _emit_progress(self) -> None:
        if not self.config.show_progress:
            return
        stats = self.store.stats()
        queue_breakdown = self._queue_breakdown()
        text_queue_size = queue_breakdown["text_queue"]
        image_queue_size = queue_breakdown["image_queue"]
        text_neighbor_queue = queue_breakdown["text_neighbor_queue"]
        image_entity_queue = queue_breakdown["image_entity_queue"]
        queue_size = text_queue_size + image_queue_size
        max_steps = self.config.max_steps
        max_nodes = self.config.max_nodes
        steps_text = f"{self.state.step}/{max_steps}" if max_steps else str(self.state.step)
        node_count = int(stats.get("nodes", 0))
        text_node_count = self._text_node_count()
        nodes_text = f"{text_node_count}/{max_nodes}" if max_nodes is not None else str(text_node_count)
        line = (
            "[progress] "
            f"steps={steps_text} "
            f"queue={queue_size} "
            f"text_queue={text_queue_size} "
            f"text_neighbor_queue={text_neighbor_queue} "
            f"image_entity_queue={image_entity_queue} "
            f"image_queue={image_queue_size} "
            f"text_nodes={nodes_text} "
            f"nodes={node_count} "
            f"edges={int(stats.get('edges', 0))} "
            f"completed={len(self.state.completed_tasks)} "
            f"failed={len(self.state.failed_tasks)} "
            f"skipped={len(self.state.skipped_tasks)}"
        )
        self._progress_width = max(self._progress_width, len(line))
        if sys.stdout.isatty():
            padded = line.ljust(self._progress_width)
            print(f"\r{padded}", end="", file=sys.stdout, flush=True)
            return
        print(line, file=sys.stdout, flush=True)

    def _emit_created_node_events(self, result: NodeExpansionResult) -> None:
        stats = self.store.stats()
        queue_breakdown = self._queue_breakdown()
        text_count = 0
        image_count = 0
        for node in self.store.list_nodes():
            node_type = node.get("node_type")
            if node_type == "text":
                text_count += 1
            elif node_type == "image":
                image_count += 1

        created_nodes: list[dict[str, Any]] = []
        if result.text_result is not None:
            created_nodes.append(
                {
                    "node_type": "text",
                    "node_id": result.text_result.node.node_id,
                    "title": result.text_result.node.title or result.text_result.node.node_id,
                }
            )
        for image_result in result.image_results:
            if image_result.image_node is None:
                continue
            created_nodes.append(
                {
                    "node_type": "image",
                    "node_id": image_result.image_node.node_id,
                    "title": image_result.image_node.title or image_result.image_node.node_id,
                }
            )

        for node in created_nodes:
            print(
                "[node-created] "
                f"type={node['node_type']} "
                f"node_id={node['node_id']!r} "
                f"title={node['title']!r} "
                f"queue={queue_breakdown['text_queue'] + queue_breakdown['image_queue']} "
                f"text_queue={queue_breakdown['text_queue']} "
                f"text_neighbor_queue={queue_breakdown['text_neighbor_queue']} "
                f"image_entity_queue={queue_breakdown['image_entity_queue']} "
                f"image_queue={queue_breakdown['image_queue']} "
                f"nodes={int(stats.get('nodes', 0))} "
                f"text_nodes={text_count} "
                f"image_nodes={image_count} "
                f"edges={int(stats.get('edges', 0))}",
                file=sys.stdout,
                flush=True,
            )

    def _emit_node_status(self, result: NodeExpansionResult) -> None:
        stats = self.store.stats()
        nodes = self.store.list_nodes()
        text_count = 0
        image_count = 0
        latest_title: str | None = None
        latest_created_at: str = ""
        queue_breakdown = self._queue_breakdown()
        text_queue_size = queue_breakdown["text_queue"]
        image_queue_size = queue_breakdown["image_queue"]
        text_neighbor_queue = queue_breakdown["text_neighbor_queue"]
        image_entity_queue = queue_breakdown["image_entity_queue"]
        for node in nodes:
            node_type = node.get("node_type")
            if node_type == "text":
                text_count += 1
            elif node_type == "image":
                image_count += 1
            created_at = str(node.get("created_at") or "")
            if created_at >= latest_created_at:
                latest_created_at = created_at
                latest_title = node.get("title") or node.get("node_id")
        task_title = result.task.title or result.task.url
        print(
            "[node-status] "
            f"task={task_title!r} "
            f"nodes={int(stats.get('nodes', 0))} "
            f"text={text_count} "
            f"image={image_count} "
            f"text_queue={text_queue_size} "
            f"text_neighbor_queue={text_neighbor_queue} "
            f"image_entity_queue={image_entity_queue} "
            f"image_queue={image_queue_size} "
            f"latest={latest_title!r}",
            file=sys.stdout,
            flush=True,
        )

    def _finish_progress(self) -> None:
        if not self.config.show_progress:
            return
        if sys.stdout.isatty() and self._progress_width > 0:
            print(file=sys.stdout, flush=True)

    @staticmethod
    def _emit_warning(result: NodeExpansionResult) -> None:
        if result.error:
            task = result.task
            print(
                "[warning] "
                f"task_failed url={task.url!r} "
                f"title={task.title!r} "
                f"depth={task.depth} "
                f"error={result.error}",
                file=sys.stdout,
                flush=True,
            )
            return
        for failure in result.parent_link_failures:
            task = result.task
            print(
                "[warning] "
                f"parent_link_missing url={task.url!r} "
                f"title={task.title!r} "
                f"depth={task.depth} "
                f"origin={(task.metadata or {}).get('task_origin')!r} "
                f"link_type={failure.get('link_type')!r} "
                f"reason={failure.get('reason')!r} "
                f"parent_node_id={failure.get('parent_node_id')!r} "
                f"source_evidence_id={failure.get('source_evidence_id')!r} "
                f"target_node_id={failure.get('target_node_id')!r} "
                f"entity_name={failure.get('entity_name')!r}",
                file=sys.stdout,
                flush=True,
            )
        if result.attribute_error:
            task = result.task
            print(
                "[warning] "
                f"attribute_extract_failed url={task.url!r} "
                f"title={task.title!r} "
                f"depth={task.depth} "
                f"error={result.attribute_error}",
                file=sys.stdout,
                flush=True,
            )

    def _should_continue(self) -> bool:
        if self.state.step >= self.config.max_steps:
            return False
        return self._has_schedulable_tasks()

    def _timing_summary(self) -> dict[str, Any]:
        records = self.state.completed_tasks + self.state.failed_tasks
        timing_records = [
            record.get("timing")
            for record in records
            if isinstance(record.get("timing"), dict)
        ]
        if not timing_records:
            return {}

        keys = sorted({key for timing in timing_records for key in timing})
        metrics: dict[str, dict[str, float]] = {}
        for key in keys:
            values = [float(timing[key]) for timing in timing_records if timing.get(key) is not None]
            if not values:
                continue
            values_sorted = sorted(values)
            metrics[key] = {
                "total_s": sum(values),
                "avg_s": sum(values) / len(values),
                "min_s": values_sorted[0],
                "max_s": values_sorted[-1],
                "p50_s": values_sorted[len(values_sorted) // 2],
            }
        return {
            "steps_with_timing": len(timing_records),
            "metrics": metrics,
        }

    @staticmethod
    def _is_fetch_failure_reason(reason: str | None) -> bool:
        text = str(reason or "").lower()
        if not text:
            return False
        needles = (
            "image_url_precheck_failed",
            "http_429",
            "http_403",
            "forbidden",
            "timeout",
            "connection error",
            "apiconnectionerror",
            "apitimeouterror",
            "non_image_content_type",
            "url_error",
            "decode_error",
            "missing_resolved_image_asset",
        )
        return any(needle in text for needle in needles)

    def _summarize_image_results(self, image_results: list[NodeExpansionResult] | list[Any]) -> dict[str, int]:
        summary = {
            "returned": 0,
            "accepted": 0,
            "rejected": 0,
            "fetch_failed": 0,
        }
        for image_result in image_results or []:
            metadata = image_result.metadata if hasattr(image_result, "metadata") else {}
            decision_log = list(metadata.get("candidate_decisions") or [])
            for item in decision_log:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "")
                if kind == "query_results":
                    summary["returned"] += int(item.get("returned") or 0)
                    continue
                if kind == "candidate_kept":
                    status = str(item.get("status") or "").lower()
                    if status == "accepted":
                        summary["accepted"] += 1
                    elif status == "rejected":
                        summary["rejected"] += 1
                    continue
                if kind in {"candidate_drop", "candidate_skip"} and self._is_fetch_failure_reason(item.get("reason")):
                    summary["fetch_failed"] += 1
        return summary

    def _image_summary(self) -> dict[str, int]:
        records = self.state.completed_tasks + self.state.failed_tasks + self.state.skipped_tasks
        summary = {
            "returned": 0,
            "accepted": 0,
            "rejected": 0,
            "fetch_failed": 0,
        }
        for record in records:
            item = record.get("image_summary")
            if not isinstance(item, dict):
                continue
            for key in summary:
                summary[key] += int(item.get(key) or 0)
        return summary

    @staticmethod
    def _task_from_record(record: dict[str, Any]) -> ExpansionTask:
        return ExpansionTask(
            url=record["url"],
            task_type=ExpansionTaskType(record.get("task_type", ExpansionTaskType.TEXT_EXPAND.value)),
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
            config=GraphRunnerConfig(max_steps=1, max_nodes=5, parallel_workers=2, batch_size=2),
            run_id="run_smoke",
            resume=False,
        )
        runner.add_seed("https://en.wikipedia.org/wiki/Runner_Seed")
        result = runner.run()
        assert result.status == "completed"
        assert result.steps == 1
        assert result.store_stats["nodes"] == 1
        assert (Path(tmpdir) / "graph_runner_state.json").exists()

    class MockSchedulingStrategy(GraphExpansionStrategy):
        def __init__(self, store: JsonlGraphStore) -> None:
            super().__init__(
                store=store,
                wiki_builder=MockWikiBuilder(store),
                config=GraphExpansionConfig(max_depth=0, enable_image_expansion=False),
            )
            self.executed_task_types: list[ExpansionTaskType] = []

        def expand_task(
            self,
            task: ExpansionTask,
            *,
            run_id: str | None = None,
        ) -> NodeExpansionResult:
            del run_id
            self.executed_task_types.append(task.task_type)
            if task.task_type == ExpansionTaskType.TEXT_EXPAND:
                node = TextNode.from_webpage(task.url, title=task.title or task.url)
                self.store.upsert_node(node)
            task.status = ExpansionTaskStatus.DONE
            return NodeExpansionResult(task=task)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlGraphStore(tmpdir)
        strategy = MockSchedulingStrategy(store)
        runner = GraphRunner(
            strategy=strategy,
            store=store,
            config=GraphRunnerConfig(
                max_steps=10,
                max_nodes=2,
                parallel_workers=4,
                batch_size=4,
                show_progress=False,
            ),
            run_id="run_text_budget_smoke",
            resume=False,
        )
        for index in range(4):
            strategy.enqueue(
                ExpansionTask(
                    url=f"https://en.wikipedia.org/wiki/Text_{index}",
                    title=f"Text {index}",
                )
            )
        for index in range(2):
            strategy.enqueue(
                ExpansionTask.from_image_expansion(
                    url=f"https://en.wikipedia.org/wiki/Image_Source_{index}",
                    title=f"Image Source {index}",
                    depth=0,
                    source_text_node_id=f"text_source_{index}",
                    source_evidence_id=f"evidence_{index}",
                )
            )
        result = runner.run()
        assert result.status == "paused"
        assert strategy.executed_task_types.count(ExpansionTaskType.TEXT_EXPAND) == 2
        assert strategy.executed_task_types.count(ExpansionTaskType.IMAGE_EXPAND) == 2
        assert strategy.queue_size(ExpansionTaskType.TEXT_EXPAND) == 2
        assert strategy.queue_size(ExpansionTaskType.IMAGE_EXPAND) == 0
        assert runner._text_node_count() == 2
    print("graph_runner smoke test passed")


if __name__ == "__main__":
    _smoke_test()
