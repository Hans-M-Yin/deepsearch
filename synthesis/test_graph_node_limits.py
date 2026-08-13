"""Tests for per-node-type limits applied by the graph runner."""

from __future__ import annotations

import tempfile
import unittest

from synthesis.graph_expansion import (
    ExpansionTask,
    ExpansionTaskStatus,
    ExpansionTaskType,
    GraphExpansionStrategy,
    NodeExpansionResult,
)
from synthesis.graph_runner import GraphRunner, GraphRunnerConfig
from synthesis.image_discovery import ImageDiscoveryResult
from synthesis.nodes import ImageNode, TextNode
from synthesis.store import JsonlGraphStore


class _NodeCreatingStrategy(GraphExpansionStrategy):
    """Small strategy double that materializes one node per task."""

    def expand_task(self, task: ExpansionTask, *, run_id: str | None = None) -> NodeExpansionResult:
        del run_id
        if task.task_type == ExpansionTaskType.IMAGE_EXPAND:
            self.store.upsert_node(ImageNode(node_id=f"image-{task.url}", title=task.title))
        else:
            self.store.upsert_node(TextNode.from_webpage(task.url, title=task.title or task.url))
        task.status = ExpansionTaskStatus.DONE
        return NodeExpansionResult(task=task)


class GraphNodeLimitTests(unittest.TestCase):
    def test_runner_limits_image_nodes_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir)
            strategy = _NodeCreatingStrategy(store=store, wiki_builder=object())
            runner = GraphRunner(
                strategy=strategy,
                store=store,
                config=GraphRunnerConfig(
                    max_steps=10,
                    max_nodes=2,
                    show_progress=False,
                ),
                run_id="image_limit_test",
                resume=False,
            )
            for index in range(4):
                strategy.enqueue(
                    ExpansionTask.from_image_expansion(
                        url=f"image-source-{index}",
                        title=f"Image {index}",
                        depth=0,
                        source_text_node_id=f"text-{index}",
                        source_evidence_id=f"evidence-{index}",
                    )
                )

            result = runner.run()

            self.assertEqual(result.status, "paused")
            self.assertEqual(store.count_nodes("image"), 2)
            self.assertEqual(strategy.queue_size(ExpansionTaskType.IMAGE_EXPAND), 2)

    def test_image_discovery_budget_handles_multiple_nodes_from_concurrent_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir)
            strategy = GraphExpansionStrategy(store=store, wiki_builder=object())
            strategy.configure_node_limits(2)

            def discover(index: int) -> ImageDiscoveryResult:
                node = ImageNode(node_id=f"budget-image-{index}", title=f"Image {index}")
                store.upsert_node(node)
                return ImageDiscoveryResult(plan_id=f"plan-{index}", image_node=node)

            first = strategy._run_image_discovery_with_budget(
                "plan-0",
                persist=True,
                discover=lambda: discover(0),
            )
            second = strategy._run_image_discovery_with_budget(
                "plan-1",
                persist=True,
                discover=lambda: discover(1),
            )
            third = strategy._run_image_discovery_with_budget(
                "plan-2",
                persist=True,
                discover=lambda: discover(2),
            )

            self.assertIsNotNone(first.image_node)
            self.assertIsNotNone(second.image_node)
            self.assertIsNone(third.image_node)
            self.assertTrue(third.metadata["image_node_limit_reached"])
            self.assertEqual(store.count_nodes("image"), 2)


if __name__ == "__main__":
    unittest.main()
