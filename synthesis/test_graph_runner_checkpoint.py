"""Tests for graph/store checkpoint synchronization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synthesis.graph_expansion import (
    ExpansionTask,
    ExpansionTaskStatus,
    GraphExpansionStrategy,
    NodeExpansionResult,
)
from synthesis.graph_runner import GraphRunner, GraphRunnerConfig
from synthesis.nodes import TextNode
from synthesis.store import JsonlGraphStore


class _FlushWritingStrategy(GraphExpansionStrategy):
    def expand_task(
        self,
        task: ExpansionTask,
        *,
        run_id: str | None = None,
    ) -> NodeExpansionResult:
        del run_id
        self.store.upsert_node(TextNode.from_webpage(task.url, title=task.title or task.url))
        self.store.maybe_flush()
        task.status = ExpansionTaskStatus.DONE
        return NodeExpansionResult(task=task)


class _InterruptingStrategy(_FlushWritingStrategy):
    def expand_task(
        self,
        task: ExpansionTask,
        *,
        run_id: str | None = None,
    ) -> NodeExpansionResult:
        del run_id
        self.store.upsert_node(TextNode.from_webpage(task.url, title=task.title or task.url))
        self.store.maybe_flush()
        raise KeyboardInterrupt


class GraphRunnerCheckpointTests(unittest.TestCase):
    def test_delta_flush_triggers_state_save_before_checkpoint_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir, flush_record_threshold=1)
            strategy = _FlushWritingStrategy(store=store, wiki_builder=object())
            runner = GraphRunner(
                strategy=strategy,
                store=store,
                config=GraphRunnerConfig(
                    max_steps=10,
                    checkpoint_every=100,
                    show_progress=False,
                ),
                run_id="flush_checkpoint_test",
                resume=False,
            )
            runner.add_seed("https://example.com/flush-checkpoint", title="Flush checkpoint")

            result = runner._run_one()[0]
            runner._handle_result(result, None)

            state = json.loads(Path(tmpdir, "graph_runner_state.json").read_text())
            self.assertEqual(state["step"], 1)
            self.assertEqual(state["queue"], [])
            self.assertTrue(Path(tmpdir, "nodes.delta.jsonl").exists())

    def test_keyboard_interrupt_flushes_delta_and_keeps_active_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir, flush_record_threshold=1)
            strategy = _InterruptingStrategy(store=store, wiki_builder=object())
            runner = GraphRunner(
                strategy=strategy,
                store=store,
                config=GraphRunnerConfig(max_steps=10, show_progress=False),
                run_id="interrupt_checkpoint_test",
                resume=False,
            )
            runner.add_seed("https://example.com/interrupt", title="Interrupt")

            with self.assertRaises(KeyboardInterrupt):
                runner.run()

            state = json.loads(Path(tmpdir, "graph_runner_state.json").read_text())
            self.assertEqual(state["status"], "paused")
            self.assertEqual(len(state["queue"]), 1)
            self.assertEqual(state["queue"][0]["url"], "https://example.com/interrupt")
            self.assertTrue(Path(tmpdir, "nodes.delta.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
