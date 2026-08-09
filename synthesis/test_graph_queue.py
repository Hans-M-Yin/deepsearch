"""Tests for FIFO queue indexing and scheduling semantics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthesis.edges import EdgeType
from synthesis.evidence import Evidence, EvidenceType, SearchEngine, SearchSnapshot
from synthesis.graph_expansion import (
    ExpansionTask,
    ExpansionTaskType,
    GraphExpansionConfig,
    GraphExpansionStrategy,
    WikiTextBuildResult,
)
from synthesis.nodes import ImageNode, NodeSource, TextNode
from synthesis.store import JsonlGraphStore
from synthesis.wiki_text_builder import WikiLinkCandidate, WikiTextBuilder


class _CachedTextBuilder:
    """Small builder double for existing-node short-circuit tests."""

    def __init__(self, store: JsonlGraphStore) -> None:
        self.store = store
        self.attribute_calls = 0

    def build_from_url(
        self,
        url: str,
        *,
        title: str | None = None,
        run_id: str | None = None,
        persist: bool = True,
    ) -> WikiTextBuildResult:
        del title, run_id, persist
        node_id = TextNode.make_id("wikipedia_page", url)
        node_record = self.store.get_node(node_id)
        if node_record is None:
            raise AssertionError(f"test builder expected an existing node: {url}")
        evidence_record = next(
            evidence
            for evidence in self.store.list_evidence()
            if node_id in (evidence.get("node_ids") or [])
        )
        return WikiTextBuildResult(
            node=WikiTextBuilder._text_node_from_record(node_record),
            text_evidence=WikiTextBuilder._evidence_from_record(evidence_record),
            snapshot=SearchSnapshot.create(SearchEngine.JINA_READER, query=url),
            from_cache=True,
        )

    def extract_attributes(self, *args, **kwargs) -> Evidence:
        del args, kwargs
        self.attribute_calls += 1
        raise AssertionError("existing text nodes must not run attribute extraction")

    @staticmethod
    def _edge_to_linked_entity(
        source_node: TextNode,
        candidate: WikiLinkCandidate,
        evidence: Evidence,
        *,
        run_id: str | None = None,
    ):
        del evidence, run_id
        from synthesis.edges import Edge

        return Edge.create(
            source_node.node_id,
            candidate.node_id,
            edge_type=EdgeType.WIKI_LINK,
            relation=candidate.anchor_text,
        )


def _persist_existing_text_node(
    store: JsonlGraphStore,
    url: str,
    *,
    title: str,
) -> tuple[TextNode, Evidence]:
    node = TextNode(
        node_id=TextNode.make_id("wikipedia_page", url),
        subtype="wiki_page",
        title=title,
        description=f"{title} page",
        source=NodeSource(source_type="wikipedia", url=url),
    )
    evidence = Evidence.create(
        EvidenceType.WEB_TEXT,
        content=f"{title} page",
        node_ids=[node.node_id],
        url=url,
    )
    store.upsert_node(node)
    store.upsert_evidence(evidence)
    return node, evidence


class GraphExpansionQueueTests(unittest.TestCase):
    def _strategy(self, *, queue_pop_strategy: str = "fifo") -> GraphExpansionStrategy:
        return GraphExpansionStrategy(
            store=JsonlGraphStore(Path(tempfile.mkdtemp())),
            wiki_builder=None,
            config=GraphExpansionConfig(queue_pop_strategy=queue_pop_strategy),
        )

    @staticmethod
    def _text(
        url: str,
        *,
        parent_node_id: str | None = None,
        origin: str | None = None,
    ) -> ExpansionTask:
        metadata = {"task_origin": origin} if origin is not None else {}
        return ExpansionTask(
            url=url,
            parent_node_id=parent_node_id,
            metadata=metadata,
        )

    def test_fifo_preserves_global_order_across_task_buckets(self) -> None:
        strategy = self._strategy()
        tasks = [
            self._text("https://example.test/neighbor-1", parent_node_id="parent"),
            self._text("https://example.test/root"),
            self._text(
                "https://example.test/image-entity",
                parent_node_id="image-parent",
                origin="image_entity",
            ),
            ExpansionTask.from_image_expansion(
                url="https://example.test/image",
                title="Image",
                depth=0,
                source_text_node_id="text-source",
                source_evidence_id="evidence-source",
            ),
            self._text("https://example.test/neighbor-2", parent_node_id="parent"),
        ]
        for task in tasks:
            self.assertTrue(strategy.enqueue(task))

        self.assertEqual(strategy.queue_size(), 5)
        self.assertEqual(strategy.queue_size(ExpansionTaskType.TEXT_EXPAND), 4)
        self.assertEqual(strategy.queue_size(ExpansionTaskType.IMAGE_EXPAND), 1)
        self.assertEqual(
            strategy.queue_breakdown(),
            {
                "text_queue": 4,
                "image_queue": 1,
                "text_neighbor_queue": 3,
                "image_entity_queue": 1,
            },
        )
        self.assertTrue(strategy.has_root_text_tasks())

        popped = [
            strategy.pop_next_task()
            for _ in range(len(tasks))
        ]
        self.assertEqual(popped, tasks)
        self.assertEqual(
            [task.enqueue_seq for task in tasks],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(strategy.queue_size(), 0)
        self.assertFalse(strategy.has_root_text_tasks())

    def test_fifo_filters_select_earliest_eligible_task(self) -> None:
        strategy = self._strategy()
        ordinary = self._text(
            "https://example.test/ordinary",
            parent_node_id="parent",
        )
        image = ExpansionTask.from_image_expansion(
            url="https://example.test/image",
            title="Image",
            depth=0,
            source_text_node_id="text-source",
            source_evidence_id="evidence-source",
        )
        root = self._text("https://example.test/root")
        image_entity = self._text(
            "https://example.test/entity",
            parent_node_id="image-parent",
            origin="image_entity",
        )
        for task in (ordinary, image, root, image_entity):
            strategy.enqueue(task)

        self.assertIs(
            strategy.pop_next_task(
                allowed_task_types={ExpansionTaskType.TEXT_EXPAND},
                text_task_origin="image_entity",
            ),
            image_entity,
        )
        self.assertIs(
            strategy.pop_next_task(
                allowed_task_types={ExpansionTaskType.TEXT_EXPAND},
                root_text_only=True,
            ),
            root,
        )
        self.assertIs(
            strategy.pop_next_task(
                allowed_task_types={ExpansionTaskType.IMAGE_EXPAND},
            ),
            image,
        )
        self.assertIs(strategy.pop_next_task(), ordinary)
        self.assertIsNone(strategy.pop_next_task())

    def test_old_queue_records_without_sequence_restore_in_order(self) -> None:
        old_tasks = [
            self._text("https://example.test/old-a"),
            self._text("https://example.test/old-b", parent_node_id="parent"),
        ]
        old_records = [task.to_dict() for task in old_tasks]
        for record in old_records:
            record.pop("enqueue_seq", None)

        restored = self._strategy()
        for record in old_records:
            restored.enqueue(
                ExpansionTask(
                    url=record["url"],
                    task_type=ExpansionTaskType(record["task_type"]),
                    parent_node_id=record.get("parent_node_id"),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
        new_task = self._text("https://example.test/new")
        restored.enqueue(new_task)

        popped = [
            restored.pop_next_task(),
            restored.pop_next_task(),
            restored.pop_next_task(),
        ]
        self.assertEqual(
            [task.url for task in popped],
            [task.url for task in old_tasks] + [new_task.url],
        )
        self.assertEqual(new_task.enqueue_seq, 2)

    def test_random_mode_keeps_legacy_queue_path(self) -> None:
        strategy = self._strategy(queue_pop_strategy="random")
        tasks = [self._text(f"https://example.test/{index}") for index in range(3)]
        for task in tasks:
            strategy.enqueue(task)

        self.assertEqual(strategy.queue_size(), 3)
        self.assertEqual(strategy.queue_breakdown()["text_queue"], 3)
        self.assertEqual(len(strategy.queue_records()), 3)

    def test_active_text_url_index_tracks_pending_link_lifecycle(self) -> None:
        pending_link = {
            "parent_node_id": "parent",
            "source_evidence_id": "evidence",
            "candidate": {"url": "https://example.test/target"},
        }
        for queue_pop_strategy in ("fifo", "random"):
            with self.subTest(queue_pop_strategy=queue_pop_strategy):
                strategy = self._strategy(queue_pop_strategy=queue_pop_strategy)
                target = self._text("https://example.test/target")
                self.assertTrue(strategy.enqueue(target))
                self.assertIs(
                    strategy._active_text_tasks_by_url[target.url],
                    target,
                )

                self.assertTrue(
                    strategy._append_pending_link_to_queued_task(
                        target.url,
                        pending_link,
                    )
                )
                self.assertEqual(
                    target.metadata["pending_parent_links"],
                    [pending_link],
                )
                self.assertFalse(
                    strategy._append_pending_link_to_queued_task(
                        target.url,
                        pending_link,
                    )
                )

                while strategy.queue_size():
                    popped = strategy.pop_next_task()
                    if popped is target:
                        break
                self.assertNotIn(target.url, strategy._active_text_tasks_by_url)
                self.assertTrue(
                    strategy._append_pending_link_to_queued_task(
                        target.url,
                        pending_link,
                    )
                )
                self.assertEqual(
                    strategy._pending_parent_links_by_url[target.url],
                    [pending_link],
                )

    def test_existing_text_task_only_materializes_text_source_edge(self) -> None:
        store = JsonlGraphStore(Path(tempfile.mkdtemp()))
        builder = _CachedTextBuilder(store)
        strategy = GraphExpansionStrategy(
            store=store,
            wiki_builder=builder,
            config=GraphExpansionConfig(
                enable_image_expansion=True,
                extract_attributes=True,
            ),
        )
        parent, source_evidence = _persist_existing_text_node(
            store,
            "https://en.wikipedia.org/wiki/Parent",
            title="Parent",
        )
        target, _ = _persist_existing_text_node(
            store,
            "https://en.wikipedia.org/wiki/Target",
            title="Target",
        )
        task = ExpansionTask.from_wiki_link(
            WikiLinkCandidate(
                title="Target",
                url="https://en.wikipedia.org/wiki/Target",
                anchor_text="Target",
                source_url="https://en.wikipedia.org/wiki/Parent",
            ),
            depth=1,
            parent_node_id=parent.node_id,
            source_evidence_id=source_evidence.evidence_id,
        )
        self.assertTrue(strategy.enqueue(task))

        result = strategy.expand_next(run_id="existing_text_test")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.task.status.value, "skipped")
        self.assertTrue(result.text_result.from_cache)
        self.assertEqual(len(result.materialized_edges), 1)
        self.assertEqual(builder.attribute_calls, 0)
        self.assertEqual(strategy.queue_size(), 0)
        edge = store.list_edges()[0]
        self.assertEqual(edge["src_node_id"], parent.node_id)
        self.assertEqual(edge["dst_node_id"], target.node_id)
        self.assertEqual(edge["edge_type"], EdgeType.WIKI_LINK.value)

    def test_existing_text_task_only_materializes_image_source_edge(self) -> None:
        store = JsonlGraphStore(Path(tempfile.mkdtemp()))
        builder = _CachedTextBuilder(store)
        strategy = GraphExpansionStrategy(
            store=store,
            wiki_builder=builder,
            config=GraphExpansionConfig(enable_image_expansion=True),
        )
        image = ImageNode(
            node_id="image_parent",
            title="Source image",
            source=NodeSource(source_type="image", url="https://example.test/image.jpg"),
        )
        store.upsert_node(image)
        source_evidence = Evidence.create(
            EvidenceType.IMAGE,
            content="grounded image entity",
            node_ids=[image.node_id],
        )
        store.upsert_evidence(source_evidence)
        target, _ = _persist_existing_text_node(
            store,
            "https://en.wikipedia.org/wiki/Target",
            title="Target",
        )
        task = ExpansionTask.from_image_entity(
            url="https://en.wikipedia.org/wiki/Target",
            title="Target",
            parent_image_node_id=image.node_id,
            source_evidence_id=source_evidence.evidence_id,
            entity={"name": "Target", "type": "person", "evidence": "Target"},
        )
        self.assertTrue(strategy.enqueue(task))

        result = strategy.expand_next(run_id="existing_image_entity_test")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.task.status.value, "skipped")
        self.assertTrue(result.text_result.from_cache)
        self.assertEqual(len(result.materialized_edges), 1)
        self.assertEqual(builder.attribute_calls, 0)
        edge = store.list_edges()[0]
        self.assertEqual(edge["src_node_id"], image.node_id)
        self.assertEqual(edge["dst_node_id"], target.node_id)
        self.assertEqual(edge["edge_type"], EdgeType.IMAGE_DEPICTS.value)


if __name__ == "__main__":
    unittest.main()
