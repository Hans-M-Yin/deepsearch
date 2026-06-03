"""Read-only graph view helpers for VQA generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from synthesis.store import JsonlGraphStore


@dataclass(slots=True)
class GraphView:
    """Convenience wrapper around ``JsonlGraphStore`` for path sampling."""

    store: JsonlGraphStore
    allowed_edge_types: set[str] | None = None

    def __post_init__(self) -> None:
        self.nodes_by_id: dict[str, dict[str, Any]] = {
            record["node_id"]: record for record in self.store.list_nodes()
        }
        self.edges_by_id: dict[str, dict[str, Any]] = {
            record["edge_id"]: record for record in self.store.list_edges()
        }
        self.out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.in_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges_by_id.values():
            if self.allowed_edge_types and edge.get("edge_type") not in self.allowed_edge_types:
                continue
            self.out_edges[edge["src_node_id"]].append(edge)
            self.in_edges[edge["dst_node_id"]].append(edge)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes_by_id.get(node_id)

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        return self.edges_by_id.get(edge_id)

    def neighbors(self, node_id: str) -> list[dict[str, Any]]:
        return list(self.out_edges.get(node_id, []))

    def node_type(self, node_id: str) -> str | None:
        node = self.get_node(node_id)
        return None if node is None else node.get("node_type")

    def list_node_ids(self, *, node_type: str | None = None) -> list[str]:
        if node_type is None:
            return list(self.nodes_by_id.keys())
        return [
            node_id
            for node_id, node in self.nodes_by_id.items()
            if node.get("node_type") == node_type
        ]
