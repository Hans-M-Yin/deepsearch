"""Inspect one image node and the entities grounded from it.

Examples:
  python debug/inspect_image_node.py \
    --graph-dir runs/example \
    --image-node-id image:abc

  python debug/inspect_image_node.py \
    --graph-dir runs/example \
    --image-id image:abc \
    --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug.list_neighbor import (  # Reuse the graph's existing status semantics.
    _build_grounded_entity_reports,
    _collect_runner_state_index,
    _load_runner_state,
)
from synthesis.store import JsonlGraphStore


IMAGE_NODE_TYPE = "image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print an image node's source text, search query, and grounded entities."
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing nodes.jsonl and edges.jsonl.",
    )
    parser.add_argument(
        "--image-node-id",
        "--image-id",
        dest="image_node_id",
        required=True,
        help="ID of the image node to inspect.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def _node_label(node: dict[str, Any]) -> str:
    for key in ("title", "caption", "summary", "canonical_id", "node_id"):
        value = node.get(key)
        if value:
            return " ".join(str(value).split())
    return "<untitled>"


def _source_text_nodes(
    *,
    store: JsonlGraphStore,
    image_node: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    image_node_id = str(image_node["node_id"])
    sources: dict[str, dict[str, Any]] = {}

    for edge in store.edges_to(image_node_id):
        source_id = str(edge.get("src_node_id") or "")
        source_node = nodes_by_id.get(source_id)
        if not source_node or source_node.get("node_type") != "text":
            continue
        sources[source_id] = {
            "node_id": source_id,
            "title": _node_label(source_node),
            "summary": source_node.get("summary"),
            "edge_id": edge.get("edge_id"),
            "edge_type": edge.get("edge_type"),
            "relation": edge.get("relation"),
        }

    # Some image records retain this provenance even when the incoming edge is absent.
    metadata = image_node.get("metadata") or {}
    fallback_id = str(metadata.get("source_text_node_id") or "")
    fallback_node = nodes_by_id.get(fallback_id)
    if fallback_node and fallback_node.get("node_type") == "text" and fallback_id not in sources:
        sources[fallback_id] = {
            "node_id": fallback_id,
            "title": _node_label(fallback_node),
            "summary": fallback_node.get("summary"),
            "edge_id": None,
            "edge_type": None,
            "relation": None,
        }

    return sorted(sources.values(), key=lambda item: (item["title"], item["node_id"]))


def build_report(graph_dir: Path, image_node_id: str) -> dict[str, Any]:
    store = JsonlGraphStore(graph_dir)
    image_node = store.get_node(image_node_id)
    if image_node is None:
        raise ValueError(f"Image node not found: {image_node_id}")
    if image_node.get("node_type") != IMAGE_NODE_TYPE:
        raise ValueError(
            f"Node {image_node_id!r} is {image_node.get('node_type')!r}, not an image node."
        )

    nodes_by_id = {str(node.get("node_id")): node for node in store.list_nodes()}
    runner_state, runner_state_error = _load_runner_state(graph_dir)
    entity_reports = _build_grounded_entity_reports(
        image_node=image_node,
        out_edges=store.edges_from(image_node_id),
        nodes_by_id=nodes_by_id,
        runner_state_index=_collect_runner_state_index(runner_state),
        summary_chars=0,
    )
    metadata = image_node.get("metadata") or {}
    source = image_node.get("source") or {}

    return {
        "graph_dir": str(graph_dir),
        "image_node": {
            "node_id": image_node_id,
            "title": _node_label(image_node),
            "caption": image_node.get("caption"),
            "image_url": source.get("url") if isinstance(source, dict) else None,
            "image_origin": metadata.get("image_origin"),
            "unique_state": image_node.get("unique_state"),
        },
        "source_text_nodes": _source_text_nodes(
            store=store, image_node=image_node, nodes_by_id=nodes_by_id
        ),
        "search_query": metadata.get("search_query"),
        "runner_state_error": runner_state_error,
        "grounded_entities": entity_reports,
    }


def print_report(report: dict[str, Any]) -> None:
    image = report["image_node"]
    print(f"Image node: {image['node_id']} ({image['title']})")
    print(f"Unique state: {image.get('unique_state') or '<missing>'}")
    print(f"Search query: {report.get('search_query') or '<missing>'}")
    print("Source text node(s):")
    sources = report["source_text_nodes"]
    if not sources:
        print("  <none recorded>")
    for source in sources:
        print(f"  - {source['node_id']}: {source['title']!r}")
        if source.get("relation"):
            print(f"    via {source['edge_type']} relation={source['relation']!r}")

    print("Grounded entities:")
    entities = report["grounded_entities"]
    if not entities:
        print("  <none recorded>")
    for index, entity in enumerate(entities, start=1):
        statuses = [entity.get("status")]
        statuses.extend(entity.get("metadata_statuses") or [])
        statuses = [str(status) for status in dict.fromkeys(statuses) if status]
        if entity.get("query_overlap_entity") and "query_overlap_entity" not in statuses:
            statuses.append("query_overlap_entity")
        print(f"  {index}. {entity.get('name')!r} ({entity.get('type') or '?'})")
        print(f"     status: {', '.join(statuses) or 'unknown'}")
        if entity.get("relation_to_image"):
            print(f"     relation_to_image: {entity['relation_to_image']!r}")
        if entity.get("evidence"):
            print(f"     evidence: {entity['evidence']!r}")
        for target in entity.get("linked_targets") or []:
            overlap = bool(target.get("query_overlap_entity"))
            print(
                "     linked_to: "
                f"{target.get('dst_node_id')} ({target.get('dst_title')!r}), "
                f"query_overlap={overlap}"
            )
        for state in entity.get("runner_state") or []:
            print(f"     runner_state: {state.get('status')} ({state.get('section')})")


def main() -> None:
    args = parse_args()
    try:
        report = build_report(Path(args.graph_dir), args.image_node_id)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
