"""List isolated nodes (no in-edges and no out-edges) from a graph.

For image nodes, also classify origin as wiki_inline / visual_plan / other, reusing
existing heuristics from debug tools.

Example:
  python debug/list_isolated_nodes.py \
    --graph-dir runs/0712_multi_seed_visual_test_8192_6 \
    --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List isolated graph nodes and classify image-node origin.")
    parser.add_argument("--graph-dir", required=True, help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of isolated nodes to print; <=0 means all.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _image_variant_sources(node: dict[str, Any]) -> list[str]:
    variants = node.get("image_variants") or []
    if not isinstance(variants, list):
        return []
    sources: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        source = str(variant.get("source") or "").strip()
        if source:
            sources.add(source)
    return sorted(sources)


def _image_origin(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    source = node.get("source") or {}
    source_type = source.get("source_type") if isinstance(source, dict) else None
    image_origin = str(metadata.get("image_origin") or "").strip().lower()
    variant_sources = _image_variant_sources(node)
    if source_type == "wikipedia_inline_image" or image_origin == "wikipedia_inline":
        return "wiki_inline"
    if "wikipedia_inline" in variant_sources:
        return "wiki_inline"
    if source_type in {"image_search_bundle", "image_search"}:
        return "visual_plan"
    if source_type:
        return f"other:{source_type}"
    return "other:unknown"


def _node_title(node: dict[str, Any]) -> str:
    for key in ("title", "caption", "summary", "canonical_id", "node_id"):
        value = node.get(key)
        if value:
            return " ".join(str(value).split())
    return "<untitled>"


def _node_degrees(edges: list[dict[str, Any]]) -> Counter[str]:
    """Calculate total degree (in-degree + out-degree) for every node."""
    degrees: Counter[str] = Counter()
    for edge in edges:
        src_node_id = edge.get("src_node_id")
        dst_node_id = edge.get("dst_node_id")
        if src_node_id:
            degrees[str(src_node_id)] += 1
        if dst_node_id:
            degrees[str(dst_node_id)] += 1
    return degrees


def _degree_category(node: dict[str, Any]) -> str | None:
    node_type = str(node.get("node_type") or "unknown")
    if node_type == "text":
        return "text"
    if node_type == "image":
        origin = _image_origin(node)
        if origin in {"wiki_inline", "visual_plan"}:
            return origin
    return None


def _average_degrees(
    nodes: list[dict[str, Any]], degrees: Counter[str]
) -> dict[str, dict[str, int | float]]:
    categories = ("wiki_inline", "visual_plan", "text")
    degree_sums: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    for node in nodes:
        category = _degree_category(node)
        if category is None:
            continue
        node_id = str(node.get("node_id") or "")
        node_counts[category] += 1
        degree_sums[category] += degrees[node_id]

    return {
        category: {
            "node_count": node_counts[category],
            "average_degree": (
                degree_sums[category] / node_counts[category]
                if node_counts[category]
                else 0.0
            ),
        }
        for category in categories
    }


def main() -> int:
    args = parse_args()
    store = JsonlGraphStore(Path(args.graph_dir))
    nodes = store.list_nodes()
    degrees = _node_degrees(store.list_edges())
    connected_node_ids = set(degrees)

    isolated: list[dict[str, Any]] = []
    node_type_counts: Counter[str] = Counter()
    image_origin_counts: Counter[str] = Counter()

    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if node_id in connected_node_ids:
            continue
        node_type = str(node.get("node_type") or "unknown")
        record = {
            "node_id": node_id,
            "node_type": node_type,
            "title": _node_title(node),
        }
        if node_type == "image":
            record["image_origin"] = _image_origin(node)
            record["image_url"] = node.get("image_url")
            record["source_page_url"] = node.get("source_page_url")
            image_origin_counts[record["image_origin"]] += 1
        node_type_counts[node_type] += 1
        isolated.append(record)

    isolated.sort(key=lambda item: (item.get("node_type") or "", item.get("image_origin") or "", item.get("title") or "", item.get("node_id") or ""))
    if args.limit and args.limit > 0:
        isolated = isolated[: args.limit]

    payload = {
        "graph_dir": str(Path(args.graph_dir).resolve()),
        "isolated_node_count": sum(node_type_counts.values()),
        "isolated_node_type_counts": dict(node_type_counts),
        "isolated_image_origin_counts": dict(image_origin_counts),
        "average_total_degree_by_category": _average_degrees(nodes, degrees),
        "nodes": isolated,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
