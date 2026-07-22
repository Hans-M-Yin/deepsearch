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


def main() -> int:
    args = parse_args()
    store = JsonlGraphStore(Path(args.graph_dir))
    nodes = store.list_nodes()

    isolated: list[dict[str, Any]] = []
    node_type_counts: Counter[str] = Counter()
    image_origin_counts: Counter[str] = Counter()

    for node in nodes:
        node_id = str(node.get("node_id") or "")
        in_edges = store.edges_to(node_id)
        out_edges = store.edges_from(node_id)
        if in_edges or out_edges:
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
        "nodes": isolated,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
