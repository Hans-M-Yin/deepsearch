"""Print sampled visual-plan image nodes for manual quality review.

A visual-plan image node is identified using the same precedence as
``debug/visualize_graph.py``: Wikipedia-inline images are excluded first, and
nodes with ``source.source_type`` equal to ``image_search`` or
``image_search_bundle`` are retained.

Example:
  python synthesis/post_process/sample_visual_plan_nodes.py \
    --graph-dir runs/my_graph \
    --num-sample 100 \
    --seed 20260729
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


VISUAL_PLAN_SOURCE_TYPES = {"image_search", "image_search_bundle"}
SEARCH_RETRIEVED = "search_retrieved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory containing nodes.jsonl and edges.jsonl.",
    )
    parser.add_argument(
        "--num-sample",
        type=int,
        required=True,
        help="Number of visual-plan image nodes to sample. <=0 prints all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
        help="Deterministic random seed for sampling.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def image_origin(node: dict[str, Any]) -> str:
    """Classify image origin with the same logic as debug/visualize_graph.py."""
    metadata = _as_dict(node.get("metadata"))
    source = _as_dict(node.get("source"))
    source_type = _clean(source.get("source_type"))
    origin = _clean(metadata.get("image_origin")).lower()
    variant_sources = {
        _clean(_as_dict(variant).get("source"))
        for variant in node.get("image_variants") or []
        if isinstance(variant, dict)
    }
    if (
        source_type == "wikipedia_inline_image"
        or origin == "wikipedia_inline"
        or "wikipedia_inline" in variant_sources
    ):
        return "wiki_inline"
    if source_type in VISUAL_PLAN_SOURCE_TYPES:
        return "visual_plan"
    return f"other:{source_type or 'unknown'}"


def image_url(node: dict[str, Any]) -> str:
    metadata = _as_dict(node.get("metadata"))
    resolved_image = _as_dict(metadata.get("resolved_image"))
    candidates = (
        resolved_image.get("resolved_url"),
        node.get("image_url"),
        resolved_image.get("original_url"),
        node.get("oss_uri"),
        node.get("thumb_oss_uri"),
    )
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    for variant in node.get("image_variants") or []:
        if not isinstance(variant, dict):
            continue
        for key in ("image_url", "thumbnail_url"):
            value = str(variant.get(key) or "").strip()
            if value:
                return value
    return "<missing-image-url>"


def source_text_node(
    image_node: dict[str, Any],
    *,
    incoming_edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the best persisted text -> image source, matching visualize_graph."""
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for edge in incoming_edges:
        source = nodes_by_id.get(str(edge.get("src_node_id") or ""))
        if source is not None and source.get("node_type") == "text":
            candidates.append((edge, source))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item[0].get("edge_type") != SEARCH_RETRIEVED,
            str(item[1].get("node_id") or ""),
        )
    )
    edge, node = candidates[0]
    source = _as_dict(node.get("source"))
    return {
        "node_id": str(node.get("node_id") or "<missing-text-node-id>"),
        "title": _clean(node.get("title") or node.get("canonical_id") or node.get("node_id")),
        "source_url": str(source.get("url") or "").strip(),
        "edge_id": str(edge.get("edge_id") or ""),
        "edge_relation": _clean(edge.get("relation")),
    }


def build_rows(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes_by_id = {
        str(node.get("node_id") or ""): node
        for node in nodes
        if node.get("node_id")
    }
    incoming: dict[str, list[dict[str, Any]]] = {}
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        dst_node_id = str(edge.get("dst_node_id") or "")
        src_node_id = str(edge.get("src_node_id") or "")
        if dst_node_id:
            incoming.setdefault(dst_node_id, []).append(edge)
        if src_node_id:
            outgoing.setdefault(src_node_id, []).append(edge)

    rows: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("node_type") != "image" or image_origin(node) != "visual_plan":
            continue
        metadata = _as_dict(node.get("metadata"))
        source = source_text_node(
            node,
            incoming_edges=incoming.get(str(node.get("node_id") or ""), []),
            nodes_by_id=nodes_by_id,
        )
        image_node_id = str(node.get("node_id") or "")
        downstream_text_relations: list[dict[str, str]] = []
        for edge in outgoing.get(image_node_id, []):
            target = nodes_by_id.get(str(edge.get("dst_node_id") or ""))
            if edge.get("edge_type") != "image_depicts" or not target or target.get("node_type") != "text":
                continue
            downstream_text_relations.append(
                {
                    "relation": _clean(edge.get("relation")) or "<missing-relation>",
                    "title": _clean(target.get("title") or target.get("canonical_id") or target.get("node_id")),
                    "edge_id": str(edge.get("edge_id") or ""),
                }
            )
        downstream_text_relations.sort(
            key=lambda item: (item["relation"].lower(), item["title"].lower(), item["edge_id"])
        )
        rows.append(
            {
                "image_node_id": image_node_id,
                "search_query": _clean(metadata.get("search_query") or metadata.get("query")) or "<missing-search-query>",
                "source_text_node": (
                    f"{source['node_id']} | {source['title']}"
                    if source is not None
                    else "<missing-source-text-node>"
                ),
                "image_url": image_url(node),
                "downstream_text_relations": downstream_text_relations,
            }
        )
    return sorted(rows, key=lambda row: row["image_node_id"])


def main() -> int:
    args = parse_args()
    graph_dir = args.graph_dir.expanduser().resolve()
    if not graph_dir.is_dir():
        raise SystemExit(f"error: graph directory does not exist: {graph_dir}")

    try:
        rows = build_rows(
            load_jsonl(graph_dir / "nodes.jsonl"),
            load_jsonl(graph_dir / "edges.jsonl"),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if not rows:
        raise SystemExit("error: no visual-plan image nodes found")

    requested = int(args.num_sample)
    if requested <= 0 or requested >= len(rows):
        selected = rows
    else:
        selected = random.Random(args.seed).sample(rows, requested)
        selected.sort(key=lambda row: row["image_node_id"])

    print(
        f"visual_plan_image_nodes={len(rows)} sampled={len(selected)} "
        f"seed={args.seed} graph_dir={graph_dir}"
    )
    print("image_node_id\tsearch_query\tsource_text_node\timage_url")
    for row in selected:
        print(
            f"{row['image_node_id']}\t{row['search_query']}\t"
            f"{row['source_text_node']}\t{row['image_url']}"
        )
        for downstream in row["downstream_text_relations"]:
            print(f"  {downstream['relation']} -> {downstream['title']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
