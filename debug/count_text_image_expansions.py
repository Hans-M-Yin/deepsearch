"""Count image nodes expanded from each text node by origin.

Examples:
  python debug/count_text_image_expansions.py \
    --graph-dir runs/my_graph_run

  python debug/count_text_image_expansions.py \
    --graph-dir runs/my_graph_run \
    --json \
    --include-zero

The graph directory may also contain visual_plans.jsonl. It is used to recover
the source text node for older wiki-inline image records that kept their
``image_variants[].source = wikipedia_inline`` marker but lost
``metadata.source_text_node_id`` during persistence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TEXT_NODE_TYPE = "text"
IMAGE_NODE_TYPE = "image"
SEARCH_RETRIEVED_EDGE_TYPE = "search_retrieved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize, for each text node, how many image nodes were expanded via "
            "visual_plan and wiki_inline from a persisted graph directory."
        )
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing nodes.jsonl and edges.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows to print after sorting. <=0 means all rows.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("total", "visual_plan", "wiki_inline", "title", "node_id"),
        default="total",
        help="Row sort key.",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include text nodes with zero image expansions in the output rows.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only count nodes and edges whose status is active.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object instead of a human-readable table.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def is_active(record: dict[str, Any]) -> bool:
    return str(record.get("status") or "active").strip().lower() == "active"


def short(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def image_variant_sources(node: dict[str, Any]) -> list[str]:
    variants = node.get("image_variants") or []
    if not isinstance(variants, list):
        return []
    sources: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        source = str(variant.get("source") or "").strip().lower()
        if source:
            sources.add(source)
    return sorted(sources)


def image_origin(node: dict[str, Any]) -> str:
    """Classify image origin using the same precedence as visualize_graph.py."""
    metadata = node.get("metadata") or {}
    source = node.get("source") or {}
    source_type = str(source.get("source_type") or "").strip().lower() if isinstance(source, dict) else ""
    origin = str(metadata.get("image_origin") or "").strip().lower()
    variant_sources = image_variant_sources(node)
    if source_type == "wikipedia_inline_image" or origin == "wikipedia_inline" or "wikipedia_inline" in variant_sources:
        return "wiki_inline"
    if source_type in {"image_search_bundle", "image_search"}:
        return "visual_plan"
    if source_type:
        return f"other:{source_type}"
    return "other:unknown"


def image_urls(node: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for key in ("image_url", "oss_uri", "thumb_oss_uri"):
        value = str(node.get(key) or "").strip()
        if value:
            urls.add(value)
    for variant in node.get("image_variants") or []:
        if not isinstance(variant, dict):
            continue
        for key in ("image_url", "thumbnail_url"):
            value = str(variant.get(key) or "").strip()
            if value:
                urls.add(value)
    return urls


def wiki_inline_plan_sources(visual_plans: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Index persisted wiki-inline plans by their original/thumbnail URLs."""
    sources_by_url: dict[str, set[str]] = defaultdict(set)
    for plan in visual_plans:
        metadata = plan.get("metadata") or {}
        planner = str(plan.get("planner") or "").strip().lower()
        plan_source = str(metadata.get("plan_source") or "").strip().lower() if isinstance(metadata, dict) else ""
        if planner != "wikipedia_inline_image_planner" and plan_source != "wikipedia_inline_image":
            continue
        source_text_node_id = str(plan.get("node_id") or "").strip()
        if not source_text_node_id:
            continue
        candidate_urls: set[str] = set()
        if isinstance(metadata, dict):
            for key in ("image_url", "thumbnail_url"):
                value = str(metadata.get(key) or "").strip()
                if value:
                    candidate_urls.add(value)
        target = plan.get("target") or {}
        if isinstance(target, dict):
            value = str(target.get("url") or "").strip()
            if value:
                candidate_urls.add(value)
            target_metadata = target.get("metadata") or {}
            if isinstance(target_metadata, dict):
                for key in ("image_url", "thumbnail_url"):
                    value = str(target_metadata.get(key) or "").strip()
                    if value:
                        candidate_urls.add(value)
        for query_spec in plan.get("query_specs") or []:
            if not isinstance(query_spec, dict):
                continue
            query_metadata = query_spec.get("metadata") or {}
            if not isinstance(query_metadata, dict):
                continue
            for key in ("image_url", "thumbnail_url"):
                value = str(query_metadata.get(key) or "").strip()
                if value:
                    candidate_urls.add(value)
        for url in candidate_urls:
            sources_by_url[url].add(source_text_node_id)
    return sources_by_url


def matched_wiki_inline_plan_sources(
    node: dict[str, Any],
    plan_sources_by_url: dict[str, set[str]],
) -> set[str]:
    matched_source_ids: set[str] = set()
    for url in image_urls(node):
        matched_source_ids.update(plan_sources_by_url.get(url, set()))
    return matched_source_ids


def node_title(node: dict[str, Any]) -> str:
    for key in ("title", "summary", "caption", "canonical_id", "node_id"):
        value = node.get(key)
        if value:
            return str(value)
    return str(node.get("node_id") or "")


def collect_visual_plan_pairs(
    *,
    edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    plan_sources_by_url: dict[str, set[str]],
) -> tuple[dict[str, set[str]], int, int]:
    pairs: dict[str, set[str]] = defaultdict(set)
    skipped_edges = 0
    excluded_wiki_inline_edges = 0
    for edge in edges:
        if str(edge.get("edge_type") or "").strip().lower() != SEARCH_RETRIEVED_EDGE_TYPE:
            continue
        src_node_id = str(edge.get("src_node_id") or "").strip()
        dst_node_id = str(edge.get("dst_node_id") or "").strip()
        if not src_node_id or not dst_node_id:
            skipped_edges += 1
            continue
        edge_source = edge.get("source") or {}
        edge_source_type = str(edge_source.get("source_type") or "").strip().lower() if isinstance(edge_source, dict) else ""
        src_node_type = str(edge.get("src_node_type") or "").strip().lower()
        dst_node_type = str(edge.get("dst_node_type") or "").strip().lower()
        if edge_source_type not in {"image_search", "image_search_bundle"}:
            continue
        if src_node_type and src_node_type != TEXT_NODE_TYPE:
            skipped_edges += 1
            continue
        if dst_node_type and dst_node_type != IMAGE_NODE_TYPE:
            skipped_edges += 1
            continue
        src_node = nodes_by_id.get(src_node_id)
        dst_node = nodes_by_id.get(dst_node_id)
        if src_node is None or dst_node is None:
            skipped_edges += 1
            continue
        if src_node.get("node_type") != TEXT_NODE_TYPE or dst_node.get("node_type") != IMAGE_NODE_TYPE:
            skipped_edges += 1
            continue
        # Match visualize_graph.py: node-level wiki-inline signals take precedence
        # over the generic image_search/image_search_bundle source type.
        if image_origin(dst_node) == "wiki_inline" or matched_wiki_inline_plan_sources(dst_node, plan_sources_by_url):
            excluded_wiki_inline_edges += 1
            continue
        pairs[src_node_id].add(dst_node_id)
    return pairs, skipped_edges, excluded_wiki_inline_edges


def collect_wiki_inline_pairs(
    *,
    nodes: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    plan_sources_by_url: dict[str, set[str]],
) -> tuple[dict[str, set[str]], int, int, int]:
    pairs: dict[str, set[str]] = defaultdict(set)
    skipped_nodes = 0
    recovered_from_plans = 0
    ambiguous_plan_matches = 0
    for node in nodes:
        if node.get("node_type") != IMAGE_NODE_TYPE:
            continue
        matched_source_ids = matched_wiki_inline_plan_sources(node, plan_sources_by_url)
        if image_origin(node) != "wiki_inline" and not matched_source_ids:
            continue
        metadata = node.get("metadata") or {}
        source_text_node_id = str(metadata.get("source_text_node_id") or "").strip()
        image_node_id = str(node.get("node_id") or "").strip()
        if not source_text_node_id:
            if len(matched_source_ids) == 1:
                source_text_node_id = next(iter(matched_source_ids))
                recovered_from_plans += 1
            elif len(matched_source_ids) > 1:
                ambiguous_plan_matches += 1
        if not source_text_node_id or not image_node_id:
            skipped_nodes += 1
            continue
        source_node = nodes_by_id.get(source_text_node_id)
        if source_node is None or source_node.get("node_type") != TEXT_NODE_TYPE:
            skipped_nodes += 1
            continue
        pairs[source_text_node_id].add(image_node_id)
    return pairs, skipped_nodes, recovered_from_plans, ambiguous_plan_matches


def build_rows(
    *,
    text_nodes: list[dict[str, Any]],
    visual_plan_pairs: dict[str, set[str]],
    wiki_inline_pairs: dict[str, set[str]],
    include_zero: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in text_nodes:
        node_id = str(node.get("node_id") or "")
        visual_plan_count = len(visual_plan_pairs.get(node_id, set()))
        wiki_inline_count = len(wiki_inline_pairs.get(node_id, set()))
        total_count = visual_plan_count + wiki_inline_count
        if not include_zero and total_count <= 0:
            continue
        rows.append(
            {
                "node_id": node_id,
                "title": node_title(node),
                "visual_plan": visual_plan_count,
                "wiki_inline": wiki_inline_count,
                "total": total_count,
            }
        )
    return rows


def sort_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "title":
        return sorted(rows, key=lambda row: (str(row["title"]).lower(), row["node_id"]))
    if sort_by == "node_id":
        return sorted(rows, key=lambda row: row["node_id"])
    return sorted(
        rows,
        key=lambda row: (
            -int(row[sort_by]),
            -int(row["total"]),
            str(row["title"]).lower(),
            row["node_id"],
        ),
    )


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<no matching text nodes>"

    title_width = min(
        72,
        max(len("title"), max(len(short(row["title"], 72)) for row in rows)),
    )
    node_width = min(
        32,
        max(len("node_id"), max(len(short(row["node_id"], 32)) for row in rows)),
    )
    header = (
        f"{'node_id':<{node_width}}  "
        f"{'visual':>6}  "
        f"{'inline':>6}  "
        f"{'total':>6}  "
        f"title"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{short(row['node_id'], node_width):<{node_width}}  "
            f"{row['visual_plan']:>6}  "
            f"{row['wiki_inline']:>6}  "
            f"{row['total']:>6}  "
            f"{short(row['title'], title_width)}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    nodes = load_jsonl(graph_dir / "nodes.jsonl")
    edges = load_jsonl(graph_dir / "edges.jsonl")
    visual_plans = load_jsonl(graph_dir / "visual_plans.jsonl")

    if args.active_only:
        nodes = [node for node in nodes if is_active(node)]
        edges = [edge for edge in edges if is_active(edge)]

    nodes_by_id = {
        str(node.get("node_id")): node
        for node in nodes
        if isinstance(node, dict) and node.get("node_id")
    }
    text_nodes = [node for node in nodes if node.get("node_type") == TEXT_NODE_TYPE]
    image_nodes = [node for node in nodes if node.get("node_type") == IMAGE_NODE_TYPE]

    plan_sources_by_url = wiki_inline_plan_sources(visual_plans)
    visual_plan_pairs, skipped_visual_edges, excluded_wiki_inline_edges = collect_visual_plan_pairs(
        edges=edges,
        nodes_by_id=nodes_by_id,
        plan_sources_by_url=plan_sources_by_url,
    )
    (
        wiki_inline_pairs,
        skipped_wiki_inline_nodes,
        wiki_inline_nodes_recovered_from_plans,
        ambiguous_wiki_inline_plan_matches,
    ) = collect_wiki_inline_pairs(
        nodes=image_nodes,
        nodes_by_id=nodes_by_id,
        plan_sources_by_url=plan_sources_by_url,
    )

    rows = build_rows(
        text_nodes=text_nodes,
        visual_plan_pairs=visual_plan_pairs,
        wiki_inline_pairs=wiki_inline_pairs,
        include_zero=args.include_zero,
    )
    all_rows = sort_rows(rows, args.sort_by)
    rows = list(all_rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    summary = {
        "graph_dir": str(graph_dir),
        "text_node_count": len(text_nodes),
        "image_node_count": len(image_nodes),
        "text_nodes_with_any_images": sum(1 for row in all_rows if row["total"] > 0),
        "visual_plan_source_text_nodes": len(visual_plan_pairs),
        "wiki_inline_source_text_nodes": len(wiki_inline_pairs),
        "visual_plan_source_image_pairs": sum(len(image_ids) for image_ids in visual_plan_pairs.values()),
        "wiki_inline_source_image_pairs": sum(len(image_ids) for image_ids in wiki_inline_pairs.values()),
        "skipped_visual_plan_edges": skipped_visual_edges,
        "excluded_wiki_inline_edges_from_visual_plan": excluded_wiki_inline_edges,
        "skipped_wiki_inline_image_nodes": skipped_wiki_inline_nodes,
        "wiki_inline_image_nodes_recovered_from_visual_plans": wiki_inline_nodes_recovered_from_plans,
        "ambiguous_wiki_inline_visual_plan_matches": ambiguous_wiki_inline_plan_matches,
        "active_only": bool(args.active_only),
    }

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"graph_dir={graph_dir}")
    print(
        "summary: "
        f"text_nodes={summary['text_node_count']} "
        f"image_nodes={summary['image_node_count']} "
        f"text_nodes_with_any_images={summary['text_nodes_with_any_images']} "
        f"visual_pairs={summary['visual_plan_source_image_pairs']} "
        f"wiki_inline_pairs={summary['wiki_inline_source_image_pairs']} "
        f"skipped_visual_edges={summary['skipped_visual_plan_edges']} "
        f"excluded_wiki_edges={summary['excluded_wiki_inline_edges_from_visual_plan']} "
        f"recovered_wiki_nodes={summary['wiki_inline_image_nodes_recovered_from_visual_plans']} "
        f"skipped_wiki_inline_nodes={summary['skipped_wiki_inline_image_nodes']} "
        f"ambiguous_wiki_matches={summary['ambiguous_wiki_inline_visual_plan_matches']}"
    )
    print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
