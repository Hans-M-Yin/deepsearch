"""Summarize graph connectivity, degree statistics, and isolated-node samples.

Examples:
  python debug/connection_check.py \
    --graph-dir runs/0712_multi_seed_visual_test4

  python debug/connection_check.py \
    --graph-dir synthesis/runs/mock_graph_review_20260712_env/query_overlap \
    --limit 10 \
    --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect graph connectivity using weakly connected components, summarize degree "
            "statistics for text/image nodes, and print representative fully isolated nodes."
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
        default=10,
        help="How many isolated-node examples to print. Use 0 to skip examples.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only count active nodes and active edges whose endpoints are active nodes.",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=140,
        help="Max characters used when printing long text fields.",
    )
    parser.add_argument(
        "--top-components",
        type=int,
        default=10,
        help="How many of the largest weakly connected components to summarize.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object instead of human-readable text.",
    )
    return parser.parse_args()


def is_active(record: dict[str, Any]) -> bool:
    return str(record.get("status") or "active") == "active"


def short(value: Any, max_len: int) -> str:
    text = " ".join(str(value or "").split())
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def source_url_of(node: dict[str, Any]) -> str | None:
    source = node.get("source") or {}
    if isinstance(source, dict):
        url = source.get("url")
        if url:
            return str(url)
    for key in ("image_url", "source_page_url", "oss_uri", "thumb_oss_uri"):
        value = node.get(key)
        if value:
            return str(value)
    return None


def image_variant_sources(node: dict[str, Any]) -> list[str]:
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


def status_distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown").strip() or "unknown"
        counter[status] += 1
    return {str(key): counter[key] for key in sorted(counter)}


def entity_key(entity: dict[str, Any]) -> tuple[str, str, str]:
    if not isinstance(entity, dict):
        return ("", "", "")
    return (
        " ".join(str(entity.get("name") or "").split()).lower(),
        " ".join(str(entity.get("relation_to_image") or "").split()).lower(),
        " ".join(str(entity.get("evidence") or "").split()).lower(),
    )


def classify_image_type(node: dict[str, Any]) -> str:
    source = node.get("source") or {}
    metadata = node.get("metadata") or {}
    source_type = str(source.get("source_type") if isinstance(source, dict) else "" or "").strip()
    image_origin = str(metadata.get("image_origin") or "").strip().lower()
    variant_sources = image_variant_sources(node)
    if source_type == "wikipedia_inline_image" or image_origin == "wikipedia_inline":
        return "wiki_inline"
    if "wikipedia_inline" in variant_sources:
        return "wiki_inline"
    if source_type in {"image_search_bundle", "image_search"}:
        return "visual_plan"
    if source_type:
        return f"other:{source_type}"
    return "other:unknown"


def node_display_title(node: dict[str, Any], *, summary_chars: int) -> str:
    for key in ("title", "caption", "summary", "canonical_id", "node_id"):
        value = node.get(key)
        if value:
            return short(value, summary_chars)
    return "<untitled>"


def degree_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(int(record.get("degree") or 0) for record in records)
    return {str(key): counter[key] for key in sorted(counter)}


def average_degree_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    total_in = sum(int(record.get("in_degree") or 0) for record in records)
    total_out = sum(int(record.get("out_degree") or 0) for record in records)
    total_degree = total_in + total_out
    return {
        "count": count,
        "avg_degree": (total_degree / count) if count else 0.0,
        "avg_in_degree": (total_in / count) if count else 0.0,
        "avg_out_degree": (total_out / count) if count else 0.0,
        "degree_distribution": degree_distribution(records),
    }


def image_grounding_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    grounded_distribution: Counter[int] = Counter()
    no_text_reason_counter: Counter[str] = Counter()
    total_grounded = 0
    total_linked_text = 0
    total_without_text = 0
    total_query_overlap = 0
    images_with_grounded = 0
    images_with_linked_text = 0
    images_with_grounded_but_no_text = 0

    for record in records:
        grounded_count = int(record.get("grounded_entity_count") or 0)
        linked_text_count = int(record.get("linked_text_edge_count") or 0)
        without_text_count = int(record.get("grounded_without_text_edge_count") or 0)
        query_overlap_count = int(record.get("query_overlap_grounded_entity_count") or 0)

        grounded_distribution[grounded_count] += 1
        total_grounded += grounded_count
        total_linked_text += linked_text_count
        total_without_text += without_text_count
        total_query_overlap += query_overlap_count
        if grounded_count > 0:
            images_with_grounded += 1
        if linked_text_count > 0:
            images_with_linked_text += 1
        if grounded_count > 0 and without_text_count > 0:
            images_with_grounded_but_no_text += 1
        no_text_reason_counter.update(record.get("grounded_no_text_reason_counts") or {})

    return {
        "total_grounded_entities": total_grounded,
        "avg_grounded_entities_per_image": (total_grounded / count) if count else 0.0,
        "images_with_grounded_entities": images_with_grounded,
        "images_with_linked_text_entities": images_with_linked_text,
        "linked_text_edge_count": total_linked_text,
        "grounded_without_text_edge_count": total_without_text,
        "images_with_grounded_but_no_text": images_with_grounded_but_no_text,
        "query_overlap_flagged_entity_count": total_query_overlap,
        "grounded_entity_distribution": {str(key): grounded_distribution[key] for key in sorted(grounded_distribution)},
        "no_text_reason_counts": {str(key): no_text_reason_counter[key] for key in sorted(no_text_reason_counter)},
    }


def merge_group_stats(
    records: list[dict[str, Any]],
    *,
    include_image_grounding: bool,
) -> dict[str, Any]:
    stats = average_degree_stats(records)
    if include_image_grounding:
        stats["grounding_stats"] = image_grounding_stats(records)
    return stats


def load_runner_state(graph_dir: Path) -> dict[str, Any]:
    state_path = graph_dir / "graph_runner_state.json"
    if not state_path.exists():
        return {
            "found": False,
            "path": str(state_path),
            "load_error": None,
            "status": None,
            "queue_size": 0,
            "image_entity_task_count": 0,
            "image_entity_pending_link_count": 0,
            "query_overlap_pending_link_count": 0,
            "queue": [],
        }
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "found": True,
            "path": str(state_path),
            "load_error": f"{exc.__class__.__name__}: {exc}",
            "status": None,
            "queue_size": 0,
            "image_entity_task_count": 0,
            "image_entity_pending_link_count": 0,
            "query_overlap_pending_link_count": 0,
            "queue": [],
        }

    queue = list(payload.get("queue") or [])
    image_entity_task_count = 0
    image_entity_pending_link_count = 0
    query_overlap_pending_link_count = 0
    for task in queue:
        if not isinstance(task, dict):
            continue
        if str(task.get("task_type") or "") != "text_expand":
            continue
        metadata = task.get("metadata") or {}
        if not isinstance(metadata, dict) or metadata.get("task_origin") != "image_entity":
            continue
        image_entity_task_count += 1
        for pending in metadata.get("pending_parent_links") or []:
            if not isinstance(pending, dict):
                continue
            if str(pending.get("link_type") or "wiki_link") != "image_entity":
                continue
            image_entity_pending_link_count += 1
            if bool(pending.get("query_overlap_entity")):
                query_overlap_pending_link_count += 1
    return {
        "found": True,
        "path": str(state_path),
        "load_error": None,
        "status": payload.get("status"),
        "queue_size": len(queue),
        "image_entity_task_count": image_entity_task_count,
        "image_entity_pending_link_count": image_entity_pending_link_count,
        "query_overlap_pending_link_count": query_overlap_pending_link_count,
        "queue": queue,
    }


def queued_image_entity_links(
    *,
    queue_records: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    links_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in queue_records:
        if not isinstance(task, dict):
            continue
        if str(task.get("task_type") or "") != "text_expand":
            continue
        metadata = task.get("metadata") or {}
        if not isinstance(metadata, dict) or metadata.get("task_origin") != "image_entity":
            continue
        for pending in metadata.get("pending_parent_links") or []:
            if not isinstance(pending, dict):
                continue
            if str(pending.get("link_type") or "wiki_link") != "image_entity":
                continue
            parent_node_id = pending.get("parent_node_id")
            if parent_node_id in nodes_by_id:
                links_by_image[str(parent_node_id)].append(pending)
    return links_by_image


def image_text_edge_counts(
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src not in nodes_by_id or dst not in nodes_by_id:
            continue
        if nodes_by_id[src].get("node_type") != "image":
            continue
        if nodes_by_id[dst].get("node_type") != "text":
            continue
        edge_type = str(edge.get("edge_type") or "").strip()
        if edge_type and edge_type != "image_depicts":
            continue
        counter[str(src)] += 1
    return counter


def linked_grounded_entities_by_image(
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grounded_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src not in nodes_by_id or dst not in nodes_by_id:
            continue
        if nodes_by_id[src].get("node_type") != "image":
            continue
        if nodes_by_id[dst].get("node_type") != "text":
            continue
        edge_type = str(edge.get("edge_type") or "").strip()
        if edge_type and edge_type != "image_depicts":
            continue
        for evidence_ref in edge.get("evidence_refs") or []:
            if not isinstance(evidence_ref, dict):
                continue
            metadata = evidence_ref.get("metadata") or {}
            grounded = metadata.get("grounded_entity") if isinstance(metadata, dict) else None
            if isinstance(grounded, dict):
                grounded_by_image[str(src)].append(grounded)
    return grounded_by_image


def derive_grounded_no_text_reason_counts(
    *,
    grounded_entities: list[dict[str, Any]],
    unresolved_grounded_entities: list[dict[str, Any]],
    query_overlap_grounded_entities: list[dict[str, Any]],
    linked_grounded_entities: list[dict[str, Any]],
    queued_pending_links: list[dict[str, Any]],
) -> dict[str, int]:
    linked_keys: Counter[tuple[str, str, str]] = Counter(
        entity_key(item) for item in linked_grounded_entities if any(entity_key(item))
    )
    unresolved_statuses_by_key: dict[tuple[str, str, str], deque[str]] = defaultdict(deque)
    for item in unresolved_grounded_entities:
        key = entity_key(item)
        if not any(key):
            continue
        status = str(item.get("status") or "unknown").strip() or "unknown"
        unresolved_statuses_by_key[key].append(status)
    queued_keys: Counter[tuple[str, str, str]] = Counter()
    for pending in queued_pending_links:
        if not isinstance(pending, dict):
            continue
        key = entity_key(pending.get("entity") or {})
        if any(key):
            queued_keys[key] += 1
    query_overlap_keys: Counter[tuple[str, str, str]] = Counter(
        entity_key(item) for item in query_overlap_grounded_entities if any(entity_key(item))
    )

    reason_counts: Counter[str] = Counter()
    for entity in grounded_entities:
        key = entity_key(entity)
        if linked_keys[key] > 0:
            linked_keys[key] -= 1
            continue
        if unresolved_statuses_by_key[key]:
            reason_counts[unresolved_statuses_by_key[key].popleft()] += 1
            continue
        if queued_keys[key] > 0:
            queued_keys[key] -= 1
            reason_counts["still_queued_image_entity"] += 1
            continue
        if query_overlap_keys[key] > 0:
            query_overlap_keys[key] -= 1
            reason_counts["query_overlap_without_text"] += 1
            continue
        reason_counts["other_unexplained"] += 1
    return {str(key): reason_counts[key] for key in sorted(reason_counts)}


def build_graph_state(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    Counter[str],
    Counter[str],
    dict[str, set[str]],
    int,
    int,
    int,
]:
    nodes_by_id = {
        record["node_id"]: record
        for record in nodes
        if isinstance(record.get("node_id"), str)
    }
    node_ids = set(nodes_by_id)
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    undirected_adj: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    edges_with_missing_src = 0
    edges_with_missing_dst = 0
    edges_between_known_nodes = 0

    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        src_known = src in node_ids
        dst_known = dst in node_ids
        if src_known:
            out_degree[str(src)] += 1
        else:
            edges_with_missing_src += 1
        if dst_known:
            in_degree[str(dst)] += 1
        else:
            edges_with_missing_dst += 1
        if not (src_known and dst_known):
            continue
        edges_between_known_nodes += 1
        if src == dst:
            continue
        undirected_adj[str(src)].add(str(dst))
        undirected_adj[str(dst)].add(str(src))

    return (
        nodes_by_id,
        in_degree,
        out_degree,
        undirected_adj,
        edges_with_missing_src,
        edges_with_missing_dst,
        edges_between_known_nodes,
    )


def summarize_node(
    node: dict[str, Any],
    *,
    in_degree: int,
    out_degree: int,
    linked_text_edge_count: int,
    linked_grounded_entities: list[dict[str, Any]],
    queued_pending_links: list[dict[str, Any]],
    summary_chars: int,
) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    image_grounding = metadata.get("image_grounding") or {}
    grounded_entities = list(metadata.get("grounded_entities") or [])
    unresolved_grounded_entities = list(metadata.get("unresolved_grounded_entities") or [])
    query_overlap_grounded_entities = list(metadata.get("query_overlap_grounded_entities") or [])
    summary = {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "status": node.get("status"),
        "title": node_display_title(node, summary_chars=summary_chars),
        "subtype": node.get("subtype"),
        "canonical_id": node.get("canonical_id"),
        "source_type": (node.get("source") or {}).get("source_type") if isinstance(node.get("source"), dict) else None,
        "source_url": source_url_of(node),
        "in_degree": int(in_degree),
        "out_degree": int(out_degree),
        "degree": int(in_degree) + int(out_degree),
        "created_at": node.get("created_at"),
        "updated_at": node.get("updated_at"),
    }
    if node.get("node_type") == "image":
        variant_sources = image_variant_sources(node)
        unresolved_status_counts = status_distribution(unresolved_grounded_entities)
        grounded_without_text_edge_count = max(0, len(grounded_entities) - int(linked_text_edge_count))
        grounded_no_text_reason_counts = derive_grounded_no_text_reason_counts(
            grounded_entities=grounded_entities,
            unresolved_grounded_entities=unresolved_grounded_entities,
            query_overlap_grounded_entities=query_overlap_grounded_entities,
            linked_grounded_entities=linked_grounded_entities,
            queued_pending_links=queued_pending_links,
        )
        summary.update(
            {
                "image_type": classify_image_type(node),
                "image_origin": metadata.get("image_origin"),
                "variant_sources": variant_sources,
                "image_url": node.get("image_url"),
                "source_page_url": node.get("source_page_url"),
                "search_query": metadata.get("search_query"),
                "visual_target": metadata.get("visual_target"),
                "grounding_check": image_grounding.get("check"),
                "grounded_entity_count": len(grounded_entities),
                "unresolved_grounded_entity_count": len(unresolved_grounded_entities),
                "query_overlap_grounded_entity_count": len(query_overlap_grounded_entities),
                "linked_text_edge_count": int(linked_text_edge_count),
                "grounded_without_text_edge_count": grounded_without_text_edge_count,
                "queued_image_entity_count": len(queued_pending_links),
                "unresolved_status_counts": unresolved_status_counts,
                "grounded_no_text_reason_counts": grounded_no_text_reason_counts,
                "grounded_entity_names": [item.get("name") for item in grounded_entities[:5] if isinstance(item, dict)],
                "unresolved_entity_names": [item.get("name") for item in unresolved_grounded_entities[:5] if isinstance(item, dict)],
                "query_overlap_entity_names": [item.get("name") for item in query_overlap_grounded_entities[:5] if isinstance(item, dict)],
            }
        )
        summary["possible_isolation_reason"] = infer_image_isolation_reason(summary)
    elif node.get("node_type") == "text":
        summary.update(
            {
                "summary": short(node.get("summary") or node.get("description"), summary_chars),
                "alias_count": len(node.get("aliases") or []),
                "attribute_count": len((node.get("attributes") or {})),
            }
        )
        summary["possible_isolation_reason"] = infer_text_isolation_reason(summary)
    else:
        summary["possible_isolation_reason"] = "node has no incoming or outgoing edges in the stored graph"
    return summary


def infer_image_isolation_reason(summary: dict[str, Any]) -> str:
    image_type = str(summary.get("image_type") or "")
    grounding_check = str(summary.get("grounding_check") or "")
    grounded_count = int(summary.get("grounded_entity_count") or 0)
    unresolved_count = int(summary.get("unresolved_grounded_entity_count") or 0)
    query_overlap_count = int(summary.get("query_overlap_grounded_entity_count") or 0)
    no_text_reasons = summary.get("grounded_no_text_reason_counts") or {}
    queued_count = int(no_text_reasons.get("still_queued_image_entity") or 0)
    query_overlap_without_text = int(no_text_reasons.get("query_overlap_without_text") or 0)
    if query_overlap_count > 0 and grounded_count == 0 and unresolved_count == 0:
        return "all grounded entities were filtered by query overlap, so no image->text edge was kept"
    if queued_count > 0:
        return "grounded entities resolved to image-entity text tasks that are still queued, so no persisted image->text edge exists yet"
    if query_overlap_without_text > 0:
        return "some grounded entities were marked as query-overlap and still did not produce a persisted image->text edge"
    if unresolved_count > 0 and grounded_count == 0:
        return "grounded entities were found but remained unresolved, so no outgoing image->text edge was persisted"
    if grounded_count == 0 and grounding_check in {
        "not_configured",
        "mllm_grounding_failed",
        "image_url_precheck_failed",
    }:
        return f"image grounding did not produce usable linked entities ({grounding_check})"
    if grounded_count == 0:
        if image_type == "wiki_inline":
            return "wiki-inline image nodes intentionally skip source text->image edges; this one also has no persisted outgoing grounded edge"
        if image_type == "visual_plan":
            return "visual-plan image has neither a persisted source text->image edge nor a persisted outgoing grounded edge"
        return "image node has no persisted incoming source edge or outgoing grounded edge"
    return "grounded entities exist in metadata, but no incident edge was ultimately persisted"


def infer_text_isolation_reason(summary: dict[str, Any]) -> str:
    subtype = str(summary.get("subtype") or "")
    if subtype == "wiki_page":
        return "text wiki-page node has no persisted incoming or outgoing wiki-link relation in the stored graph"
    if subtype:
        return f"text node of subtype {subtype!r} has no incoming or outgoing edge in the stored graph"
    return "text node has no incoming or outgoing edge in the stored graph"


def component_summaries(
    *,
    components: list[list[str]],
    nodes_by_id: dict[str, dict[str, Any]],
    top_components: int,
    summary_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    previews: list[dict[str, Any]] = []
    size_distribution: Counter[int] = Counter(len(component) for component in components)
    ranked = sorted(components, key=lambda ids: (-len(ids), tuple(ids)))
    for rank, component in enumerate(ranked[: max(0, top_components)], start=1):
        type_counter: Counter[str] = Counter()
        image_type_counter: Counter[str] = Counter()
        sample_nodes: list[dict[str, Any]] = []
        for node_id in sorted(component):
            node = nodes_by_id[node_id]
            node_type = str(node.get("node_type") or "unknown")
            type_counter[node_type] += 1
            if node_type == "image":
                image_type_counter[classify_image_type(node)] += 1
        for node_id in sorted(component)[:3]:
            node = nodes_by_id[node_id]
            sample_nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node.get("node_type"),
                    "title": node_display_title(node, summary_chars=summary_chars),
                }
            )
        previews.append(
            {
                "rank": rank,
                "size": len(component),
                "node_type_counts": dict(sorted(type_counter.items())),
                "image_type_counts": dict(sorted(image_type_counter.items())),
                "sample_nodes": sample_nodes,
            }
        )
    return previews, {str(size): size_distribution[size] for size in sorted(size_distribution)}


def weakly_connected_components(undirected_adj: dict[str, set[str]]) -> list[list[str]]:
    visited: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(undirected_adj):
        if start in visited:
            continue
        queue: deque[str] = deque([start])
        visited.add(start)
        component: list[str] = []
        while queue:
            node_id = queue.popleft()
            component.append(node_id)
            for neighbor in sorted(undirected_adj.get(node_id, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def pick_isolated_examples(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not records:
        return []

    def bucket_key(record: dict[str, Any]) -> str:
        node_type = str(record.get("node_type") or "unknown")
        if node_type == "image":
            return f"image:{record.get('image_type') or 'other'}"
        return node_type

    def sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(record.get("title") or ""),
            str(record.get("source_url") or ""),
            str(record.get("node_id") or ""),
        )

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[bucket_key(record)].append(record)
    for items in buckets.values():
        items.sort(key=sort_key)

    preferred_order = [
        "image:visual_plan",
        "image:wiki_inline",
        "image:other:unknown",
        "text",
        "region",
        "unknown",
    ]
    seen_ids: set[str] = set()
    chosen: list[dict[str, Any]] = []

    def pop_from_bucket(name: str) -> None:
        items = buckets.get(name) or []
        while items:
            record = items.pop(0)
            node_id = str(record.get("node_id") or "")
            if node_id and node_id not in seen_ids:
                chosen.append(record)
                seen_ids.add(node_id)
                return

    while len(chosen) < limit:
        progressed = False
        for name in preferred_order + sorted(key for key in buckets if key not in preferred_order):
            before = len(chosen)
            pop_from_bucket(name)
            if len(chosen) > before:
                progressed = True
                if len(chosen) >= limit:
                    break
        if not progressed:
            break
    return chosen


def build_report(
    *,
    graph_dir: Path,
    active_only: bool,
    summary_chars: int,
    top_components: int,
    isolated_limit: int,
) -> dict[str, Any]:
    store = JsonlGraphStore(graph_dir)
    nodes = store.list_nodes()
    edges = store.list_edges()
    runner_state = load_runner_state(graph_dir)
    if active_only:
        nodes = [node for node in nodes if is_active(node)]
        active_node_ids = {node["node_id"] for node in nodes if isinstance(node.get("node_id"), str)}
        edges = [
            edge
            for edge in edges
            if is_active(edge)
            and edge.get("src_node_id") in active_node_ids
            and edge.get("dst_node_id") in active_node_ids
        ]

    (
        nodes_by_id,
        in_degree,
        out_degree,
        undirected_adj,
        edges_with_missing_src,
        edges_with_missing_dst,
        edges_between_known_nodes,
    ) = build_graph_state(nodes=nodes, edges=edges)
    linked_text_edges = image_text_edge_counts(nodes_by_id=nodes_by_id, edges=edges)
    linked_grounded_by_image = linked_grounded_entities_by_image(nodes_by_id=nodes_by_id, edges=edges)
    queued_links_by_image = queued_image_entity_links(
        queue_records=runner_state.get("queue") or [],
        nodes_by_id=nodes_by_id,
    )

    summaries: list[dict[str, Any]] = []
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        summaries.append(
            summarize_node(
                node,
                in_degree=in_degree.get(node_id, 0),
                out_degree=out_degree.get(node_id, 0),
                linked_text_edge_count=linked_text_edges.get(node_id, 0),
                linked_grounded_entities=linked_grounded_by_image.get(node_id, []),
                queued_pending_links=queued_links_by_image.get(node_id, []),
                summary_chars=summary_chars,
            )
        )

    text_nodes = [record for record in summaries if record.get("node_type") == "text"]
    image_nodes = [record for record in summaries if record.get("node_type") == "image"]
    image_wiki_inline = [record for record in image_nodes if record.get("image_type") == "wiki_inline"]
    image_visual_plan = [record for record in image_nodes if record.get("image_type") == "visual_plan"]
    image_other = [
        record
        for record in image_nodes
        if record.get("image_type") not in {"wiki_inline", "visual_plan"}
    ]

    components = weakly_connected_components(undirected_adj)
    component_previews, component_size_distribution = component_summaries(
        components=components,
        nodes_by_id=nodes_by_id,
        top_components=top_components,
        summary_chars=summary_chars,
    )
    largest_component_size = max((len(component) for component in components), default=0)
    isolated_nodes = [
        record
        for record in summaries
        if int(record.get("in_degree") or 0) == 0 and int(record.get("out_degree") or 0) == 0
    ]
    isolated_samples = pick_isolated_examples(isolated_nodes, limit=isolated_limit)

    return {
        "graph_dir": str(graph_dir),
        "active_only": active_only,
        "degree_definition": "degree = in_degree + out_degree",
        "node_count": len(summaries),
        "edge_count": len(edges),
        "edges_between_known_nodes": edges_between_known_nodes,
        "edges_with_missing_src": edges_with_missing_src,
        "edges_with_missing_dst": edges_with_missing_dst,
        "runner_state": {
            key: value
            for key, value in runner_state.items()
            if key != "queue"
        },
        "group_stats": {
            "text": merge_group_stats(text_nodes, include_image_grounding=False),
            "image": merge_group_stats(image_nodes, include_image_grounding=True),
            "image_wiki_inline": merge_group_stats(image_wiki_inline, include_image_grounding=True),
            "image_visual_plan": merge_group_stats(image_visual_plan, include_image_grounding=True),
            "image_other": merge_group_stats(image_other, include_image_grounding=True),
        },
        "connectivity": {
            "type": "weakly_connected_components",
            "component_count": len(components),
            "largest_component_size": largest_component_size,
            "largest_component_fraction": (largest_component_size / len(summaries)) if summaries else 0.0,
            "component_size_distribution": component_size_distribution,
            "top_components": component_previews,
            "isolated_node_count": len(isolated_nodes),
        },
        "isolated_samples": isolated_samples,
    }


def print_distribution(label: str, distribution: dict[str, int]) -> None:
    indent = " " * (len(label) - len(label.lstrip(" ")) + 2)
    print(label)
    if not distribution:
        print(f"{indent}<empty>")
        return
    text = ", ".join(f"{degree}:{count}" for degree, count in distribution.items())
    print(f"{indent}{text}")


def print_group_stats(label: str, stats: dict[str, Any]) -> None:
    print(label)
    print(f"  count={stats.get('count', 0)}")
    print(f"  avg_degree={stats.get('avg_degree', 0.0):.3f}")
    print(f"  avg_in_degree={stats.get('avg_in_degree', 0.0):.3f}")
    print(f"  avg_out_degree={stats.get('avg_out_degree', 0.0):.3f}")
    print_distribution("  degree_distribution:", stats.get("degree_distribution") or {})
    grounding = stats.get("grounding_stats") or {}
    if grounding:
        print("  grounded_entity_stats:")
        print(f"    total_grounded_entities={grounding.get('total_grounded_entities', 0)}")
        print(
            "    avg_grounded_entities_per_image="
            f"{grounding.get('avg_grounded_entities_per_image', 0.0):.3f}"
        )
        print(f"    images_with_grounded_entities={grounding.get('images_with_grounded_entities', 0)}")
        print(f"    images_with_linked_text_entities={grounding.get('images_with_linked_text_entities', 0)}")
        print(f"    linked_text_edge_count={grounding.get('linked_text_edge_count', 0)}")
        print(
            "    grounded_without_text_edge_count="
            f"{grounding.get('grounded_without_text_edge_count', 0)}"
        )
        print(
            "    images_with_grounded_but_no_text="
            f"{grounding.get('images_with_grounded_but_no_text', 0)}"
        )
        print(
            "    query_overlap_flagged_entity_count="
            f"{grounding.get('query_overlap_flagged_entity_count', 0)}"
        )
        print_distribution(
            "    grounded_entity_distribution:",
            grounding.get("grounded_entity_distribution") or {},
        )
        print_distribution(
            "    no_text_reason_counts:",
            grounding.get("no_text_reason_counts") or {},
        )


def print_isolated_samples(samples: list[dict[str, Any]]) -> None:
    print(f"\nIsolated Samples ({len(samples)})")
    if not samples:
        print("  <none>")
        return
    for index, record in enumerate(samples, start=1):
        print(
            f"  {index:>2}. {record.get('node_id')} [{record.get('node_type')}] {record.get('title')!r}"
        )
        print(
            f"      status={record.get('status')} degree={record.get('degree')} "
            f"source_type={record.get('source_type')}"
        )
        if record.get("source_url"):
            print(f"      source_url={record.get('source_url')}")
        if record.get("node_type") == "image":
            print(
                f"      image_type={record.get('image_type')} grounding_check={record.get('grounding_check')} "
                f"grounded={record.get('grounded_entity_count')} linked_text={record.get('linked_text_edge_count')} "
                f"queued={record.get('queued_image_entity_count')} "
                f"grounded_without_text={record.get('grounded_without_text_edge_count')} "
                f"unresolved={record.get('unresolved_grounded_entity_count')} "
                f"query_overlap={record.get('query_overlap_grounded_entity_count')}"
            )
            if record.get("image_origin") or record.get("variant_sources"):
                print(
                    f"      image_origin={record.get('image_origin')!r} "
                    f"variant_sources={record.get('variant_sources') or []}"
                )
            if record.get("grounded_no_text_reason_counts"):
                print(
                    "      grounded_no_text_reason_counts="
                    f"{record.get('grounded_no_text_reason_counts')}"
                )
            if record.get("search_query"):
                print(f"      search_query={record.get('search_query')!r}")
            if record.get("visual_target"):
                print(f"      visual_target={record.get('visual_target')!r}")
            if record.get("grounded_entity_names"):
                print(f"      grounded_entity_names={record.get('grounded_entity_names')}")
            if record.get("unresolved_entity_names"):
                print(f"      unresolved_entity_names={record.get('unresolved_entity_names')}")
            if record.get("query_overlap_entity_names"):
                print(f"      query_overlap_entity_names={record.get('query_overlap_entity_names')}")
        elif record.get("node_type") == "text":
            if record.get("subtype"):
                print(f"      subtype={record.get('subtype')} canonical_id={record.get('canonical_id')}")
            if record.get("summary"):
                print(f"      summary={record.get('summary')!r}")
        print(f"      possible_isolation_reason={record.get('possible_isolation_reason')}")


def print_report(report: dict[str, Any]) -> None:
    print("Overview")
    print(f"  graph_dir={report.get('graph_dir')}")
    print(f"  active_only={report.get('active_only')}")
    print(f"  degree_definition={report.get('degree_definition')}")
    print(f"  node_count={report.get('node_count')}")
    print(f"  edge_count={report.get('edge_count')}")
    print(f"  edges_between_known_nodes={report.get('edges_between_known_nodes')}")
    print(f"  edges_with_missing_src={report.get('edges_with_missing_src')}")
    print(f"  edges_with_missing_dst={report.get('edges_with_missing_dst')}")
    runner_state = report.get("runner_state") or {}
    print(
        f"  runner_state_found={runner_state.get('found')} "
        f"status={runner_state.get('status')} queue_size={runner_state.get('queue_size')}"
    )
    if runner_state.get("path"):
        print(f"  runner_state_path={runner_state.get('path')}")
    if runner_state.get("load_error"):
        print(f"  runner_state_load_error={runner_state.get('load_error')}")
    print(
        "  runner_image_entity_queue="
        f"tasks={runner_state.get('image_entity_task_count')} "
        f"pending_links={runner_state.get('image_entity_pending_link_count')} "
        f"query_overlap_pending_links={runner_state.get('query_overlap_pending_link_count')}"
    )

    print("\nDegree Stats")
    group_stats = report.get("group_stats") or {}
    print_group_stats("text:", group_stats.get("text") or {})
    print_group_stats("image:", group_stats.get("image") or {})
    print_group_stats("image_wiki_inline:", group_stats.get("image_wiki_inline") or {})
    print_group_stats("image_visual_plan:", group_stats.get("image_visual_plan") or {})
    other_stats = group_stats.get("image_other") or {}
    if other_stats.get("count"):
        print_group_stats("image_other:", other_stats)

    connectivity = report.get("connectivity") or {}
    print("\nConnectivity")
    print(f"  type={connectivity.get('type')}")
    print(f"  component_count={connectivity.get('component_count')}")
    print(f"  largest_component_size={connectivity.get('largest_component_size')}")
    print(f"  largest_component_fraction={connectivity.get('largest_component_fraction', 0.0):.4f}")
    print(f"  isolated_node_count={connectivity.get('isolated_node_count')}")
    print_distribution("  component_size_distribution:", connectivity.get("component_size_distribution") or {})

    top_components = connectivity.get("top_components") or []
    print(f"  top_components ({len(top_components)}):")
    if not top_components:
        print("    <none>")
    for component in top_components:
        print(
            f"    rank={component.get('rank')} size={component.get('size')} "
            f"node_type_counts={component.get('node_type_counts')} image_type_counts={component.get('image_type_counts')}"
        )
        sample_nodes = component.get("sample_nodes") or []
        if sample_nodes:
            sample_text = ", ".join(
                f"{item.get('node_id')}[{item.get('node_type')}] {item.get('title')!r}"
                for item in sample_nodes
            )
            print(f"      sample_nodes={sample_text}")

    print_isolated_samples(report.get("isolated_samples") or [])


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    if not graph_dir.exists():
        print(f"Graph directory does not exist: {graph_dir}", file=sys.stderr)
        return 1
    report = build_report(
        graph_dir=graph_dir,
        active_only=bool(args.active_only),
        summary_chars=max(20, int(args.summary_chars)),
        top_components=max(0, int(args.top_components)),
        isolated_limit=max(0, int(args.limit)),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        try:
            print_report(report)
        except BrokenPipeError:
            return 141
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
