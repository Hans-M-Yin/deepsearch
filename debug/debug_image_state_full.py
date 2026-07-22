"""Audit persisted visual-plan and image-grounding state for one graph run.

The graph store does not persist every candidate-level decision.  In particular,
an image rejected by the visual-plan post-grounding filter is never written to
``nodes.jsonl``.  This tool therefore separates exact persisted facts from
stages that cannot be reconstructed from an existing graph directory.

Example:
  python debug/debug_image_state_full.py \
    --graph-dir runs/example \
    --pretty
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from debug.list_neighbor import (
    _build_grounded_entity_reports,
    _collect_runner_state_index,
)


IMAGE_DEPICTS = "image_depicts"
SEARCH_RETRIEVED = "search_retrieved"
ORIGINS = ("visual_plan", "wiki_inline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize visual-plan filtering and persisted image-grounding state."
    )
    parser.add_argument("--graph-dir", required=True, help="Directory containing graph JSONL files.")
    parser.add_argument("--visual-plans-file", default="visual_plans.jsonl")
    parser.add_argument("--state-file", default="graph_runner_state.json")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _image_variant_sources(node: dict[str, Any]) -> set[str]:
    variants = node.get("image_variants") or []
    if not isinstance(variants, list):
        return set()
    return {
        str(item.get("source") or "").strip()
        for item in variants
        if isinstance(item, dict) and str(item.get("source") or "").strip()
    }


def image_origin(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    source = node.get("source") or {}
    source_type = source.get("source_type") if isinstance(source, dict) else None
    image_origin = str(metadata.get("image_origin") or "").strip().lower()
    if source_type == "wikipedia_inline_image" or image_origin == "wikipedia_inline":
        return "wiki_inline"
    if "wikipedia_inline" in _image_variant_sources(node):
        return "wiki_inline"
    if source_type in {"image_search", "image_search_bundle"}:
        return "visual_plan"
    return f"other:{source_type}" if source_type else "other:unknown"


def plan_origin(plan: dict[str, Any]) -> str:
    metadata = plan.get("metadata") or {}
    if plan.get("planner") == "wikipedia_inline_image_planner" or metadata.get("plan_source") == "wikipedia_inline_image":
        return "wiki_inline"
    return "visual_plan"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _image_summary(state: dict[str, Any] | None) -> dict[str, int]:
    summary: Counter[str] = Counter()
    for section in ("completed_tasks", "failed_tasks", "skipped_tasks"):
        for record in (state or {}).get(section) or []:
            if not isinstance(record, dict):
                continue
            item = record.get("image_summary") or {}
            if not isinstance(item, dict):
                continue
            for key in ("returned", "accepted", "rejected", "fetch_failed"):
                summary[key] += int(item.get(key) or 0)
    return {key: summary[key] for key in ("returned", "accepted", "rejected", "fetch_failed")}


def _plan_binding_indexes(plans: Iterable[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_query: dict[str, set[str]] = defaultdict(set)
    by_url: dict[str, set[str]] = defaultdict(set)
    for plan in plans:
        plan_id = str(plan.get("plan_id") or "")
        if not plan_id:
            continue
        for query in plan.get("queries") or []:
            normalized = _normalize(query)
            if normalized:
                by_query[normalized].add(plan_id)
        target = plan.get("target") or {}
        url = str(target.get("url") or "").strip() if isinstance(target, dict) else ""
        if url:
            by_url[url].add(plan_id)
    return by_query, by_url


def _binding_counts(images: list[dict[str, Any]], plans: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_query, by_url = _plan_binding_indexes(plans)
    counts: dict[str, Counter[str]] = {origin: Counter() for origin in ORIGINS}
    for node in images:
        origin = image_origin(node)
        if origin not in counts:
            continue
        metadata = node.get("metadata") or {}
        if origin == "visual_plan":
            matches = by_query.get(_normalize(metadata.get("search_query")), set())
        else:
            matches = by_url.get(str(node.get("image_url") or "").strip(), set())
        counts[origin]["image_nodes"] += 1
        if len(matches) == 1:
            counts[origin]["heuristically_unique_plan_match"] += 1
        elif len(matches) > 1:
            counts[origin]["heuristically_ambiguous_plan_match"] += 1
        else:
            counts[origin]["no_heuristic_plan_match"] += 1
    return {
        origin: {
            "image_nodes": counts[origin]["image_nodes"],
            "heuristically_unique_plan_match": counts[origin]["heuristically_unique_plan_match"],
            "heuristically_ambiguous_plan_match": counts[origin]["heuristically_ambiguous_plan_match"],
            "no_heuristic_plan_match": counts[origin]["no_heuristic_plan_match"],
        }
        for origin in ORIGINS
    }


def _image_node_stats(
    images: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    state: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge.get("edge_type") == IMAGE_DEPICTS:
            out_edges[str(edge.get("src_node_id") or "")].append(edge)
    runner_index = _collect_runner_state_index(state)

    output: dict[str, dict[str, Any]] = {}
    for origin in ORIGINS:
        image_subset = [node for node in images if image_origin(node) == origin]
        entity_statuses: Counter[str] = Counter()
        metadata_statuses: Counter[str] = Counter()
        post_filter_reasons: Counter[str] = Counter()
        uniqueness_block_reasons: Counter[str] = Counter()
        node_counts: Counter[str] = Counter()
        initial_grounded_entity_count = 0

        for node in image_subset:
            node_id = str(node.get("node_id") or "")
            metadata = node.get("metadata") or {}
            initial_grounded_entity_count += sum(
                1 for entity in metadata.get("grounded_entities") or [] if isinstance(entity, dict)
            )
            reports = _build_grounded_entity_reports(
                image_node=node,
                out_edges=out_edges.get(node_id, []),
                nodes_by_id=nodes_by_id,
                runner_state_index=runner_index,
                summary_chars=0,
            )
            node_counts["image_nodes"] += 1
            if reports:
                node_counts["nodes_with_initial_grounded_entities"] += 1
            if out_edges.get(node_id):
                node_counts["nodes_with_materialized_text_edge"] += 1
            if any(report.get("status") == "queued_pending" for report in reports):
                node_counts["nodes_with_pending_text_expansion"] += 1

            for report in reports:
                entity_statuses[str(report.get("status") or "grounded_only")] += 1
                for status in report.get("metadata_statuses") or []:
                    metadata_statuses[str(status)] += 1

            post_filter = metadata.get("visual_plan_post_grounding_filter") or {}
            if isinstance(post_filter, dict):
                reason = str(post_filter.get("filter_reason") or "kept_or_no_reason")
                post_filter_reasons[reason] += 1

            uniqueness = metadata.get("wiki_inline_entity_uniqueness_filter") or {}
            if isinstance(uniqueness, dict):
                for item in uniqueness.get("blocked_entities") or []:
                    if isinstance(item, dict):
                        uniqueness_block_reasons[str(item.get("status") or "blocked")] += 1

        # "linked" is the only state proving that a downstream text-node edge
        # exists now.  Pending/completed are retained separately because a
        # completed task can still have a parent-link failure.
        output[origin] = {
            "image_nodes": {
                "total": node_counts["image_nodes"],
                "with_initial_grounded_entities": node_counts["nodes_with_initial_grounded_entities"],
                "with_materialized_text_edge": node_counts["nodes_with_materialized_text_edge"],
                "with_pending_text_expansion": node_counts["nodes_with_pending_text_expansion"],
            },
            "initial_grounded_entities": initial_grounded_entity_count,
            "entity_final_statuses": _counter_dict(entity_statuses),
            "entity_metadata_filter_statuses": _counter_dict(metadata_statuses),
            "materialized_text_node_entities": entity_statuses["linked"],
            "pending_text_expansion_entities": entity_statuses["queued_pending"],
            "expandable_or_materialized_entities": (
                entity_statuses["linked"]
                + entity_statuses["queued_pending"]
                + entity_statuses["task_completed"]
            ),
            "post_grounding_filter_reasons_observed_on_persisted_nodes": _counter_dict(post_filter_reasons),
            "wiki_inline_uniqueness_block_reasons_observed_on_persisted_nodes": _counter_dict(uniqueness_block_reasons),
        }
    return output


def build_report(
    *,
    graph_dir: Path,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    images = [node for node in nodes if node.get("node_type") == "image"]
    nodes_by_id = {str(node.get("node_id") or ""): node for node in nodes if node.get("node_id")}
    plan_counts = Counter(plan_origin(plan) for plan in plans)
    plan_ids = [str(plan.get("plan_id") or "") for plan in plans]
    duplicate_plan_ids = len(plan_ids) - len(set(plan_ids))

    return {
        "graph_dir": str(graph_dir),
        "input_records": {
            "nodes": len(nodes),
            "edges": len(edges),
            "image_nodes": len(images),
            "visual_plan_records": len(plans),
            "duplicate_or_missing_visual_plan_ids": duplicate_plan_ids + sum(not item for item in plan_ids),
            "runner_state_present": state is not None,
        },
        "visual_plan_counts": {
            "total": len(plans),
            "by_origin": {origin: plan_counts[origin] for origin in ORIGINS},
            "by_planner": _counter_dict(Counter(str(plan.get("planner") or "unknown") for plan in plans)),
            "image_search_stage": {
                "visual_plan_records_sent_to_discover_for_plan": plan_counts["visual_plan"],
                "wiki_inline_records_sent_to_inline_image_discovery": plan_counts["wiki_inline"],
                "note": (
                    "A persisted plan is generated by _expand_images and is normally executed once. "
                    "The graph format does not retain a per-plan completion event."
                ),
            },
        },
        "candidate_image_summary_from_runner_state": _image_summary(state),
        "persisted_image_node_state": _image_node_stats(images, edges, nodes_by_id, state),
        "plan_to_image_binding_diagnostics": _binding_counts(images, plans),
        "filter_stage_observability": {
            "exact_from_persisted_graph": [
                "final image nodes by origin",
                "initial grounded entity count on retained image nodes",
                "per-entity linked / queued / failed / unresolved / query-overlap status",
                "filter metadata retained on surviving nodes",
            ],
            "not_reconstructable_from_existing_graph_dir": [
                "per-plan candidate selection before node materialization",
                "visual-plan post-grounding rejections, because rejected nodes are not persisted",
                "wiki-inline images discarded before persistence or by page cap",
            ],
            "why": (
                "graph_runner_state.json stores only aggregate image_summary counts; "
                "candidate_decisions and discarded ImageDiscoveryResult objects are not persisted."
            ),
        },
    }


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    if not graph_dir.exists():
        raise SystemExit(f"Graph directory does not exist: {graph_dir}")
    report = build_report(
        graph_dir=graph_dir,
        nodes=_load_jsonl(graph_dir / "nodes.jsonl"),
        edges=_load_jsonl(graph_dir / "edges.jsonl"),
        plans=_load_jsonl(graph_dir / args.visual_plans_file),
        state=_load_json(graph_dir / args.state_file),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
