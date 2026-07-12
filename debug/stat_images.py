import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect fully isolated graph nodes. For isolated image nodes, report grounded entities, "
            "their persisted selection status, matched wiki URL if recorded, and the inferred source text node."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing nodes.jsonl / edges.jsonl and optional graph_runner_state.json / visual_plans.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of isolated nodes to inspect in detail. Use 0 to inspect all isolated nodes.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only count nodes/edges whose status is active.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object instead of human-readable text.",
    )
    parser.add_argument(
        "--state-file",
        default="graph_runner_state.json",
        help="Optional runner state filename inside --run-dir.",
    )
    parser.add_argument(
        "--visual-plans-file",
        default="visual_plans.jsonl",
        help="Optional visual plan JSONL filename inside --run-dir.",
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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_active(record: dict[str, Any]) -> bool:
    return str(record.get("status") or "active") == "active"


def short(text: Any, max_len: int = 160) -> str:
    value = "" if text is None else str(text).replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


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


def normalized_entity_name(entity_or_name: Any) -> str:
    if isinstance(entity_or_name, dict):
        return normalize_text(entity_or_name.get("name"))
    return normalize_text(entity_or_name)


def summarize_node(node: dict[str, Any], in_deg: int, out_deg: int) -> dict[str, Any]:
    source = node.get("source") or {}
    metadata = node.get("metadata") or {}
    image_grounding = metadata.get("image_grounding") or {}
    image_variants = node.get("image_variants") or []
    accepted_image_ids = node.get("accepted_image_ids") or []
    rejected_image_ids = node.get("rejected_image_ids") or []

    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "status": node.get("status"),
        "title": node.get("title"),
        "summary": node.get("summary"),
        "caption": node.get("caption") or node.get("summary"),
        "subtype": node.get("subtype"),
        "canonical_id": node.get("canonical_id"),
        "in_degree": in_deg,
        "out_degree": out_deg,
        "source_type": source.get("source_type") if isinstance(source, dict) else None,
        "source_url": source_url_of(node),
        "image_url": node.get("image_url"),
        "source_page_url": node.get("source_page_url"),
        "width": node.get("width"),
        "height": node.get("height"),
        "storage_status": node.get("storage_status"),
        "image_variant_count": len(image_variants),
        "accepted_image_count": len(accepted_image_ids),
        "rejected_image_count": len(rejected_image_ids),
        "search_query": metadata.get("search_query"),
        "visual_target": metadata.get("visual_target"),
        "grounding_check": image_grounding.get("check"),
        "grounded_entities": list(metadata.get("grounded_entities") or []),
        "unresolved_grounded_entities": list(metadata.get("unresolved_grounded_entities") or []),
        "query_overlap_grounded_entities": list(metadata.get("query_overlap_grounded_entities") or []),
        "created_at": node.get("created_at"),
    }


def build_graph_indexes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Counter[str], Counter[str], dict[str, list[dict[str, Any]]], int, int]:
    node_ids = {node["node_id"] for node in nodes if isinstance(node.get("node_id"), str)}
    nodes_by_id = {node["node_id"]: node for node in nodes if isinstance(node.get("node_id"), str)}

    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_src = 0
    unknown_dst = 0

    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src in node_ids:
            out_degree[src] += 1
            out_edges[src].append(edge)
        else:
            unknown_src += 1
        if dst in node_ids:
            in_degree[dst] += 1
        else:
            unknown_dst += 1

    return nodes_by_id, in_degree, out_degree, out_edges, unknown_src, unknown_dst


def build_text_nodes_by_source_url(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("node_type") != "text":
            continue
        url = source_url_of(node)
        if url:
            result[url].append(node)
    return result


def build_visual_plan_indexes(visual_plans: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    indexes = {
        "by_query": defaultdict(list),
        "by_target_description": defaultdict(list),
        "by_expected_visual": defaultdict(list),
        "by_meta_source_page_url": defaultdict(list),
        "by_node_source_url": defaultdict(list),
    }
    for plan in visual_plans:
        if not isinstance(plan, dict):
            continue

        for raw_query in plan.get("queries") or []:
            key = normalize_text(raw_query)
            if key:
                indexes["by_query"][key].append(plan)

        target_description = normalize_text(plan.get("target_description"))
        if target_description:
            indexes["by_target_description"][target_description].append(plan)

        expected_visual = normalize_text(plan.get("expected_visual"))
        if expected_visual:
            indexes["by_expected_visual"][expected_visual].append(plan)

        metadata = plan.get("metadata") or {}
        if isinstance(metadata, dict):
            meta_source_page_url = str(metadata.get("source_page_url") or "").strip()
            if meta_source_page_url:
                indexes["by_meta_source_page_url"][meta_source_page_url].append(plan)

        node_source_url = str(plan.get("node_source_url") or "").strip()
        if node_source_url:
            indexes["by_node_source_url"][node_source_url].append(plan)
    return indexes


def infer_image_source_candidates(
    image_record: dict[str, Any],
    *,
    visual_plan_indexes: dict[str, dict[str, list[dict[str, Any]]]],
    text_nodes_by_source_url: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    raw_search_query = str(image_record.get("search_query") or "").strip()
    raw_visual_target = str(image_record.get("visual_target") or "").strip()
    raw_source_page_url = str(image_record.get("source_page_url") or "").strip()

    search_query = normalize_text(raw_search_query)
    visual_target = normalize_text(raw_visual_target)
    source_page_url = raw_source_page_url

    candidates: dict[str, dict[str, Any]] = {}
    relevant_plans: dict[str, dict[str, Any]] = {}

    def candidate_key(*, source_node_id: Any, source_url: Any, source_title: Any) -> str:
        if source_node_id:
            return f"node:{source_node_id}"
        if source_url:
            return f"url:{source_url}"
        return f"title:{source_title}"

    def add_candidate(
        *,
        source_node_id: Any,
        source_title: Any,
        source_url: Any,
        score: int,
        matched_by: str | list[str],
        plan_id: Any = None,
    ) -> None:
        key = candidate_key(source_node_id=source_node_id, source_url=source_url, source_title=source_title)
        item = candidates.get(key)
        if item is None:
            item = {
                "source_node_id": source_node_id,
                "source_title": source_title,
                "source_url": source_url,
                "score": 0,
                "matched_by": [],
                "plan_ids": [],
            }
            candidates[key] = item
        item["score"] += int(score)
        matched_values = [matched_by] if isinstance(matched_by, str) else list(matched_by or [])
        for value in matched_values:
            if value and value not in item["matched_by"]:
                item["matched_by"].append(value)
        if plan_id and plan_id not in item["plan_ids"]:
            item["plan_ids"].append(plan_id)

    def attach_plan(plan: dict[str, Any], score: int, matched_by: str) -> None:
        plan_key = str(plan.get("plan_id") or f"no_plan::{plan.get('node_id')}::{plan.get('node_source_url')}")
        item = relevant_plans.get(plan_key)
        if item is None:
            item = {"plan": plan, "score": 0, "matched_by": []}
            relevant_plans[plan_key] = item
        item["score"] += int(score)
        if matched_by not in item["matched_by"]:
            item["matched_by"].append(matched_by)

    if source_page_url:
        for node in text_nodes_by_source_url.get(source_page_url, []):
            add_candidate(
                source_node_id=node.get("node_id"),
                source_title=node.get("title") or node.get("canonical_id"),
                source_url=source_url_of(node),
                score=100,
                matched_by="exact_source_page_text_url",
            )

    if search_query:
        for plan in visual_plan_indexes["by_query"].get(search_query, []):
            attach_plan(plan, 40, "search_query")
    if visual_target:
        for plan in visual_plan_indexes["by_target_description"].get(visual_target, []):
            attach_plan(plan, 20, "target_description")
        for plan in visual_plan_indexes["by_expected_visual"].get(visual_target, []):
            attach_plan(plan, 20, "expected_visual")
    if source_page_url:
        for plan in visual_plan_indexes["by_meta_source_page_url"].get(source_page_url, []):
            attach_plan(plan, 15, "plan.metadata.source_page_url")
        for plan in visual_plan_indexes["by_node_source_url"].get(source_page_url, []):
            attach_plan(plan, 10, "node_source_url")

    for item in relevant_plans.values():
        plan = item["plan"]
        add_candidate(
            source_node_id=plan.get("node_id"),
            source_title=plan.get("node_title"),
            source_url=plan.get("node_source_url"),
            score=item["score"],
            matched_by=item["matched_by"],
            plan_id=plan.get("plan_id"),
        )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -int(item.get("score") or 0),
            str(item.get("source_title") or ""),
            str(item.get("source_url") or ""),
        ),
    )
    for item in ranked:
        item["matched_by"] = list(item.get("matched_by") or [])
        item["plan_ids"] = list(item.get("plan_ids") or [])
    return ranked


def collect_queued_image_entity_tasks(state: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(state, dict):
        return by_parent

    for task in state.get("queue") or []:
        if not isinstance(task, dict):
            continue
        metadata = task.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        pending_links = metadata.get("pending_parent_links") or []
        for pending in pending_links:
            if not isinstance(pending, dict):
                continue
            if (pending.get("link_type") or "") != "image_entity":
                continue
            parent_id = pending.get("parent_node_id") or task.get("parent_node_id")
            if not isinstance(parent_id, str) or not parent_id:
                continue

            entity = pending.get("entity") or {}
            resolved_target = pending.get("resolved_target") or {}
            by_parent[parent_id].append(
                {
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                    "relation_to_image": entity.get("relation_to_image"),
                    "evidence": entity.get("evidence"),
                    "query_overlap_entity": bool(pending.get("query_overlap_entity")),
                    "wiki_title": resolved_target.get("title") or task.get("title"),
                    "wiki_url": resolved_target.get("url") or task.get("url"),
                }
            )
    return by_parent


def collect_image_entity_failures(state: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(state, dict):
        return by_parent

    for bucket in ("completed_tasks", "failed_tasks", "skipped_tasks"):
        for record in state.get(bucket) or []:
            if not isinstance(record, dict):
                continue
            task = record.get("task") or {}
            for failure in record.get("parent_link_failures") or []:
                if not isinstance(failure, dict):
                    continue
                if (failure.get("link_type") or "") != "image_entity":
                    continue
                parent_id = failure.get("parent_node_id")
                if not isinstance(parent_id, str) or not parent_id:
                    continue
                by_parent[parent_id].append(
                    {
                        "entity_name": failure.get("entity_name"),
                        "entity_type": failure.get("entity_type"),
                        "reason": failure.get("reason"),
                        "target_title": failure.get("target_title"),
                        "target_url": failure.get("target_url"),
                        "step": record.get("step"),
                        "task_url": task.get("url") if isinstance(task, dict) else None,
                    }
                )
    return by_parent


def merge_grounded_entities(image_record: dict[str, Any]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def ensure_entry(raw: dict[str, Any], *, extra_status: str | None = None) -> None:
        if not isinstance(raw, dict):
            return
        key = normalized_entity_name(raw)
        if not key:
            return
        entry = entries.get(key)
        if entry is None:
            entry = {
                "name": raw.get("name"),
                "type": raw.get("type"),
                "relation_to_image": raw.get("relation_to_image"),
                "evidence": raw.get("evidence"),
                "recorded_statuses": set(),
            }
            entries[key] = entry
            order.append(key)
        else:
            if not entry.get("type") and raw.get("type"):
                entry["type"] = raw.get("type")
            if not entry.get("relation_to_image") and raw.get("relation_to_image"):
                entry["relation_to_image"] = raw.get("relation_to_image")
            if not entry.get("evidence") and raw.get("evidence"):
                entry["evidence"] = raw.get("evidence")

        status = str(raw.get("status") or "").strip()
        if status:
            entry["recorded_statuses"].add(status)
        if extra_status:
            entry["recorded_statuses"].add(extra_status)

    for raw in image_record.get("grounded_entities") or []:
        ensure_entry(raw, extra_status="grounded_entity")
    for raw in image_record.get("unresolved_grounded_entities") or []:
        ensure_entry(raw)
    for raw in image_record.get("query_overlap_grounded_entities") or []:
        ensure_entry(raw, extra_status="query_overlap_entity")

    merged: list[dict[str, Any]] = []
    for key in order:
        item = dict(entries[key])
        item["recorded_statuses"] = sorted(item.get("recorded_statuses") or [])
        merged.append(item)
    return merged


def build_outgoing_entity_edge_index(image_node_id: str, out_edges: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in out_edges.get(image_node_id, []):
        metadata = edge.get("metadata") or {}
        key = normalized_entity_name(metadata.get("entity_name"))
        if key:
            result[key].append(edge)
    return result


def group_records_by_entity_name(records: list[dict[str, Any]], *, field: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = normalized_entity_name(record.get(field))
        if key:
            result[key].append(record)
    return result


def build_grounded_entity_diagnostics(
    image_record: dict[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    out_edges: dict[str, list[dict[str, Any]]],
    queued_image_entity_tasks: dict[str, list[dict[str, Any]]],
    image_entity_failures: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    image_node_id = str(image_record.get("node_id") or "")
    merged_entities = merge_grounded_entities(image_record)
    edge_index = build_outgoing_entity_edge_index(image_node_id, out_edges)
    queued_index = group_records_by_entity_name(queued_image_entity_tasks.get(image_node_id, []), field="entity_name")
    failure_index = group_records_by_entity_name(image_entity_failures.get(image_node_id, []), field="entity_name")

    diagnostics: list[dict[str, Any]] = []
    for entity in merged_entities:
        key = normalized_entity_name(entity)
        statuses = set(entity.get("recorded_statuses") or [])
        query_overlap_entity = "query_overlap_entity" in statuses or "filtered_by_query_entity_overlap" in statuses

        edge_matches = edge_index.get(key) or []
        queued_matches = queued_index.get(key) or []
        failure_matches = failure_index.get(key) or []

        selection_status = "no_persisted_record"
        selection_reason = (
            "No outgoing edge, no queued expansion, no parent-link failure, and no unresolved/query-overlap status "
            "were found in persisted artifacts."
        )
        matched_wiki_title = None
        matched_wiki_url = None
        target_node_id = None

        if edge_matches:
            edge = edge_matches[0]
            target_node = nodes_by_id.get(edge.get("dst_node_id"))
            target_node_id = edge.get("dst_node_id")
            matched_wiki_title = (target_node or {}).get("title") or (target_node or {}).get("canonical_id")
            matched_wiki_url = source_url_of(target_node or {})
            selection_status = "selected_connected"
            selection_reason = "Grounded entity was selected and connected to an existing text node."
        elif queued_matches:
            queued = queued_matches[0]
            matched_wiki_title = queued.get("wiki_title")
            matched_wiki_url = queued.get("wiki_url")
            selection_status = "selected_queued"
            selection_reason = "Grounded entity was selected and is still queued for text expansion."
        elif failure_matches:
            failure = failure_matches[0]
            matched_wiki_title = failure.get("target_title")
            matched_wiki_url = failure.get("target_url")
            selection_status = "selected_but_failed_to_materialize"
            failure_reason = str(failure.get("reason") or "unknown_parent_link_failure")
            selection_reason = f"Grounded entity was selected, but edge materialization failed: {failure_reason}."
        elif "filtered_out" in statuses:
            selection_status = "not_selected_filtered_out"
            selection_reason = "Grounded entity was filtered out before resolution."
        elif query_overlap_entity:
            selection_status = "not_selected_query_overlap"
            selection_reason = "Grounded entity overlaps with the retrieve/search query and was marked as query-overlap."
        elif "unresolved" in statuses:
            selection_status = "not_selected_unresolved"
            selection_reason = (
                "Grounded entity reached resolution but no final matched target was persisted. "
                "The resolved wiki URL is not available in persisted artifacts."
            )

        diagnostics.append(
            {
                "name": entity.get("name"),
                "type": entity.get("type"),
                "relation_to_image": entity.get("relation_to_image"),
                "evidence": entity.get("evidence"),
                "recorded_statuses": sorted(statuses),
                "query_overlap_entity": query_overlap_entity,
                "selected": selection_status.startswith("selected_"),
                "selection_status": selection_status,
                "selection_reason": selection_reason,
                "matched_wiki_title": matched_wiki_title,
                "matched_wiki_url": matched_wiki_url,
                "target_node_id": target_node_id,
            }
        )

    return diagnostics


def build_image_detail(
    image_record: dict[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    out_edges: dict[str, list[dict[str, Any]]],
    queued_image_entity_tasks: dict[str, list[dict[str, Any]]],
    image_entity_failures: dict[str, list[dict[str, Any]]],
    visual_plan_indexes: dict[str, dict[str, list[dict[str, Any]]]],
    text_nodes_by_source_url: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_candidates = infer_image_source_candidates(
        image_record,
        visual_plan_indexes=visual_plan_indexes,
        text_nodes_by_source_url=text_nodes_by_source_url,
    )
    best_source = source_candidates[0] if source_candidates else None
    grounded_entity_diagnostics = build_grounded_entity_diagnostics(
        image_record,
        nodes_by_id=nodes_by_id,
        out_edges=out_edges,
        queued_image_entity_tasks=queued_image_entity_tasks,
        image_entity_failures=image_entity_failures,
    )
    return {
        **image_record,
        "inferred_source_node": best_source,
        "inferred_source_candidate_count": len(source_candidates),
        "grounded_entity_diagnostics": grounded_entity_diagnostics,
    }


def print_summary(counter: Counter[str], name: str) -> None:
    print(name)
    if not counter:
        print("  (none)")
        return
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {key}: {value}")


def print_image_detail(record: dict[str, Any]) -> None:
    if record.get("search_query"):
        print(f"   search_query: {short(record.get('search_query'), 160)}")
    if record.get("visual_target"):
        print(f"   visual_target: {short(record.get('visual_target'), 160)}")
    if record.get("grounding_check"):
        print(f"   image_grounding.check: {record.get('grounding_check')}")

    source_node = record.get("inferred_source_node")
    print(f"   inferred_source_candidate_count: {record.get('inferred_source_candidate_count', 0)}")
    if isinstance(source_node, dict):
        print(
            f"   inferred_source: {short(source_node.get('source_title'), 120)} | "
            f"node_id={source_node.get('source_node_id')} | url={source_node.get('source_url')}"
        )
        matched_by = ", ".join(source_node.get("matched_by") or [])
        if matched_by:
            print(f"      matched_by: {matched_by}")

    diagnostics = list(record.get("grounded_entity_diagnostics") or [])
    print(f"   grounded_entity_count: {len(diagnostics)}")
    for idx, item in enumerate(diagnostics, start=1):
        print(
            f"   GE{idx}. name={short(item.get('name'), 60)} | "
            f"selected={item.get('selected')} | "
            f"selection_status={item.get('selection_status')}"
        )
        if item.get("type"):
            print(f"      type: {short(item.get('type'), 80)}")
        if item.get("relation_to_image"):
            print(f"      relation: {short(item.get('relation_to_image'), 120)}")
        if item.get("recorded_statuses"):
            print(f"      recorded_statuses: {', '.join(item.get('recorded_statuses') or [])}")
        if item.get("query_overlap_entity"):
            print("      query_overlap_entity: true")
        print(f"      reason: {short(item.get('selection_reason'), 200)}")
        if item.get("matched_wiki_title"):
            print(f"      matched_wiki_title: {short(item.get('matched_wiki_title'), 120)}")
        if item.get("matched_wiki_url"):
            print(f"      matched_wiki_url: {item.get('matched_wiki_url')}")
        if item.get("target_node_id"):
            print(f"      target_node_id: {item.get('target_node_id')}")
        if item.get("evidence"):
            print(f"      evidence: {short(item.get('evidence'), 160)}")


def print_nodes(records: list[dict[str, Any]]) -> None:
    print(f"\n=== Fully Isolated Nodes ({len(records)}) ===")
    if not records:
        print("(none)")
        return

    for idx, record in enumerate(records, start=1):
        print(
            f"{idx}. [{record.get('node_type')}][{record.get('status')}] "
            f"{short(record.get('title') or record.get('node_id'), 120)}"
        )
        print(f"   node_id: {record.get('node_id')}")
        print(f"   degree: in={record.get('in_degree')} out={record.get('out_degree')}")
        if record.get("canonical_id"):
            print(f"   canonical_id: {record.get('canonical_id')}")
        if record.get("source_type"):
            print(f"   source_type: {record.get('source_type')}")
        if record.get("source_url"):
            print(f"   source_url: {record.get('source_url')}")
        if record.get("image_url") and record.get("image_url") != record.get("source_url"):
            print(f"   image_url: {record.get('image_url')}")
        if record.get("source_page_url") and record.get("source_page_url") != record.get("source_url"):
            print(f"   source_page_url: {record.get('source_page_url')}")
        if record.get("width") or record.get("height"):
            print(f"   size: {record.get('width')} x {record.get('height')}")
        if record.get("storage_status"):
            print(f"   storage_status: {record.get('storage_status')}")
        if record.get("image_variant_count"):
            print(
                "   image_variants: "
                f"{record.get('image_variant_count')} "
                f"(accepted={record.get('accepted_image_count')}, rejected={record.get('rejected_image_count')})"
            )
        if record.get("created_at"):
            print(f"   created_at: {record.get('created_at')}")
        if record.get("node_type") == "image":
            print_image_detail(record)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()

    nodes = load_jsonl(run_dir / "nodes.jsonl")
    edges = load_jsonl(run_dir / "edges.jsonl")
    visual_plans = load_jsonl(run_dir / args.visual_plans_file)
    runner_state = load_json(run_dir / args.state_file)

    if args.active_only:
        nodes = [node for node in nodes if is_active(node)]
        edges = [edge for edge in edges if is_active(edge)]

    nodes_by_id, in_degree, out_degree, out_edges, unknown_src, unknown_dst = build_graph_indexes(nodes, edges)
    text_nodes_by_source_url = build_text_nodes_by_source_url(nodes)
    visual_plan_indexes = build_visual_plan_indexes(visual_plans)
    queued_image_entity_tasks = collect_queued_image_entity_tasks(runner_state)
    image_entity_failures = collect_image_entity_failures(runner_state)

    isolated_all: list[dict[str, Any]] = []
    isolated_type_counter: Counter[str] = Counter()
    isolated_status_counter: Counter[str] = Counter()

    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        in_deg = in_degree.get(node_id, 0)
        out_deg = out_degree.get(node_id, 0)
        if in_deg != 0 or out_deg != 0:
            continue
        item = summarize_node(node, in_deg, out_deg)
        isolated_all.append(item)
        isolated_type_counter[str(node.get("node_type"))] += 1
        isolated_status_counter[str(node.get("status"))] += 1

    isolated_all.sort(
        key=lambda item: (
            str(item.get("node_type")),
            str(item.get("created_at") or ""),
            str(item.get("node_id") or ""),
        )
    )

    detail_limit = int(args.limit)
    if detail_limit > 0:
        isolated_for_detail = isolated_all[:detail_limit]
    else:
        isolated_for_detail = list(isolated_all)

    processed_records: list[dict[str, Any]] = []
    entity_status_counter: Counter[str] = Counter()
    for item in isolated_for_detail:
        if item.get("node_type") == "image":
            detailed = build_image_detail(
                item,
                nodes_by_id=nodes_by_id,
                out_edges=out_edges,
                queued_image_entity_tasks=queued_image_entity_tasks,
                image_entity_failures=image_entity_failures,
                visual_plan_indexes=visual_plan_indexes,
                text_nodes_by_source_url=text_nodes_by_source_url,
            )
            for entity in detailed.get("grounded_entity_diagnostics") or []:
                entity_status_counter[str(entity.get("selection_status"))] += 1
            processed_records.append(detailed)
            continue
        processed_records.append(item)

    summary = {
        "run_dir": str(run_dir),
        "active_only": args.active_only,
        "nodes": len(nodes),
        "edges": len(edges),
        "fully_isolated_nodes": len(isolated_all),
        "processed_isolated_nodes": len(processed_records),
        "visual_plan_records": len(visual_plans),
        "runner_state_loaded": runner_state is not None,
        "queued_image_entity_parent_count": len(queued_image_entity_tasks),
        "image_entity_failure_parent_count": len(image_entity_failures),
        "unknown_edge_src_refs": unknown_src,
        "unknown_edge_dst_refs": unknown_dst,
        "isolated_by_type": dict(isolated_type_counter),
        "isolated_by_status": dict(isolated_status_counter),
        "processed_entity_selection_status": dict(entity_status_counter),
    }

    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "records": processed_records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("=== Fully Isolated Node Summary ===")
    for key in (
        "run_dir",
        "active_only",
        "nodes",
        "edges",
        "fully_isolated_nodes",
        "processed_isolated_nodes",
        "visual_plan_records",
        "runner_state_loaded",
        "queued_image_entity_parent_count",
        "image_entity_failure_parent_count",
        "unknown_edge_src_refs",
        "unknown_edge_dst_refs",
    ):
        print(f"{key}: {summary[key]}")

    print_summary(Counter(node.get("node_type") for node in nodes), "\nnode_type_counts:")
    print_summary(isolated_type_counter, "\nfully_isolated_by_type:")
    print_summary(isolated_status_counter, "\nfully_isolated_by_status:")
    print_summary(entity_status_counter, "\nprocessed_entity_selection_status:")
    print_nodes(processed_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
