"""List image nodes and summarize grounded-entity status.

Examples:
  python debug/list_neighbor.py \
    --graph-dir runs/0712_multi_seed_visual_test4 \
    --limit 10

  python debug/list_neighbor.py \
    --graph-dir synthesis/runs/mock_graph_review_20260712_env/query_overlap \
    --limit 5 \
    --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


IMAGE_NODE_TYPE = "image"
IMAGE_DEPICTS_EDGE_TYPE = "image_depicts"
DEFAULT_STATE_FILE_NAME = "graph_runner_state.json"

STATUS_PRIORITY = {
    "linked": 100,
    "parent_link_failed": 90,
    "queued_pending": 80,
    "task_failed": 70,
    "task_skipped": 60,
    "task_completed": 50,
    "unresolved": 40,
    "filtered_by_query_entity_overlap": 35,
    "filtered_out": 30,
    "query_overlap_entity": 20,
    "grounded_only": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize image nodes, their origin, and grounded-entity status."
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing nodes.jsonl and edges.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of image nodes to print. <=0 means all image nodes.",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=180,
        help="Max characters used when printing long title/evidence snippets.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object instead of human-readable text.",
    )
    return parser.parse_args()


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _entity_key(entity: dict[str, Any]) -> tuple[str, str, str, str] | None:
    name = _normalize_text(entity.get("name"))
    entity_type = _normalize_text(entity.get("type"))
    relation = _normalize_text(entity.get("relation_to_image") or entity.get("relation"))
    evidence = _normalize_text(entity.get("evidence"))
    if not any((name, entity_type, relation, evidence)):
        return None
    return (name, entity_type, relation, evidence)


def _entity_sort_label(entity: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _normalize_text(entity.get("name")),
        _normalize_text(entity.get("type")),
        _normalize_text(entity.get("relation_to_image") or entity.get("relation")),
    )


def _source_url(node: dict[str, Any]) -> str | None:
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


def _node_title(node: dict[str, Any], *, summary_chars: int) -> str:
    for key in ("title", "caption", "summary", "canonical_id", "node_id"):
        value = node.get(key)
        if value:
            return _short(value, summary_chars)
    return "<untitled>"


def _image_origin(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    source = node.get("source") or {}
    source_type = source.get("source_type") if isinstance(source, dict) else None
    if (
        source_type == "wikipedia_inline_image"
        or metadata.get("image_origin") == "wikipedia_inline"
        or metadata.get("wiki_inline_keep_in_graph") is not None
    ):
        return "wiki_inline"
    if metadata.get("search_query") or metadata.get("visual_target") or source_type == "image_search_bundle":
        return "visual_plan"
    return "unknown"


def _load_runner_state(graph_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    state_path = graph_dir / DEFAULT_STATE_FILE_NAME
    if not state_path.exists():
        return None, None
    try:
        return json.loads(state_path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def _task_entity_records(task_record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = task_record.get("metadata") or {}
    if metadata.get("task_origin") != "image_entity":
        return []

    parent_node_id = task_record.get("parent_node_id")
    pending_links = metadata.get("pending_parent_links") or []
    records: list[dict[str, Any]] = []

    for pending in pending_links:
        if not isinstance(pending, dict):
            continue
        if (pending.get("link_type") or "wiki_link") != "image_entity":
            continue
        entity = pending.get("entity")
        if not isinstance(entity, dict):
            entity = {
                "name": metadata.get("entity_name"),
                "type": metadata.get("entity_type"),
            }
        key = _entity_key(entity)
        if key is None:
            continue
        records.append(
            {
                "image_node_id": pending.get("parent_node_id") or parent_node_id,
                "entity": dict(entity),
                "entity_key": key,
                "resolved_target": pending.get("resolved_target"),
            }
        )

    if records:
        return records

    entity = {
        "name": metadata.get("entity_name"),
        "type": metadata.get("entity_type"),
    }
    key = _entity_key(entity)
    if key is None or not parent_node_id:
        return []
    return [
        {
            "image_node_id": parent_node_id,
            "entity": entity,
            "entity_key": key,
            "resolved_target": None,
        }
    ]


def _matching_failure(failures: list[dict[str, Any]], entity: dict[str, Any]) -> dict[str, Any] | None:
    entity_name = _normalize_text(entity.get("name"))
    entity_type = _normalize_text(entity.get("type"))
    for failure in failures:
        failure_name = _normalize_text(failure.get("entity_name"))
        failure_type = _normalize_text(failure.get("entity_type"))
        if entity_name and failure_name and entity_name == failure_name:
            return failure
        if entity_name and failure_name:
            continue
        if entity_type and failure_type and entity_type == failure_type:
            return failure
    return None


def _runner_status_from_section(section: str, record: dict[str, Any], entity: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if section == "queue":
        return "queued_pending", {}
    if section == "failed_tasks":
        return "task_failed", {"error": record.get("error")}
    if section == "skipped_tasks":
        return "task_skipped", {"attribute_error": record.get("attribute_error")}
    if section == "completed_tasks":
        failures = [item for item in (record.get("parent_link_failures") or []) if isinstance(item, dict)]
        failure = _matching_failure(failures, entity)
        if failure is not None:
            return "parent_link_failed", {"failure": failure}
        return "task_completed", {"materialized_edge_count": record.get("materialized_edge_count")}
    return "grounded_only", {}


def _collect_runner_state_index(state: dict[str, Any] | None) -> dict[tuple[str, tuple[str, str, str, str]], list[dict[str, Any]]]:
    if not isinstance(state, dict):
        return {}

    indexed: dict[tuple[str, tuple[str, str, str, str]], list[dict[str, Any]]] = defaultdict(list)
    sections = (
        ("queue", state.get("queue") or []),
        ("failed_tasks", state.get("failed_tasks") or []),
        ("skipped_tasks", state.get("skipped_tasks") or []),
        ("completed_tasks", state.get("completed_tasks") or []),
    )

    for section_name, items in sections:
        for raw in items:
            task_record = raw if section_name == "queue" else raw.get("task") or {}
            for task_item in _task_entity_records(task_record):
                image_node_id = task_item.get("image_node_id")
                entity_key = task_item.get("entity_key")
                entity = task_item.get("entity")
                if not image_node_id or entity_key is None or not isinstance(entity, dict):
                    continue
                status, extra = _runner_status_from_section(section_name, raw if isinstance(raw, dict) else {}, entity)
                indexed[(str(image_node_id), entity_key)].append(
                    {
                        "status": status,
                        "entity": dict(entity),
                        "resolved_target": task_item.get("resolved_target"),
                        "section": section_name,
                        **extra,
                    }
                )
    return indexed


def _collect_linked_entities(
    *,
    edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    summary_chars: int,
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    linked: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge.get("edge_type") != IMAGE_DEPICTS_EDGE_TYPE:
            continue
        target_node = nodes_by_id.get(edge.get("dst_node_id"))
        edge_metadata = edge.get("metadata") or {}
        evidence_refs = edge.get("evidence_refs") or []
        matched = False

        for evidence_ref in evidence_refs:
            if not isinstance(evidence_ref, dict):
                continue
            ref_metadata = evidence_ref.get("metadata") or {}
            entity = ref_metadata.get("grounded_entity")
            if not isinstance(entity, dict):
                continue
            key = _entity_key(entity)
            if key is None:
                continue
            linked[key].append(
                {
                    "edge_id": edge.get("edge_id"),
                    "relation": edge.get("relation"),
                    "dst_node_id": edge.get("dst_node_id"),
                    "dst_title": _node_title(target_node or {}, summary_chars=summary_chars) if target_node else None,
                    "query_overlap_entity": bool(
                        ref_metadata.get("query_overlap_entity")
                        or edge_metadata.get("query_overlap_entity")
                    ),
                    "resolved_target": ref_metadata.get("resolved_target"),
                }
            )
            matched = True

        if matched:
            continue

        fallback_entity = {
            "name": edge_metadata.get("entity_name"),
            "type": edge_metadata.get("entity_type"),
            "relation_to_image": edge.get("relation"),
        }
        key = _entity_key(fallback_entity)
        if key is None:
            continue
        linked[key].append(
            {
                "edge_id": edge.get("edge_id"),
                "relation": edge.get("relation"),
                "dst_node_id": edge.get("dst_node_id"),
                "dst_title": _node_title(target_node or {}, summary_chars=summary_chars) if target_node else None,
                "query_overlap_entity": bool(edge_metadata.get("query_overlap_entity")),
                "resolved_target": None,
            }
        )
    return linked


def _ensure_entity_report(
    reports: dict[tuple[str, str, str, str], dict[str, Any]],
    entity: dict[str, Any],
    *,
    summary_chars: int,
) -> dict[str, Any] | None:
    key = _entity_key(entity)
    if key is None:
        return None
    report = reports.get(key)
    if report is None:
        report = {
            "entity_key": list(key),
            "name": entity.get("name"),
            "type": entity.get("type"),
            "relation_to_image": _short(entity.get("relation_to_image") or entity.get("relation"), summary_chars),
            "evidence": _short(entity.get("evidence"), summary_chars),
            "status": "grounded_only",
            "query_overlap_entity": False,
            "metadata_statuses": [],
            "linked_targets": [],
            "runner_state": [],
            "present_in_grounded_entities": False,
            "present_in_unresolved_grounded_entities": False,
            "present_in_query_overlap_grounded_entities": False,
        }
        reports[key] = report
        return report

    if not report.get("name") and entity.get("name"):
        report["name"] = entity.get("name")
    if not report.get("type") and entity.get("type"):
        report["type"] = entity.get("type")
    if not report.get("relation_to_image") and (entity.get("relation_to_image") or entity.get("relation")):
        report["relation_to_image"] = _short(entity.get("relation_to_image") or entity.get("relation"), summary_chars)
    if not report.get("evidence") and entity.get("evidence"):
        report["evidence"] = _short(entity.get("evidence"), summary_chars)
    return report


def _choose_entity_status(report: dict[str, Any]) -> str:
    if report.get("linked_targets"):
        return "linked"

    runner_statuses = {str(item.get("status") or "") for item in (report.get("runner_state") or [])}
    for candidate in ("parent_link_failed", "queued_pending", "task_failed", "task_skipped", "task_completed"):
        if candidate in runner_statuses:
            return candidate

    metadata_statuses = {str(item or "") for item in (report.get("metadata_statuses") or [])}
    for candidate in ("unresolved", "filtered_by_query_entity_overlap", "filtered_out", "query_overlap_entity"):
        if candidate in metadata_statuses:
            return candidate
    return "grounded_only"


def _status_rank(status: str) -> int:
    return STATUS_PRIORITY.get(status, 0)


def _build_grounded_entity_reports(
    *,
    image_node: dict[str, Any],
    out_edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    runner_state_index: dict[tuple[str, tuple[str, str, str, str]], list[dict[str, Any]]],
    summary_chars: int,
) -> list[dict[str, Any]]:
    metadata = image_node.get("metadata") or {}
    raw_grounded = [item for item in (metadata.get("grounded_entities") or []) if isinstance(item, dict)]
    unresolved = [item for item in (metadata.get("unresolved_grounded_entities") or []) if isinstance(item, dict)]
    query_overlap = [item for item in (metadata.get("query_overlap_grounded_entities") or []) if isinstance(item, dict)]

    linked_by_key = _collect_linked_entities(
        edges=out_edges,
        nodes_by_id=nodes_by_id,
        summary_chars=summary_chars,
    )

    reports: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for entity in raw_grounded:
        report = _ensure_entity_report(reports, entity, summary_chars=summary_chars)
        if report is not None:
            report["present_in_grounded_entities"] = True

    for entity in unresolved:
        report = _ensure_entity_report(reports, entity, summary_chars=summary_chars)
        if report is not None:
            report["present_in_unresolved_grounded_entities"] = True
            status = str(entity.get("status") or "unresolved")
            report["metadata_statuses"].append(status)
            if status in {"query_overlap_entity", "filtered_by_query_entity_overlap"}:
                report["query_overlap_entity"] = True

    for entity in query_overlap:
        report = _ensure_entity_report(reports, entity, summary_chars=summary_chars)
        if report is not None:
            report["present_in_query_overlap_grounded_entities"] = True
            report["query_overlap_entity"] = True
            report["metadata_statuses"].append(str(entity.get("status") or "query_overlap_entity"))

    for entity_key, linked_targets in linked_by_key.items():
        report = reports.get(entity_key)
        if report is None:
            fallback_entity = {
                "name": entity_key[0],
                "type": entity_key[1],
                "relation_to_image": entity_key[2],
                "evidence": entity_key[3],
            }
            report = _ensure_entity_report(reports, fallback_entity, summary_chars=summary_chars)
        if report is None:
            continue
        report["linked_targets"].extend(linked_targets)
        if any(item.get("query_overlap_entity") for item in linked_targets):
            report["query_overlap_entity"] = True

    image_node_id = str(image_node.get("node_id") or "")
    for entity_key, report in list(reports.items()):
        runner_items = runner_state_index.get((image_node_id, entity_key), [])
        if runner_items:
            report["runner_state"].extend(runner_items)
            if any(item.get("status") == "queued_pending" for item in runner_items):
                report["query_overlap_entity"] = report.get("query_overlap_entity", False)

    # Include entities that only show up in runner state or linked edges.
    known_keys = set(reports)
    for key, linked_targets in linked_by_key.items():
        if key in known_keys:
            continue
        fallback_entity = {
            "name": key[0],
            "type": key[1],
            "relation_to_image": key[2],
            "evidence": key[3],
        }
        report = _ensure_entity_report(reports, fallback_entity, summary_chars=summary_chars)
        if report is not None:
            report["linked_targets"].extend(linked_targets)
    for (parent_node_id, key), runner_items in runner_state_index.items():
        if parent_node_id != image_node_id:
            continue
        if key in reports:
            continue
        entity = runner_items[0].get("entity") if runner_items else {}
        report = _ensure_entity_report(reports, entity if isinstance(entity, dict) else {}, summary_chars=summary_chars)
        if report is not None:
            report["runner_state"].extend(runner_items)

    grounded_entities = list(reports.values())
    for report in grounded_entities:
        report["metadata_statuses"] = sorted(set(str(item) for item in report.get("metadata_statuses") or [] if item))
        report["status"] = _choose_entity_status(report)
        report["linked_targets"] = sorted(
            report.get("linked_targets") or [],
            key=lambda item: (
                str(item.get("dst_title") or ""),
                str(item.get("dst_node_id") or ""),
                str(item.get("edge_id") or ""),
            ),
        )
        report["runner_state"] = sorted(
            report.get("runner_state") or [],
            key=lambda item: (
                -_status_rank(str(item.get("status") or "")),
                str(item.get("section") or ""),
                str(((item.get("resolved_target") or {}).get("title") or "")),
            ),
        )

    grounded_entities.sort(
        key=lambda item: (
            -_status_rank(str(item.get("status") or "")),
            *_entity_sort_label(item),
        )
    )
    return grounded_entities


def _image_report(
    *,
    image_node: dict[str, Any],
    store: JsonlGraphStore,
    nodes_by_id: dict[str, dict[str, Any]],
    runner_state_index: dict[tuple[str, tuple[str, str, str, str]], list[dict[str, Any]]],
    summary_chars: int,
) -> dict[str, Any]:
    source = image_node.get("source") or {}
    metadata = image_node.get("metadata") or {}
    out_edges = list(store.edges_from(str(image_node.get("node_id") or "")))
    grounded_entities = _build_grounded_entity_reports(
        image_node=image_node,
        out_edges=out_edges,
        nodes_by_id=nodes_by_id,
        runner_state_index=runner_state_index,
        summary_chars=summary_chars,
    )
    status_counts = Counter(str(item.get("status") or "unknown") for item in grounded_entities)

    return {
        "node_id": image_node.get("node_id"),
        "node_type": image_node.get("node_type"),
        "title": _node_title(image_node, summary_chars=summary_chars),
        "status": image_node.get("status"),
        "created_at": image_node.get("created_at"),
        "source_type": source.get("source_type") if isinstance(source, dict) else None,
        "origin": _image_origin(image_node),
        "image_url": image_node.get("image_url") or _source_url(image_node),
        "source_page_url": image_node.get("source_page_url"),
        "summary": _short(image_node.get("summary") or image_node.get("caption"), summary_chars),
        "search_query": _short(metadata.get("search_query"), summary_chars),
        "visual_target": _short(metadata.get("visual_target"), summary_chars),
        "grounded_entity_count": len(grounded_entities),
        "grounded_entity_status_counts": dict(sorted(status_counts.items())),
        "grounded_entities": grounded_entities,
    }


def build_report(*, graph_dir: Path, limit: int, summary_chars: int) -> dict[str, Any]:
    store = JsonlGraphStore(graph_dir)
    nodes_by_id = {record["node_id"]: record for record in store.list_nodes()}
    image_nodes = [node for node in nodes_by_id.values() if node.get("node_type") == IMAGE_NODE_TYPE]
    image_nodes.sort(
        key=lambda node: (
            str(node.get("created_at") or ""),
            str(node.get("node_id") or ""),
        ),
        reverse=True,
    )
    selected_nodes = image_nodes if limit <= 0 else image_nodes[:limit]

    runner_state, runner_state_error = _load_runner_state(graph_dir)
    runner_state_index = _collect_runner_state_index(runner_state)

    images = [
        _image_report(
            image_node=node,
            store=store,
            nodes_by_id=nodes_by_id,
            runner_state_index=runner_state_index,
            summary_chars=summary_chars,
        )
        for node in selected_nodes
    ]

    return {
        "graph_dir": str(graph_dir),
        "limit": limit,
        "total_image_nodes": len(image_nodes),
        "returned_image_nodes": len(images),
        "runner_state_present": runner_state is not None,
        "runner_state_error": runner_state_error,
        "images": images,
    }


def _print_grounded_entities(grounded_entities: list[dict[str, Any]]) -> None:
    print(f"  grounded_entity_count={len(grounded_entities)}")
    if not grounded_entities:
        print("  grounded_entities=<none>")
        return

    print("  grounded_entities:")
    for index, entity in enumerate(grounded_entities, start=1):
        name = entity.get("name") or "<unnamed>"
        entity_type = entity.get("type") or "?"
        status = entity.get("status") or "unknown"
        print(
            f"    {index:>2}. status={status} query_overlap={bool(entity.get('query_overlap_entity'))} "
            f"name={name!r} type={entity_type!r}"
        )
        if entity.get("relation_to_image"):
            print(f"        relation_to_image={entity.get('relation_to_image')!r}")
        if entity.get("evidence"):
            print(f"        evidence={entity.get('evidence')!r}")
        metadata_statuses = entity.get("metadata_statuses") or []
        if metadata_statuses:
            print(f"        metadata_statuses={metadata_statuses}")
        linked_targets = entity.get("linked_targets") or []
        if linked_targets:
            for target in linked_targets:
                print(
                    "        linked_to="
                    f"{target.get('dst_node_id')} title={target.get('dst_title')!r} "
                    f"edge_id={target.get('edge_id')} relation={target.get('relation')!r}"
                )
                resolved_target = target.get("resolved_target") or {}
                if isinstance(resolved_target, dict) and resolved_target.get("url"):
                    print(
                        "          resolved_target="
                        f"{resolved_target.get('title')!r} url={resolved_target.get('url')}"
                    )
        runner_state = entity.get("runner_state") or []
        if runner_state:
            for item in runner_state:
                print(
                    f"        runner_state status={item.get('status')} section={item.get('section')}"
                )
                resolved_target = item.get("resolved_target") or {}
                if isinstance(resolved_target, dict) and resolved_target.get("url"):
                    print(
                        "          resolved_target="
                        f"{resolved_target.get('title')!r} url={resolved_target.get('url')}"
                    )
                if item.get("error"):
                    print(f"          error={item.get('error')!r}")
                if item.get("attribute_error"):
                    print(f"          attribute_error={item.get('attribute_error')!r}")
                failure = item.get("failure")
                if isinstance(failure, dict):
                    print(f"          failure_reason={failure.get('reason')!r}")


def print_report(report: dict[str, Any]) -> None:
    print("Image Node Summary")
    print(f"  graph_dir={report.get('graph_dir')}")
    print(f"  total_image_nodes={report.get('total_image_nodes')}")
    print(f"  returned_image_nodes={report.get('returned_image_nodes')}")
    print(f"  limit={report.get('limit')}")
    print(f"  runner_state_present={report.get('runner_state_present')}")
    if report.get("runner_state_error"):
        print(f"  runner_state_error={report.get('runner_state_error')!r}")

    images = report.get("images") or []
    if not images:
        print("\n<no image nodes>")
        return

    for index, image in enumerate(images, start=1):
        print(f"\n[{index}] {image.get('node_id')}")
        print(f"  title={image.get('title')!r}")
        print(f"  status={image.get('status')}")
        print(f"  origin={image.get('origin')} source_type={image.get('source_type')}")
        print(f"  image_url={image.get('image_url')}")
        if image.get("source_page_url"):
            print(f"  source_page_url={image.get('source_page_url')}")
        if image.get("search_query"):
            print(f"  search_query={image.get('search_query')!r}")
        if image.get("visual_target"):
            print(f"  visual_target={image.get('visual_target')!r}")
        if image.get("summary"):
            print(f"  summary={image.get('summary')!r}")
        print(f"  grounded_entity_status_counts={image.get('grounded_entity_status_counts')}")
        _print_grounded_entities(image.get("grounded_entities") or [])


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    if not graph_dir.exists():
        print(f"Graph directory does not exist: {graph_dir}", file=sys.stderr)
        return 1

    report = build_report(
        graph_dir=graph_dir,
        limit=int(args.limit),
        summary_chars=max(40, int(args.summary_chars)),
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
