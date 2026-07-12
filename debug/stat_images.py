import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.store import JsonlGraphStore


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in isolated-node debug")

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in isolated-node debug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect fully isolated nodes and show image grounding / source wiki URLs / resolver candidates."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing nodes.jsonl / edges.jsonl / visual_plans.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of fully isolated nodes to print",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only count nodes / edges whose status is active",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print node details as JSON lines instead of human-readable text",
    )
    parser.add_argument(
        "--state-file",
        default="graph_runner_state.json",
        help="Runner state filename inside --run-dir",
    )
    parser.add_argument(
        "--visual-plans-file",
        default="visual_plans.jsonl",
        help="Visual plan JSONL filename inside --run-dir",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="Optional env file to load before wiki resolver debug",
    )
    parser.add_argument(
        "--override-env",
        action="store_true",
        help="Let --env-file override existing env vars",
    )
    return parser.parse_args()


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


def entity_brief(entity: dict[str, Any]) -> str:
    name = short(entity.get("name") or "", 60)
    entity_type = short(entity.get("type") or "", 24)
    status = short(entity.get("status") or "", 24)
    relation = short(entity.get("relation_to_image") or "", 80)

    parts: list[str] = []
    if name:
        parts.append(f"name={name}")
    if entity_type:
        parts.append(f"type={entity_type}")
    if status:
        parts.append(f"status={status}")
    if relation:
        parts.append(f"relation={relation}")
    return " | ".join(parts) if parts else "{}"


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


def load_visual_plans(run_dir: str, file_name: str) -> list[dict[str, Any]]:
    return load_jsonl(Path(run_dir) / file_name)


def load_runner_state(run_dir: str, file_name: str) -> dict[str, Any] | None:
    path = Path(run_dir) / file_name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def collect_queued_image_entity_tasks(
    state: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(state, dict):
        return by_parent

    for task in state.get("queue") or []:
        if not isinstance(task, dict):
            continue
        if task.get("task_type") != "text_expand":
            continue

        metadata = task.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue

        pending_links = metadata.get("pending_parent_links") or []
        for pending in pending_links:
            if not isinstance(pending, dict):
                continue
            if pending.get("link_type") != "image_entity":
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
                    "wiki_title": resolved_target.get("title") or task.get("title"),
                    "wiki_url": resolved_target.get("url") or task.get("url"),
                    "query_overlap_entity": pending.get("query_overlap_entity"),
                }
            )
    return by_parent


def build_text_nodes_by_source_url(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        if node.get("node_type") != "text":
            continue
        source = node.get("source") or {}
        if not isinstance(source, dict):
            continue
        url = source.get("url")
        if not isinstance(url, str) or not url:
            continue
        result.setdefault(url, []).append(node)
    return result


def build_graph_indexes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Counter[str], Counter[str], dict[str, list[dict[str, Any]]], int, int]:
    node_ids = {node["node_id"] for node in nodes if isinstance(node.get("node_id"), str)}
    nodes_by_id: dict[str, dict[str, Any]] = {
        node["node_id"]: node for node in nodes if isinstance(node.get("node_id"), str)
    }

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


def infer_image_source_candidates(
    image_record: dict[str, Any],
    visual_plans: list[dict[str, Any]],
    text_nodes_by_source_url: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    raw_search_query = str(image_record.get("search_query") or "").strip()
    raw_visual_target = str(image_record.get("visual_target") or "").strip()
    raw_source_page_url = str(image_record.get("source_page_url") or "").strip()

    search_query = normalize_text(raw_search_query)
    visual_target = normalize_text(raw_visual_target)
    source_page_url = raw_source_page_url

    by_source_node_id: dict[str, dict[str, Any]] = {}

    for plan in visual_plans:
        source_node_id = plan.get("node_id")
        if not isinstance(source_node_id, str) or not source_node_id:
            continue

        score = 0
        matched_by: list[str] = []

        plan_queries = [
            normalize_text(item)
            for item in (plan.get("queries") or [])
            if isinstance(item, str)
        ]
        if search_query and search_query in plan_queries:
            score += 4
            matched_by.append("search_query")

        if visual_target and visual_target == normalize_text(plan.get("target_description")):
            score += 2
            matched_by.append("target_description")

        if visual_target and visual_target == normalize_text(plan.get("expected_visual")):
            score += 2
            matched_by.append("expected_visual")

        metadata = plan.get("metadata") or {}
        if isinstance(metadata, dict):
            meta_source_page_url = str(metadata.get("source_page_url") or "").strip()
            if source_page_url and source_page_url == meta_source_page_url:
                score += 2
                matched_by.append("plan.metadata.source_page_url")

        node_source_url = str(plan.get("node_source_url") or "").strip()
        if source_page_url and source_page_url == node_source_url:
            score += 1
            matched_by.append("node_source_url")

        if score <= 0:
            continue

        candidate = {
            "method": "visual_plan_match",
            "score": score,
            "matched_by": matched_by,
            "source_node_id": source_node_id,
            "source_title": plan.get("node_title"),
            "source_url": plan.get("node_source_url"),
            "plan_id": plan.get("plan_id"),
        }

        existing = by_source_node_id.get(source_node_id)
        if existing is None or candidate["score"] > existing["score"]:
            by_source_node_id[source_node_id] = candidate

    if source_page_url:
        for node in text_nodes_by_source_url.get(source_page_url, []):
            source_node_id = node.get("node_id")
            if not isinstance(source_node_id, str) or not source_node_id:
                continue
            candidate = {
                "method": "source_page_url_exact_text_url",
                "score": 1,
                "matched_by": ["source_page_url"],
                "source_node_id": source_node_id,
                "source_title": node.get("title") or node.get("canonical_id"),
                "source_url": ((node.get("source") or {}).get("url") if isinstance(node.get("source"), dict) else None),
                "plan_id": None,
            }
            existing = by_source_node_id.get(source_node_id)
            if existing is None or candidate["score"] > existing["score"]:
                by_source_node_id[source_node_id] = candidate

    return sorted(
        by_source_node_id.values(),
        key=lambda item: (
            -int(item.get("score") or 0),
            str(item.get("source_title") or ""),
            str(item.get("source_node_id") or ""),
        ),
    )


def debug_resolver_candidates(
    *,
    builder: Any,
    entity: dict[str, Any],
    source_node_title: str,
    image_caption: str | None,
) -> dict[str, Any]:
    resolver = builder.wiki_resolver
    label = (entity.get("name") or "").strip()
    context_parts = [part for part in (entity.get("evidence"), image_caption, source_node_title) if part]
    context = " ".join(context_parts)

    if not label:
        return {
            "label": label,
            "reason": "empty_label",
            "queries": [],
            "per_query_candidates": [],
            "errors": [],
            "merged_ranked_candidates": [],
            "local_url_matched_candidates": [],
        }

    queries = resolver._build_queries(
        label,
        entity_type=entity.get("type"),
        source_title=source_node_title,
        context=context,
    )
    per_query: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for query in queries:
        try:
            query_candidates = resolver._search(query, limit=5)
            per_query.append(
                {
                    "query": query,
                    "candidate_count": len(query_candidates),
                    "candidates": [candidate.to_dict() for candidate in query_candidates],
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "query": query,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    try:
        ranked = resolver.search_candidates(
            label,
            entity_type=entity.get("type"),
            source_title=source_node_title,
            context=context,
            limit=5,
        )
    except Exception as exc:
        ranked = []
        errors.append(
            {
                "query": "<merged_search>",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )

    local_candidates = builder._find_text_nodes_by_candidate_urls(ranked)

    return {
        "label": label,
        "entity_type": entity.get("type"),
        "source_node_title": source_node_title,
        "context": context,
        "queries": queries,
        "per_query_candidates": per_query,
        "errors": errors,
        "merged_ranked_candidates": [candidate.to_dict() for candidate in ranked[:10]],
        "local_url_matched_candidates": local_candidates,
    }


def build_grounded_entity_diagnostics(
    image_record: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    out_edges: dict[str, list[dict[str, Any]]],
    queued_image_entity_tasks: dict[str, list[dict[str, Any]]],
    *,
    builder: Any,
    source_node_title: str,
) -> list[dict[str, Any]]:
    grounded_entities = list(image_record.get("grounded_entities") or [])
    unresolved_grounded_entities = list(image_record.get("unresolved_grounded_entities") or [])
    query_overlap_grounded_entities = list(image_record.get("query_overlap_grounded_entities") or [])

    image_node_id = image_record.get("node_id")
    image_caption = image_record.get("caption")

    edges_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in out_edges.get(image_node_id, []):
        edge_meta = edge.get("metadata") or {}
        key = normalize_text(edge_meta.get("entity_name"))
        if key:
            edges_by_name[key].append(edge)

    queued_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in queued_image_entity_tasks.get(image_node_id, []):
        key = normalize_text(item.get("entity_name"))
        if key:
            queued_by_name[key].append(item)

    unresolved_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in unresolved_grounded_entities:
        key = normalize_text(item.get("name"))
        if key:
            unresolved_by_name[key].append(item)

    query_overlap_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in query_overlap_grounded_entities:
        key = normalize_text(item.get("name"))
        if key:
            query_overlap_by_name[key].append(item)

    diagnostics: list[dict[str, Any]] = []
    for entity in grounded_entities:
        key = normalize_text(entity.get("name"))
        matching_edges = edges_by_name.get(key) or []
        queued_targets = queued_by_name.get(key) or []
        unresolved_items = unresolved_by_name.get(key) or []

        resolver_debug = debug_resolver_candidates(
            builder=builder,
            entity=entity,
            source_node_title=source_node_title,
            image_caption=image_caption,
        )

        diag = {
            "name": entity.get("name"),
            "type": entity.get("type"),
            "relation_to_image": entity.get("relation_to_image"),
            "evidence": entity.get("evidence"),
            "recorded_wiki_title": None,
            "recorded_wiki_url": None,
            "final_outcome": None,
            "query_overlap_entity": bool(query_overlap_by_name.get(key)),
            "resolver_queries": resolver_debug.get("queries") or [],
            "resolver_per_query_candidates": resolver_debug.get("per_query_candidates") or [],
            "resolver_errors": resolver_debug.get("errors") or [],
            "merged_ranked_candidates": resolver_debug.get("merged_ranked_candidates") or [],
            "local_url_matched_candidates": resolver_debug.get("local_url_matched_candidates") or [],
            "debug_failure_reason": None,
        }

        if matching_edges:
            edge = matching_edges[0]
            target_node = nodes_by_id.get(edge.get("dst_node_id"))
            target_source = (target_node or {}).get("source") or {}
            diag["final_outcome"] = "connected_existing_text_node"
            diag["recorded_wiki_title"] = (target_node or {}).get("title") or (target_node or {}).get("canonical_id")
            diag["recorded_wiki_url"] = target_source.get("url") if isinstance(target_source, dict) else None
        elif queued_targets:
            queued = queued_targets[0]
            diag["final_outcome"] = "queued_text_node_expansion"
            diag["recorded_wiki_title"] = queued.get("wiki_title")
            diag["recorded_wiki_url"] = queued.get("wiki_url")
        elif any(item.get("status") == "filtered_out" for item in unresolved_items):
            diag["final_outcome"] = "filtered_out_before_resolution"
        elif any(item.get("status") == "unresolved" for item in unresolved_items):
            diag["final_outcome"] = "unresolved_after_resolution"
        elif query_overlap_by_name.get(key):
            diag["final_outcome"] = "query_overlap_entity_without_target"
        else:
            diag["final_outcome"] = "no_edge_and_no_target_url"

        merged = diag["merged_ranked_candidates"]
        if diag["final_outcome"] == "no_edge_and_no_target_url":
            if not merged:
                diag["debug_failure_reason"] = "no_wikipedia_candidates_found"
            else:
                diag["debug_failure_reason"] = "has_wikipedia_candidates_but_no_recorded_edge_or_queue"

        diagnostics.append(diag)

    return diagnostics


def summarize_node(node: dict[str, Any], in_deg: int, out_deg: int) -> dict[str, Any]:
    source = node.get("source") or {}
    metadata = node.get("metadata") or {}
    image_variants = node.get("image_variants") or []
    accepted_image_ids = node.get("accepted_image_ids") or []
    rejected_image_ids = node.get("rejected_image_ids") or []
    image_grounding = metadata.get("image_grounding") or {}

    grounded_entities = list(metadata.get("grounded_entities") or [])
    unresolved_grounded_entities = list(metadata.get("unresolved_grounded_entities") or [])
    query_overlap_grounded_entities = list(metadata.get("query_overlap_grounded_entities") or [])

    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "status": node.get("status"),
        "title": node.get("title"),
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
        "metadata_keys": sorted(metadata.keys())[:20] if isinstance(metadata, dict) else [],
        "grounding_check": image_grounding.get("check"),
        "grounded_entities": grounded_entities,
        "unresolved_grounded_entities": unresolved_grounded_entities,
        "query_overlap_grounded_entities": query_overlap_grounded_entities,
        "search_query": metadata.get("search_query"),
        "visual_target": metadata.get("visual_target"),
        "raw_metadata": metadata,
        "created_at": node.get("created_at"),
        "updated_at": node.get("updated_at"),
    }


def print_summary(counter: Counter[str], name: str) -> None:
    print(name)
    if not counter:
        print("  (none)")
        return
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {key}: {value}")


def print_image_grounding(record: dict[str, Any]) -> None:
    diagnostics = list(record.get("grounded_entity_diagnostics") or [])
    unresolved_grounded_entities = list(record.get("unresolved_grounded_entities") or [])
    query_overlap_grounded_entities = list(record.get("query_overlap_grounded_entities") or [])

    print(f"   image_grounding.check: {record.get('grounding_check')}")
    print(f"   grounded_entity_count: {len(diagnostics)}")
    for idx, item in enumerate(diagnostics, start=1):
        parts = [f"name={short(item.get('name') or '', 60)}"]
        if item.get("type"):
            parts.append(f"type={short(item.get('type'), 24)}")
        if item.get("relation_to_image"):
            parts.append(f"relation={short(item.get('relation_to_image'), 80)}")
        if item.get("final_outcome"):
            parts.append(f"outcome={item.get('final_outcome')}")
        print(f"   GE{idx}. {' | '.join(parts)}")

        if item.get("recorded_wiki_title"):
            print(f"      recorded_wiki_title: {short(item.get('recorded_wiki_title'), 120)}")
        if item.get("recorded_wiki_url"):
            print(f"      recorded_wiki_url: {item.get('recorded_wiki_url')}")
        if item.get("query_overlap_entity"):
            print("      query_overlap_entity: true")
        evidence = short(item.get("evidence") or "", 120)
        if evidence:
            print(f"      evidence: {evidence}")

        queries = list(item.get("resolver_queries") or [])
        if queries:
            print(f"      wiki_match_query_count: {len(queries)}")
            for q_idx, query in enumerate(queries[:5], start=1):
                print(f"         Q{q_idx}: {short(query, 140)}")

        per_query = list(item.get("resolver_per_query_candidates") or [])
        if per_query:
            for q_idx, payload in enumerate(per_query[:5], start=1):
                query = payload.get("query")
                candidates = list(payload.get("candidates") or [])
                print(
                    f"         Q{q_idx}_returned: query={short(query, 100)} "
                    f"count={payload.get('candidate_count')}"
                )
                for c_idx, cand in enumerate(candidates[:3], start=1):
                    print(
                        f"            Q{q_idx}.C{c_idx}: "
                        f"title={short(cand.get('title'), 80)} | "
                        f"score={cand.get('score')} | "
                        f"url={cand.get('url')}"
                    )

        merged = list(item.get("merged_ranked_candidates") or [])
        print(f"      merged_wiki_candidate_count: {len(merged)}")
        if merged:
            for c_idx, cand in enumerate(merged[:5], start=1):
                print(
                    f"         M{c_idx}: "
                    f"title={short(cand.get('title'), 80)} | "
                    f"score={cand.get('score')} | "
                    f"url={cand.get('url')}"
                )

        local_matches = list(item.get("local_url_matched_candidates") or [])
        if local_matches:
            print(f"      local_url_matched_candidate_count: {len(local_matches)}")
            for l_idx, local in enumerate(local_matches[:3], start=1):
                node = local.get("node") or {}
                candidate = local.get("candidate") or {}
                print(
                    f"         L{l_idx}: "
                    f"node_id={node.get('node_id')} | "
                    f"local_title={short(node.get('title'), 80)} | "
                    f"candidate_title={short(candidate.get('title'), 80)} | "
                    f"url={local.get('url')}"
                )

        if item.get("debug_failure_reason"):
            print(f"      debug_failure_reason: {item.get('debug_failure_reason')}")

        resolver_errors = list(item.get("resolver_errors") or [])
        if resolver_errors:
            print(f"      resolver_error_count: {len(resolver_errors)}")
            for e_idx, err in enumerate(resolver_errors[:3], start=1):
                print(
                    f"         E{e_idx}: "
                    f"query={short(err.get('query'), 100)} | "
                    f"error={short(err.get('error'), 160)}"
                )

    print(f"   unresolved_grounded_entity_count: {len(unresolved_grounded_entities)}")
    if unresolved_grounded_entities:
        preview = ", ".join(entity_brief(entity) for entity in unresolved_grounded_entities[:5])
        print(f"   unresolved_grounded_entities: {preview}")
        remaining = len(unresolved_grounded_entities) - min(len(unresolved_grounded_entities), 5)
        if remaining > 0:
            print(f"   unresolved_grounded_entities_more: {remaining}")

    print(f"   query_overlap_grounded_entity_count: {len(query_overlap_grounded_entities)}")
    if query_overlap_grounded_entities:
        preview = ", ".join(entity_brief(entity) for entity in query_overlap_grounded_entities[:5])
        print(f"   query_overlap_grounded_entities: {preview}")
        remaining = len(query_overlap_grounded_entities) - min(len(query_overlap_grounded_entities), 5)
        if remaining > 0:
            print(f"   query_overlap_grounded_entities_more: {remaining}")


def print_image_source(record: dict[str, Any]) -> None:
    if record.get("search_query"):
        print(f"   search_query: {short(record.get('search_query'), 140)}")
    if record.get("visual_target"):
        print(f"   visual_target: {short(record.get('visual_target'), 140)}")

    candidates = list(record.get("inferred_source_candidates") or [])
    print(f"   inferred_source_candidate_count: {len(candidates)}")
    if not candidates:
        print("   inferred_source_wiki_url: not found")
        return

    best = candidates[0]
    print(f"   inferred_source_wiki_title: {short(best.get('source_title'), 120)}")
    print(f"   inferred_source_wiki_url: {best.get('source_url')}")
    if best.get("plan_id"):
        print(f"   inferred_source_plan_id: {best.get('plan_id')}")

    for idx, item in enumerate(candidates[:3], start=1):
        matched_by = ", ".join(item.get("matched_by") or [])
        print(
            f"   source{idx}: method={item.get('method')} "
            f"score={item.get('score')} "
            f"matched_by={matched_by}"
        )
        print(f"      source_node_id: {item.get('source_node_id')}")
        print(f"      source_title: {short(item.get('source_title'), 120)}")
        if item.get("source_url"):
            print(f"      source_url: {item.get('source_url')}")
        if item.get("plan_id"):
            print(f"      plan_id: {item.get('plan_id')}")


def print_nodes(records: list[dict[str, Any]], label: str, limit: int, as_json: bool) -> None:
    print(f"\n=== {label} ({len(records)}) ===")
    if not records:
        print("(none)")
        return

    for idx, record in enumerate(records[:limit], start=1):
        if as_json:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
            continue

        print(
            f"{idx}. [{record.get('node_type')}][{record.get('status')}] "
            f"{short(record.get('title') or record.get('node_id'), 120)}"
        )
        print(f"   node_id: {record.get('node_id')}")
        print(f"   degree: in={record.get('in_degree')} out={record.get('out_degree')}")
        if record.get("subtype"):
            print(f"   subtype: {record.get('subtype')}")
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
        if record.get("metadata_keys"):
            print(f"   metadata_keys: {', '.join(record.get('metadata_keys')[:10])}")
        print(f"   created_at: {record.get('created_at')}")

        if record.get("node_type") == "image":
            print_image_grounding(record)
            print_image_source(record)

    remaining = len(records) - min(len(records), limit)
    if remaining > 0:
        print(f"... truncated {remaining} more")


def main() -> int:
    args = parse_args()

    env_path = Path(args.env_file).expanduser().resolve()
    loaded_env = load_env_file(env_path, override=args.override_env)

    from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig

    store = JsonlGraphStore(args.run_dir)
    builder = ImageDiscoveryBuilder(
        store=store,
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=False,
            try_source_page_recovery=False,
        ),
    )

    nodes = store.list_nodes()
    edges = store.list_edges()

    if args.active_only:
        nodes = [node for node in nodes if is_active(node)]
        edges = [edge for edge in edges if is_active(edge)]

    visual_plans = load_visual_plans(args.run_dir, args.visual_plans_file)
    runner_state = load_runner_state(args.run_dir, args.state_file)
    queued_image_entity_tasks = collect_queued_image_entity_tasks(runner_state)
    text_nodes_by_source_url = build_text_nodes_by_source_url(nodes)

    nodes_by_id, in_degree, out_degree, out_edges, unknown_src, unknown_dst = build_graph_indexes(
        nodes, edges
    )

    isolated: list[dict[str, Any]] = []
    isolated_type_counter: Counter[str] = Counter()
    isolated_status_counter: Counter[str] = Counter()

    for node in nodes:
        node_id = node["node_id"]
        in_deg = in_degree.get(node_id, 0)
        out_deg = out_degree.get(node_id, 0)
        if in_deg != 0 or out_deg != 0:
            continue

        item = summarize_node(node, in_deg, out_deg)
        if item.get("node_type") == "image":
            item["inferred_source_candidates"] = infer_image_source_candidates(
                item,
                visual_plans=visual_plans,
                text_nodes_by_source_url=text_nodes_by_source_url,
            )
            inferred_sources = item.get("inferred_source_candidates") or []
            source_node_title = ""
            if inferred_sources:
                source_node_title = str(inferred_sources[0].get("source_title") or "").strip()

            item["grounded_entity_diagnostics"] = build_grounded_entity_diagnostics(
                item,
                nodes_by_id=nodes_by_id,
                out_edges=out_edges,
                queued_image_entity_tasks=queued_image_entity_tasks,
                builder=builder,
                source_node_title=source_node_title,
            )
        isolated.append(item)
        isolated_type_counter[str(node.get("node_type"))] += 1
        isolated_status_counter[str(node.get("status"))] += 1

    isolated.sort(
        key=lambda item: (
            str(item.get("node_type")),
            str(item.get("created_at")),
            str(item.get("node_id")),
        )
    )

    print("=== Fully Isolated Node Summary ===")
    print(f"run_dir: {args.run_dir}")
    print(f"env_file: {env_path}")
    print(f"env_loaded_count: {len(loaded_env)}")
    print(f"active_only: {args.active_only}")
    print(f"nodes: {len(nodes)}")
    print(f"edges: {len(edges)}")
    print(f"fully_isolated_nodes: {len(isolated)}")
    print(f"visual_plan_records: {len(visual_plans)}")
    print(f"runner_state_loaded: {'yes' if runner_state is not None else 'no'}")
    print(f"queued_image_entity_parent_count: {len(queued_image_entity_tasks)}")
    print(f"unknown_edge_src_refs: {unknown_src}")
    print(f"unknown_edge_dst_refs: {unknown_dst}")

    print_summary(Counter(node.get("node_type") for node in nodes), "\nnode_type_counts:")
    print_summary(isolated_type_counter, "\nfully_isolated_by_type:")
    print_summary(isolated_status_counter, "\nfully_isolated_by_status:")

    print_nodes(isolated, "Fully Isolated Nodes", args.limit, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
