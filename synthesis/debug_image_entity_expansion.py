"""Debug how one image is grounded, filtered, and expanded into graph edges.

Examples:
  python -m synthesis.debug_image_entity_expansion \
    --graph-dir runs/kobe_text_only \
    --image-url "https://example.com/test.jpg" \
    --image-title "1999: Man United's miracle at Camp Nou" \
    --image-snippet "Manchester United players celebrate with the Champions League trophy" \
    --source-query-text "Manchester United comeback celebration 1999 UEFA Champions League final Camp Nou against Bayern Munich" \
    --source-node-title "Camp Nou" \
    --skip-check

  python -m synthesis.debug_image_entity_expansion \
    --graph-dir runs/kobe_text_only \
    --image-node-id image_123 \
    --image-url "https://example.com/test.jpg" \
    --image-title "manual image" \
    --image-snippet "players holding a silver trophy" \
    --source-query-text "Manchester United comeback celebration 1999 UEFA Champions League final Camp Nou against Bayern Munich" \
    --source-node-title "Camp Nou" \
    --grounded-entities-json '[{"name":"UEFA Champions League","relation_to_image":"the silver trophy being held by the players in the center","evidence":"The large silver cup in the middle is the UEFA Champions League trophy."}]'
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.evidence import Evidence, EvidenceType
from synthesis.edges import Edge, EdgeSource, EdgeType, EvidenceRef
from synthesis.image_discovery import (
    ImageCandidateStatus,
    ImageDiscoveryBuilder,
    ImageDiscoveryConfig,
    ImageValidationResult,
    ResolvedImageAsset,
)
from synthesis.nodes import ImageNode
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.search_client import ImageSearchResult
from synthesis.store import JsonlGraphStore
from synthesis.visual_planner import SearchQuerySpec, VisualSearchPlan
from synthesis.wiki_text_builder import EnhancedReaderClient, InvalidWikiPageError, WikiTextBuilder


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in debug_image_entity_expansion")

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in debug_image_entity_expansion")


def _load_grounded_entities(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.grounded_entities_file:
        path = Path(args.grounded_entities_file).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(args.grounded_entities_json)
    if not isinstance(payload, list):
        raise ValueError("grounded entities input must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def _build_search_result(args: argparse.Namespace) -> ImageSearchResult:
    return ImageSearchResult(
        title=args.image_title or None,
        image_url=args.image_url,
        source_page_url=args.source_page_url or None,
        snippet=args.image_snippet or None,
        source="debug_image_entity_expansion",
        rank=1,
        raw={"debug_image_entity_expansion": True},
    )


def _build_plan(args: argparse.Namespace, search_result: ImageSearchResult) -> VisualSearchPlan:
    target_text = (args.target_text or args.image_snippet or args.image_title or args.source_query_text).strip()
    query_text = (args.source_query_text or target_text).strip()
    target = Evidence.create(
        EvidenceType.WEB_TEXT,
        content=target_text,
        url=search_result.source_page_url,
        extractor="debug_image_entity_expansion",
        evidence_key=f"debug_target:{target_text}",
    )
    query = SearchQuerySpec.create(
        query_text,
        target.evidence_id,
        expected_visual=target_text,
        source="debug_image_entity_expansion",
    )
    return VisualSearchPlan.create(
        target,
        queries=[query],
        planner="debug_image_entity_expansion",
        metadata={"debug_image_entity_expansion": True},
    )


def _seed_local_resolved_asset(
    builder: ImageDiscoveryBuilder,
    search_result: ImageSearchResult,
    image_path: Path,
) -> ResolvedImageAsset:
    payload = image_path.read_bytes()
    content_type = mimetypes.guess_type(str(image_path))[0] or builder._sniff_content_type(payload) or "image/jpeg"
    cache_key = builder._resolved_image_cache_key(search_result.image_url, search_result.source_page_url)
    cache_path = builder._write_image_cache_file(cache_key, payload, content_type)
    model_content_type, model_payload = builder._prepare_model_payload(
        payload=payload,
        content_type=content_type,
        max_edge=builder.config.model_image_max_edge,
    )
    asset = ResolvedImageAsset(
        cache_key=cache_key,
        original_url=search_result.image_url,
        resolved_url=search_result.image_url,
        source_page_url=search_result.source_page_url,
        model_url=builder._data_url(model_content_type, model_payload),
        asset_uri=cache_path,
        cache_path=cache_path,
        content_type=content_type,
        strategy="local_file",
    )
    builder._resolved_image_cache[cache_key] = asset
    return asset


def _run_image_check(
    *,
    builder: ImageDiscoveryBuilder,
    plan: VisualSearchPlan,
    search_result: ImageSearchResult,
    local_image_path: Path | None,
) -> ImageValidationResult:
    if local_image_path is None:
        return builder.image_check(
            plan=plan,
            query=plan.queries[0],
            search_result=search_result,
            run_id="debug_image_entity_expansion",
        )

    resolved_asset = _seed_local_resolved_asset(builder, search_result, local_image_path)
    result = builder._image_check_with_mllm(
        plan=plan,
        search_result=search_result,
        model_alias=builder.image_check_model_alias or os.environ.get("IMAGE_CHECK_MODEL"),
        run_id="debug_image_entity_expansion",
        resolved_asset=resolved_asset,
    )
    result.metadata = dict(result.metadata or {})
    result.metadata["resolved_image_key"] = resolved_asset.cache_key
    result.metadata["resolved_image"] = resolved_asset.to_metadata()
    return result


def _force_accept_validation(
    builder: ImageDiscoveryBuilder,
    search_result: ImageSearchResult,
    local_image_path: Path | None,
) -> ImageValidationResult:
    metadata: dict[str, Any] = {"check": "manual_force_accept"}
    if local_image_path is not None:
        resolved_asset = _seed_local_resolved_asset(builder, search_result, local_image_path)
        metadata["resolved_image_key"] = resolved_asset.cache_key
        metadata["resolved_image"] = resolved_asset.to_metadata()
    return ImageValidationResult(
        status=ImageCandidateStatus.ACCEPTED,
        confidence=1.0,
        reason="manual_force_accept",
        metadata=metadata,
    )


def _source_node_title(args: argparse.Namespace) -> str:
    return (args.source_node_title or args.target_text or args.image_title or "manual_test_source").strip()


def _filter_grounded_entities(
    *,
    builder: ImageDiscoveryBuilder,
    args: argparse.Namespace,
    grounded_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    query_text = (args.source_query_text or args.target_text or args.image_snippet or args.image_title or "").strip()
    source_node_title = _source_node_title(args)
    blocked_query_entities = builder._query_implied_entity_labels(
        query_text,
        source_node_title=source_node_title,
        grounded_entities=grounded_entities,
    )

    kept: list[dict[str, Any]] = []
    filtered_out: list[dict[str, Any]] = []
    for entity in grounded_entities:
        if not builder._should_expand_entity(entity):
            filtered_out.append({**entity, "status": "filtered_out"})
            continue
        if builder._is_query_implied_entity(entity, blocked_query_entities):
            filtered_out.append({**entity, "status": "filtered_by_query_entity_overlap"})
            continue
        kept.append(entity)
    return {
        "query_text": query_text,
        "source_node_title": source_node_title,
        "blocked_query_entities": sorted(blocked_query_entities),
        "kept_grounded_entities": kept,
        "filtered_out_entities": filtered_out,
    }


def _simulate_queued_text_expansions(
    *,
    store: JsonlGraphStore,
    queued_tasks: list[dict[str, Any]],
    image_node: ImageNode,
    image_evidence: Evidence,
    reader_base_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reader = EnhancedReaderClient(base_url=reader_base_url)
    wiki_builder = WikiTextBuilder(
        reader=reader,
        store=store,
    )

    simulated_nodes: list[dict[str, Any]] = []
    simulated_edges: list[dict[str, Any]] = []
    for pending in queued_tasks:
        url = pending.get("url")
        title = pending.get("title")
        pending_link = pending.get("pending_link") or {}
        entity = pending_link.get("entity") or {}
        if not url or not isinstance(pending_link, dict) or not isinstance(entity, dict):
            continue
        try:
            text_result = wiki_builder.build_from_url(
                url,
                title=title,
                run_id="debug_image_entity_expansion",
                persist=False,
            )
            simulated_nodes.append(
                {
                    "url": url,
                    "requested_title": title,
                    "node_id": text_result.node.node_id,
                    "node_title": text_result.node.title,
                    "source_url": text_result.node.source.url if text_result.node.source else None,
                    "from_cache": text_result.from_cache,
                }
            )
            edge = Edge.create(
                image_node.node_id,
                text_result.node.node_id,
                edge_type=EdgeType.IMAGE_DEPICTS,
                relation=entity.get("relation_to_image") or "depicts",
                src_node_type="image",
                dst_node_type="text",
                evidence_refs=[
                    EvidenceRef(
                        evidence_id=image_evidence.evidence_id,
                        quote=entity.get("evidence"),
                        metadata={
                            "grounded_entity": entity,
                            "resolved_target": pending_link.get("resolved_target"),
                        },
                    )
                ],
                source=EdgeSource(
                    source_type="image_grounding_delayed_debug",
                    url=image_node.image_url,
                    run_id="debug_image_entity_expansion",
                    builder="debug_image_entity_expansion",
                ),
                extractor="debug_image_entity_expansion",
                metadata={
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                    "link_type": "image_entity",
                    "debug_materialized": True,
                },
                evidence_key=f"{image_evidence.evidence_id}:{entity.get('name')}:{text_result.node.node_id}",
            )
            simulated_edges.append(
                {
                    "status": "would_materialize",
                    "entity_name": entity.get("name"),
                    "target_node_id": text_result.node.node_id,
                    "target_title": text_result.node.title,
                    "edge": edge.to_dict(),
                }
            )
        except InvalidWikiPageError as exc:
            simulated_nodes.append(
                {
                    "url": url,
                    "requested_title": title,
                    "status": "skipped_invalid_page",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
        except Exception as exc:
            simulated_nodes.append(
                {
                    "url": url,
                    "requested_title": title,
                    "status": "failed_to_build",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
    return simulated_nodes, simulated_edges


def _build_image_node(args: argparse.Namespace, store: JsonlGraphStore) -> ImageNode:
    if args.image_node_id:
        record = store.get_node(args.image_node_id)
        if record is None:
            raise ValueError(f"image node not found: {args.image_node_id}")
        if record.get("node_type") != "image":
            raise ValueError(f"node is not an image node: {args.image_node_id}")
        return ImageNode(
            node_id=record["node_id"],
            title=record.get("title"),
            summary=record.get("summary"),
            source=record.get("source"),
            asset_refs=record.get("asset_refs") or [],
            metadata=dict(record.get("metadata") or {}),
            status=record.get("status", "active"),
            created_at=record.get("created_at"),
            updated_at=record.get("updated_at"),
            image_url=record.get("image_url"),
            source_page_url=record.get("source_page_url"),
            oss_uri=record.get("oss_uri"),
            thumb_oss_uri=record.get("thumb_oss_uri"),
            caption=record.get("caption"),
            width=record.get("width"),
            height=record.get("height"),
            content_type=record.get("content_type"),
            phash=record.get("phash"),
            storage_status=record.get("storage_status", "url_only"),
            primary_image_id=record.get("primary_image_id"),
            accepted_image_ids=record.get("accepted_image_ids") or [],
            rejected_image_ids=record.get("rejected_image_ids") or [],
            image_variants=[],
        )
    return ImageNode.from_url(
        args.image_url or "https://example.com/manual-debug-image.jpg",
        source_page_url=args.source_page_url or None,
        caption=args.image_snippet or None,
        title=args.image_title or None,
        metadata={},
    )


def _debug_entity_statuses(
    *,
    builder: ImageDiscoveryBuilder,
    image_node: ImageNode,
    grounded_entities: list[dict[str, Any]],
    source_query_text: str,
    source_node_title: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    blocked = builder._query_implied_entity_labels(
        source_query_text,
        source_node_title=source_node_title,
        grounded_entities=grounded_entities,
    )
    statuses: list[dict[str, Any]] = []
    for entity in grounded_entities:
        record = {"entity": entity}
        label = (entity.get("name") or "").strip()
        record["normalized_label"] = builder._normalize_entity_label(label)
        record["query_overlap_entity"] = builder._is_query_implied_entity(entity, blocked)
        record["resolver_debug"] = _debug_resolver_candidates(
            builder=builder,
            entity=entity,
            source_node_title=source_node_title,
            source_query_text=source_query_text,
            image_caption=image_node.caption,
        )
        if not builder._should_expand_entity(entity):
            record["status"] = "filtered_out"
            statuses.append(record)
            continue
        if builder._is_query_implied_entity(entity, blocked):
            record["status"] = "filtered_by_query_entity_overlap"
            statuses.append(record)
            continue
        resolution = builder._resolve_grounded_entity_link_target(
            entity,
            source_node_title=source_node_title,
            source_query_text=source_query_text,
            image_caption=image_node.caption,
        )
        if resolution is None:
            record["status"] = "unresolved"
            statuses.append(record)
            continue
        matched_node = resolution.get("matched_node")
        resolved_target = resolution.get("resolved_target")
        record["resolution_debug"] = resolution.get("debug")
        if matched_node is not None:
            record["status"] = "matched_existing_node_by_url_llm"
            record["matched_node_id"] = matched_node.get("node_id")
            record["matched_title"] = matched_node.get("title")
            statuses.append(record)
            continue
        if resolved_target is None:
            record["status"] = "unresolved"
            statuses.append(record)
            continue
        record["status"] = "queued_for_expansion"
        record["resolved_target"] = resolved_target
        statuses.append(record)
    return statuses, blocked


def _debug_existing_text_node_matches(
    *,
    builder: ImageDiscoveryBuilder,
    label: str,
) -> dict[str, Any]:
    if builder.store is None:
        return {"label": label, "reason": "missing_store"}
    needle = builder._normalize_entity_label(label)
    if not needle:
        return {"label": label, "reason": "empty_normalized_label"}

    exact_matches: list[dict[str, Any]] = []
    contains_matches: list[dict[str, Any]] = []
    scanned_count = 0
    for node in builder.store.list_nodes():
        if node.get("node_type") != "text":
            continue
        scanned_count += 1
        title = node.get("title") or ""
        aliases = node.get("aliases") or []
        labels = [title, *aliases]
        normalized_labels = [builder._normalize_entity_label(item) for item in labels if item]
        if needle in normalized_labels:
            exact_matches.append(
                {
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                    "aliases": aliases,
                    "matched_labels": [item for item, normalized in zip(labels, normalized_labels) if normalized == needle],
                }
            )
            continue
        for raw_label, normalized_label in zip(labels, normalized_labels):
            if builder._is_unique_contains_match(needle, normalized_label):
                contains_matches.append(
                    {
                        "node_id": node.get("node_id"),
                        "title": node.get("title"),
                        "aliases": aliases,
                        "matched_label": raw_label,
                        "normalized_matched_label": normalized_label,
                    }
                )
                break
    return {
        "label": label,
        "normalized_label": needle,
        "scanned_text_node_count": scanned_count,
        "exact_match_count": len(exact_matches),
        "contains_match_count": len(contains_matches),
        "exact_matches": exact_matches[:10],
        "contains_matches": contains_matches[:10],
    }


def _debug_resolver_candidates(
    *,
    builder: ImageDiscoveryBuilder,
    entity: dict[str, Any],
    source_node_title: str,
    source_query_text: str,
    image_caption: str | None,
) -> dict[str, Any]:
    resolver = builder.wiki_resolver
    label = (entity.get("name") or "").strip()
    context_parts = [part for part in (entity.get("evidence"), image_caption, source_node_title) if part]
    context = " ".join(context_parts)
    if not label:
        return {"label": label, "reason": "empty_label"}

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

    ranked = resolver.search_candidates(
        label,
        entity_type=entity.get("type"),
        source_title=source_node_title,
        context=context,
        limit=5,
    )
    local_candidates = builder._find_text_nodes_by_candidate_urls(ranked)
    resolution = builder._resolve_grounded_entity_link_target(
        entity,
        source_node_title=source_node_title,
        source_query_text=source_query_text,
        image_caption=image_caption,
    )

    return {
        "label": label,
        "entity_type": entity.get("type"),
        "source_node_title": source_node_title,
        "source_query_text": source_query_text,
        "context": context,
        "queries": queries,
        "per_query_candidates": per_query,
        "errors": errors,
        "merged_ranked_candidates": [candidate.to_dict() for candidate in ranked[:10]],
        "local_url_matched_candidates": local_candidates,
        "resolution_result": resolution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug image grounded-entity expansion.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=str, default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--image-node-id", type=str, default="")
    parser.add_argument("--image-title", type=str, default="")
    parser.add_argument("--image-snippet", type=str, default="")
    parser.add_argument("--image-url", type=str, default="")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--source-page-url", type=str, default="")
    parser.add_argument("--source-query-text", type=str, required=True)
    parser.add_argument("--source-node-title", type=str, default="")
    parser.add_argument("--target-text", type=str, default="")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--reader-base-url", type=str, default="http://127.0.0.1:8004")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--grounded-entities-json", type=str)
    group.add_argument("--grounded-entities-file", type=str)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    store = JsonlGraphStore(args.graph_dir)
    builder = ImageDiscoveryBuilder(
        store=store,
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=not bool(args.image_path),
            try_source_page_recovery=False,
        ),
        image_check_model_alias=os.environ.get("IMAGE_CHECK_MODEL"),
    )
    image_node = _build_image_node(args, store)
    search_result = _build_search_result(args)
    plan = _build_plan(args, search_result)
    local_image_path = Path(args.image_path).expanduser().resolve() if args.image_path else None

    validation_elapsed_s = 0.0
    grounding_elapsed_s = 0.0
    validation: ImageValidationResult | None = None
    grounding: dict[str, Any] = {}

    if args.grounded_entities_file or args.grounded_entities_json:
        grounded_entities = _load_grounded_entities(args)
    else:
        if args.skip_check:
            validation = _force_accept_validation(builder, search_result, local_image_path)
        else:
            validation_started_at = time.perf_counter()
            validation = _run_image_check(
                builder=builder,
                plan=plan,
                search_result=search_result,
                local_image_path=local_image_path,
            )
            validation_elapsed_s = time.perf_counter() - validation_started_at
        grounding_started_at = time.perf_counter()
        grounding = builder.image_ground(
            plan=plan,
            search_result=search_result,
            image_node=image_node,
            validation=validation,
            run_id="debug_image_entity_expansion",
        )
        grounding_elapsed_s = time.perf_counter() - grounding_started_at
        grounded_entities = list(grounding.get("grounded_entities") or [])

    image_evidence = Evidence.create(
        EvidenceType.IMAGE,
        content=image_node.caption or image_node.title or image_node.node_id,
        node_ids=[image_node.node_id],
        url=image_node.image_url,
        extractor="debug_image_entity_expansion",
    )

    statuses, blocked = _debug_entity_statuses(
        builder=builder,
        image_node=image_node,
        grounded_entities=grounded_entities,
        source_query_text=args.source_query_text,
        source_node_title=_source_node_title(args),
    )
    edges, queued_tasks = builder._link_or_queue_grounded_entities(
        image_node=image_node,
        grounded_entities=grounded_entities,
        image_evidence=image_evidence,
        run_id="debug_image_entity_expansion",
        source_node_title=_source_node_title(args) or None,
        source_query_text=args.source_query_text,
    )
    simulated_nodes, simulated_edges = _simulate_queued_text_expansions(
        store=store,
        queued_tasks=queued_tasks,
        image_node=image_node,
        image_evidence=image_evidence,
        reader_base_url=args.reader_base_url,
    )

    output = {
        "image": {
            "node_id": image_node.node_id,
            "image_url": image_node.image_url,
            "title": image_node.title,
            "snippet": args.image_snippet,
            "caption": image_node.caption,
            "source_page_url": image_node.source_page_url,
        },
        "query": {
            "source_query_text": args.source_query_text,
            "source_node_title": _source_node_title(args),
            "target_text": args.target_text or None,
        },
        "validation": validation.to_dict() if validation is not None else None,
        "grounding": grounding or {
            "grounded_entities": grounded_entities,
            "raw_model_output": None,
            "caption": image_node.caption,
        },
        "filtered_grounding": _filter_grounded_entities(
            builder=builder,
            args=args,
            grounded_entities=list(grounded_entities),
        ),
        "blocked_query_entities": sorted(blocked),
        "entity_statuses": statuses,
        "created_edges": [edge.to_dict() for edge in edges],
        "queued_tasks": queued_tasks,
        "simulated_expanded_text_nodes": simulated_nodes,
        "simulated_materialized_edges": simulated_edges,
        "expansion_summary": {
            "grounded_entity_count": len(grounded_entities),
            "created_edge_count": len(edges),
            "queued_task_count": len(queued_tasks),
            "simulated_expanded_text_node_count": sum(
                1 for node in simulated_nodes if node.get("node_id")
            ),
            "simulated_materialized_edge_count": len(simulated_edges),
            "connected_target_node_ids": [edge.dst_node_id for edge in edges],
            "queued_target_titles": [task.get("title") for task in queued_tasks],
            "simulated_connected_target_node_ids": [
                item.get("target_node_id") for item in simulated_edges
            ],
        },
        "timing": {
            "image_check_s": validation_elapsed_s,
            "image_ground_s": grounding_elapsed_s,
            "total_s": validation_elapsed_s + grounding_elapsed_s,
        },
        "image_node_metadata_after_link_or_queue": image_node.metadata,
    }
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
