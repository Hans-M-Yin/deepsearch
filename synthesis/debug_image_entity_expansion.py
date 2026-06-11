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
        if not builder._should_expand_entity(entity):
            record["status"] = "filtered_out"
            statuses.append(record)
            continue
        if builder._is_query_implied_entity(entity, blocked):
            record["status"] = "filtered_by_query_entity_overlap"
            statuses.append(record)
            continue
        matched_node = builder._match_text_node(entity.get("name"))
        if matched_node is not None:
            record["status"] = "matched_existing_node"
            record["matched_node_id"] = matched_node.get("node_id")
            record["matched_title"] = matched_node.get("title")
            statuses.append(record)
            continue
        resolved_target = builder._resolve_grounded_entity(
            entity,
            source_node_title=source_node_title,
            image_caption=image_node.caption,
        )
        if resolved_target is None:
            record["status"] = "unresolved"
            statuses.append(record)
            continue
        existing_by_url = builder._find_text_node_by_url(resolved_target["url"])
        if existing_by_url is not None:
            record["status"] = "matched_existing_node_by_url"
            record["matched_node_id"] = existing_by_url.get("node_id")
            record["matched_title"] = existing_by_url.get("title")
            record["resolved_target"] = resolved_target
            statuses.append(record)
            continue
        record["status"] = "queued_for_expansion"
        record["resolved_target"] = resolved_target
        statuses.append(record)
    return statuses, blocked


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
        "expansion_summary": {
            "grounded_entity_count": len(grounded_entities),
            "created_edge_count": len(edges),
            "queued_task_count": len(queued_tasks),
            "connected_target_node_ids": [edge.dst_node_id for edge in edges],
            "queued_target_titles": [task.get("title") for task in queued_tasks],
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
