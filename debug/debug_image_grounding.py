"""Run image grounding on a single image and show only Wikipedia candidate URLs.

Examples:
  python debug/debug_image_grounding.py \
    --image-url https://cdn.nba.com/manage/2021/08/kobe-to-shaq.jpg \
    --title "Kobe to Shaq alley-oop" \
    --query-text "Kobe Bryant throwing the alley-oop pass to Shaquille O'Neal in Game 7 of the 2000 Western Conference Finals" \
    --source-node-title "Kobe Bryant" \
    --pretty

  python debug/debug_image_grounding.py \
    --image-path /tmp/kobe_shaq.jpg \
    --title "Kobe to Shaq alley-oop" \
    --skip-check \
    --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig, ImageValidationResult
from synthesis.nodes import ImageNode
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.test_image_grounding import (
    _UnusedSearchClient,
    _build_plan,
    _build_search_result,
    _force_accept_validation,
    _run_image_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run image grounding and show per-entity Wikipedia candidate URLs without LLM resolution debug."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-path", type=str, help="Local image file path.")
    group.add_argument("--image-url", type=str, help="Remote image URL.")
    parser.add_argument("--title", "--image-title", dest="title", type=str, default="", help="Image title.")
    parser.add_argument(
        "--snippet",
        "--image-snippet",
        dest="snippet",
        type=str,
        default="",
        help="Image snippet or caption.",
    )
    parser.add_argument("--source-page-url", type=str, default="", help="Optional source page URL.")
    parser.add_argument(
        "--target-text",
        type=str,
        default="",
        help="Visual target text. Defaults to snippet/title if omitted.",
    )
    parser.add_argument(
        "--query-text",
        "--source-query-text",
        dest="query_text",
        type=str,
        default="",
        help="Visual-plan query text used for overlap filtering. Defaults to target-text.",
    )
    parser.add_argument(
        "--source-node-title",
        type=str,
        default="",
        help="Optional source text-node title used by query-overlap filtering and wiki search hints.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(DEFAULT_ENV_PATH),
        help="Path to synthesis env file.",
    )
    parser.add_argument(
        "--reader-base-url",
        type=str,
        default="http://127.0.0.1:8004",
        help="Enhanced Reader base URL used to fetch source-page context for grounding.",
    )
    parser.add_argument(
        "--wiki-limit",
        type=int,
        default=5,
        help="Max number of Wikipedia candidates to keep per query and after merge.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip image_check and force image_ground on the provided image.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the final JSON result.",
    )
    return parser.parse_args()


def _source_node_title(args: argparse.Namespace) -> str:
    return (args.source_node_title or args.target_text or args.title or "manual_test_source").strip()


def _query_text(args: argparse.Namespace) -> str:
    return (args.query_text or args.target_text or args.snippet or args.title or "").strip()


def _compact_validation(validation: ImageValidationResult) -> dict[str, Any]:
    metadata = dict(validation.metadata or {})
    return {
        "status": validation.status.value if hasattr(validation.status, "value") else str(validation.status),
        "confidence": validation.confidence,
        "reason": validation.reason,
        "drop_candidate": validation.drop_candidate,
        "check": metadata.get("check"),
        "model_alias": metadata.get("model_alias"),
    }


def _compact_candidate(candidate: Any) -> dict[str, Any]:
    raw = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate or {})
    return {
        "title": raw.get("title") or "",
        "url": raw.get("url") or "",
    }


def _entity_filter_status(
    *,
    builder: ImageDiscoveryBuilder,
    entity: dict[str, Any],
    blocked_query_entities: set[str],
) -> str:
    if not builder._should_expand_entity(entity):
        return "filtered_out"
    if builder._is_query_implied_entity(entity, blocked_query_entities):
        return "filtered_by_query_entity_overlap"
    return "kept"


def _resolver_candidates_for_entity(
    *,
    builder: ImageDiscoveryBuilder,
    entity: dict[str, Any],
    source_node_title: str,
    image_caption: str | None,
    wiki_limit: int,
    filter_status: str,
) -> dict[str, Any]:
    resolver = builder.wiki_resolver
    label = (entity.get("name") or "").strip()
    entity_type = entity.get("type")
    relation = entity.get("relation_to_image")
    context_parts = [part for part in (entity.get("evidence"), image_caption, source_node_title) if part]
    context = " ".join(context_parts)

    item: dict[str, Any] = {
        "name": label,
        "type": entity_type,
        "relation_to_image": relation,
        "filter_status": filter_status,
        "wiki_queries": [],
        "per_query_candidates": [],
        "merged_ranked_candidates": [],
    }
    if not label:
        item["reason"] = "empty_entity_name"
        return item

    queries = resolver._build_queries(
        label,
        entity_type=entity_type,
        source_title=source_node_title,
        context=context,
    )
    item["wiki_queries"] = queries

    errors: list[dict[str, Any]] = []
    for query in queries:
        try:
            candidates = resolver._search(query, limit=wiki_limit)
            item["per_query_candidates"].append(
                {
                    "query": query,
                    "candidate_count": len(candidates),
                    "candidates": [_compact_candidate(candidate) for candidate in candidates],
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
        merged = resolver.search_candidates(
            label,
            entity_type=entity_type,
            source_title=source_node_title,
            context=context,
            limit=wiki_limit,
        )
        item["merged_ranked_candidates"] = [_compact_candidate(candidate) for candidate in merged]
    except Exception as exc:
        errors.append(
            {
                "query": "<merged_search>",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )

    if errors:
        item["errors"] = errors
    return item


def _emit_output(payload: dict[str, Any], *, pretty: bool) -> int:
    text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
    try:
        print(text)
    except BrokenPipeError:
        return 141
    return 0


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))

    builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=not bool(args.image_path),
            try_source_page_recovery=False,
            image_grounding_reader_base_url=args.reader_base_url,
        ),
        image_check_model_alias=os.environ.get("IMAGE_CHECK_MODEL"),
    )

    search_result = _build_search_result(args)
    plan = _build_plan(args, search_result)
    image_node = ImageNode.from_url(
        search_result.image_url or "",
        source_page_url=search_result.source_page_url,
        title=search_result.title,
        caption=search_result.snippet or search_result.title,
        run_id="debug_image_grounding",
        metadata={"debug_image_grounding": True},
    )

    local_image_path = Path(args.image_path).expanduser().resolve() if args.image_path else None

    if args.skip_check:
        validation = _force_accept_validation(builder, search_result, local_image_path)
        validation_elapsed_s = 0.0
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
        run_id="debug_image_grounding",
    )
    grounding_elapsed_s = time.perf_counter() - grounding_started_at

    grounded_entities = list(grounding.get("grounded_entities") or [])
    source_node_title = _source_node_title(args)
    query_text = _query_text(args)
    blocked_query_entities = builder._query_implied_entity_labels(
        query_text,
        source_node_title=source_node_title,
        grounded_entities=grounded_entities,
    )

    grounding_entities_summary: list[dict[str, Any]] = []
    resolver_candidates: list[dict[str, Any]] = []
    for entity in grounded_entities:
        filter_status = _entity_filter_status(
            builder=builder,
            entity=entity,
            blocked_query_entities=blocked_query_entities,
        )
        grounding_entities_summary.append(
            {
                "name": entity.get("name"),
                "type": entity.get("type"),
                "relation_to_image": entity.get("relation_to_image"),
                "filter_status": filter_status,
            }
        )
        resolver_candidates.append(
            _resolver_candidates_for_entity(
                builder=builder,
                entity=entity,
                source_node_title=source_node_title,
                image_caption=image_node.caption,
                wiki_limit=max(1, min(int(args.wiki_limit), 10)),
                filter_status=filter_status,
            )
        )

    output = {
        "image": {
            "image_url": search_result.image_url,
            "title": search_result.title,
            "snippet": search_result.snippet,
            "source_page_url": search_result.source_page_url,
        },
        "query": {
            "query_text": query_text,
            "source_node_title": source_node_title,
            "target_text": (args.target_text or args.snippet or args.title or "").strip() or None,
        },
        "validation": _compact_validation(validation),
        "grounding": {
            "check": grounding.get("check"),
            "model_alias": grounding.get("model_alias"),
            "caption": grounding.get("caption"),
            "grounded_entity_count": len(grounded_entities),
            "blocked_query_entities": sorted(blocked_query_entities),
            "grounded_entities": grounding_entities_summary,
        },
        "wiki_resolution_candidates": resolver_candidates,
        "timing": {
            "image_check_s": validation_elapsed_s,
            "image_ground_s": grounding_elapsed_s,
            "total_s": validation_elapsed_s + grounding_elapsed_s,
        },
    }
    return _emit_output(output, pretty=args.pretty)


if __name__ == "__main__":
    raise SystemExit(main())
