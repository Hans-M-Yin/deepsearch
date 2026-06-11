"""Run a minimal text -> image -> text backlink test without full graph BFS.

This script is designed to answer one narrow question:
starting from a single seed text node, can the image-expansion path produce
image-entity text tasks that successfully materialize image->text edges?

Flow:
1. Build and persist exactly one seed text node.
2. Expand only images from that seed node.
3. Execute the queued image-entity text tasks in dry-run mode (no persistence).
4. Report which tasks would materialize image->text edges back to the image node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.model_worker import LLM_WORKER
from synthesis.run_min_graph import DEFAULT_ENV_PATH, check_reader_service, load_env_file
from synthesis.search_client import CommonsImageSearchClient, CommonsSerpApiSearchClient, SerpApiSearchClient
from synthesis.store import JsonlGraphStore
from synthesis.visual_planner import LLMVisualSearchPlanner
from synthesis.wiki_text_builder import EnhancedReaderClient, WikiTextBuilder
from synthesis.graph_expansion import GraphExpansionConfig, GraphExpansionStrategy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-url", required=True, help="Seed Wikipedia URL.")
    parser.add_argument("--seed-title", default="", help="Optional seed page title.")
    parser.add_argument("--store-dir", required=True, help="Output graph directory.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing environment variables.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--reader-check-timeout", type=float, default=60.0, help="Enhanced Reader preflight timeout in seconds.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip preflight reader reachability check.")
    parser.add_argument(
        "--image-backend",
        choices=("commons", "commons_serpapi", "serpapi"),
        default="commons_serpapi",
        help="Image search backend for the minimal test.",
    )
    parser.add_argument("--per-query-image-limit", type=int, default=3, help="Image search results per visual query.")
    parser.add_argument("--max-images-per-plan", type=int, default=3, help="Accepted images per visual plan.")
    parser.add_argument("--image-budget-chars", type=int, default=8000, help="Visual planner text budget.")
    parser.add_argument("--force-accept-images", action="store_true", help="Bypass semantic image rejection for debugging.")
    parser.add_argument("--max-links", type=int, default=20, help="Wiki links extracted for the seed page.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print final JSON.")
    return parser


def _has_serpapi_credentials() -> bool:
    import os

    return bool(
        os.environ.get("SERPAPI_AK")
        or os.environ.get("AIDP_SERP_AK")
        or os.environ.get("SERPAPI_API_KEY")
        or os.environ.get("SERP_API_KEY")
    )


def _build_backend(name: str):
    if name == "commons":
        return CommonsImageSearchClient()
    if name == "commons_serpapi":
        if not _has_serpapi_credentials():
            raise ValueError("Image backend 'commons_serpapi' requires SerpApi credentials.")
        return CommonsSerpApiSearchClient()
    if name == "serpapi":
        if not _has_serpapi_credentials():
            raise ValueError("Image backend 'serpapi' requires SerpApi credentials.")
        return SerpApiSearchClient()
    raise ValueError(f"Unsupported image backend: {name}")


def _edge_summary(edge: Any) -> dict[str, Any]:
    payload = edge.to_dict()
    return {
        "edge_id": payload.get("edge_id"),
        "edge_type": payload.get("edge_type"),
        "src_node_id": payload.get("src_node_id"),
        "dst_node_id": payload.get("dst_node_id"),
        "relation": payload.get("relation"),
        "metadata": payload.get("metadata"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file).expanduser().resolve()
    load_env_file(env_path, override=args.override_env)

    if not args.skip_reader_check:
        ok, message = check_reader_service(
            args.reader_base_url,
            test_url=args.seed_url,
            timeout_s=args.reader_check_timeout,
        )
        if not ok:
            print(f"[preflight] Enhanced Reader unavailable at {args.reader_base_url}: {message}", file=sys.stderr)
            return 2

    store_dir = Path(args.store_dir).expanduser().resolve()
    store = JsonlGraphStore(store_dir)
    reader = EnhancedReaderClient(base_url=args.reader_base_url)

    wiki_builder = WikiTextBuilder(
        reader=reader,
        store=store,
        model_client=LLM_WORKER,
        max_links=args.max_links,
    )
    visual_planner = LLMVisualSearchPlanner(
        model_client=LLM_WORKER,
        target_chars_per_budget=args.image_budget_chars,
    )
    image_builder = ImageDiscoveryBuilder(
        store=store,
        search_client=_build_backend(args.image_backend),
        config=ImageDiscoveryConfig(
            per_query_limit=args.per_query_image_limit,
            max_images_per_plan=args.max_images_per_plan,
            force_accept_images=args.force_accept_images,
            image_grounding_reader_base_url=args.reader_base_url,
        ),
        model_client=LLM_WORKER,
    )

    persist_strategy = GraphExpansionStrategy(
        store=store,
        wiki_builder=wiki_builder,
        visual_planner=visual_planner,
        image_builder=image_builder,
        config=GraphExpansionConfig(
            max_depth=0,
            max_new_text_neighbors=0,
            extract_attributes=False,
            enable_image_expansion=True,
            persist=True,
        ),
    )

    dryrun_strategy = GraphExpansionStrategy(
        store=store,
        wiki_builder=wiki_builder,
        visual_planner=visual_planner,
        image_builder=image_builder,
        config=GraphExpansionConfig(
            max_depth=0,
            max_new_text_neighbors=0,
            extract_attributes=False,
            enable_image_expansion=False,
            persist=False,
        ),
    )

    seed_result = wiki_builder.build_from_url(
        args.seed_url,
        title=args.seed_title or None,
        run_id="test_min_image_backlink",
        persist=True,
    )
    plans, image_results, queued_tasks = persist_strategy._expand_images(
        seed_result,
        run_id="test_min_image_backlink",
    )

    dryrun_results: list[dict[str, Any]] = []
    for task in queued_tasks:
        result = dryrun_strategy.expand_task(
            task,
            run_id="test_min_image_backlink",
        )
        dryrun_results.append(
            {
                "task_url": task.url,
                "task_title": task.title,
                "error": result.error,
                "text_node_id": result.text_result.node.node_id if result.text_result else None,
                "text_node_title": result.text_result.node.title if result.text_result else None,
                "materialized_edge_count": len(result.materialized_edges),
                "materialized_edges": [_edge_summary(edge) for edge in result.materialized_edges],
                "parent_link_failures": [dict(item) for item in result.parent_link_failures],
            }
        )

    image_node_summaries: list[dict[str, Any]] = []
    queued_titles: list[str] = []
    for image_result in image_results:
        if image_result.image_node is not None:
            image_node_summaries.append(
                {
                    "node_id": image_result.image_node.node_id,
                    "title": image_result.image_node.title,
                    "caption": image_result.image_node.caption,
                    "image_url": image_result.image_node.image_url,
                }
            )
        for task in image_result.queued_tasks:
            queued_titles.append(task.get("title"))

    output = {
        "seed": {
            "url": args.seed_url,
            "title": seed_result.node.title,
            "node_id": seed_result.node.node_id,
        },
        "phase_1": {
            "text_node_persisted": True,
            "store_stats_after_seed": store.stats(),
        },
        "phase_2": {
            "visual_plan_count": len(plans),
            "image_result_count": len(image_results),
            "image_nodes": image_node_summaries,
            "queued_image_entity_task_count": len(queued_tasks),
            "queued_image_entity_task_titles": queued_titles,
        },
        "phase_3": {
            "dryrun_task_count": len(dryrun_results),
            "dryrun_results": dryrun_results,
        },
        "summary": {
            "image_nodes_created": len(image_node_summaries),
            "queued_image_entity_text_tasks": len(queued_tasks),
            "dryrun_successful_text_builds": sum(1 for item in dryrun_results if item.get("text_node_id")),
            "dryrun_materialized_edge_count": sum(int(item.get("materialized_edge_count") or 0) for item in dryrun_results),
            "dryrun_failed_tasks": sum(1 for item in dryrun_results if item.get("error")),
        },
    }

    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
