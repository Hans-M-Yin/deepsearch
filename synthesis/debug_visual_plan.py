"""Inspect visual planning output for one Wikipedia URL.

Run from the repository root, for example:

    python synthesis/debug_visual_plan.py \
      --url https://en.wikipedia.org/wiki/Kobe_Bryant
"""

from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.model_worker import LLM_WORKER
from synthesis.run_min_graph import DEFAULT_ENV_PATH, check_reader_service, load_env_file
from synthesis.visual_planner import LLMVisualSearchPlanner
from synthesis.wiki_text_builder import EnhancedReaderClient, WikiTextBuilder


def _short(text: str | None, limit: int) -> str:
    raw = " ".join((text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _plan_record(node: dict[str, Any], plan: Any) -> dict[str, Any]:
    target = plan.target
    return {
        "node_id": node.get("node_id"),
        "node_title": node.get("title") or node.get("canonical_id"),
        "node_source_url": (node.get("source") or {}).get("url"),
        "plan_id": plan.plan_id,
        "target_description": target.content,
        "target_type": target.metadata.get("target_type"),
        "downstream_use": target.metadata.get("downstream_use"),
        "source_passage": target.metadata.get("source_passage"),
        "source_quote": target.metadata.get("source_quote"),
        "uniqueness": target.metadata.get("uniqueness"),
        "reason": target.metadata.get("reason"),
        "plan_judge_reason": target.metadata.get("plan_judge_reason"),
        "expected_visual": target.metadata.get("expected_visual") or target.metadata.get("query"),
        "queries": [query.query for query in plan.queries],
        "query_specs": [query.to_dict() for query in plan.queries],
        "target": target.to_dict(),
        "planner": plan.planner,
        "metadata": plan.metadata,
    }


def _print_plan_summary(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No visual plans produced.")
        return
    for index, record in enumerate(records, start=1):
        print(f"\n=== Plan {index} ===")
        print(f"target: {record.get('target_description') or ''}")
        print(f"type: {record.get('target_type') or '-'}")
        print(f"use: {record.get('downstream_use') or '-'}")
        print(f"uniqueness: {record.get('uniqueness') or '-'}")
        print(f"reason: {record.get('reason') or '-'}")
        print(f"plan_judge_reason: {record.get('plan_judge_reason') or '-'}")
        print(f"source_passage: {_short(record.get('source_passage') or '', 300)}")
        print(f"source_quote: {_short(record.get('source_quote') or '', 300)}")
        print(f"expected_visual: {record.get('expected_visual') or '-'}")
        queries = list(record.get("queries") or [])
        if not queries:
            print("queries: -")
            continue
        print("queries:")
        for query_index, query in enumerate(queries, start=1):
            print(f"  {query_index}. {query}")


def _print_raw_planner_trace(trace: dict[str, Any]) -> None:
    print("\n=== Raw Visual Planner Output ===")
    raw_output = trace.get("raw_model_output")
    print(raw_output or "(no planner call)")
    print("\n=== Parsed Raw Plans ===")
    for index, candidate in enumerate(trace.get("raw_candidates") or [], start=1):
        print(f"{index}. query: {candidate.get('query') or ''}")
        print(f"   reason: {candidate.get('reason') or ''}")

    print("\n=== Plan Uniqueness Judge ===")
    for index, candidate in enumerate(trace.get("judge_results") or [], start=1):
        print(f"{index}. keep: {'yes' if candidate.get('plan_judge_keep') else 'no'}")
        print(f"   query: {candidate.get('query') or ''}")
        print(f"   reason: {candidate.get('plan_judge_reason') or ''}")
        print(f"   raw_output: {_short(candidate.get('plan_judge_raw_output'), 800)}")


def _print_candidate_list(title: str, candidates: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not candidates:
        print("(none)")
        return
    for index, candidate in enumerate(candidates, start=1):
        search_result = candidate.get("search_result") or {}
        validation = candidate.get("validation") or {}
        print(f"{index}. title: {search_result.get('title') or ''}")
        print(f"   image_url: {search_result.get('image_url') or ''}")
        print(f"   status: {validation.get('status') or ''}")
        print(f"   confidence: {validation.get('confidence')}")
        print(f"   reason: {validation.get('reason') or ''}")


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _print_search_snapshots(snapshots: list[Any]) -> None:
    print("\n=== Serper Search Snapshots ===")
    if not snapshots:
        print("(none)")
        return
    for index, snapshot in enumerate(snapshots, start=1):
        metadata = getattr(snapshot, "metadata", {}) or {}
        print(f"{index}. engine: {_enum_value(getattr(snapshot, 'engine', ''))}")
        print(f"   raw_engine: {metadata.get('raw_engine') or '-'}")
        print(f"   query: {getattr(snapshot, 'query', '') or ''}")
        print(f"   status: {_enum_value(getattr(snapshot, 'status', ''))}")
        print(f"   status_code: {getattr(snapshot, 'status_code', None)}")
        print(f"   result_count: {getattr(snapshot, 'result_count', None)}")
        print(f"   error: {getattr(snapshot, 'error', None) or ''}")
        print(f"   response_preview: {_short(getattr(snapshot, 'response_preview', None), 1200)}")


def _print_candidate_decisions(decisions: list[dict[str, Any]]) -> None:
    print("\n=== Search Decision Log ===")
    if not decisions:
        print("(none)")
        return
    for index, decision in enumerate(decisions, start=1):
        print(f"{index}. kind: {decision.get('kind') or ''}")
        print(f"   query: {decision.get('query') or ''}")
        if "returned" in decision:
            print(f"   returned: {decision.get('returned')}")
        if "fallback_used" in decision:
            print(f"   fallback_used: {decision.get('fallback_used')}")
        if "reason" in decision:
            print(f"   reason: {decision.get('reason') or ''}")
        if "status" in decision:
            print(f"   status: {decision.get('status') or ''}")
        if "result_index" in decision:
            print(f"   result_index: {decision.get('result_index')}")
        if "rank" in decision:
            print(f"   rank: {decision.get('rank')}")
        if "title" in decision:
            print(f"   title: {decision.get('title') or ''}")
        if "url" in decision:
            print(f"   url: {decision.get('url') or ''}")
        if "check" in decision:
            print(f"   check: {decision.get('check') or ''}")
        if "raw_model_output" in decision:
            print(f"   raw_model_output: {_short(decision.get('raw_model_output'), 800)}")


def _print_image_result(plan_index: int, plan: Any, result: Any) -> None:
    print(f"\n{'=' * 16} Image Pipeline: Plan {plan_index} {'=' * 16}")
    print(f"query: {plan.target.content or ''}")
    metadata = result.metadata or {}
    _print_search_snapshots(list(result.snapshots or []))
    _print_candidate_decisions(list(metadata.get("candidate_decisions") or []))
    _print_candidate_list(
        "Serper Results After Per-Image Content Check",
        list(metadata.get("content_checked_candidates") or []),
    )

    consistency = metadata.get("retrieval_consistency") or {}
    print("\n=== Retrieval Consistency Judge ===")
    print(f"decision: {consistency.get('decision') or ''}")
    print(f"candidate_count: {consistency.get('candidate_count')}")
    print(f"required_consistent_count: {consistency.get('required_consistent_count')}")
    print(f"consistent_indexes: {consistency.get('consistent_indexes') or []}")
    print(f"reason: {consistency.get('judge_reason') or ''}")
    print(f"raw_output: {_short(consistency.get('raw_model_output'), 1200)}")

    _print_candidate_list(
        "Candidates After Retrieval Consistency Check",
        [candidate.to_dict() for candidate in result.candidates],
    )

    print("\n=== Final Grounding And Link/Expansion Decisions ===")
    if result.image_node is None:
        print("image_node: not created")
        return
    node = result.image_node
    node_metadata = node.metadata or {}
    print(f"image_node_id: {node.node_id}")
    print(f"image_url: {node.image_url or ''}")
    print(f"caption: {node.caption or ''}")
    print("grounded_entities:")
    print(json.dumps(node_metadata.get("grounded_entities") or [], ensure_ascii=False, indent=2))
    print("linked_text_edges:")
    print(json.dumps([edge.to_dict() for edge in result.grounded_edges], ensure_ascii=False, indent=2))
    print("queued_text_expansions:")
    print(json.dumps(result.queued_tasks, ensure_ascii=False, indent=2))
    print("unresolved_grounded_entities:")
    print(json.dumps(node_metadata.get("unresolved_grounded_entities") or [], ensure_ascii=False, indent=2))
    print("query_overlap_grounded_entities:")
    print(json.dumps(node_metadata.get("query_overlap_grounded_entities") or [], ensure_ascii=False, indent=2))


def _build_image_search_client(name: str) -> Any:
    from synthesis.search_client import (
        CommonsImageSearchClient,
        CommonsSerpApiSearchClient,
        OpenSerpSearchClient,
        SerperAdapterSearchClient,
        SerperSearchClient,
        SerpApiSearchClient,
    )

    builders = {
        "commons": CommonsImageSearchClient,
        "commons_serpapi": CommonsSerpApiSearchClient,
        "serpapi": SerpApiSearchClient,
        "serper": SerperSearchClient,
        "openserp": OpenSerpSearchClient,
        "serper_adapter": SerperAdapterSearchClient,
    }
    return builders[name]()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Wikipedia page URL to inspect.")
    parser.add_argument("--title", default="", help="Optional page title override.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--reader-check-timeout", type=float, default=60.0, help="Enhanced Reader preflight timeout in seconds.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip Enhanced Reader preflight check.")
    parser.add_argument("--max-links", type=int, default=60, help="Maximum Wikipedia links extracted while building the text node.")
    parser.add_argument("--max-targets", type=int, default=5, help="Maximum visual targets proposed by the planner.")
    parser.add_argument("--max-queries-per-target", type=int, default=4, help="Maximum image queries per target.")
    parser.add_argument("--max-content-chars", type=int, default=12000, help="Maximum text chars passed to the planner. <=0 disables truncation.")
    parser.add_argument("--model-alias", default=None, help="Optional model alias overriding VISUAL_PLANNER_MODEL.")
    parser.add_argument("--plans-only", action="store_true", help="Stop after raw planning and plan-level uniqueness filtering.")
    parser.add_argument(
        "--image-backend",
        choices=("commons", "commons_serpapi", "serpapi", "serper", "openserp", "serper_adapter"),
        default="serper",
        help="Image search backend used by the full debug pipeline.",
    )
    parser.add_argument("--per-query-image-limit", type=int, default=6, help="Image results requested for each kept visual plan.")
    parser.add_argument("--max-images-per-plan", type=int, default=6, help="Maximum image candidates passed into consistency filtering.")
    parser.add_argument("--run-id", default="debug_visual_plan", help="Run id recorded in generated evidence.")
    parser.add_argument("--pretty", action="store_true", help="Also print the full JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file).expanduser().resolve()
    loaded_env = load_env_file(env_path, override=args.override_env)

    if not args.skip_reader_check:
        ok, message = check_reader_service(
            args.reader_base_url,
            test_url=args.url,
            timeout_s=args.reader_check_timeout,
        )
        if not ok:
            print(f"[preflight] Enhanced Reader unavailable at {args.reader_base_url}: {message}", file=sys.stderr)
            return 2

    from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
    from synthesis.store import JsonlGraphStore

    with tempfile.TemporaryDirectory(prefix="debug_visual_plan_") as tmpdir:
        temp_root = Path(tmpdir)
        store = JsonlGraphStore(temp_root / "graph")
        reader = EnhancedReaderClient(base_url=args.reader_base_url)
        wiki_builder = WikiTextBuilder(
            reader=reader,
            store=store,
            model_client=LLM_WORKER,
            max_links=args.max_links,
        )
        visual_planner = LLMVisualSearchPlanner(
            model_client=LLM_WORKER,
            model_alias=args.model_alias,
            max_targets=args.max_targets,
            max_queries_per_target=args.max_queries_per_target,
            min_content_chars_for_images=min(
                1000,
                args.max_content_chars if args.max_content_chars and args.max_content_chars > 0 else 1000,
            ),
        )

        text_result = wiki_builder.build_from_url(
            args.url,
            title=args.title or None,
            run_id=args.run_id,
            persist=False,
        )
        store.upsert_node(text_result.node)
        page_text = text_result.node.description or text_result.node.summary or ""
        if args.max_content_chars and args.max_content_chars > 0:
            page_text = page_text[: args.max_content_chars]

        plans = visual_planner.plan(
            node=text_result.node.to_dict(),
            page_text=page_text,
            source_evidence_ids=[text_result.text_evidence.evidence_id],
            run_id=args.run_id,
        )
        records = [_plan_record(text_result.node.to_dict(), plan) for plan in plans]

        print("=== debug visual plan pipeline ===")
        print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
        print(f"url: {args.url}")
        print(f"title: {text_result.node.title or ''}")
        print(f"node_id: {text_result.node.node_id}")
        print(f"content_chars: {len(text_result.node.description or '')}")
        print(f"planner_input_chars: {len(page_text)}")
        _print_raw_planner_trace(visual_planner.last_plan_trace)
        print(f"\n=== Plans Kept After LLM Filtering ({len(records)}) ===")
        _print_plan_summary(records)

        image_results: list[Any] = []
        if not args.plans_only:
            image_builder = ImageDiscoveryBuilder(
                store=store,
                search_client=_build_image_search_client(args.image_backend),
                config=ImageDiscoveryConfig(
                    per_query_limit=args.per_query_image_limit,
                    max_images_per_plan=args.max_images_per_plan,
                    cache_dir=str(temp_root / "image_cache"),
                    upload_cached_images=False,
                    image_grounding_reader_base_url=args.reader_base_url,
                ),
                model_client=LLM_WORKER,
            )
            for index, plan in enumerate(plans, start=1):
                try:
                    image_result = image_builder.discover_for_plan(
                        plan,
                        run_id=args.run_id,
                        persist=False,
                    )
                except Exception as exc:
                    print(f"\n=== Image Pipeline: Plan {index} Failed ===")
                    print(f"{exc.__class__.__name__}: {exc}")
                    continue
                image_results.append(image_result)
                _print_image_result(index, plan, image_result)

        if args.pretty:
            payload = {
                "url": args.url,
                "title": text_result.node.title,
                "node_id": text_result.node.node_id,
                "planner_input_chars": len(page_text),
                "planner_trace": visual_planner.last_plan_trace,
                "plans": records,
                "image_results": [result.to_dict() for result in image_results],
            }
            print("\n=== json ===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
