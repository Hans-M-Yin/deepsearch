"""Inspect visual planning output for one Wikipedia URL.

Run from the repository root, for example:

    python synthesis/debug_visual_plan.py \
      --url https://en.wikipedia.org/wiki/Kobe_Bryant
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--max-targets", type=int, default=7, help="Maximum visual targets proposed by the planner.")
    parser.add_argument("--max-queries-per-target", type=int, default=4, help="Maximum image queries per target.")
    parser.add_argument("--max-content-chars", type=int, default=12000, help="Maximum text chars passed to the planner. <=0 disables truncation.")
    parser.add_argument("--model-alias", default=None, help="Optional model alias overriding VISUAL_PLANNER_MODEL.")
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

    reader = EnhancedReaderClient(base_url=args.reader_base_url)
    wiki_builder = WikiTextBuilder(
        reader=reader,
        store=None,
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

    print("=== debug visual plan ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"url: {args.url}")
    print(f"title: {text_result.node.title or ''}")
    print(f"node_id: {text_result.node.node_id}")
    print(f"content_chars: {len(text_result.node.description or '')}")
    print(f"planner_input_chars: {len(page_text)}")
    print(f"plan_count: {len(records)}")
    _print_plan_summary(records)

    if args.pretty:
        payload = {
            "url": args.url,
            "title": text_result.node.title,
            "node_id": text_result.node.node_id,
            "planner_input_chars": len(page_text),
            "plans": records,
        }
        print("\n=== json ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
