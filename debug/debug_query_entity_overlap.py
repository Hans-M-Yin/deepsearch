"""Inspect query-overlap decisions for a list of grounded entity names.

Examples:
  python debug/debug_query_entity_overlap.py \
    --query 'The cover of the album "Zoomer" by Schneider TM' \
    --source-node-title 'Schneider TM' \
    --entity 'Dirk Dresselhaus' \
    --entity 'Schneider TM' \
    --entity 'Zoomer (album)'

  # Reproduce the lexical fallback against the text nodes in an existing graph.
  python debug/debug_query_entity_overlap.py \
    --graph-dir runs/example \
    --query '...' \
    --source-node-title '...' \
    --entity '...' \
    --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.store import JsonlGraphStore


class _UnusedSearchClient:
    def search_text(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This debug tool only evaluates query overlap.")

    def search_image(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This debug tool only evaluates query overlap.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show LLM, lexical fallback, and final query-overlap decisions."
    )
    parser.add_argument("--query", required=True, help="The text-to-image search query.")
    parser.add_argument(
        "--entity",
        action="append",
        required=True,
        help="Grounded entity name. Repeat this flag for each entity.",
    )
    parser.add_argument(
        "--source-node-title",
        default="",
        help="Optional source text-node title supplied to the overlap filter.",
    )
    parser.add_argument(
        "--graph-dir",
        help=(
            "Use the graph's existing text-node titles and aliases for lexical fallback. "
            "Without this option, supplied entity names are used as temporary text-node labels."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="Synthesis env file used to configure the optional LLM filter.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM filter and show the lexical fallback only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def _temporary_store(source_title: str, entity_names: list[str]) -> JsonlGraphStore:
    root = Path(tempfile.mkdtemp(prefix="query_overlap_debug_"))
    store = JsonlGraphStore(root)
    labels = [source_title, *entity_names]
    for index, label in enumerate(dict.fromkeys(item.strip() for item in labels if item.strip())):
        store.upsert_node(
            {
                "node_id": f"debug_text_{index}",
                "node_type": "text",
                "title": label,
            }
        )
    return store


def _model_alias(builder: ImageDiscoveryBuilder) -> str | None:
    return (
        os.environ.get("IMAGE_QUERY_ENTITY_FILTER_MODEL")
        or os.environ.get("IMAGE_GROUND_MODEL")
        or builder.image_check_model_alias
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(Path(args.env_file))
    entity_names = [str(name).strip() for name in args.entity if str(name).strip()]
    source_title = str(args.source_node_title or "").strip()
    store = JsonlGraphStore(args.graph_dir) if args.graph_dir else _temporary_store(source_title, entity_names)
    builder = ImageDiscoveryBuilder(
        store=store,
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(),
        image_check_model_alias=os.environ.get("IMAGE_CHECK_MODEL"),
    )
    entities = [{"name": name} for name in entity_names]

    # Passing no candidates intentionally bypasses the LLM branch, yielding the
    # exact lexical fallback used when the LLM produces no blocked entities.
    fallback_blocked = builder._query_implied_entity_labels(
        args.query,
        source_node_title=source_title,
        grounded_entities=[],
    )
    model_alias = _model_alias(builder)
    llm_blocked: set[str] = set()
    if not args.no_llm and model_alias:
        llm_blocked = builder._query_implied_entity_labels_with_llm(
            args.query,
            source_node_title=source_title,
            grounded_entities=entities,
        )

    # This is the production selection rule: use the LLM set only if it blocks
    # at least one candidate; otherwise use the lexical fallback set.
    final_blocked = llm_blocked if llm_blocked else fallback_blocked
    decision_source = "llm" if llm_blocked else "lexical_fallback"
    results = []
    for name in entity_names:
        normalized = builder._normalize_entity_label(name)
        results.append(
            {
                "name": name,
                "normalized_name": normalized,
                "llm_blocked": normalized in llm_blocked,
                "lexical_fallback_blocked": normalized in fallback_blocked,
                "query_overlap_entity": normalized in final_blocked,
            }
        )
    return {
        "query": args.query,
        "source_node_title": source_title or None,
        "graph_dir": str(args.graph_dir) if args.graph_dir else None,
        "lexical_label_source": "graph text nodes and aliases" if args.graph_dir else "temporary labels from source title and supplied entities",
        "llm_requested": not args.no_llm,
        "llm_model_alias": model_alias,
        "llm_blocked_labels": sorted(llm_blocked),
        "lexical_fallback_blocked_labels": sorted(fallback_blocked),
        "final_decision_source": decision_source,
        "final_blocked_labels": sorted(final_blocked),
        "entities": results,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Query: {report['query']}")
    print(f"Source title: {report['source_node_title'] or '<none>'}")
    print(f"LLM requested: {report['llm_requested']}; model: {report['llm_model_alias'] or '<none>'}")
    print(f"LLM blocked: {report['llm_blocked_labels']}")
    print(f"Lexical fallback blocked: {report['lexical_fallback_blocked_labels']}")
    print(f"Final decision source: {report['final_decision_source']}")
    print("Entities:")
    for entity in report["entities"]:
        print(
            f"  - {entity['name']!r}: overlap={entity['query_overlap_entity']} "
            f"(llm={entity['llm_blocked']}, fallback={entity['lexical_fallback_blocked']})"
        )


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
