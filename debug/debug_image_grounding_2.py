"""Extract persisted image-grounding inputs for one image node from a graph run.

Examples:
  python debug/debug_image_grounding_2.py \
    --graph-dir runs/0712_multi_seed_visual_test6 \
    --title "Image: The first Airbus A330neo being delivered to its launch customer, TAP Air Portugal, in 2018" \
    --pretty
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one image node's persisted image-grounding inputs and emit a "
            "best-effort debug/debug_image_grounding.py reproduction command."
        )
    )
    parser.add_argument("--graph-dir", required=True, help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--title", required=True, help="Exact image node title to inspect.")
    parser.add_argument(
        "--origin",
        default="visual_plan",
        help="Expected image origin. Default: visual_plan. Use empty string to disable origin filtering.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON payload.")
    return parser.parse_args()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _image_variant_sources(node: dict[str, Any]) -> list[str]:
    variants = node.get("image_variants") or []
    if not isinstance(variants, list):
        return []
    sources: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        source = str(variant.get("source") or "").strip()
        if source:
            sources.add(source)
    return sorted(sources)


def _image_origin(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    source = node.get("source") or {}
    source_type = source.get("source_type") if isinstance(source, dict) else None
    image_origin = str(metadata.get("image_origin") or "").strip().lower()
    variant_sources = _image_variant_sources(node)
    if source_type == "wikipedia_inline_image" or image_origin == "wikipedia_inline":
        return "wiki_inline"
    if "wikipedia_inline" in variant_sources:
        return "wiki_inline"
    if source_type in {"image_search_bundle", "image_search"}:
        return "visual_plan"
    if source_type:
        return f"other:{source_type}"
    return "other:unknown"


def _choose_image_input(node: dict[str, Any], resolved_image: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    cache_path = str(resolved_image.get("cache_path") or "").strip()
    if cache_path and Path(cache_path).exists():
        return "image_path", cache_path, "resolved_image.cache_path"

    for value, label in (
        (node.get("image_url"), "node.image_url"),
        (resolved_image.get("asset_uri"), "resolved_image.asset_uri"),
        (resolved_image.get("original_url"), "resolved_image.original_url"),
        (resolved_image.get("resolved_url"), "resolved_image.resolved_url"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        if text.startswith("file://"):
            parsed = urlparse(text)
            local_path = Path(unquote(parsed.path))
            if local_path.exists():
                return "image_path", str(local_path), label
        local_path = Path(text)
        if local_path.is_absolute() and local_path.exists():
            return "image_path", str(local_path), label
        if text.startswith(("http://", "https://")):
            return "image_url", text, label

    return None, None, None


def _find_parent_search_edges(edges: list[dict[str, Any]], image_node_id: str) -> list[dict[str, Any]]:
    matches = []
    for edge in edges:
        if edge.get("dst_node_id") != image_node_id:
            continue
        if edge.get("edge_type") != "search_retrieved":
            continue
        if edge.get("src_node_type") != "text":
            continue
        if edge.get("dst_node_type") != "image":
            continue
        matches.append(edge)
    matches.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("edge_id") or "")))
    return matches


def _pick_parent_edge(parent_edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not parent_edges:
        return None
    preferred = [edge for edge in parent_edges if (edge.get("metadata") or {}).get("query")]
    if preferred:
        return preferred[-1]
    return parent_edges[-1]


def _build_repro_command(
    *,
    image_arg_name: str | None,
    image_arg_value: str | None,
    title: str | None,
    snippet: str | None,
    source_page_url: str | None,
    target_text: str | None,
    query_text: str | None,
    source_node_title: str | None,
) -> str:
    parts = ["python", "debug/debug_image_grounding.py"]
    if image_arg_name and image_arg_value:
        parts.extend([f"--{image_arg_name.replace('_', '-')}", image_arg_value])
    if title:
        parts.extend(["--title", title])
    if snippet:
        parts.extend(["--snippet", snippet])
    if source_page_url:
        parts.extend(["--source-page-url", source_page_url])
    if target_text:
        parts.extend(["--target-text", target_text])
    if query_text:
        parts.extend(["--query-text", query_text])
    if source_node_title:
        parts.extend(["--source-node-title", source_node_title])
    parts.append("--pretty")
    return " ".join(shlex.quote(part) for part in parts)


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    if not graph_dir.exists():
        raise SystemExit(f"Graph directory does not exist: {graph_dir}")

    store = JsonlGraphStore(graph_dir)
    nodes = store.list_nodes()
    edges = store.list_edges()
    nodes_by_id = {str(node.get("node_id") or ""): node for node in nodes}

    wanted_title = _normalize_text(args.title)
    matches = []
    for node in nodes:
        if node.get("node_type") != "image":
            continue
        if _normalize_text(node.get("title")) != wanted_title:
            continue
        origin = _image_origin(node)
        if args.origin and origin != args.origin:
            continue
        matches.append(node)

    if not matches:
        raise SystemExit(
            "No image node matched the requested title/origin. "
            f"title={args.title!r} origin={args.origin!r}"
        )
    if len(matches) > 1:
        payload = {
            "error": "multiple_matching_image_nodes",
            "match_count": len(matches),
            "matches": [
                {
                    "node_id": item.get("node_id"),
                    "title": item.get("title"),
                    "origin": _image_origin(item),
                    "image_url": item.get("image_url"),
                }
                for item in matches
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    node = matches[0]
    metadata = node.get("metadata") or {}
    resolved_image = metadata.get("resolved_image") or {}
    image_grounding = metadata.get("image_grounding") or {}
    grounding_context = image_grounding.get("context") or metadata.get("image_grounding_context") or {}
    prompt = metadata.get("image_grounding_prompt") or {}

    parent_edges = _find_parent_search_edges(edges, str(node.get("node_id") or ""))
    parent_edge = _pick_parent_edge(parent_edges)
    parent_node = nodes_by_id.get(str(parent_edge.get("src_node_id") or "")) if parent_edge else None

    image_title = (
        (grounding_context.get("metadata") or {}).get("image_title")
        or node.get("title")
        or node.get("caption")
    )
    image_snippet = (
        (grounding_context.get("metadata") or {}).get("image_snippet")
        or node.get("caption")
        or node.get("summary")
    )
    source_page_url = (
        (grounding_context.get("metadata") or {}).get("source_page_url")
        or node.get("source_page_url")
        or resolved_image.get("source_page_url")
    )
    source_node_title = (parent_node or {}).get("title") or metadata.get("source_text_node_id")
    query_text = (
        ((parent_edge or {}).get("metadata") or {}).get("query")
        or metadata.get("search_query")
    )
    target_text = metadata.get("visual_target")

    image_arg_name, image_arg_value, image_arg_source = _choose_image_input(node, resolved_image)
    repro_command = _build_repro_command(
        image_arg_name=image_arg_name,
        image_arg_value=image_arg_value,
        title=image_title,
        snippet=image_snippet,
        source_page_url=source_page_url,
        target_text=target_text,
        query_text=query_text,
        source_node_title=source_node_title,
    )

    payload = {
        "graph_dir": str(graph_dir),
        "match": {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "origin": _image_origin(node),
            "image_url": node.get("image_url"),
            "source_page_url": node.get("source_page_url"),
            "caption": node.get("caption"),
        },
        "persisted_grounding_inputs": {
            "resolved_image": resolved_image,
            "image_grounding_context": grounding_context,
            "image_grounding_prompt": prompt,
            "image_grounding": image_grounding,
            "grounded_entities": metadata.get("grounded_entities") or [],
        },
        "parent_visual_plan_context": {
            "parent_edge": parent_edge,
            "parent_node": {
                "node_id": (parent_node or {}).get("node_id"),
                "title": (parent_node or {}).get("title"),
                "source": (parent_node or {}).get("source"),
            },
            "candidate_parent_edge_count": len(parent_edges),
        },
        "reproduction": {
            "image_argument": {
                "name": image_arg_name,
                "value": image_arg_value,
                "chosen_from": image_arg_source,
            },
            "title": image_title,
            "snippet": image_snippet,
            "source_page_url": source_page_url,
            "target_text": target_text,
            "query_text": query_text,
            "source_node_title": source_node_title,
            "debug_image_grounding_command": repro_command,
            "limitations": [
                "debug_image_grounding.py will rebuild grounding context from title/snippet/source_page_url; if the source page content has changed since the original run, reproduction will only be approximate.",
                "If the chosen image argument is a remote URL instead of a persisted cache_path, the fetched image may differ from the exact cached asset used in the original run.",
            ],
        },
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
