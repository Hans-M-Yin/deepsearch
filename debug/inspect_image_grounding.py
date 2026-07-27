"""Print the persisted image-grounding request and response for one image node.

Example:
  python debug/inspect_image_grounding.py \
    --graph-dir synthesis/runs/my_graph \
    --node-id image_0123456789abcdef \
    --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory containing nodes.jsonl.")
    parser.add_argument("--node-id", required=True, help="Image node ID whose persisted grounding call to inspect.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _image_input_references(node: dict[str, Any], resolved_image: dict[str, Any]) -> dict[str, Any]:
    """Return persisted references for rebuilding the image portion of the request."""
    return {
        "image_url": node.get("image_url"),
        "oss_uri": node.get("oss_uri"),
        "thumb_oss_uri": node.get("thumb_oss_uri"),
        "source_page_url": node.get("source_page_url"),
        "resolved_image": {
            key: resolved_image.get(key)
            for key in (
                "original_url",
                "resolved_url",
                "asset_uri",
                "cache_path",
                "content_type",
                "width",
                "height",
                "strategy",
            )
            if resolved_image.get(key) is not None
        },
    }


def build_report(graph_dir: Path, node_id: str) -> dict[str, Any]:
    graph_dir = graph_dir.expanduser().resolve()
    store = JsonlGraphStore(graph_dir)
    node = store.get_node(node_id)
    if node is None:
        raise ValueError(f"Image node not found: {node_id}")
    if node.get("node_type") != "image":
        raise ValueError(f"Node {node_id!r} has type {node.get('node_type')!r}, not 'image'.")

    metadata = node.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    grounding = metadata.get("image_grounding") or {}
    if not isinstance(grounding, dict):
        grounding = {}
    legacy_prompt = metadata.get("image_grounding_prompt") or {}
    if not isinstance(legacy_prompt, dict):
        legacy_prompt = {}
    grounding_context = grounding.get("context") or metadata.get("image_grounding_context") or {}
    if not isinstance(grounding_context, dict):
        grounding_context = {}
    resolved_image = metadata.get("resolved_image") or {}
    if not isinstance(resolved_image, dict):
        resolved_image = {}

    system_prompt = grounding.get("debug_prompt_system") or legacy_prompt.get("system")
    user_text = grounding.get("debug_prompt_user_text") or legacy_prompt.get("user_text")
    raw_output = grounding.get("raw_model_output")
    persisted = bool(system_prompt or user_text or raw_output or grounding)

    return {
        "graph_dir": str(graph_dir),
        "node": {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "caption": node.get("caption"),
        },
        "grounding_request": {
            "system_prompt": system_prompt,
            "user_text": user_text,
            "image_input_references": _image_input_references(node, resolved_image),
            "grounding_context": grounding_context,
            "note": (
                "The runtime data:image/... base64 URL is not persisted. Use one of the image "
                "references above to reconstruct the image attachment."
            ),
        },
        "grounding_response": {
            "persisted": persisted,
            "check": grounding.get("check"),
            "model_alias": grounding.get("model_alias"),
            "usage": grounding.get("usage"),
            "raw_model_output": raw_output,
            "grounded_entities": metadata.get("grounded_entities") or [],
            "run_id": grounding.get("run_id"),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args.graph_dir, args.node_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
