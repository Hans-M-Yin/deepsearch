"""Print sampled image-grounding audit prompts from a persisted graph.

The script is offline: it only reads ``nodes.jsonl`` and ``edges.jsonl`` from
``--graph-dir``. It does not call models, download images, or modify the graph.

Example:
  python debug/export_image_grounding_audit_prompts.py \
    --graph-dir runs/my_graph \
    --sample-num 100 \
    --seed 20260728
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


DEFAULT_SAMPLE_NUM = 100
IMAGE_DEPICTS = "image_depicts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory containing graph JSONL files.")
    parser.add_argument(
        "--sample-num",
        type=int,
        default=DEFAULT_SAMPLE_NUM,
        help=f"Number of eligible image nodes to sample; <=0 means all (default: {DEFAULT_SAMPLE_NUM}).",
    )
    parser.add_argument("--seed", type=int, default=20260728, help="Deterministic random seed for sampling.")
    parser.add_argument(
        "--include-inactive-edges",
        action="store_true",
        help="Include grounded image_depicts edges with non-active status. Default audits active edges only.",
    )
    return parser.parse_args()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _url(value: Any) -> str:
    return str(value or "").strip()


def _is_active(edge: dict[str, Any]) -> bool:
    return _text(edge.get("status") or "active").lower() == "active"


def _source_url(node: dict[str, Any]) -> str:
    metadata = _as_dict(node.get("metadata"))
    resolved = _as_dict(metadata.get("resolved_image"))
    source = _as_dict(node.get("source"))
    # The resolved URL is normally the most directly usable external image URL.
    # Keep alternatives in the prompt if it was stored only as a local asset path.
    return (
        _url(resolved.get("resolved_url"))
        or _url(node.get("image_url"))
        or _url(resolved.get("original_url"))
        or _url(source.get("url"))
        or _url(node.get("oss_uri"))
    )


def _image_sources(node: dict[str, Any]) -> dict[str, str]:
    metadata = _as_dict(node.get("metadata"))
    resolved = _as_dict(metadata.get("resolved_image"))
    return {
        "image_url": _url(node.get("image_url")),
        "resolved_image_url": _url(resolved.get("resolved_url")),
        "source_page_url": _url(node.get("source_page_url")),
        "original_image_url": _url(resolved.get("original_url")),
        "asset_uri": _url(resolved.get("asset_uri")) or _url(node.get("oss_uri")),
    }


def _edge_entity(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    metadata = _as_dict(edge.get("metadata"))
    grounded_entity = {}
    for ref in _as_list(edge.get("evidence_refs")):
        ref_metadata = _as_dict(_as_dict(ref).get("metadata"))
        candidate = _as_dict(ref_metadata.get("grounded_entity"))
        if candidate:
            grounded_entity = candidate
            break
    target = nodes_by_id.get(_url(edge.get("dst_node_id")), {})
    return {
        "edge_id": _url(edge.get("edge_id")),
        "entity_name": _text(metadata.get("entity_name") or grounded_entity.get("name") or target.get("title")),
        "entity_type": _text(metadata.get("entity_type") or grounded_entity.get("type")),
        "relation_to_image": _text(edge.get("relation") or grounded_entity.get("relation_to_image")),
        "grounding_evidence": _text(
            grounded_entity.get("evidence")
            or next(
                (
                    _as_dict(ref).get("quote")
                    for ref in _as_list(edge.get("evidence_refs"))
                    if _as_dict(ref).get("quote")
                ),
                "",
            )
        ),
        "target_text_node_id": _url(edge.get("dst_node_id")),
        "target_text_node_title": _text(target.get("title")),
        "edge_status": _text(edge.get("status") or "active"),
    }


def _eligible_image_records(
    store: JsonlGraphStore,
    *,
    include_inactive_edges: bool,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    nodes_by_id = {str(node.get("node_id") or ""): node for node in store.list_nodes()}
    entities_by_image_id: dict[str, list[dict[str, str]]] = {}
    for edge in store.list_edges():
        if edge.get("edge_type") != IMAGE_DEPICTS:
            continue
        if not include_inactive_edges and not _is_active(edge):
            continue
        image_node_id = _url(edge.get("src_node_id"))
        image_node = nodes_by_id.get(image_node_id)
        if not image_node or image_node.get("node_type") != "image":
            continue
        entity = _edge_entity(edge, nodes_by_id)
        # A valid sample should contain a named entity and a persisted edge.
        if not entity["entity_name"]:
            continue
        entities_by_image_id.setdefault(image_node_id, []).append(entity)

    records: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for image_node_id, entities in entities_by_image_id.items():
        image_node = nodes_by_id[image_node_id]
        entities.sort(key=lambda item: item["edge_id"])
        records.append((image_node, entities))
    return sorted(records, key=lambda item: str(item[0].get("node_id") or ""))


def _prompt(image_node: dict[str, Any], entities: list[dict[str, str]]) -> str:
    sources = _image_sources(image_node)
    primary_url = _source_url(image_node)
    metadata = _as_dict(image_node.get("metadata"))
    lines = [
        "You are auditing the correctness of an image-grounding system.",
        "",
        "Use the image URL below and conduct web research as needed. Search for the image itself, reverse-search it when possible, inspect its source page, and consult reliable sources about the depicted scene, people, objects, event, or work.",
        "",
        "Your task is to independently determine whether each claimed grounded entity is actually visible in, clearly represented by, or otherwise visually supported by this exact image at the stated locator. Do not accept a claim merely because the entity is mentioned in surrounding webpage text or is generally associated with the image topic.",
        "",
        "For every candidate, return one row in a Markdown table with exactly these columns:",
        "edge_id | entity | verdict (correct / incorrect / uncertain) | confidence (0-1) | visual assessment | web evidence and sources",
        "",
        "Evaluation rules:",
        "- correct: the entity is visibly supported by the image and the locator is materially accurate.",
        "- incorrect: the entity is absent, misidentified, or the locator points to the wrong person/object/scene.",
        "- uncertain: available image and web evidence do not support a confident conclusion.",
        "- Treat a text-only association, caption-only inference, or source-page claim without visual support as insufficient; use uncertain or incorrect as appropriate.",
        "- Cite concrete evidence: source titles, URLs, reverse-image-match information, captions, or visual comparison details. Explain the visual cue that supports your verdict.",
        "- Do not evaluate entities that are not listed below, and do not rewrite the claims before judging them.",
        "",
        "Image record:",
        f"- Image URL: {primary_url or '[missing; inspect alternative references below]'}",
        f"- Image title: {_text(image_node.get('title')) or '[missing]'}",
        f"- Image caption: {_text(image_node.get('caption')) or '[missing]'}",
        f"- Source page URL: {sources['source_page_url'] or '[missing]'}",
        f"- Search query: {_text(metadata.get('search_query')) or '[missing]'}",
        f"- Visual target: {_text(metadata.get('visual_target')) or '[missing]'}",
        "- Alternative image references:",
        f"  - persisted image_url: {sources['image_url'] or '[missing]'}",
        f"  - resolved image URL: {sources['resolved_image_url'] or '[missing]'}",
        f"  - original image URL: {sources['original_image_url'] or '[missing]'}",
        f"  - asset URI: {sources['asset_uri'] or '[missing]'}",
        "",
        "Grounding claims to audit:",
    ]
    for index, entity in enumerate(entities, start=1):
        lines.extend(
            [
                f"{index}. edge_id: {entity['edge_id']}",
                f"   entity: {entity['entity_name']}",
                f"   entity_type: {entity['entity_type'] or '[not recorded]'}",
                f"   locator / position in image: {entity['relation_to_image'] or '[not recorded]'}",
                f"   grounding evidence from system: {entity['grounding_evidence'] or '[not recorded]'}",
                f"   linked text node: {entity['target_text_node_id']} | {entity['target_text_node_title'] or '[not recorded]'}",
            ]
        )
    lines.extend(
        [
            "",
            "After the table, provide:",
            "1. a one-sentence overall assessment of this image node's grounding quality; and",
            "2. counts for correct, incorrect, and uncertain claims.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    graph_dir = args.graph_dir.expanduser().resolve()
    if not graph_dir.is_dir():
        raise SystemExit(f"error: graph directory does not exist: {graph_dir}")

    store = JsonlGraphStore(graph_dir)
    records = _eligible_image_records(store, include_inactive_edges=bool(args.include_inactive_edges))
    if not records:
        raise SystemExit(
            "error: no eligible image nodes with image_depicts grounding edges found. "
            "Use --include-inactive-edges if only soft-deleted grounding edges remain."
        )

    sample_num = int(args.sample_num)
    if sample_num <= 0 or sample_num >= len(records):
        selected = records
    else:
        selected = random.Random(args.seed).sample(records, sample_num)
        selected.sort(key=lambda item: str(item[0].get("node_id") or ""))

    print("Image Grounding External Audit Prompt Export")
    print(f"graph_dir: {graph_dir}")
    print(
        f"eligible_image_nodes: {len(records)} | sampled_image_nodes: {len(selected)} "
        f"| seed: {args.seed} | include_inactive_edges: {bool(args.include_inactive_edges)}"
    )
    print()

    for index, (image_node, entities) in enumerate(selected, start=1):
        edge_ids = [entity["edge_id"] for entity in entities]
        print("=" * 110)
        print(f"[{index}/{len(selected)}] image_node_id: {image_node.get('node_id')}")
        print(f"grounding_edge_ids (locator / position claims): {', '.join(edge_ids)}")
        print()
        print(_prompt(image_node, entities))
        print("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
