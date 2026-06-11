"""Debug how grounded image entities are filtered and expanded.

Examples:
  python -m synthesis.debug_image_entity_expansion \
    --graph-dir runs/kobe_text_only \
    --image-node-id image_123 \
    --source-query-text "Manchester United comeback celebration 1999 UEFA Champions League final Camp Nou against Bayern Munich" \
    --source-node-title "Camp Nou" \
    --grounded-entities-file /tmp/entities.json

  python -m synthesis.debug_image_entity_expansion \
    --graph-dir runs/kobe_text_only \
    --image-title "manual image" \
    --image-caption "players holding a silver trophy" \
    --image-url "https://example.com/test.jpg" \
    --source-query-text "Manchester United comeback celebration 1999 UEFA Champions League final Camp Nou against Bayern Munich" \
    --source-node-title "Camp Nou" \
    --grounded-entities-json '[{"name":"UEFA Champions League","relation_to_image":"the silver trophy being held by the players in the center","evidence":"The large silver cup in the middle is the UEFA Champions League trophy."}]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.evidence import Evidence, EvidenceType
from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.nodes import ImageNode
from synthesis.store import JsonlGraphStore


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
        caption=args.image_caption or None,
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
    parser.add_argument("--image-node-id", type=str, default="")
    parser.add_argument("--image-title", type=str, default="")
    parser.add_argument("--image-caption", type=str, default="")
    parser.add_argument("--image-url", type=str, default="")
    parser.add_argument("--source-page-url", type=str, default="")
    parser.add_argument("--source-query-text", type=str, required=True)
    parser.add_argument("--source-node-title", type=str, default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grounded-entities-json", type=str)
    group.add_argument("--grounded-entities-file", type=str)
    args = parser.parse_args()

    store = JsonlGraphStore(args.graph_dir)
    builder = ImageDiscoveryBuilder(
        store=store,
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(),
    )
    grounded_entities = _load_grounded_entities(args)
    image_node = _build_image_node(args, store)
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
        source_node_title=args.source_node_title,
    )
    edges, queued_tasks = builder._link_or_queue_grounded_entities(
        image_node=image_node,
        grounded_entities=grounded_entities,
        image_evidence=image_evidence,
        run_id="debug_image_entity_expansion",
        source_node_title=args.source_node_title or None,
        source_query_text=args.source_query_text,
    )

    print("image_node:")
    print(json.dumps({"node_id": image_node.node_id, "title": image_node.title, "caption": image_node.caption}, ensure_ascii=False, indent=2))
    print("blocked_query_entities:")
    print(json.dumps(sorted(blocked), ensure_ascii=False, indent=2))
    print("entity_statuses:")
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    print("created_edges:")
    print(json.dumps([edge.to_dict() for edge in edges], ensure_ascii=False, indent=2))
    print("queued_tasks:")
    print(json.dumps(queued_tasks, ensure_ascii=False, indent=2))
    print("image_node_metadata_after_link_or_queue:")
    print(json.dumps(image_node.metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
