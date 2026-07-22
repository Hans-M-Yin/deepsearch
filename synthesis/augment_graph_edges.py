"""Augment an existing synthesis graph with additional shortcut edges.

This script copies an existing graph store into a new output directory, then:

1. Completes directed text-text links by re-reading each existing Wikipedia
   text node and checking whether its markdown links to other text nodes already
   present in the graph. No synthetic reverse edges are created.
2. Completes local image->text links by resolving grounded image entities to
   Wikipedia URLs and linking them only when the corresponding text node is
   already present in the graph.

The augmented graph is written only to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.edges import Edge, EdgeSource, EdgeType, EvidenceRef
from synthesis.evidence import Evidence, EvidenceType
from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.model_worker import LLM_WORKER
from synthesis.nodes import NodeType
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.store import JsonlGraphStore
from synthesis.wiki_text_builder import ReaderDocument, WikiLinkCandidate, WikiTextBuilder


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


class RawMarkdownReaderClient:
    """Minimal client for the raw 8003 markdown reader endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8003",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def read(self, url: str, **kwargs: Any) -> ReaderDocument:
        del kwargs
        target = url if url.startswith(("http://", "https://")) else f"https://{url}"
        request = Request(
            f"{self.base_url}/{target}",
            headers={"Accept": "text/plain"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            markdown = response.read().decode("utf-8")
        return ReaderDocument(
            url=target,
            title=WikiTextBuilder._title_from_url(target),
            content=markdown,
            raw_markdown=markdown,
            raw={"reader": "raw_markdown_reader"},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment an existing synthesis graph with extra edges.")
    parser.add_argument("--input-graph-dir", type=Path, required=True, help="Existing graph store directory.")
    parser.add_argument("--output-graph-dir", type=Path, required=True, help="New graph store directory to write.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH, help="Env file to load before running.")
    parser.add_argument(
        "--reader-base-url",
        type=str,
        default="http://127.0.0.1:8003",
        help="Raw markdown reader base URL used for text-text augmentation.",
    )
    parser.add_argument("--reader-timeout-s", type=float, default=180.0)
    parser.add_argument("--skip-text-text", action="store_true")
    parser.add_argument("--skip-image-text", action="store_true")
    parser.add_argument(
        "--limit-text-nodes",
        type=int,
        default=0,
        help="Optional debug limit on how many text nodes to process.",
    )
    parser.add_argument(
        "--limit-image-nodes",
        type=int,
        default=0,
        help="Optional debug limit on how many image nodes to process.",
    )
    return parser.parse_args()


def _copy_graph_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Input graph dir does not exist: {src}")
    if dst.exists():
        if any(dst.iterdir()):
            raise FileExistsError(f"Output graph dir must not already contain files: {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def _is_wikipedia_text_node(node: dict[str, Any]) -> bool:
    if node.get("node_type") != NodeType.TEXT.value:
        return False
    source = node.get("source") or {}
    url = source.get("url") or ""
    return "/wiki/" in url and "wikipedia.org" in url


def _normalize_wiki_url(url: str | None) -> str | None:
    if not url:
        return None
    return WikiTextBuilder._normalize_wikipedia_url(url)


def _existing_edge_pairs(store: JsonlGraphStore) -> set[tuple[str, str]]:
    return {(edge.get("src_node_id"), edge.get("dst_node_id")) for edge in store.list_edges()}


def _text_nodes_by_url(store: JsonlGraphStore) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for node in store.list_nodes():
        if not _is_wikipedia_text_node(node):
            continue
        source = node.get("source") or {}
        normalized = _normalize_wiki_url(source.get("url"))
        if normalized:
            mapping[normalized] = node
    return mapping


def _build_text_link_evidence(
    *,
    src_node: dict[str, Any],
    dst_node: dict[str, Any],
    anchor_text: str,
    context: str | None,
) -> Evidence:
    return Evidence.create(
        EvidenceType.WEB_TEXT,
        content=context or anchor_text,
        node_ids=[src_node["node_id"], dst_node["node_id"]],
        url=(src_node.get("source") or {}).get("url"),
        extractor="graph_edge_augmenter",
        metadata={
            "augmentation_mode": "text_text_wiki_link",
            "anchor_text": anchor_text,
            "target_title": dst_node.get("title"),
        },
        evidence_key=f"augment_text:{src_node['node_id']}:{dst_node['node_id']}:{anchor_text}",
    )


def _upsert_text_edge(
    *,
    store: JsonlGraphStore,
    existing_pairs: set[tuple[str, str]],
    src_node: dict[str, Any],
    dst_node: dict[str, Any],
    relation: str,
    evidence: Evidence,
) -> bool:
    pair = (src_node["node_id"], dst_node["node_id"])
    if pair in existing_pairs:
        return False
    store.upsert_evidence(evidence)
    edge = Edge.create(
        src_node["node_id"],
        dst_node["node_id"],
        edge_type=EdgeType.WIKI_LINK,
        relation=relation,
        src_node_type=NodeType.TEXT.value,
        dst_node_type=NodeType.TEXT.value,
        evidence_refs=[
            EvidenceRef(
                evidence_id=evidence.evidence_id,
                quote=evidence.content,
                url=evidence.url,
                metadata={"augmentation_mode": "text_text_wiki_link"},
            )
        ],
        source=EdgeSource(
            source_type="graph_augmentation",
            url=evidence.url,
            builder="augment_graph_edges",
        ),
        extractor="augment_graph_edges",
        metadata={
            "augmented": True,
            "augmentation_mode": "text_text_wiki_link",
        },
        evidence_key=evidence.evidence_id,
    )
    store.upsert_edge(edge)
    existing_pairs.add(pair)
    return True


def augment_text_text_edges(
    *,
    store: JsonlGraphStore,
    reader: RawMarkdownReaderClient,
    limit_text_nodes: int = 0,
) -> dict[str, Any]:
    nodes_by_url = _text_nodes_by_url(store)
    text_nodes = list(nodes_by_url.values())
    if limit_text_nodes > 0:
        text_nodes = text_nodes[:limit_text_nodes]

    existing_pairs = _existing_edge_pairs(store)
    relation_builder = WikiTextBuilder(reader=reader, model_client=LLM_WORKER)
    stats = {
        "text_nodes_processed": 0,
        "reader_failures": 0,
        "edges_added": 0,
        "existing_pairs_skipped": 0,
    }

    for src_node in text_nodes:
        source_url = _normalize_wiki_url((src_node.get("source") or {}).get("url"))
        if not source_url:
            continue
        stats["text_nodes_processed"] += 1
        try:
            document = reader.read(source_url)
        except Exception:
            stats["reader_failures"] += 1
            continue
        markdown = document.raw_markdown or document.content or ""
        if not markdown:
            continue
        source_text_node = WikiTextBuilder._text_node_from_record(src_node)
        for anchor_text, href, start, end in WikiTextBuilder._iter_markdown_links(markdown):
            target_url = WikiTextBuilder._wiki_url_from_href(href, source_url=source_url)
            if not target_url:
                continue
            normalized_target_url = _normalize_wiki_url(target_url)
            if not normalized_target_url:
                continue
            dst_node = nodes_by_url.get(normalized_target_url)
            if dst_node is None or dst_node["node_id"] == src_node["node_id"]:
                continue
            pair = (src_node["node_id"], dst_node["node_id"])
            if pair in existing_pairs:
                stats["existing_pairs_skipped"] += 1
                continue
            context = WikiTextBuilder._context(markdown, start, end)
            candidate = WikiLinkCandidate(
                title=dst_node.get("title") or WikiTextBuilder._title_from_url(normalized_target_url) or "",
                url=normalized_target_url,
                anchor_text=anchor_text.strip(),
                source_url=source_url,
                context=context,
            )
            relation_info = relation_builder._extract_relation_for_link(source_text_node, candidate)
            relation = relation_info.get("relation") or candidate.anchor_text or (dst_node.get("title") or "related article")
            forward_evidence = _build_text_link_evidence(
                src_node=src_node,
                dst_node=dst_node,
                anchor_text=anchor_text.strip() or (dst_node.get("title") or "related article"),
                context=context,
            )
            edge_added = _upsert_text_edge(
                store=store,
                existing_pairs=existing_pairs,
                src_node=src_node,
                dst_node=dst_node,
                relation=relation,
                evidence=forward_evidence,
            )
            if edge_added:
                stats["edges_added"] += 1

    store.flush()
    return stats


def _grounded_entities_from_image_node(image_node: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = image_node.get("metadata") or {}
    grounded_entities = metadata.get("grounded_entities") or []
    query_overlap_entities = metadata.get("query_overlap_grounded_entities") or []
    unresolved_entities = metadata.get("unresolved_grounded_entities") or []

    combined: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append(entity: dict[str, Any], *, query_overlap_entity: bool) -> None:
        key = (entity.get("name") or "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        combined.append({**entity, "query_overlap_entity": query_overlap_entity})

    for raw in query_overlap_entities:
        if isinstance(raw, dict):
            _append(raw, query_overlap_entity=True)
    for raw in unresolved_entities:
        if isinstance(raw, dict) and raw.get("status") == "filtered_by_query_entity_overlap":
            _append(raw, query_overlap_entity=True)
    for raw in grounded_entities:
        if isinstance(raw, dict):
            _append(raw, query_overlap_entity=False)

    return combined


def augment_image_text_edges(
    *,
    store: JsonlGraphStore,
    limit_image_nodes: int = 0,
) -> dict[str, Any]:
    builder = ImageDiscoveryBuilder(
        store=store,
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=True,
            try_source_page_recovery=False,
        ),
    )
    existing_pairs = _existing_edge_pairs(store)
    text_nodes_by_url = {
        _normalize_wiki_url((node.get("source") or {}).get("url")): node
        for node in store.list_nodes()
        if node.get("node_type") == NodeType.TEXT.value
        and _normalize_wiki_url((node.get("source") or {}).get("url"))
    }
    image_nodes = [node for node in store.list_nodes() if node.get("node_type") == NodeType.IMAGE.value]
    if limit_image_nodes > 0:
        image_nodes = image_nodes[:limit_image_nodes]

    stats = {
        "image_nodes_processed": 0,
        "grounded_entities_seen": 0,
        "entities_resolved": 0,
        "entities_matched_existing_nodes": 0,
        "edges_added": 0,
    }

    for image_node in image_nodes:
        stats["image_nodes_processed"] += 1
        grounded_entities = _grounded_entities_from_image_node(image_node)
        stats["grounded_entities_seen"] += len(grounded_entities)
        for entity in grounded_entities:
            resolved_target = builder._resolve_grounded_entity(
                entity,
                source_node_title=None,
                image_caption=image_node.get("caption") or image_node.get("summary"),
            )
            if resolved_target is None:
                continue
            stats["entities_resolved"] += 1
            dst_node = text_nodes_by_url.get(_normalize_wiki_url(resolved_target.get("url")))
            if dst_node is None:
                continue
            stats["entities_matched_existing_nodes"] += 1
            pair = (image_node["node_id"], dst_node["node_id"])
            if pair in existing_pairs:
                continue
            evidence = Evidence.create(
                EvidenceType.VLM_OUTPUT,
                content=entity.get("evidence") or entity.get("relation_to_image"),
                node_ids=[image_node["node_id"], dst_node["node_id"]],
                url=image_node.get("image_url"),
                extractor="augment_graph_edges",
                metadata={
                    "augmentation_mode": "image_text_grounded_entity_link",
                    "grounded_entity": entity,
                    "resolved_target": resolved_target,
                    "query_overlap_entity": bool(entity.get("query_overlap_entity")),
                },
                evidence_key=f"augment_image_grounded:{image_node['node_id']}:{dst_node['node_id']}:{entity.get('name')}",
            )
            store.upsert_evidence(evidence)
            edge = Edge.create(
                image_node["node_id"],
                dst_node["node_id"],
                edge_type=EdgeType.IMAGE_DEPICTS,
                relation=entity.get("relation_to_image") or "depicted in image",
                src_node_type=NodeType.IMAGE.value,
                dst_node_type=NodeType.TEXT.value,
                evidence_refs=[
                    EvidenceRef(
                        evidence_id=evidence.evidence_id,
                        quote=entity.get("evidence"),
                        metadata={
                            "augmentation_mode": "image_text_grounded_entity_link",
                            "candidate_title": dst_node.get("title"),
                            "grounded_entity": entity,
                            "resolved_target": resolved_target,
                            "query_overlap_entity": bool(entity.get("query_overlap_entity")),
                        },
                    )
                ],
                source=EdgeSource(
                    source_type="graph_augmentation",
                    url=image_node.get("image_url"),
                    builder="augment_graph_edges",
                ),
                extractor="augment_graph_edges",
                metadata={
                    "augmented": True,
                    "augmentation_mode": "image_text_grounded_entity_link",
                    "entity_name": entity.get("name"),
                    "query_overlap_entity": bool(entity.get("query_overlap_entity")),
                },
                evidence_key=f"{evidence.evidence_id}:{dst_node['node_id']}",
            )
            store.upsert_edge(edge)
            existing_pairs.add(pair)
            stats["edges_added"] += 1

    store.flush()
    return stats


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))

    input_graph_dir = args.input_graph_dir.resolve()
    output_graph_dir = args.output_graph_dir.resolve()
    _copy_graph_dir(input_graph_dir, output_graph_dir)

    store = JsonlGraphStore(output_graph_dir)
    started_at = time.perf_counter()
    summary: dict[str, Any] = {
        "input_graph_dir": str(input_graph_dir),
        "output_graph_dir": str(output_graph_dir),
        "started_at": time.time(),
    }

    if not args.skip_text_text:
        reader = RawMarkdownReaderClient(base_url=args.reader_base_url, timeout_s=args.reader_timeout_s)
        summary["text_text"] = augment_text_text_edges(
            store=store,
            reader=reader,
            limit_text_nodes=args.limit_text_nodes,
        )

    if not args.skip_image_text:
        summary["image_text"] = augment_image_text_edges(
            store=store,
            limit_image_nodes=args.limit_image_nodes,
        )

    summary["elapsed_s"] = time.perf_counter() - started_at
    summary["store_stats"] = store.stats()
    summary_path = output_graph_dir / "augmentation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
