"""Augment an existing synthesis graph with additional shortcut edges.

This script copies an existing graph store into a new output directory, then:

1. Completes text->text links by re-reading each existing Wikipedia text node
   and checking whether its markdown contains outgoing links to other text nodes
   already present in the graph.
2. Completes local image->text links by asking a vision model whether an image
   contains candidate text entities from the image's source-text neighborhood.

The augmented graph is written only to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
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
from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
from synthesis.nodes import NodeType
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.search_client import ImageSearchResult
from synthesis.store import JsonlGraphStore
from synthesis.visual_planner import SearchQuerySpec, VisualSearchPlan
from synthesis.wiki_text_builder import EnhancedReaderClient, ReaderDocument, WikiLinkCandidate, WikiTextBuilder


PROMPT_IMAGE_TEXT_EDGE_VERIFY = """You are verifying whether an image visibly contains specific candidate entities.

You are given:
- an image
- optional image metadata
- a list of existing graph entities

Task:
Decide which candidate entities are clearly visible or directly represented in the image.

Rules:
- Be conservative. Only mark YES when the entity is clearly present or directly represented.
- For people, teams, brands, logos, landmarks, organizations, and objects, rely on visible evidence.
- Do not guess based only on likely association.
- If the image does not clearly show the entity, mark NO.
- The locator must help a user immediately find the entity in the image.
- Keep locator short and concrete.
- Keep evidence to one short sentence.

Output exactly one block:
<verify>
entity: candidate title | yes|no | locator | short evidence
entity: candidate title | yes|no | locator | short evidence
</verify>
"""


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


class RawMarkdownReaderClient:
    """Minimal client for the raw 8002 markdown reader endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8002",
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
        default="http://127.0.0.1:8002",
        help="Raw markdown reader base URL used for text-text augmentation.",
    )
    parser.add_argument("--reader-timeout-s", type=float, default=180.0)
    parser.add_argument("--skip-text-text", action="store_true")
    parser.add_argument("--skip-image-text", action="store_true")
    parser.add_argument(
        "--image-text-model",
        type=str,
        default="",
        help="Optional model alias for image->text verification. Defaults to IMAGE_TEXT_EDGE_MODEL, then IMAGE_GROUND_MODEL, then IMAGE_CHECK_MODEL.",
    )
    parser.add_argument(
        "--image-candidate-hop-radius",
        type=int,
        default=2,
        help="How many text-text hops from source text nodes to gather image->text candidates.",
    )
    parser.add_argument(
        "--max-image-candidates-per-image",
        type=int,
        default=24,
        help="Maximum candidate text nodes verified for each image.",
    )
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
    reverse: bool = False,
) -> bool:
    pair = (src_node["node_id"], dst_node["node_id"])
    if pair in existing_pairs:
        return False
    store.upsert_evidence(evidence)
    edge = Edge.create(
        src_node["node_id"],
        dst_node["node_id"],
        edge_type=EdgeType.DERIVED if reverse else EdgeType.WIKI_LINK,
        relation=relation,
        src_node_type=NodeType.TEXT.value,
        dst_node_type=NodeType.TEXT.value,
        evidence_refs=[
            EvidenceRef(
                evidence_id=evidence.evidence_id,
                quote=evidence.content,
                url=evidence.url,
                metadata={"augmentation_mode": "text_text_wiki_link_reverse" if reverse else "text_text_wiki_link"},
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
            "augmentation_mode": "text_text_wiki_link_reverse" if reverse else "text_text_wiki_link",
        },
        evidence_key=f"{evidence.evidence_id}:{'reverse' if reverse else 'forward'}",
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
        "forward_edges_added": 0,
        "reverse_edges_added": 0,
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
            context = WikiTextBuilder._context(markdown, start, end)
            candidate = WikiLinkCandidate(
                title=dst_node.get("title") or WikiTextBuilder._title_from_url(normalized_target_url) or "",
                url=normalized_target_url,
                anchor_text=anchor_text.strip(),
                source_url=source_url,
                context=context,
            )
            relation_info = relation_builder._extract_relation_for_link(source_text_node, candidate)
            relation = relation_info.get("predicate") or candidate.anchor_text or (dst_node.get("title") or "related article")
            forward_evidence = _build_text_link_evidence(
                src_node=src_node,
                dst_node=dst_node,
                anchor_text=anchor_text.strip() or (dst_node.get("title") or "related article"),
                context=context,
            )
            forward_added = _upsert_text_edge(
                store=store,
                existing_pairs=existing_pairs,
                src_node=src_node,
                dst_node=dst_node,
                relation=relation,
                evidence=forward_evidence,
                reverse=False,
            )
            if forward_added:
                stats["forward_edges_added"] += 1

            reverse_evidence = Evidence.create(
                EvidenceType.WEB_TEXT,
                content=context or anchor_text,
                node_ids=[dst_node["node_id"], src_node["node_id"]],
                url=(src_node.get("source") or {}).get("url"),
                extractor="graph_edge_augmenter",
                metadata={
                    "augmentation_mode": "text_text_wiki_link_reverse",
                    "forward_anchor_text": anchor_text.strip(),
                    "forward_source_title": src_node.get("title"),
                },
                evidence_key=f"augment_text_reverse:{dst_node['node_id']}:{src_node['node_id']}:{anchor_text}",
            )
            reverse_added = _upsert_text_edge(
                store=store,
                existing_pairs=existing_pairs,
                src_node=dst_node,
                dst_node=src_node,
                relation=f"reverse of {relation}",
                evidence=reverse_evidence,
                reverse=True,
            )
            if reverse_added:
                stats["reverse_edges_added"] += 1

    store.flush()
    return stats


def _text_adjacency(store: JsonlGraphStore) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for node in store.list_nodes():
        if node.get("node_type") == NodeType.TEXT.value:
            adjacency[node["node_id"]] = set()
    for edge in store.list_edges():
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def _image_source_text_ids(store: JsonlGraphStore, image_node: dict[str, Any]) -> list[str]:
    image_node_id = image_node["node_id"]
    ids: list[str] = []
    for edge in store.list_edges():
        if edge.get("dst_node_id") != image_node_id:
            continue
        if edge.get("src_node_type") == NodeType.TEXT.value and edge.get("edge_type") in {
            EdgeType.SEARCH_RETRIEVED.value,
            EdgeType.IMAGE_SOURCE_PAGE.value,
        }:
            ids.append(edge["src_node_id"])
    if ids:
        return list(dict.fromkeys(ids))
    source_page_url = image_node.get("source_page_url")
    if not source_page_url:
        return []
    normalized = _normalize_wiki_url(source_page_url)
    if not normalized:
        return []
    for node in store.list_nodes():
        if node.get("node_type") != NodeType.TEXT.value:
            continue
        node_url = _normalize_wiki_url((node.get("source") or {}).get("url"))
        if node_url == normalized:
            ids.append(node["node_id"])
    return list(dict.fromkeys(ids))


def _bfs_text_candidates(
    *,
    adjacency: dict[str, set[str]],
    start_ids: list[str],
    radius: int,
) -> list[str]:
    if radius <= 0:
        return list(dict.fromkeys(start_ids))
    visited: set[str] = set(start_ids)
    frontier = list(start_ids)
    for _ in range(radius):
        next_frontier: list[str] = []
        for node_id in frontier:
            for neighbor in adjacency.get(node_id, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    ordered = list(dict.fromkeys(start_ids + [node_id for node_id in visited if node_id not in set(start_ids)]))
    return ordered


def _build_image_plan(image_node: dict[str, Any]) -> VisualSearchPlan:
    image_label = image_node.get("title") or image_node.get("caption") or image_node.get("node_id")
    target = Evidence.create(
        EvidenceType.VISUAL_TARGET,
        content=image_label,
        node_ids=[],
        extractor="augment_graph_edges",
        metadata={"augmentation_mode": "image_text_verify"},
        evidence_key=f"augment_image_target:{image_node['node_id']}",
    )
    query = SearchQuerySpec.create(
        image_label,
        target.evidence_id,
        source="augment_graph_edges",
        expected_visual=image_label,
    )
    return VisualSearchPlan.create(
        target,
        queries=[query],
        source_node_id=None,
        planner="augment_graph_edges",
        metadata={"augmentation_mode": "image_text_verify"},
    )


def _build_image_search_result(image_node: dict[str, Any]) -> ImageSearchResult:
    return ImageSearchResult(
        title=image_node.get("title"),
        image_url=image_node.get("image_url"),
        source_page_url=image_node.get("source_page_url"),
        snippet=image_node.get("caption") or image_node.get("summary"),
        source="augment_graph_edges",
        raw={"augmentation_mode": "image_text_verify"},
    )


def _verify_image_candidates(
    *,
    builder: ImageDiscoveryBuilder,
    model_alias: str,
    image_node: dict[str, Any],
    candidate_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidate_nodes:
        return [], {"model_alias": model_alias, "usage": None, "raw_model_output": None}

    search_result = _build_image_search_result(image_node)
    resolved_asset, error = builder._resolve_image_asset(search_result)
    if resolved_asset is None or error is not None:
        raise RuntimeError(f"failed_to_resolve_image_asset:{error}")

    prompt_lines = [
        f"Image title: {image_node.get('title') or ''}",
        f"Image caption: {image_node.get('caption') or image_node.get('summary') or ''}",
        "Candidate entities:",
    ]
    for node in candidate_nodes:
        prompt_lines.append(f"- {node.get('title') or node['node_id']}")
    prompt = "\n".join(prompt_lines)

    response = builder.model_client.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=PROMPT_IMAGE_TEXT_EDGE_VERIFY),
                ModelMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": resolved_asset.model_url}},
                    ],
                ),
            ],
            temperature=0.0,
            max_tokens=1200,
            metadata={"trace_label": f"augment_image_text:{image_node.get('node_id')}"},
        )
    )

    name_to_node = {
        (node.get("title") or "").strip(): node
        for node in candidate_nodes
        if (node.get("title") or "").strip()
    }
    matches: list[dict[str, Any]] = []
    text = response.content
    block = text
    lower_text = text.lower()
    start = lower_text.find("<verify>")
    end = lower_text.find("</verify>")
    if start >= 0 and end > start:
        block = text[start + len("<verify>") : end]
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("entity:"):
            continue
        payload = line.split(":", 1)[1].strip()
        parts = [part.strip() for part in payload.split("|")]
        if len(parts) < 2:
            continue
        name = parts[0]
        decision = parts[1].lower()
        node = name_to_node.get(name)
        if node is None or decision != "yes":
            continue
        matches.append(
            {
                "node": node,
                "relation_to_image": parts[2] if len(parts) > 2 and parts[2] else "depicted in image",
                "evidence": parts[3] if len(parts) > 3 else None,
            }
        )
    return matches, {
        "model_alias": model_alias,
        "usage": response.usage,
        "raw_model_output": response.content,
        "resolved_image": resolved_asset.to_metadata(),
    }


def augment_image_text_edges(
    *,
    store: JsonlGraphStore,
    model_alias: str,
    hop_radius: int,
    max_candidates_per_image: int,
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
    adjacency = _text_adjacency(store)
    existing_pairs = _existing_edge_pairs(store)
    text_nodes_by_id = {
        node["node_id"]: node
        for node in store.list_nodes()
        if node.get("node_type") == NodeType.TEXT.value
    }
    image_nodes = [node for node in store.list_nodes() if node.get("node_type") == NodeType.IMAGE.value]
    if limit_image_nodes > 0:
        image_nodes = image_nodes[:limit_image_nodes]

    stats = {
        "image_nodes_processed": 0,
        "verification_failures": 0,
        "edges_added": 0,
    }

    for image_node in image_nodes:
        stats["image_nodes_processed"] += 1
        source_text_ids = _image_source_text_ids(store, image_node)
        if not source_text_ids:
            continue
        candidate_ids = _bfs_text_candidates(
            adjacency=adjacency,
            start_ids=source_text_ids,
            radius=hop_radius,
        )
        candidate_nodes: list[dict[str, Any]] = []
        for node_id in candidate_ids:
            if (image_node["node_id"], node_id) in existing_pairs:
                continue
            node = text_nodes_by_id.get(node_id)
            if node is None or not node.get("title"):
                continue
            candidate_nodes.append(node)
            if len(candidate_nodes) >= max_candidates_per_image:
                break
        if not candidate_nodes:
            continue
        try:
            matches, diagnostics = _verify_image_candidates(
                builder=builder,
                model_alias=model_alias,
                image_node=image_node,
                candidate_nodes=candidate_nodes,
            )
        except Exception:
            stats["verification_failures"] += 1
            continue

        verify_evidence = Evidence.create(
            EvidenceType.VLM_OUTPUT,
            content=diagnostics.get("raw_model_output"),
            node_ids=[image_node["node_id"]],
            url=image_node.get("image_url"),
            extractor="augment_graph_edges",
            metadata={
                "augmentation_mode": "image_text_verify",
                "model_alias": diagnostics.get("model_alias"),
                "usage": diagnostics.get("usage"),
                "resolved_image": diagnostics.get("resolved_image"),
            },
            evidence_key=f"augment_image_verify:{image_node['node_id']}:{model_alias}",
        )
        store.upsert_evidence(verify_evidence)

        for match in matches:
            dst_node = match["node"]
            pair = (image_node["node_id"], dst_node["node_id"])
            if pair in existing_pairs:
                continue
            edge = Edge.create(
                image_node["node_id"],
                dst_node["node_id"],
                edge_type=EdgeType.IMAGE_DEPICTS,
                relation=match["relation_to_image"],
                src_node_type=NodeType.IMAGE.value,
                dst_node_type=NodeType.TEXT.value,
                evidence_refs=[
                    EvidenceRef(
                        evidence_id=verify_evidence.evidence_id,
                        quote=match.get("evidence"),
                        metadata={
                            "augmentation_mode": "image_text_verify",
                            "candidate_title": dst_node.get("title"),
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
                    "augmentation_mode": "image_text_verify",
                    "model_alias": diagnostics.get("model_alias"),
                },
                evidence_key=f"{verify_evidence.evidence_id}:{dst_node['node_id']}",
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
        image_model_alias = (
            args.image_text_model
            or os.environ.get("IMAGE_TEXT_EDGE_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
            or os.environ.get("IMAGE_CHECK_MODEL")
        )
        if not image_model_alias:
            raise ValueError(
                "Image-text augmentation requires --image-text-model or IMAGE_TEXT_EDGE_MODEL/IMAGE_GROUND_MODEL/IMAGE_CHECK_MODEL."
            )
        summary["image_text"] = augment_image_text_edges(
            store=store,
            model_alias=image_model_alias,
            hop_radius=args.image_candidate_hop_radius,
            max_candidates_per_image=args.max_image_candidates_per_image,
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
