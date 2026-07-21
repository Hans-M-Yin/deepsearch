from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelWorkerClient
from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.store import JsonlGraphStore
from synthesis.test_image_grounding import _UnusedSearchClient
from synthesis.wiki_text_builder import EnhancedReaderClient, WikiTextBuilder

PREPARE_PROMPT = """You are preparing verification evidence for one Wikipedia entity page.

Your job is NOT to judge whether the graph edge is correct.
Your job is only to extract compact, visually useful evidence about the entity.

Use the Wikipedia page text and the image title as weak context.
Keep only information that is useful for visual identity checking or event disambiguation.
Do not include broad biography, achievements, or unrelated background.
If the image title appears unrelated, say so explicitly.

Output exactly one JSON object with keys:
- entity_title: string
- entity_type: string
- title_relevance: { relevant: boolean, reason: string }
- identity_summary: string
- visual_profile: string[]
- event_context: string[]
- disambiguation_cues: string[]
"""

REFERENCE_IMAGE_PROMPT = """You are checking whether a reference image from a Wikipedia page is useful for identity verification of the target entity.

Decide whether the image clearly contains the target entity/object AND whether it is visually useful as an identity anchor.
A useful identity anchor should make it possible to recognize what the entity looks like, or what stable visual form it has.

Output exactly one JSON object with keys:
- keep: boolean
- target_localization: string
- why_relevant: string
- identity_anchor_strength: strong|medium|weak
- target_visibility: clear|partial|poor
"""

JUDGE_PROMPT = """You are verifying one graph edge from an image node to a text/entity node.

Task:
Determine whether the object/person referred to by the graph relation in the graph image is truly the target entity.

Important rules:
- Judge the specific object/person indicated by relation_to_image and grounding_evidence, not the whole image loosely.
- Use the reference images only as identity anchors.
- Use textual evidence only to support visual verification, not to replace it.
- If the graph image does not provide enough evidence, return insufficient.
- If the target in the graph image appears to be a different entity/object, return contradict.

Output exactly one JSON object with keys:
- decision: support|contradict|insufficient
- error_type: none|wrong_identity|wrong_relation|ambiguous|insufficient_evidence
- confidence: number
- reason: string
- evidence_for: string[]
- evidence_against: string[]
"""


@dataclass
class VerificationResult:
    edge_id: str
    image_node_id: str
    text_node_id: str
    decision: str
    error_type: str
    confidence: float | None
    reason: str
    evidence_for: list[str]
    evidence_against: list[str]
    judge_model_alias: str | None
    prepare_model_alias: str | None
    kept_reference_image_count: int
    source_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify image->text graph edges with wiki-based visual evidence.")
    parser.add_argument("--graph-dir", required=True, help="Directory containing graph JSONL tables.")
    parser.add_argument("--image-node-id", default="", help="Optional image node id; when set, verify only edges from this image node.")
    parser.add_argument("--env-file", type=str, default=str(DEFAULT_ENV_PATH), help="Path to env file.")
    parser.add_argument("--reader-base-url", type=str, default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--prepare-model", type=str, default=os.environ.get("IMAGE_EDGE_VERIFY_PREPARE_MODEL") or os.environ.get("TEXT_PROCESS_MODEL") or "", help="Model alias for prepare steps.")
    parser.add_argument("--judge-model", type=str, default=os.environ.get("IMAGE_EDGE_VERIFY_JUDGE_MODEL") or os.environ.get("IMAGE_GROUND_MODEL") or os.environ.get("IMAGE_CHECK_MODEL") or "", help="Model alias for final judge.")
    parser.add_argument("--max-reference-images", type=int, default=6, help="Max kept wiki reference images per entity.")
    parser.add_argument("--write-back", action="store_true", help="Write verification results back into edge metadata and optionally remove bad edges.")
    parser.add_argument("--dry-run", action="store_true", help="Run verification only; do not modify graph files. Print planned removals.")
    parser.add_argument("--drop-on", default="contradict", choices=["contradict", "contradict_or_insufficient", "never"], help="Edge removal policy when --write-back is set.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", str(value or "").lower())).strip()


def _is_wikipedia_url(url: str | None) -> bool:
    parsed = urlparse(str(url or "").strip())
    return bool(parsed.netloc.endswith("wikipedia.org") and parsed.path.startswith("/wiki/"))


def _iter_candidate_edges(store: JsonlGraphStore, image_node_id: str | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for edge in store.list_edges():
        if edge.get("edge_type") != "image_depicts":
            continue
        if image_node_id and edge.get("src_node_id") != image_node_id:
            continue
        src_node = store.get_node(str(edge.get("src_node_id") or ""))
        dst_node = store.get_node(str(edge.get("dst_node_id") or ""))
        if not src_node or not dst_node:
            continue
        if src_node.get("node_type") != "image":
            continue
        if dst_node.get("node_type") != "text":
            continue
        source = edge.get("source") or {}
        if str(source.get("source_type") or "") not in {"image_grounding", "image_grounding_delayed", "image_grounding_delayed_debug"}:
            continue
        results.append(edge)
    return results


def _image_node_debug_stats(store: JsonlGraphStore, image_node_id: str) -> dict[str, Any]:
    node = store.get_node(image_node_id)
    out_edges = store.edges_from(image_node_id)
    edge_type_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    sample_edges: list[dict[str, Any]] = []
    for edge in out_edges:
        edge_type = str(edge.get("edge_type") or "<missing>")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        source = edge.get("source") or {}
        source_type = str((source or {}).get("source_type") or "<missing>")
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        if len(sample_edges) < 10:
            sample_edges.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "edge_type": edge.get("edge_type"),
                    "relation": edge.get("relation"),
                    "dst_node_id": edge.get("dst_node_id"),
                    "source_type": (source or {}).get("source_type"),
                }
            )
    metadata = (node or {}).get("metadata") or {}
    return {
        "image_node_exists": node is not None,
        "image_node_type": (node or {}).get("node_type"),
        "out_edge_count": len(out_edges),
        "out_edge_type_counts": edge_type_counts,
        "out_edge_source_type_counts": source_type_counts,
        "grounded_entity_count": len(metadata.get("grounded_entities") or []),
        "unresolved_grounded_entity_count": len(metadata.get("unresolved_grounded_entities") or []),
        "query_overlap_grounded_entity_count": len(metadata.get("query_overlap_grounded_entities") or []),
        "sample_out_edges": sample_edges,
    }


def _find_grounded_entity(image_node: dict[str, Any], text_node: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any] | None:
    grounded = ((image_node.get("metadata") or {}).get("grounded_entities") or [])
    title = str(text_node.get("title") or "").strip()
    aliases = [str(item).strip() for item in (text_node.get("aliases") or []) if str(item).strip()]
    needles = {_normalize_label(title), *(_normalize_label(item) for item in aliases)}
    for item in grounded:
        normalized = _normalize_label((item or {}).get("name"))
        if normalized and normalized in needles:
            return dict(item)
    relation = ((edge.get("metadata") or {}).get("relation") or "").strip()
    for item in grounded:
        if relation and relation == str((item or {}).get("relation_to_image") or "").strip():
            return dict(item)
    return None


def _extract_reference_images(document_content: str, source_url: str, limit: int) -> list[dict[str, Any]]:
    candidates = WikiTextBuilder.extract_wiki_inline_images(document_content or "", source_url=source_url)
    return [candidate.to_dict() for candidate in candidates[: max(1, limit * 2)]]


def _prepare_entity_context(model_client: ModelWorkerClient, model_alias: str, *, text_node: dict[str, Any], wiki_document: dict[str, Any], image_title: str) -> dict[str, Any]:
    content = str(wiki_document.get("content") or "")
    prompt = (
        f"Entity title:\n{text_node.get('title') or ''}\n\n"
        f"Entity aliases:\n{json.dumps(text_node.get('aliases') or [], ensure_ascii=False)}\n\n"
        f"Image title:\n{image_title or ''}\n\n"
        f"Wikipedia page content:\n{content[:12000]}"
    )
    response = model_client.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=PREPARE_PROMPT),
                ModelMessage(role="user", content=prompt),
            ],
            metadata={"trace_label": f"image_edge_verify_prepare:{text_node.get('node_id') or ''}"},
        )
    )
    return _parse_json_object(response.content, default={"raw_model_output": response.content})


def _prepare_reference_image(model_client: ModelWorkerClient, model_alias: str, *, entity_title: str, visual_profile: list[str], event_context: list[str], image_item: dict[str, Any]) -> dict[str, Any]:
    text = (
        f"Entity title:\n{entity_title}\n\n"
        f"Visual profile:\n" + "\n".join(f"- {item}" for item in (visual_profile or [])) + "\n\n"
        f"Event context:\n" + "\n".join(f"- {item}" for item in (event_context or [])) + "\n\n"
        f"Reference image metadata:\ncaption: {image_item.get('caption') or ''}\nalt_text: {image_item.get('alt_text') or ''}\n"
    )
    response = model_client.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=REFERENCE_IMAGE_PROMPT),
                ModelMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_item.get("model_url") or image_item.get("image_url") or image_item.get("thumbnail_url") or ""}},
                    ],
                ),
            ],
            metadata={"trace_label": f"image_edge_verify_reference:{entity_title[:80]}"},
        )
    )
    payload = _parse_json_object(response.content, default={"keep": False, "raw_model_output": response.content})
    payload["raw_model_output"] = response.content
    return payload


def _judge_edge(model_client: ModelWorkerClient, model_alias: str, *, image_node: dict[str, Any], text_node: dict[str, Any], edge: dict[str, Any], grounded_entity: dict[str, Any] | None, prepared_context: dict[str, Any], reference_images: list[dict[str, Any]]) -> dict[str, Any]:
    image_metadata = image_node.get("metadata") or {}
    image_grounding = image_metadata.get("image_grounding") or {}
    image_context = image_grounding.get("context") or image_metadata.get("image_grounding_context") or {}
    user_text = {
        "entity_title": text_node.get("title"),
        "entity_aliases": text_node.get("aliases") or [],
        "edge_relation": edge.get("relation") or (grounded_entity or {}).get("relation_to_image"),
        "grounding_evidence": (grounded_entity or {}).get("evidence"),
        "grounded_entity_name": (grounded_entity or {}).get("name"),
        "image_title": image_node.get("title") or "",
        "image_caption": image_node.get("caption") or image_node.get("summary") or "",
        "image_source_page_url": image_node.get("source_page_url") or "",
        "image_grounding_context": image_context,
        "prepared_context": prepared_context,
        "reference_images": reference_images,
    }
    image_url = image_node.get("image_url") or (((image_metadata.get("resolved_image") or {}).get("asset_uri")) or ((image_metadata.get("resolved_image") or {}).get("resolved_url")) or "")
    content = [{"type": "text", "text": json.dumps(user_text, ensure_ascii=False)}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    for item in reference_images:
        reference_url = item.get("model_url") or item.get("image_url") or item.get("thumbnail_url") or ""
        if reference_url:
            content.append({"type": "image_url", "image_url": {"url": reference_url}})
    response = model_client.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=JUDGE_PROMPT),
                ModelMessage(role="user", content=content),
            ],
            metadata={"trace_label": f"image_edge_verify_judge:{edge.get('edge_id') or ''}"},
        )
    )
    payload = _parse_json_object(response.content, default={"decision": "insufficient", "error_type": "insufficient_evidence", "reason": response.content})
    payload["raw_model_output"] = response.content
    return payload


def _parse_json_object(text: str, default: dict[str, Any]) -> dict[str, Any]:
    payload = str(text or "").strip()
    if not payload:
        return dict(default)
    try:
        return json.loads(payload)
    except Exception:
        pass
    match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if not match:
        return dict(default)
    try:
        return json.loads(match.group(0))
    except Exception:
        return dict(default)


def _delete_edge(store: JsonlGraphStore, edge_id: str) -> bool:
    with store._lock:  # reuse existing store lock for surgical deletion
        existed = edge_id in store._tables["edges"]
        if not existed:
            return False
        del store._tables["edges"][edge_id]
        store._dirty.add("edges")
        store._pending_write_count += 1
        return True


def _resolve_reference_image(
    builder: ImageDiscoveryBuilder,
    *,
    image_item: dict[str, Any],
    page_title: str,
    entity_title: str,
) -> dict[str, Any] | None:
    image_url = str(image_item.get("image_url") or image_item.get("thumbnail_url") or "").strip()
    if not image_url:
        return None
    from synthesis.search_client import ImageSearchResult

    search_result = ImageSearchResult(
        title=str(image_item.get("caption") or entity_title or page_title or ""),
        image_url=image_url,
        thumbnail_url=str(image_item.get("thumbnail_url") or "") or None,
        source_page_url=str(image_item.get("source_page_url") or "") or None,
        snippet=str(image_item.get("caption") or image_item.get("alt_text") or "") or None,
        rank=image_item.get("rank"),
    )
    asset, error = builder._resolve_image_asset(
        search_result,
        persist_asset=False,
        recovery_query=f"{entity_title} {page_title}".strip() or entity_title or page_title,
    )
    if asset is None:
        return None
    return {
        **image_item,
        "model_url": asset.model_url,
        "resolved_image": asset.to_metadata(),
        "resolve_strategy": asset.strategy,
        "resolved_error": error,
    }


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    if not args.prepare_model:
        raise SystemExit("Missing prepare model. Set --prepare-model or IMAGE_EDGE_VERIFY_PREPARE_MODEL/TEXT_PROCESS_MODEL.")
    if not args.judge_model:
        raise SystemExit("Missing judge model. Set --judge-model or IMAGE_EDGE_VERIFY_JUDGE_MODEL/IMAGE_GROUND_MODEL/IMAGE_CHECK_MODEL.")

    store = JsonlGraphStore(Path(args.graph_dir))
    reader = EnhancedReaderClient(base_url=args.reader_base_url)
    model_client = LLM_WORKER
    image_builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(precheck_image_urls=True, try_source_page_recovery=True),
        model_client=model_client,
        image_check_model_alias=os.environ.get("IMAGE_CHECK_MODEL"),
    )

    results: list[dict[str, Any]] = []
    planned_removals: list[dict[str, Any]] = []
    edge_records = _iter_candidate_edges(store, image_node_id=args.image_node_id or None)
    started = time.perf_counter()

    for edge in edge_records:
        image_node = store.get_node(str(edge.get("src_node_id") or ""))
        text_node = store.get_node(str(edge.get("dst_node_id") or ""))
        if not image_node or not text_node:
            continue
        source = text_node.get("source") or {}
        text_url = text_node.get("url") or source.get("url") or source.get("source_url") or ""
        if not _is_wikipedia_url(text_url):
            continue
        grounded_entity = _find_grounded_entity(image_node, text_node, edge)
        try:
            wiki_doc = reader.read(text_url).to_dict()
        except Exception as exc:
            results.append(
                VerificationResult(
                    edge_id=str(edge.get("edge_id") or ""),
                    image_node_id=str(image_node.get("node_id") or ""),
                    text_node_id=str(text_node.get("node_id") or ""),
                    decision="insufficient",
                    error_type="insufficient_evidence",
                    confidence=None,
                    reason=f"reader_error:{exc.__class__.__name__}:{exc}",
                    evidence_for=[],
                    evidence_against=["failed to read wikipedia page"],
                    judge_model_alias=args.judge_model,
                    prepare_model_alias=args.prepare_model,
                    kept_reference_image_count=0,
                    source_type=str((edge.get("metadata") or {}).get("source_type") or ""),
                ).to_dict()
            )
            continue

        prepared_context = _prepare_entity_context(
            model_client,
            args.prepare_model,
            text_node=text_node,
            wiki_document=wiki_doc,
            image_title=str(image_node.get("title") or ""),
        )
        raw_reference_images = _extract_reference_images(wiki_doc.get("raw_markdown") or wiki_doc.get("content") or "", text_url, args.max_reference_images)
        resolved_reference_images: list[dict[str, Any]] = []
        for image_item in raw_reference_images:
            resolved = _resolve_reference_image(
                image_builder,
                image_item=image_item,
                page_title=str(wiki_doc.get("title") or text_node.get("title") or ""),
                entity_title=str(text_node.get("title") or ""),
            )
            if resolved is not None:
                resolved_reference_images.append(resolved)
        kept_reference_images: list[dict[str, Any]] = []
        for image_item in resolved_reference_images:
            decision = _prepare_reference_image(
                model_client,
                args.prepare_model,
                entity_title=str(text_node.get("title") or ""),
                visual_profile=list(prepared_context.get("visual_profile") or []),
                event_context=list(prepared_context.get("event_context") or []),
                image_item=image_item,
            )
            if decision.get("keep") is True:
                kept_reference_images.append({**image_item, **decision})
            if len(kept_reference_images) >= max(1, args.max_reference_images):
                break

        judged = _judge_edge(
            model_client,
            args.judge_model,
            image_node=image_node,
            text_node=text_node,
            edge=edge,
            grounded_entity=grounded_entity,
            prepared_context=prepared_context,
            reference_images=kept_reference_images,
        )
        result = VerificationResult(
            edge_id=str(edge.get("edge_id") or ""),
            image_node_id=str(image_node.get("node_id") or ""),
            text_node_id=str(text_node.get("node_id") or ""),
            decision=str(judged.get("decision") or "insufficient"),
            error_type=str(judged.get("error_type") or "insufficient_evidence"),
            confidence=float(judged.get("confidence")) if str(judged.get("confidence") or "").strip() else None,
            reason=str(judged.get("reason") or ""),
            evidence_for=[str(item) for item in (judged.get("evidence_for") or [])],
            evidence_against=[str(item) for item in (judged.get("evidence_against") or [])],
            judge_model_alias=args.judge_model,
            prepare_model_alias=args.prepare_model,
            kept_reference_image_count=len(kept_reference_images),
            source_type=str((edge.get("metadata") or {}).get("source_type") or ""),
        )
        record = result.to_dict()
        record["prepared_context"] = prepared_context
        record["grounded_entity"] = grounded_entity
        record["reference_images"] = kept_reference_images
        results.append(record)

        should_drop = False
        if args.drop_on == "contradict" and result.decision == "contradict":
            should_drop = True
        elif args.drop_on == "contradict_or_insufficient" and result.decision in {"contradict", "insufficient"}:
            should_drop = True
        if should_drop:
            planned_removals.append(
                {
                    "edge_id": result.edge_id,
                    "image_node_id": result.image_node_id,
                    "text_node_id": result.text_node_id,
                    "decision": result.decision,
                    "error_type": result.error_type,
                    "reason": result.reason,
                }
            )

        if args.write_back and not args.dry_run:
            edge_record = store.get_edge(result.edge_id) or edge
            edge_meta = dict(edge_record.get("metadata") or {})
            edge_meta["post_verify_image_text"] = {
                "decision": result.decision,
                "error_type": result.error_type,
                "confidence": result.confidence,
                "reason": result.reason,
                "evidence_for": result.evidence_for,
                "evidence_against": result.evidence_against,
                "judge_model_alias": result.judge_model_alias,
                "prepare_model_alias": result.prepare_model_alias,
                "kept_reference_image_count": result.kept_reference_image_count,
                "verified_at_unix": time.time(),
            }
            edge_record["metadata"] = edge_meta
            store.upsert_edge(edge_record)
            if args.drop_on == "contradict" and result.decision == "contradict":
                _delete_edge(store, result.edge_id)
            elif args.drop_on == "contradict_or_insufficient" and result.decision in {"contradict", "insufficient"}:
                _delete_edge(store, result.edge_id)

    if args.write_back and not args.dry_run and store.has_pending_writes():
        store.flush()

    payload = {
        "graph_dir": str(Path(args.graph_dir).resolve()),
        "image_node_id": args.image_node_id or None,
        "prepare_model": args.prepare_model,
        "judge_model": args.judge_model,
        "dry_run": bool(args.dry_run),
        "write_back": bool(args.write_back and not args.dry_run),
        "edge_count": len(results),
        "elapsed_s": time.perf_counter() - started,
        "planned_removal_count": len(planned_removals),
        "planned_removals": planned_removals,
        "results": results,
    }
    if args.image_node_id and not results:
        payload["empty_result_debug"] = _image_node_debug_stats(store, args.image_node_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    if args.dry_run and planned_removals:
        print("\n[verify_image_text_edges][dry-run] planned removals:", file=sys.stderr)
        for item in planned_removals:
            print(
                "- "
                f"edge_id={item['edge_id']} image_node_id={item['image_node_id']} text_node_id={item['text_node_id']} "
                f"decision={item['decision']} error_type={item['error_type']} reason={item['reason']}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
