"""Verify image-to-text graph edges with wiki-based visual evidence.

Restore edges previously soft-deleted by this verifier:

    python synthesis/post_process/verify_image_text_edges.py \
      --graph-dir runs/my_graph \
      --restore-post-verify-rejected \
      --pretty

Restore selected edges only:

    python synthesis/post_process/verify_image_text_edges.py \
      --graph-dir runs/my_graph \
      --restore-edge-id edge_aaa \
      --restore-edge-id edge_bbb \
      --pretty

Add ``--dry-run`` to preview the restore without modifying ``edges.jsonl``.
"""

from __future__ import annotations

import argparse
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelWorkerClient
from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.store import JsonlGraphStore
from synthesis.test_image_grounding import _UnusedSearchClient
from synthesis.wiki_text_builder import EnhancedReaderClient, WikiTextBuilder

_VERIFY_FIXED_REQUEST_ID = "3200636808"

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

Output exactly one JSON object, with no Markdown or extra prose, containing:
- decision: support|contradict|insufficient
- error_type: none|wrong_identity|wrong_relation|ambiguous|insufficient_evidence
- confidence: number from 0.0 to 1.0
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


def _verify_worker_metadata(trace_label: str) -> dict[str, str]:
    """Attach stable gateway routing fields so repeated verifier prompts can cache."""
    return {
        "trace_label": trace_label,
        "session_id": _VERIFY_FIXED_REQUEST_ID,
        "prompt_cache_key": _VERIFY_FIXED_REQUEST_ID,
        "user_id": _VERIFY_FIXED_REQUEST_ID,
        "x_tt_logid": _VERIFY_FIXED_REQUEST_ID,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify image->text graph edges with wiki-based visual evidence.")
    parser.add_argument("--graph-dir", required=True, help="Directory containing graph JSONL tables.")
    parser.add_argument("--image-node-id", default="", help="Optional image node id; when set, verify only edges from this image node.")
    parser.add_argument(
        "--relation-override",
        default="",
        help=(
            "Dry-run-only replacement relation/locator to test against every selected edge. "
            "The graph is not modified."
        ),
    )
    parser.add_argument("--env-file", type=str, default=str(DEFAULT_ENV_PATH), help="Path to env file.")
    parser.add_argument("--reader-base-url", type=str, default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--prepare-model", type=str, default=os.environ.get("IMAGE_EDGE_VERIFY_PREPARE_MODEL") or os.environ.get("TEXT_PROCESS_MODEL") or "", help="Model alias for prepare steps.")
    parser.add_argument("--judge-model", type=str, default=os.environ.get("IMAGE_EDGE_VERIFY_JUDGE_MODEL") or os.environ.get("IMAGE_GROUND_MODEL") or os.environ.get("IMAGE_CHECK_MODEL") or "", help="Model alias for final judge.")
    parser.add_argument("--max-reference-images", type=int, default=6, help="Max kept wiki reference images per entity.")
    parser.add_argument(
        "--max-image-nodes",
        type=int,
        default=0,
        help="Randomly sample at most this many candidate image nodes before verification; <=0 verifies all.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used with --max-image-nodes so an image-node sample can be reproduced.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help=(
            "Append one completed edge result at a time to this JSONL checkpoint file. "
            "Use with --resume to continue an interrupted run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed edge records from --results-jsonl and only verify remaining edges.",
    )
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="Do not skip graph edges that already contain metadata.post_verify_image_text.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for per-edge verification.")
    parser.add_argument("--write-back", action="store_true", help="Write verification results back into edge metadata and soft-delete matching rejected edges.")
    parser.add_argument("--hard-delete", action="store_true", help="Physically delete matching edges instead of the default reversible soft delete. Requires --write-back.")
    parser.add_argument("--restore-post-verify-rejected", action="store_true", help="Restore all verifier-soft-deleted edges in --graph-dir without running models.")
    parser.add_argument("--restore-edge-id", action="append", default=[], help="Restore one verifier-soft-deleted edge ID. Repeatable; implies restore mode.")
    parser.add_argument("--dry-run", action="store_true", help="Run verification or restoration planning only; do not modify graph files.")
    parser.add_argument("--drop-on", default="contradict", choices=["contradict", "contradict_or_insufficient", "never"], help="Edge soft/hard deletion policy when --write-back is set.")
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
        # Never re-verify an edge already hidden by any lifecycle policy.
        if str(edge.get("status") or "active").lower() != "active":
            continue
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


def _edge_has_post_verification(edge: dict[str, Any]) -> bool:
    return isinstance((edge.get("metadata") or {}).get("post_verify_image_text"), dict)


def _load_checkpoint_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load the most recent valid result for every edge from a JSONL checkpoint."""
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[verify_image_text_edges] ignoring invalid checkpoint JSON on line {line_number}: {path}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict):
                continue
            edge_id = str(record.get("edge_id") or "").strip()
            if edge_id:
                results[edge_id] = record
    return results


def _append_checkpoint_result(path: Path, record: dict[str, Any]) -> None:
    """Durably append one result so completed edges survive interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sample_image_node_ids(
    edge_records: list[dict[str, Any]],
    *,
    max_image_nodes: int,
    random_seed: int,
) -> list[str]:
    """Return a reproducible random subset of image nodes represented by edges."""
    image_node_ids = sorted(
        {
            str(edge.get("src_node_id") or "").strip()
            for edge in edge_records
            if str(edge.get("src_node_id") or "").strip()
        }
    )
    if max_image_nodes <= 0 or len(image_node_ids) <= max_image_nodes:
        return image_node_ids
    return sorted(random.Random(random_seed).sample(image_node_ids, max_image_nodes))


def _grounding_entity_counts(
    store: JsonlGraphStore,
    image_node_ids: list[str],
) -> dict[str, int]:
    """Count persisted grounding entities for the image nodes being verified."""
    total = 0
    images_with_grounding = 0
    for image_node_id in image_node_ids:
        node = store.get_node(image_node_id) or {}
        metadata = node.get("metadata") or {}
        grounded_entities = metadata.get("grounded_entities") or [] if isinstance(metadata, dict) else []
        if grounded_entities:
            images_with_grounding += 1
        total += len(grounded_entities)
    return {
        "grounded_entity_count": total,
        "image_node_count_with_grounded_entities": images_with_grounding,
    }


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
            metadata=_verify_worker_metadata(
                f"image_edge_verify_prepare:{text_node.get('node_id') or ''}"
            ),
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
    # Reference images are resolved before this point.  Do not fall back to a
    # remote URL here: some Gemini-compatible endpoints translate it into
    # fileData without a MIME type.
    model_url = str(image_item.get("model_url") or "").strip()
    if not _is_model_image_data_url(model_url):
        return {
            "keep": False,
            "error_type": "reference_image_unavailable",
            "reason": "reference image could not be resolved to a typed data URL",
        }
    response = model_client.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=REFERENCE_IMAGE_PROMPT),
                ModelMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": model_url}},
                    ],
                ),
            ],
            metadata=_verify_worker_metadata(
                f"image_edge_verify_reference:{entity_title[:80]}"
            ),
        )
    )
    payload = _parse_json_object(response.content, default={"keep": False, "raw_model_output": response.content})
    payload["raw_model_output"] = response.content
    return payload


def _is_model_image_data_url(value: Any) -> bool:
    """Return whether ``value`` is a MIME-typed image data URL for an LLM."""
    url = str(value or "").strip().lower()
    return url.startswith("data:image/") and ";base64," in url


def _judge_edge(model_client: ModelWorkerClient, model_alias: str, *, image_node: dict[str, Any], text_node: dict[str, Any], edge: dict[str, Any], grounded_entity: dict[str, Any] | None, prepared_context: dict[str, Any], reference_images: list[dict[str, Any]], primary_image_model_url: str | None = None) -> dict[str, Any]:
    if not _is_model_image_data_url(primary_image_model_url):
        # The judge must never make a visual contradiction decision without the
        # graph image.  This also avoids sending a naked remote image URL to
        # Gemini/Vertex, whose OpenAI adapter may produce fileData with an empty
        # MIME type.
        return {
            "decision": "insufficient",
            "error_type": "insufficient_evidence",
            "reason": "primary_image_unavailable_for_model",
        }
    image_metadata = image_node.get("metadata") or {}
    image_grounding = image_metadata.get("image_grounding") or {}
    image_context = image_grounding.get("context") or image_metadata.get("image_grounding_context") or {}
    compact_prepared_context = {
        "entity_title": prepared_context.get("entity_title") or text_node.get("title"),
        "entity_type": prepared_context.get("entity_type"),
        "title_relevance": prepared_context.get("title_relevance"),
        "identity_summary": prepared_context.get("identity_summary"),
        "visual_profile": list(prepared_context.get("visual_profile") or []),
        "event_context": list(prepared_context.get("event_context") or []),
        "disambiguation_cues": list(prepared_context.get("disambiguation_cues") or []),
    }
    compact_reference_images = [
        {
            "caption": item.get("caption"),
            "alt_text": item.get("alt_text"),
            "target_localization": item.get("target_localization"),
            "why_relevant": item.get("why_relevant"),
            "identity_anchor_strength": item.get("identity_anchor_strength"),
            "target_visibility": item.get("target_visibility"),
            "resolve_strategy": item.get("resolve_strategy"),
        }
        for item in reference_images
    ]
    compact_grounding_context = {
        "provider": image_context.get("provider"),
        "metadata": {
            "image_title": ((image_context.get("metadata") or {}).get("image_title")),
            "image_snippet": ((image_context.get("metadata") or {}).get("image_snippet")),
            "source_page_url": ((image_context.get("metadata") or {}).get("source_page_url")),
            "page_title": ((image_context.get("metadata") or {}).get("page_title")),
            "fallback_reason": ((image_context.get("metadata") or {}).get("fallback_reason")),
        },
    }
    user_text = {
        "entity_title": text_node.get("title"),
        "entity_aliases": text_node.get("aliases") or [],
        "edge_relation": edge.get("relation") or (grounded_entity or {}).get("relation_to_image"),
        "grounding_evidence": (grounded_entity or {}).get("evidence"),
        "grounded_entity_name": (grounded_entity or {}).get("name"),
        "image_title": image_node.get("title") or "",
        "image_caption": image_node.get("caption") or image_node.get("summary") or "",
        "image_source_page_url": image_node.get("source_page_url") or "",
        "image_grounding_context": compact_grounding_context,
        "prepared_context": compact_prepared_context,
        "reference_images": compact_reference_images,
    }
    content = [{"type": "text", "text": json.dumps(user_text, ensure_ascii=False)}]
    content.append({"type": "image_url", "image_url": {"url": primary_image_model_url}})
    for item in reference_images:
        reference_url = item.get("model_url")
        if _is_model_image_data_url(reference_url):
            content.append({"type": "image_url", "image_url": {"url": reference_url}})
    response = model_client.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=JUDGE_PROMPT),
                ModelMessage(role="user", content=content),
            ],
            response_format={"type": "json_object"},
            metadata=_verify_worker_metadata(
                f"image_edge_verify_judge:{edge.get('edge_id') or ''}"
            ),
        )
    )
    payload = _parse_judge_response(response.content)
    payload["raw_model_output"] = response.content
    return payload


def _compact_reference_images_for_output(reference_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "caption": item.get("caption"),
            "alt_text": item.get("alt_text"),
            "target_localization": item.get("target_localization"),
            "why_relevant": item.get("why_relevant"),
            "identity_anchor_strength": item.get("identity_anchor_strength"),
            "target_visibility": item.get("target_visibility"),
            "resolve_strategy": item.get("resolve_strategy"),
            "source_page_url": item.get("source_page_url"),
            "file_page_url": item.get("file_page_url"),
            "rank": item.get("rank"),
        }
        for item in reference_images
    ]


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


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse and validate the final judge response without mistaking format errors for weak evidence."""
    parsed = _parse_json_object(text, default={})
    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"support", "contradict", "insufficient"}:
        return {
            "decision": "insufficient",
            "error_type": "judge_output_parse_failed",
            "reason": "judge did not return a valid JSON decision",
            "evidence_for": [],
            "evidence_against": [],
        }
    error_type = str(parsed.get("error_type") or "").strip() or (
        "none" if decision == "support" else "insufficient_evidence"
    )
    try:
        confidence = float(parsed["confidence"]) if parsed.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {
        "decision": decision,
        "error_type": error_type,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip(),
        "evidence_for": [str(item) for item in (parsed.get("evidence_for") or [])],
        "evidence_against": [str(item) for item in (parsed.get("evidence_against") or [])],
    }


_POST_VERIFY_ACTOR = "verify_image_text_edges"


def _delete_edge(store: JsonlGraphStore, edge_id: str) -> bool:
    """Permanently delete an edge; used only with explicit --hard-delete."""
    with store._lock:  # reuse existing store lock for surgical deletion
        existed = edge_id in store._tables["edges"]
        if not existed:
            return False
        del store._tables["edges"][edge_id]
        store._dirty.add("edges")
        store._pending_write_count += 1
        return True


def _lifecycle_history(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("edge_lifecycle_history")
    return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _soft_delete_verified_edge(edge_record: dict[str, Any], *, decision: str) -> dict[str, Any]:
    """Make a verifier-rejected edge invisible to consumers while retaining it."""
    metadata = dict(edge_record.get("metadata") or {})
    previous_status = str(edge_record.get("status") or "active")
    now = time.time()
    event = {
        "action": "soft_deleted",
        "actor": _POST_VERIFY_ACTOR,
        "reason_code": f"post_verify_{decision}",
        "previous_status": previous_status,
        "changed_at_unix": now,
    }
    history = _lifecycle_history(metadata)
    history.append(event)
    metadata["edge_lifecycle_history"] = history
    metadata["edge_lifecycle"] = {
        "current_action": "soft_deleted",
        "actor": _POST_VERIFY_ACTOR,
        "reason_code": event["reason_code"],
        "previous_status": previous_status,
        "changed_at_unix": now,
    }
    edge_record["metadata"] = metadata
    edge_record["status"] = "rejected"
    return edge_record


def _is_verifier_soft_deleted(edge_record: dict[str, Any]) -> bool:
    if str(edge_record.get("status") or "active").lower() != "rejected":
        return False
    lifecycle = (edge_record.get("metadata") or {}).get("edge_lifecycle")
    return (
        isinstance(lifecycle, dict)
        and lifecycle.get("current_action") == "soft_deleted"
        and lifecycle.get("actor") == _POST_VERIFY_ACTOR
        and str(lifecycle.get("reason_code") or "").startswith("post_verify_")
    )


def _restore_verified_edge(edge_record: dict[str, Any]) -> dict[str, Any]:
    """Restore only an edge previously soft-deleted by this verifier."""
    if not _is_verifier_soft_deleted(edge_record):
        raise ValueError("edge is not currently soft-deleted by verify_image_text_edges")
    metadata = dict(edge_record.get("metadata") or {})
    lifecycle = dict(metadata.get("edge_lifecycle") or {})
    restored_status = str(lifecycle.get("previous_status") or "active")
    now = time.time()
    history = _lifecycle_history(metadata)
    history.append(
        {
            "action": "restored",
            "actor": _POST_VERIFY_ACTOR,
            "reason_code": "manual_restore",
            "restored_to_status": restored_status,
            "changed_at_unix": now,
        }
    )
    metadata["edge_lifecycle_history"] = history
    metadata["edge_lifecycle"] = {
        "current_action": "restored",
        "actor": _POST_VERIFY_ACTOR,
        "reason_code": "manual_restore",
        "restored_to_status": restored_status,
        "changed_at_unix": now,
    }
    edge_record["metadata"] = metadata
    edge_record["status"] = restored_status
    return edge_record


def _restore_post_verify_edges(
    store: JsonlGraphStore,
    *,
    edge_ids: list[str],
    restore_all: bool,
    dry_run: bool,
) -> dict[str, Any]:
    requested = {str(edge_id).strip() for edge_id in edge_ids if str(edge_id).strip()}
    if requested and restore_all:
        raise ValueError("use either --restore-post-verify-rejected or --restore-edge-id, not both")
    candidates = [
        edge for edge in store.list_edges()
        if _is_verifier_soft_deleted(edge)
        and (restore_all or str(edge.get("edge_id") or "") in requested)
    ]
    if requested:
        found = {str(edge.get("edge_id") or "") for edge in candidates}
        missing = sorted(requested - found)
    else:
        missing = []
    restored: list[str] = []
    if not dry_run:
        for edge in candidates:
            restored_edge = _restore_verified_edge(dict(edge))
            store.upsert_edge(restored_edge)
            restored.append(str(restored_edge.get("edge_id") or ""))
        if store.has_pending_writes():
            store.flush()
    return {
        "mode": "restore_post_verify_rejected",
        "dry_run": bool(dry_run),
        "requested_edge_ids": sorted(requested),
        "candidate_edge_ids": [str(edge.get("edge_id") or "") for edge in candidates],
        "restored_edge_ids": restored,
        "missing_or_ineligible_edge_ids": missing,
    }


def _should_drop(decision: str, drop_on: str) -> bool:
    if drop_on == "contradict":
        return decision == "contradict"
    if drop_on == "contradict_or_insufficient":
        return decision in {"contradict", "insufficient"}
    return False


def _verify_single_edge(
    *,
    edge: dict[str, Any],
    image_node: dict[str, Any],
    text_node: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    reader = EnhancedReaderClient(base_url=args.reader_base_url)
    model_client = LLM_WORKER
    image_builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(precheck_image_urls=True, try_source_page_recovery=True),
        model_client=model_client,
        image_check_model_alias=os.environ.get("IMAGE_CHECK_MODEL"),
    )

    source = text_node.get("source") or {}
    text_url = text_node.get("url") or source.get("url") or source.get("source_url") or ""
    if not _is_wikipedia_url(text_url):
        return None
    grounded_entity = _find_grounded_entity(image_node, text_node, edge)
    try:
        wiki_doc = reader.read(text_url).to_dict()
    except Exception as exc:
        return VerificationResult(
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
            source_type=str(((edge.get("source") or {}).get("source_type")) or ""),
        ).to_dict()

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
        primary_image_model_url=_resolve_image_node_for_model(image_builder, image_node=image_node),
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
        source_type=str(((edge.get("source") or {}).get("source_type")) or ""),
    )
    record = result.to_dict()
    if edge.get("_debug_original_relation") is not None:
        record["original_relation"] = edge.get("_debug_original_relation")
        record["relation_override"] = edge.get("relation")
    record["image_url"] = image_node.get("image_url")
    record["prepared_context"] = prepared_context
    record["grounded_entity"] = grounded_entity
    record["reference_images"] = _compact_reference_images_for_output(kept_reference_images)
    return record


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


def _resolve_image_node_for_model(
    builder: ImageDiscoveryBuilder,
    *,
    image_node: dict[str, Any],
) -> str | None:
    """Resolve the graph image to the typed data URL required by vision models.

    Persisted graph records deliberately omit ``model_url`` because it embeds
    base64 image bytes.  Re-resolve it at verification time instead of passing
    the raw OSS URL through the model-worker API.
    """
    metadata = image_node.get("metadata") or {}
    resolved = metadata.get("resolved_image") or {}
    if not isinstance(resolved, dict):
        resolved = {}

    # Graph construction may persist an already-downloaded cache path as either
    # ``resolved_image.cache_path`` / ``asset_uri`` or ``image_url``.  Do not
    # send such paths through the HTTP downloader; turn their bytes directly
    # into the typed data URL required by vision models.
    for value in (
        resolved.get("cache_path"),
        image_node.get("image_url"),
        image_node.get("oss_uri"),
        resolved.get("asset_uri"),
        resolved.get("original_url"),
        resolved.get("resolved_url"),
    ):
        local_path = _local_image_path(value)
        if local_path is not None:
            return _local_image_path_to_model_url(
                builder,
                local_path,
                content_type_hint=resolved.get("content_type") or image_node.get("content_type"),
            )

    image_url = str(
        image_node.get("image_url")
        or image_node.get("oss_uri")
        or resolved.get("asset_uri")
        or resolved.get("original_url")
        or resolved.get("resolved_url")
        or ""
    ).strip()
    if not image_url:
        return None
    image_item = {
        "image_url": image_url,
        "thumbnail_url": image_node.get("thumbnail_url"),
        "source_page_url": image_node.get("source_page_url"),
        "caption": image_node.get("caption") or image_node.get("summary"),
        "alt_text": image_node.get("title"),
    }
    resolved_item = _resolve_reference_image(
        builder,
        image_item=image_item,
        page_title=str(image_node.get("title") or ""),
        entity_title=str(image_node.get("title") or ""),
    )
    model_url = str((resolved_item or {}).get("model_url") or "").strip()
    return model_url if _is_model_image_data_url(model_url) else None


def _local_image_path(value: Any) -> Path | None:
    """Return an existing local path for an absolute path or file URI."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("file://"):
        path = Path(unquote(urlparse(text).path))
    else:
        path = Path(text)
    return path if path.is_absolute() and path.is_file() else None


def _local_image_path_to_model_url(
    builder: ImageDiscoveryBuilder,
    image_path: Path,
    *,
    content_type_hint: Any,
) -> str | None:
    """Read a persisted local image and convert it to a model data URL."""
    try:
        payload = image_path.read_bytes()
        builder._image_dimensions(payload, verify=True)
    except Exception:
        return None

    content_type = (
        builder._sniff_content_type(payload)
        or str(content_type_hint or "").strip().lower()
        or mimetypes.guess_type(str(image_path))[0]
        or "image/jpeg"
    )
    if not content_type.startswith("image/"):
        return None
    try:
        content_type, payload = builder._prepare_model_payload(
            payload=payload,
            content_type=content_type,
            max_edge=builder.config.model_image_max_edge,
        )
    except Exception:
        return None
    return builder._data_url(content_type, payload)


def _worker_exception_record(edge: dict[str, Any], args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    """Make a failed worker resumable instead of aborting the whole batch."""
    return VerificationResult(
        edge_id=str(edge.get("edge_id") or ""),
        image_node_id=str(edge.get("src_node_id") or ""),
        text_node_id=str(edge.get("dst_node_id") or ""),
        decision="insufficient",
        error_type="worker_exception",
        confidence=None,
        reason=f"{exc.__class__.__name__}: {exc}",
        evidence_for=[],
        evidence_against=[],
        judge_model_alias=args.judge_model,
        prepare_model_alias=args.prepare_model,
        kept_reference_image_count=0,
        source_type=str(((edge.get("source") or {}).get("source_type")) or ""),
    ).to_dict()


def _emit_image_node_complete_status(
    *,
    image_node_id: str,
    records: list[dict[str, Any]],
    completed_image_node_count: int,
    total_image_node_count: int,
    elapsed_s: float,
) -> None:
    """Log one concise stderr status line after all scheduled edges of an image finish."""
    decisions = {name: 0 for name in ("support", "contradict", "insufficient")}
    for record in records:
        decision = str(record.get("decision") or "insufficient")
        decisions[decision] = decisions.get(decision, 0) + 1
    worker_error_count = sum(
        1 for record in records if record.get("error_type") == "worker_exception"
    )
    print(
        "[verify_image_text_edges] "
        f"image_node_complete image_node_id={image_node_id!r} "
        f"edge_count={len(records)} "
        f"support={decisions['support']} "
        f"contradict={decisions['contradict']} "
        f"insufficient={decisions['insufficient']} "
        f"worker_exceptions={worker_error_count} "
        f"completed_image_nodes={completed_image_node_count}/{total_image_node_count} "
        f"elapsed_s={elapsed_s:.1f}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    args = parse_args()
    restore_requested = bool(args.restore_post_verify_rejected or args.restore_edge_id)
    if restore_requested:
        if args.write_back or args.results_jsonl or args.resume or args.reverify:
            raise SystemExit("restore mode cannot be combined with --write-back, checkpoint, --resume, or --reverify.")
        try:
            payload = _restore_post_verify_edges(
                JsonlGraphStore(Path(args.graph_dir)),
                edge_ids=args.restore_edge_id,
                restore_all=bool(args.restore_post_verify_rejected),
                dry_run=bool(args.dry_run),
            )
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0

    if args.hard_delete and not args.write_back:
        raise SystemExit("--hard-delete requires --write-back.")
    relation_override = str(args.relation_override or "").strip()
    if relation_override and args.write_back:
        raise SystemExit("--relation-override is dry-run-only and cannot be combined with --write-back.")
    load_env_file(Path(args.env_file))
    if not args.prepare_model:
        raise SystemExit("Missing prepare model. Set --prepare-model or IMAGE_EDGE_VERIFY_PREPARE_MODEL/TEXT_PROCESS_MODEL.")
    if not args.judge_model:
        raise SystemExit("Missing judge model. Set --judge-model or IMAGE_EDGE_VERIFY_JUDGE_MODEL/IMAGE_GROUND_MODEL/IMAGE_CHECK_MODEL.")

    checkpoint_path = args.results_jsonl.expanduser().resolve() if args.results_jsonl else None
    if args.resume and checkpoint_path is None:
        raise SystemExit("--resume requires --results-jsonl.")
    if checkpoint_path is not None and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0 and not args.resume:
        raise SystemExit(f"Checkpoint already exists; pass --resume to reuse it: {checkpoint_path}")
    checkpoint_results = _load_checkpoint_results(checkpoint_path) if args.resume and checkpoint_path else {}

    store = JsonlGraphStore(Path(args.graph_dir))
    # The graph is immutable during dry-run verification. Build one shared
    # read-only node index here instead of having every edge worker construct a
    # JsonlGraphStore and reparse every graph JSONL table from disk.
    nodes_by_id = {
        str(node.get("node_id") or ""): node
        for node in store.list_nodes()
        if str(node.get("node_id") or "")
    }
    results_by_edge_id: dict[str, dict[str, Any]] = {}
    planned_removals: list[dict[str, Any]] = []
    all_edge_records = _iter_candidate_edges(store, image_node_id=args.image_node_id or None)
    graph_verified_edges = [edge for edge in all_edge_records if _edge_has_post_verification(edge)]
    graph_pending_edge_records = [
        edge for edge in all_edge_records if args.reverify or not _edge_has_post_verification(edge)
    ]
    selected_image_node_ids = _sample_image_node_ids(
        graph_pending_edge_records,
        max_image_nodes=args.max_image_nodes,
        random_seed=args.random_seed,
    )
    selected_image_node_id_set = set(selected_image_node_ids)
    selected_edge_records = [
        edge
        for edge in graph_pending_edge_records
        if str(edge.get("src_node_id") or "") in selected_image_node_id_set
    ]
    checkpoint_edge_ids = set(checkpoint_results)
    edge_records = [
        edge
        for edge in selected_edge_records
        if (args.reverify or not _edge_has_post_verification(edge))
        and str(edge.get("edge_id") or "") not in checkpoint_edge_ids
    ]
    if relation_override:
        edge_records = [
            {
                **edge,
                "relation": relation_override,
                "_debug_original_relation": edge.get("relation"),
            }
            for edge in edge_records
        ]
    worker_inputs_by_edge_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    missing_worker_nodes = 0
    for edge in edge_records:
        edge_id = str(edge.get("edge_id") or "")
        image_node = nodes_by_id.get(str(edge.get("src_node_id") or ""))
        text_node = nodes_by_id.get(str(edge.get("dst_node_id") or ""))
        if not edge_id or image_node is None or text_node is None:
            missing_worker_nodes += 1
            continue
        worker_inputs_by_edge_id[edge_id] = (image_node, text_node)
    if missing_worker_nodes:
        print(
            "[verify_image_text_edges] "
            f"skipping_edges_with_missing_preloaded_nodes={missing_worker_nodes}",
            file=sys.stderr,
            flush=True,
        )
    edge_records = [
        edge for edge in edge_records
        if str(edge.get("edge_id") or "") in worker_inputs_by_edge_id
    ]
    for edge_id, record in checkpoint_results.items():
        if any(str(edge.get("edge_id") or "") == edge_id for edge in selected_edge_records):
            results_by_edge_id[edge_id] = record
    grounding_counts = _grounding_entity_counts(store, selected_image_node_ids)
    started = time.perf_counter()

    max_workers = max(1, int(args.workers))
    pending_edge_counts_by_image_node: dict[str, int] = {}
    records_by_image_node: dict[str, list[dict[str, Any]]] = {}
    for edge in edge_records:
        image_node_id = str(edge.get("src_node_id") or "")
        pending_edge_counts_by_image_node[image_node_id] = (
            pending_edge_counts_by_image_node.get(image_node_id, 0) + 1
        )
    total_scheduled_image_nodes = len(pending_edge_counts_by_image_node)
    completed_scheduled_image_nodes = 0
    if total_scheduled_image_nodes:
        print(
            "[verify_image_text_edges] "
            f"scheduled_image_nodes={total_scheduled_image_nodes} "
            f"scheduled_edges={len(edge_records)} workers={max_workers} "
            f"preloaded_nodes={len(nodes_by_id)}",
            file=sys.stderr,
            flush=True,
        )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_edge = {
            executor.submit(
                _verify_single_edge,
                edge=edge,
                image_node=worker_inputs_by_edge_id[str(edge.get("edge_id") or "")][0],
                text_node=worker_inputs_by_edge_id[str(edge.get("edge_id") or "")][1],
                args=args,
            ): edge
            for edge in edge_records
        }
        for future in as_completed(future_to_edge):
            edge = future_to_edge[future]
            try:
                record = future.result()
            except Exception as exc:
                record = _worker_exception_record(edge, args, exc)
            if record is None:
                image_node_id = str(edge.get("src_node_id") or "")
                pending_edge_counts_by_image_node[image_node_id] -= 1
                continue
            edge_id = str(record.get("edge_id") or "").strip()
            if edge_id:
                results_by_edge_id[edge_id] = record
            if checkpoint_path is not None:
                _append_checkpoint_result(checkpoint_path, record)
            image_node_id = str(edge.get("src_node_id") or "")
            node_records = records_by_image_node.setdefault(image_node_id, [])
            node_records.append(record)
            pending_edge_counts_by_image_node[image_node_id] -= 1
            if pending_edge_counts_by_image_node[image_node_id] == 0:
                completed_scheduled_image_nodes += 1
                _emit_image_node_complete_status(
                    image_node_id=image_node_id,
                    records=node_records,
                    completed_image_node_count=completed_scheduled_image_nodes,
                    total_image_node_count=total_scheduled_image_nodes,
                    elapsed_s=time.perf_counter() - started,
                )

    results = sorted(
        results_by_edge_id.values(),
        key=lambda item: (str(item.get("image_node_id") or ""), str(item.get("edge_id") or "")),
    )
    for record in results:
        if _should_drop(str(record.get("decision") or ""), args.drop_on):
            planned_removals.append(
                {
                    "edge_id": record.get("edge_id"),
                    "image_node_id": record.get("image_node_id"),
                    "text_node_id": record.get("text_node_id"),
                    "decision": record.get("decision"),
                    "error_type": record.get("error_type"),
                    "reason": record.get("reason"),
                }
            )
    contradict_count = sum(1 for record in results if record.get("decision") == "contradict")
    grounded_entity_count = grounding_counts["grounded_entity_count"]

    for record in results:
        if args.write_back and not args.dry_run:
            edge_record = store.get_edge(str(record.get("edge_id") or ""))
            if edge_record is None:
                continue
            edge_meta = dict(edge_record.get("metadata") or {})
            edge_meta["post_verify_image_text"] = {
                "decision": record.get("decision"),
                "error_type": record.get("error_type"),
                "confidence": record.get("confidence"),
                "reason": record.get("reason"),
                "evidence_for": record.get("evidence_for") or [],
                "evidence_against": record.get("evidence_against") or [],
                "judge_model_alias": record.get("judge_model_alias"),
                "prepare_model_alias": record.get("prepare_model_alias"),
                "kept_reference_image_count": record.get("kept_reference_image_count"),
                "verified_at_unix": time.time(),
            }
            edge_record["metadata"] = edge_meta
            if _should_drop(str(record.get("decision") or ""), args.drop_on):
                if args.hard_delete:
                    _delete_edge(store, str(record.get("edge_id") or ""))
                else:
                    store.upsert_edge(
                        _soft_delete_verified_edge(
                            edge_record,
                            decision=str(record.get("decision") or ""),
                        )
                    )
            else:
                store.upsert_edge(edge_record)

    if args.write_back and not args.dry_run and store.has_pending_writes():
        store.flush()

    payload = {
        "graph_dir": str(Path(args.graph_dir).resolve()),
        "image_node_id": args.image_node_id or None,
        "prepare_model": args.prepare_model,
        "judge_model": args.judge_model,
        "dry_run": bool(args.dry_run),
        "write_back": bool(args.write_back and not args.dry_run),
        "results_jsonl": str(checkpoint_path) if checkpoint_path else None,
        "resume": bool(args.resume),
        "reverify": bool(args.reverify),
        "relation_override": relation_override or None,
        "skip_graph_post_verified": not bool(args.reverify),
        "candidate_image_node_count": len(
            {str(edge.get("src_node_id") or "") for edge in all_edge_records if edge.get("src_node_id")}
        ),
        "pending_graph_post_verification_image_node_count": len(
            {str(edge.get("src_node_id") or "") for edge in graph_pending_edge_records if edge.get("src_node_id")}
        ),
        "sampled_image_node_count": len(selected_image_node_ids),
        "sampled_image_node_ids": selected_image_node_ids,
        "max_image_nodes": args.max_image_nodes,
        "random_seed": args.random_seed,
        "edge_count": len(results),
        "selected_candidate_edge_count": len(selected_edge_records),
        "candidate_edge_count": len(edge_records),
        "skipped_graph_post_verified_edge_count": 0 if args.reverify else len(graph_verified_edges),
        "skipped_checkpoint_edge_count": len(
            {str(edge.get("edge_id") or "") for edge in selected_edge_records} & checkpoint_edge_ids
        ),
        "grounded_entity_count": grounded_entity_count,
        "image_node_count_with_grounded_entities": grounding_counts["image_node_count_with_grounded_entities"],
        "contradict_count": contradict_count,
        "contradict_grounded_entity_ratio": (
            contradict_count / grounded_entity_count if grounded_entity_count else None
        ),
        "elapsed_s": time.perf_counter() - started,
        "drop_mode": "hard_delete" if args.hard_delete else "soft_delete",
        "planned_removal_count": len(planned_removals),
        "planned_removals": planned_removals,
        "results": results,
    }
    if args.image_node_id and not results:
        payload["empty_result_debug"] = _image_node_debug_stats(store, args.image_node_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    if args.dry_run and planned_removals:
        print("\n[verify_image_text_edges][dry-run] planned edge hides/deletions:", file=sys.stderr)
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
