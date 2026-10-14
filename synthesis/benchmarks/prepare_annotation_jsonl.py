#!/usr/bin/env python3
"""Materialize VQA samples into evidence-complete bilingual annotation JSONL.

``run_batch`` intentionally keeps the question-writing artifacts separate from
the graph.  This command joins the selected VQA samples back to the graph and
emits one compact, visualization-friendly record per question.  The graph
tables used by a synthesis run can be several gigabytes, so this module does
not construct :class:`JsonlGraphStore` (which is an in-memory store).  It first
collects the node/edge IDs referenced by the VQA directory and then streams
the graph JSONL tables once to find only those records.

The output schema is deliberately free of internal graph IDs.  Nodes are
identified inside a path by their zero-based position, and edges refer to
``source_index``/``target_index``.  The only sample identifiers retained at
the top level are ``question_id`` and ``path_id``.

Example::

    python -m synthesis.benchmarks.prepare_annotation_jsonl \
        --vqa-dir ../data/my_bench_test_1 \
        --graph-dir runs/0712_multi_seed_visual_test_8192_6 \
        --output-jsonl ../data/my_bench_test_1/annotation_ready.jsonl \
        --translation-model-alias gpt54_internal_azure \
        --translation-workers 16

The translation cache and audit/report files are written beside the output
JSONL.  Rerunning the command resumes completed translation requests and does
not download remote images.  Local image paths require the same OSS uploader
environment used by the synthesis project (``OSS_RELAY_URL`` or the
``OSS_*`` credentials).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Protocol
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest  # noqa: E402


LOGGER = logging.getLogger("prepare_annotation_jsonl")
SCHEMA_VERSION = "annotation_v1"
DEFAULT_TRANSLATION_MODEL = "gpt54_internal_azure"
DEFAULT_TRANSLATION_MAX_TOKENS = 8192
DEFAULT_TRANSLATION_WORKERS = 8
DEFAULT_TRANSLATION_SOURCE_MAX_CHARS = 12000
MAX_ERROR_TEXT = 2000


class ModelClient(Protocol):
    def generate(self, request: ModelRequest) -> Any:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield item


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _read_jsonl_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in _jsonl_records(path):
        value = item.get(key)
        if value is not None:
            result[str(value)] = item
    return result


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _first_text(*values: Any) -> str:
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return ""


def _normalise_url(value: Any) -> str | None:
    text = _safe_text(value)
    if text.startswith(("http://", "https://", "data:image/")):
        return text
    return None


def _is_remote_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:image/"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_values(value: Any) -> Iterator[Any]:
    """Yield values recursively; used only on small node/evidence records."""

    if isinstance(value, dict):
        for item in value.values():
            yield item
            yield from _nested_values(item)
    elif isinstance(value, list):
        for item in value:
            yield item
            yield from _nested_values(item)


def _extract_url_from_mapping(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _normalise_url(mapping.get(key))
        if value:
            return value
    for value in _nested_values(dict(mapping)):
        if isinstance(value, dict):
            continue
        candidate = _normalise_url(value)
        if candidate:
            return candidate
    return None


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _graph_table_paths(graph_dir: Path, table: str) -> list[Path]:
    """Return base and delta files in replay order."""

    paths: list[Path] = []
    for name in (f"{table}.jsonl", f"{table}.delta.jsonl"):
        candidate = graph_dir / name
        if candidate.exists():
            paths.append(candidate)
    return paths


def _scan_table_for_ids(
    paths: Iterable[Path],
    *,
    key: str,
    wanted: set[str],
    label: str,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Stream one graph table and retain only records whose key is wanted."""

    found: dict[str, dict[str, Any]] = {}
    record_count = 0
    parse_error_count = 0
    for path in paths:
        LOGGER.info("Scanning %s for %d referenced %s", path, len(wanted), label)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record_count += 1
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    parse_error_count += 1
                    LOGGER.warning("Ignoring invalid JSON in %s:%d", path, line_number)
                    continue
                if not isinstance(item, dict):
                    continue
                record_id = item.get(key)
                if record_id is not None and str(record_id) in wanted:
                    # Delta files are scanned after base files and therefore
                    # naturally implement the graph store's last-write-wins
                    # semantics.
                    found[str(record_id)] = item
    LOGGER.info("Found %d/%d referenced %s records", len(found), len(wanted), label)
    return found, record_count, parse_error_count


def _project_node(node: Mapping[str, Any]) -> dict[str, Any]:
    """Drop large raw page/debug fields after the streaming lookup.

    Text-node descriptions and image-node grounding prompts can be much larger
    than the fields needed by the annotation export.  Keeping those fields in
    ``GraphLookup`` would make a 900-sample export unnecessarily memory heavy.
    """

    result = {
        key: node.get(key)
        for key in (
            "node_id",
            "node_type",
            "title",
            "summary",
            "source",
            "asset_refs",
            "image_url",
            "source_page_url",
            "oss_uri",
            "caption",
            "storage_status",
        )
        if key in node
    }
    if str(node.get("node_type") or "") == "image":
        compact_variants: list[dict[str, Any]] = []
        for variant in node.get("image_variants") or []:
            if not isinstance(variant, dict):
                continue
            item = {
                key: variant.get(key)
                for key in ("image_url", "source_page_url", "is_primary", "variant_id")
                if key in variant
            }
            resolved = _as_dict(_as_dict(variant.get("metadata")).get("resolved_image"))
            if resolved:
                item["metadata"] = {
                    "resolved_image": {
                        key: resolved.get(key)
                        for key in (
                            "asset_uri",
                            "original_url",
                            "resolved_url",
                            "source_page_url",
                        )
                        if resolved.get(key)
                    }
                }
            compact_variants.append(item)
        result["image_variants"] = compact_variants
    return result


def _project_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: edge.get(key)
        for key in ("edge_id", "src_node_id", "dst_node_id", "relation", "status")
        if key in edge
    }
    source = _as_dict(edge.get("source"))
    if source.get("url"):
        result["source"] = {"url": source.get("url")}
    refs: list[dict[str, Any]] = []
    for ref in edge.get("evidence_refs") or []:
        if not isinstance(ref, dict):
            continue
        item = {
            key: ref.get(key)
            for key in ("evidence_id", "url", "quote")
            if ref.get(key) is not None
        }
        metadata = _as_dict(ref.get("metadata"))
        relation_info = _as_dict(metadata.get("relation_info"))
        compact_metadata: dict[str, Any] = {}
        if relation_info.get("evidence"):
            compact_metadata["relation_info"] = {"evidence": relation_info["evidence"]}
        if metadata.get("source_page_url"):
            compact_metadata["source_page_url"] = metadata["source_page_url"]
        if compact_metadata:
            item["metadata"] = compact_metadata
        refs.append(item)
    result["evidence_refs"] = refs
    return result


def _project_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: evidence.get(key)
        for key in ("evidence_id", "evidence_type", "content", "url")
        if key in evidence
    }
    metadata = _as_dict(evidence.get("metadata"))
    compact_metadata = {
        key: metadata.get(key)
        for key in ("source_page_url", "quote")
        if metadata.get(key) is not None
    }
    if compact_metadata:
        result["metadata"] = compact_metadata
    return result


def _project_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: asset.get(key)
        for key in ("asset_id", "uri", "original_url")
        if key in asset
    }
    metadata = _as_dict(asset.get("metadata"))
    if metadata.get("source_page_url"):
        result["metadata"] = {"source_page_url": metadata["source_page_url"]}
    return result


def _collect_referenced_ids(samples: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    for sample in samples:
        path = _as_dict(sample.get("path"))
        for node_id in path.get("node_ids") or []:
            if node_id:
                node_ids.add(str(node_id))
        for edge_id in path.get("edge_ids") or []:
            if edge_id:
                edge_ids.add(str(edge_id))
        # ``question_hop_chain`` can contain a normalized/merged hop whose
        # edge_id is written as ``edge_a|edge_b``.  The export follows the
        # actual path.edge_ids and should not treat that display-only alias as
        # a graph edge ID.  The construction hop_chain is sufficient as a
        # fallback for older VQA records without path IDs.
        for hop_key in ("hop_chain",):
            for hop in sample.get(hop_key) or []:
                if not isinstance(hop, dict):
                    continue
                for key in ("src_node_id", "dst_node_id"):
                    if hop.get(key):
                        node_ids.add(str(hop[key]))
                if hop.get("edge_id"):
                    edge_ids.add(str(hop["edge_id"]))
        for key in ("entry_hop", "question_target_ask", "target_ask"):
            value = sample.get(key)
            if not isinstance(value, dict):
                continue
            for node_key in ("src_node_id", "dst_node_id", "target_node_id", "ask_target_node_id"):
                if value.get(node_key):
                    node_ids.add(str(value[node_key]))
            if value.get("edge_id"):
                edge_ids.add(str(value["edge_id"]))
    return node_ids, edge_ids


@dataclass(slots=True)
class GraphLookup:
    nodes: dict[str, dict[str, Any]]
    edges: dict[str, dict[str, Any]]
    evidence: dict[str, dict[str, Any]]
    assets: dict[str, dict[str, Any]]
    report: dict[str, Any]

    @classmethod
    def load(cls, graph_dir: Path, samples: list[dict[str, Any]]) -> "GraphLookup":
        node_ids, edge_ids = _collect_referenced_ids(samples)
        nodes, node_count, node_errors = _scan_table_for_ids(
            _graph_table_paths(graph_dir, "nodes"),
            key="node_id",
            wanted=node_ids,
            label="node",
        )
        edges, edge_count, edge_errors = _scan_table_for_ids(
            _graph_table_paths(graph_dir, "edges"),
            key="edge_id",
            wanted=edge_ids,
            label="edge",
        )

        evidence_ids: set[str] = set()
        asset_ids: set[str] = set()
        for edge in edges.values():
            for ref in edge.get("evidence_refs") or []:
                if isinstance(ref, dict) and ref.get("evidence_id"):
                    evidence_ids.add(str(ref["evidence_id"]))
        for node in nodes.values():
            for ref in node.get("asset_refs") or []:
                if isinstance(ref, dict) and ref.get("asset_id"):
                    asset_ids.add(str(ref["asset_id"]))

        evidence, evidence_count, evidence_errors = _scan_table_for_ids(
            _graph_table_paths(graph_dir, "evidence"),
            key="evidence_id",
            wanted=evidence_ids,
            label="evidence",
        )
        assets, asset_count, asset_errors = _scan_table_for_ids(
            _graph_table_paths(graph_dir, "assets"),
            key="asset_id",
            wanted=asset_ids,
            label="asset",
        )
        nodes = {key: _project_node(value) for key, value in nodes.items()}
        edges = {key: _project_edge(value) for key, value in edges.items()}
        evidence = {key: _project_evidence(value) for key, value in evidence.items()}
        assets = {key: _project_asset(value) for key, value in assets.items()}
        report = {
            "graph_dir": str(graph_dir.resolve()),
            "referenced_node_ids": len(node_ids),
            "found_node_ids": len(nodes),
            "missing_node_ids": sorted(node_ids - set(nodes)),
            "referenced_edge_ids": len(edge_ids),
            "found_edge_ids": len(edges),
            "missing_edge_ids": sorted(edge_ids - set(edges)),
            "referenced_evidence_ids": len(evidence_ids),
            "found_evidence_ids": len(evidence),
            "missing_evidence_ids": sorted(evidence_ids - set(evidence)),
            "referenced_asset_ids": len(asset_ids),
            "found_asset_ids": len(assets),
            "scan_records": {
                "nodes": node_count,
                "edges": edge_count,
                "evidence": evidence_count,
                "assets": asset_count,
            },
            "scan_parse_errors": {
                "nodes": node_errors,
                "edges": edge_errors,
                "evidence": evidence_errors,
                "assets": asset_errors,
            },
        }
        return cls(nodes=nodes, edges=edges, evidence=evidence, assets=assets, report=report)


@dataclass(slots=True)
class TextSegments:
    """Build bilingual placeholders and retain the source text for one record."""

    record_key: str
    values: dict[str, str] = field(default_factory=dict)

    def wrap(self, field_name: str, value: Any) -> dict[str, str] | str:
        text = _safe_text(value)
        if not text:
            return ""
        segment_id = f"{self.record_key}:{field_name}"
        self.values[segment_id] = text
        return {"__translation_id": segment_id, "en": text}


def _parse_model_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    # Parse the response itself first: valid JSON values may contain Markdown
    # fences, which must not be mistaken for the response's outer wrapper.
    candidates = [text]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"translation model did not return a JSON object: {text[:500]}")


def _extract_translations(payload: Mapping[str, Any], expected: set[str]) -> dict[str, str]:
    translations = payload.get("translations")
    if not isinstance(translations, dict):
        translations = payload.get("translation")
    if not isinstance(translations, dict):
        raise ValueError("translation JSON must contain an object named 'translations'")
    result: dict[str, str] = {}
    for key, value in translations.items():
        key_text = str(key)
        if key_text in expected:
            result[key_text] = _safe_text(value)
    missing = expected - set(result)
    if missing:
        raise ValueError(f"translation JSON is missing {len(missing)} segment(s)")
    return result


def _translation_prompt(segments: Mapping[str, str], max_source_chars: int) -> str:
    payload = []
    for segment_id, source in segments.items():
        clipped = source
        if len(clipped) > max_source_chars:
            clipped = clipped[:max_source_chars] + "\n[English source clipped only for translation input]"
        payload.append({"id": segment_id, "english": clipped})
    return (
        "Translate every English field below into natural, faithful Simplified Chinese. "
        "Return JSON only in the exact shape {\"translations\": {segment_id: chinese_text}}. "
        "Do not omit any id. Preserve proper names, numbers, quotation marks, markdown, "
        "and URLs whenever they occur. Do not answer or explain the question; translate only.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _translate_one(
    record_key: str,
    segments: Mapping[str, str],
    *,
    model_alias: str,
    max_tokens: int,
    max_source_chars: int,
    worker: ModelClient,
    attempts: int,
) -> dict[str, Any]:
    if not segments:
        return {"record_key": record_key, "status": "complete", "translations": {}}
    prompt = _translation_prompt(segments, max_source_chars)
    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = worker.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(
                            role="system",
                            content="You are a precise English-to-Simplified-Chinese translator for an annotation dataset.",
                        ),
                        ModelMessage(role="user", content=prompt),
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    metadata={"trace_label": "annotation_jsonl_translation"},
                )
            )
            parsed = _parse_model_json(str(getattr(response, "content", response) or ""))
            translations = _extract_translations(parsed, set(segments))
            return {
                "record_key": record_key,
                "status": "complete",
                "translations": translations,
                "attempts": attempt,
            }
        except Exception as exc:  # retry parse failures as well as transient errors
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max(1, attempts):
                time.sleep(min(2.0 * attempt, 10.0))
    return {
        "record_key": record_key,
        "status": "error",
        "translations": {},
        "attempts": max(1, attempts),
        "error": last_error[:MAX_ERROR_TEXT],
    }


def _apply_translations(value: Any, translations: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__translation_id", "en"}:
            segment_id = str(value["__translation_id"])
            return {"en": value["en"], "zh": translations.get(segment_id, "")}
        return {key: _apply_translations(item, translations) for key, item in value.items()}
    if isinstance(value, list):
        return [_apply_translations(item, translations) for item in value]
    return value


def _source_url(node: Mapping[str, Any]) -> str | None:
    source = _as_dict(node.get("source"))
    metadata = _as_dict(node.get("metadata"))
    return _first_url(
        source.get("url"),
        node.get("wiki_url"),
        node.get("url"),
        metadata.get("source_url"),
    )


def _first_url(*values: Any) -> str | None:
    for value in values:
        url = _normalise_url(value)
        if url:
            return url
    return None


def _node_name(node: Mapping[str, Any]) -> str:
    node_type = str(node.get("node_type") or "").lower()
    if node_type == "image":
        title = _first_text(node.get("caption"), node.get("title"), node.get("summary"))
        if title.lower().startswith("image:"):
            title = title.split(":", 1)[1].strip()
        return title
    return _first_text(node.get("title"), node.get("name"), node.get("summary"), node.get("description"))


def _resolve_local_path(value: str, roots: Iterable[Path]) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        return None
    for root in roots:
        resolved = root / candidate
        if resolved.is_file():
            return resolved.resolve()
    return None


class LocalImageUploader:
    """Upload local image paths through the repository's existing OSS ABI."""

    def __init__(
        self,
        cache_path: Path,
        *,
        user: str,
        mode: str,
        retries: int = 5,
        resume: bool = True,
    ) -> None:
        self.cache_path = cache_path
        self.user = user
        self.mode = mode
        self.retries = max(1, retries)
        self._cache = _read_jsonl_map(cache_path, "cache_key") if resume else {}
        self._lock = threading.Lock()

    @staticmethod
    def configured() -> bool:
        """Return whether the repository uploader has enough OSS settings."""

        if _safe_text(os.environ.get("OSS_RELAY_URL")):
            return True
        return all(
            _safe_text(os.environ.get(name))
            for name in (
                "OSS_ACCESS_KEY_ID",
                "OSS_ACCESS_KEY_SECRET",
                "OSS_ENDPOINT",
                "OSS_BUCKET_NAME",
            )
        )

    @staticmethod
    def _cache_key(path: Path) -> str:
        stat = path.stat()
        return _stable_hash({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})

    def upload(self, path: Path) -> str:
        cache_key = self._cache_key(path)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached and _normalise_url(cached.get("url")):
            return str(cached["url"])

        try:
            from opensearch_vl.opensearch_infer.upload import upload_cos
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(f"cannot import OSS uploader: {exc}") from exc

        error = ""
        # The uploader's object key is based on the filename.  Include a
        # content/path-derived suffix so two local files named ``image.jpg``
        # cannot overwrite each other in OSS.
        filename = f"{path.stem}_{cache_key[:16]}{path.suffix or '.bin'}"
        for attempt in range(1, self.retries + 1):
            try:
                object_key, public_url = upload_cos(
                    str(path),
                    filename,
                    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    self.mode,
                    self.user,
                    use_direct_url=True,
                )
                url = _normalise_url(public_url)
                if not url:
                    raise RuntimeError(f"OSS uploader returned no public URL (key={object_key!r})")
                item = {
                    "cache_key": cache_key,
                    "local_path": str(path),
                    "url": url,
                    "object_key": object_key,
                    "attempts": attempt,
                }
                with self._lock:
                    self._cache[cache_key] = item
                    _append_jsonl(self.cache_path, item)
                return url
            except Exception as exc:  # pragma: no cover - exercised with real OSS
                error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(min(float(attempt), 5.0))
        raise RuntimeError(error or "OSS upload failed")


def _image_urls(
    node: Mapping[str, Any],
    *,
    graph: GraphLookup,
    uploader: LocalImageUploader | None,
    path_roots: Iterable[Path],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    metadata = _as_dict(node.get("metadata"))
    primary_variant: dict[str, Any] = {}
    variants = node.get("image_variants") or []
    for variant in variants:
        if isinstance(variant, dict) and variant.get("is_primary"):
            primary_variant = variant
            break
    if not primary_variant and isinstance(variants, list):
        primary_variant = next((item for item in variants if isinstance(item, dict)), {})
    variant_metadata = _as_dict(primary_variant.get("metadata"))
    resolved = _as_dict(variant_metadata.get("resolved_image"))

    image_value = _first_text(
        node.get("image_url"),
        node.get("oss_uri"),
        primary_variant.get("image_url"),
        resolved.get("asset_uri"),
    )
    original_url = _first_url(
        resolved.get("original_url"),
        resolved.get("resolved_url"),
        primary_variant.get("image_url"),
        node.get("original_url"),
    )
    source_page_url = _first_url(
        node.get("source_page_url"),
        resolved.get("source_page_url"),
        primary_variant.get("source_page_url"),
    )
    image_url: str | None = _normalise_url(image_value)
    if image_url is None and image_value:
        local_path = _resolve_local_path(image_value, path_roots)
        if local_path is None:
            warnings.append(f"local_image_not_found:{image_value}")
        elif uploader is None:
            warnings.append(f"local_image_upload_not_configured:{local_path}")
        else:
            try:
                image_url = uploader.upload(local_path)
            except Exception as exc:
                warnings.append(f"local_image_upload_failed:{local_path}:{exc}")
    if image_url and not original_url:
        original_url = image_url

    # If the primary image is remote but the variant's URL is a different
    # original URL, keep both.  The asset URL is the one annotators can render;
    # the original URL is the provenance link.
    image: dict[str, Any] = {
        "image_url": image_url or "",
        "original_url": original_url or "",
        "source_page_url": source_page_url or "",
    }
    # Asset records are a fallback for older graph writers that did not copy
    # original_url/source_page_url onto the image node.
    asset_refs = node.get("asset_refs") or []
    for ref in asset_refs:
        asset_id = ref.get("asset_id") if isinstance(ref, dict) else None
        asset = graph.assets.get(str(asset_id)) if asset_id else None
        if not asset:
            continue
        if not image["image_url"]:
            image["image_url"] = _first_url(asset.get("uri")) or ""
        if not image["original_url"]:
            image["original_url"] = _first_url(asset.get("original_url")) or ""
        if not image["source_page_url"]:
            image["source_page_url"] = _first_url(_as_dict(asset.get("metadata")).get("source_page_url")) or ""
    return image, warnings


def _evidence_text(ref: Mapping[str, Any], evidence: Mapping[str, Any] | None) -> tuple[str, str]:
    ref_metadata = _as_dict(ref.get("metadata"))
    evidence_metadata = _as_dict((evidence or {}).get("metadata"))
    relation_info = _as_dict(ref_metadata.get("relation_info"))
    text = _first_text(
        ref.get("quote"),
        (evidence or {}).get("content"),
        relation_info.get("evidence"),
        evidence_metadata.get("quote"),
    )
    relation_excerpt = _first_text(relation_info.get("evidence"), evidence_metadata.get("quote"))
    return text, relation_excerpt


def _edge_evidence(
    edge: Mapping[str, Any] | None,
    *,
    graph: GraphLookup,
    segments: TextSegments,
    prefix: str,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not edge:
        return []
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, ref in enumerate(edge.get("evidence_refs") or []):
        if not isinstance(ref, dict):
            continue
        evidence_id = str(ref.get("evidence_id") or "")
        evidence = graph.evidence.get(evidence_id)
        text, relation_excerpt = _evidence_text(ref, evidence)
        if evidence_id and evidence is None and not text:
            if warnings is not None:
                warnings.append(f"evidence_not_found:{evidence_id}")
            continue
        url = _first_url(
            ref.get("url"),
            (evidence or {}).get("url"),
            _as_dict((evidence or {}).get("metadata")).get("source_url"),
        )
        ref_metadata = _as_dict(ref.get("metadata"))
        evidence_metadata = _as_dict((evidence or {}).get("metadata"))
        source_page_url = _first_url(
            evidence_metadata.get("source_page_url"),
            ref_metadata.get("source_page_url"),
            url,
        )
        dedupe_key = (text, url or "")
        if not text and not url:
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        item: dict[str, Any] = {
            "text": segments.wrap(f"{prefix}.evidence[{index}].text", text),
            "url": url or "",
            "source_page_url": source_page_url or "",
        }
        if relation_excerpt and relation_excerpt != text:
            item["relation_excerpt"] = segments.wrap(
                f"{prefix}.evidence[{index}].relation_excerpt", relation_excerpt
            )
        if evidence_id:
            item["evidence_type"] = str((evidence or {}).get("evidence_type") or "")
        result.append(item)
    return result


def _node_wiki_url(node: Mapping[str, Any]) -> str:
    node_type = str(node.get("node_type") or "").lower()
    source_url = (
        _first_url(node.get("source_page_url"))
        if node_type == "image"
        else _source_url(node)
    )
    if source_url and "wikipedia.org" in urlparse(source_url).netloc:
        return source_url
    value = _first_url(node.get("wiki_url"))
    return value or ""


def _node_record(
    node: Mapping[str, Any] | None,
    *,
    index: int,
    segments: TextSegments,
    graph: GraphLookup,
    uploader: LocalImageUploader | None,
    path_roots: Iterable[Path],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    node = node or {}
    node_type = str(node.get("node_type") or "unknown")
    name = _node_name(node)
    if not name:
        warnings.append(f"node_name_missing:node_index={index}")
    record: dict[str, Any] = {
        "index": index,
        "node_type": node_type,
        "name": segments.wrap(f"search_chain.nodes[{index}].name", name),
        "wiki_url": _node_wiki_url(node),
    }
    if node_type == "image":
        image, image_warnings = _image_urls(
            node,
            graph=graph,
            uploader=uploader,
            path_roots=path_roots,
        )
        record.update(image)
        warnings.extend(image_warnings)
        if not record["image_url"]:
            warnings.append(f"image_url_missing:node_index={index}")
    return record, warnings


def _path_node_ids(sample: Mapping[str, Any]) -> list[str]:
    path = _as_dict(sample.get("path"))
    node_ids = [str(value) for value in path.get("node_ids") or [] if value]
    if node_ids:
        return node_ids
    hops = [item for item in sample.get("hop_chain") or [] if isinstance(item, dict)]
    if not hops:
        return []
    result = [str(hops[0].get("src_node_id") or "")]
    result.extend(str(item.get("dst_node_id") or "") for item in hops)
    return [item for item in result if item]


def _path_edge_ids(sample: Mapping[str, Any]) -> list[str]:
    path = _as_dict(sample.get("path"))
    edge_ids = [str(value) for value in path.get("edge_ids") or [] if value]
    if edge_ids:
        return edge_ids
    return [str(item.get("edge_id")) for item in sample.get("hop_chain") or [] if isinstance(item, dict) and item.get("edge_id")]


def _hop_for_edge(sample: Mapping[str, Any], edge_id: str, fallback_index: int) -> dict[str, Any]:
    for key in ("hop_chain", "question_hop_chain"):
        for hop in sample.get(key) or []:
            if isinstance(hop, dict) and str(hop.get("edge_id") or "") == edge_id:
                return hop
    for hop in sample.get("hop_chain") or []:
        if isinstance(hop, dict) and int(hop.get("hop_index") or -1) == fallback_index:
            return hop
    return {}


def _build_record(
    question: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    graph: GraphLookup,
    uploader: LocalImageUploader | None,
    path_roots: Iterable[Path],
) -> tuple[dict[str, Any], TextSegments, list[str]]:
    question_id = _first_text(question.get("question_id"), sample.get("question_id"))
    path = _as_dict(sample.get("path"))
    path_id = _first_text(path.get("path_id"), question.get("path_id"))
    if not question_id or not path_id:
        raise ValueError("each VQA sample must have question_id and path_id")
    record_key = f"{question_id}::{path_id}"
    segments = TextSegments(record_key)
    warnings: list[str] = []
    node_ids = _path_node_ids(sample)
    edge_ids = _path_edge_ids(sample)
    if not node_ids:
        warnings.append("path_nodes_missing")
    if not edge_ids:
        warnings.append("path_edges_missing")
    if len(edge_ids) != max(0, len(node_ids) - 1):
        warnings.append(f"path_length_mismatch:nodes={len(node_ids)},edges={len(edge_ids)}")

    nodes: list[dict[str, Any]] = []
    node_index_by_id: dict[str, int] = {}
    for index, node_id in enumerate(node_ids):
        node_index_by_id[node_id] = index
        node_record, node_warnings = _node_record(
            graph.nodes.get(node_id),
            index=index,
            segments=segments,
            graph=graph,
            uploader=uploader,
            path_roots=path_roots,
        )
        nodes.append(node_record)
        if node_id not in graph.nodes:
            warnings.append(f"node_not_found:{node_id}")
        warnings.extend(node_warnings)

    edges: list[dict[str, Any]] = []
    for index, edge_id in enumerate(edge_ids):
        edge = graph.edges.get(edge_id)
        hop = _hop_for_edge(sample, edge_id, index)
        src_id = str((edge or {}).get("src_node_id") or hop.get("src_node_id") or "")
        dst_id = str((edge or {}).get("dst_node_id") or hop.get("dst_node_id") or "")
        source_index = node_index_by_id.get(src_id, index if index < len(nodes) else None)
        target_index = node_index_by_id.get(dst_id, index + 1 if index + 1 < len(nodes) else None)
        if source_index is None or target_index is None:
            warnings.append(f"edge_endpoint_not_found:{edge_id}")
            continue
        relation = _first_text((edge or {}).get("relation"), hop.get("relation"))
        relation_value = segments.wrap(f"search_chain.edges[{index}].relation", relation)
        item: dict[str, Any] = {
            "source_index": source_index,
            "target_index": target_index,
            "relation": relation_value,
            "evidence": _edge_evidence(
                edge,
                graph=graph,
                segments=segments,
                prefix=f"search_chain.edges[{index}]",
                warnings=warnings,
            ),
        }
        if not edge:
            warnings.append(f"edge_not_found:{edge_id}")
        source_url = _first_url(_as_dict((edge or {}).get("source")).get("url"))
        if source_url:
            item["source_url"] = source_url
        edges.append(item)

    def node_ref(node_index: int | None) -> dict[str, Any]:
        if node_index is None or node_index < 0 or node_index >= len(nodes):
            return {"node_index": -1}
        node = nodes[node_index]
        return {
            "node_index": node_index,
            "name": node.get("name", ""),
            "wiki_url": node.get("wiki_url", ""),
            "node_type": node.get("node_type", ""),
            **({key: node[key] for key in ("image_url", "original_url", "source_page_url") if node.get(key)} if node.get("node_type") == "image" else {}),
        }

    entry_hop = sample.get("entry_hop")
    if not isinstance(entry_hop, dict):
        entry_hop = _hop_for_edge(sample, edge_ids[0], 0) if edge_ids else {}
    entry_edge_id = str(entry_hop.get("edge_id") or (edge_ids[0] if edge_ids else ""))
    entry_edge = graph.edges.get(entry_edge_id)
    entry_src = str((entry_edge or {}).get("src_node_id") or entry_hop.get("src_node_id") or (node_ids[0] if node_ids else ""))
    entry_dst = str((entry_edge or {}).get("dst_node_id") or entry_hop.get("dst_node_id") or (node_ids[1] if len(node_ids) > 1 else ""))
    entry_src_index = node_index_by_id.get(entry_src)
    entry_dst_index = node_index_by_id.get(entry_dst)
    opening_hop = {
        "source": node_ref(entry_src_index),
        "target": node_ref(entry_dst_index),
        "relation": segments.wrap("opening_hop.relation", _first_text((entry_edge or {}).get("relation"), entry_hop.get("relation"))),
        "statement": segments.wrap("opening_hop.statement", entry_hop.get("statement")),
        "evidence": _edge_evidence(
            entry_edge,
            graph=graph,
            segments=segments,
            prefix="opening_hop",
            warnings=warnings,
        ),
    }
    if not entry_edge and entry_edge_id:
        warnings.append(f"opening_edge_not_found:{entry_edge_id}")

    final_edge_id = edge_ids[-1] if edge_ids else ""
    final_edge = graph.edges.get(final_edge_id)
    final_hop = _hop_for_edge(sample, final_edge_id, max(0, len(edge_ids) - 1)) if final_edge_id else {}
    target_id = str(
        path.get("target_node_id")
        or (final_edge or {}).get("dst_node_id")
        or final_hop.get("dst_node_id")
        or (node_ids[-1] if node_ids else "")
    )
    target_index = node_index_by_id.get(target_id, len(nodes) - 1 if nodes else None)
    target_ask = sample.get("target_ask")
    if not isinstance(target_ask, dict):
        target_ask = _as_dict(sample.get("question_target_ask"))
    ask_target = _first_text(target_ask.get("ask_target"), sample.get("final_ask"))
    answer = _first_text(target_ask.get("answer"), question.get("answer"))
    final_ask = {
        "ask_target": segments.wrap("final_ask.ask_target", ask_target),
        "target": node_ref(target_index),
        "answer": segments.wrap("final_ask.answer", answer),
        "answer_evidence": _edge_evidence(
            final_edge,
            graph=graph,
            segments=segments,
            prefix="final_ask",
            warnings=warnings,
        ),
    }
    if target_ask.get("question_type"):
        final_ask["question_type"] = str(target_ask["question_type"])
    if not final_ask["answer_evidence"]:
        warnings.append("final_answer_evidence_missing")

    input_images: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if node.get("node_type") == "image":
            input_images.append({
                "node_index": index,
                "image_url": node.get("image_url", ""),
                "original_url": node.get("original_url", ""),
                "source_page_url": node.get("source_page_url", ""),
            })

    record = {
        "schema_version": SCHEMA_VERSION,
        "question_id": question_id,
        "path_id": path_id,
        "question": segments.wrap("question", question.get("question")),
        "answer": segments.wrap("answer", _first_text(question.get("answer"), target_ask.get("answer"))),
        "input_images": input_images,
        "search_chain": {
            "hop_count": len(edge_ids),
            "nodes": nodes,
            "edges": edges,
        },
        "opening_hop": opening_hop,
        "final_ask": final_ask,
    }
    return record, segments, sorted(set(warnings))


def _load_vqa(vqa_dir: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    if not questions_path.exists() or not samples_path.exists():
        raise FileNotFoundError(f"{vqa_dir} must contain questions.jsonl and samples.jsonl")
    questions = list(_jsonl_records(questions_path))
    samples = list(_jsonl_records(samples_path))
    questions_by_sample_id = {
        str(item.get("sample_id")): item
        for item in questions
        if item.get("sample_id") is not None
    }
    joined: list[dict[str, Any]] = []
    joined_samples: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        question = questions_by_sample_id.get(sample_id)
        if question is None:
            LOGGER.warning("No question record for sample_id=%s", sample_id)
            continue
        joined.append(question)
        joined_samples.append(sample)
        if limit is not None and len(joined) >= limit:
            break
    return joined, joined_samples


def _load_translation_cache(path: Path) -> dict[str, dict[str, Any]]:
    return _read_jsonl_map(path, "record_key")


def _translate_records(
    records: list[tuple[dict[str, Any], TextSegments, list[str]]],
    *,
    cache_path: Path,
    model_alias: str,
    workers: int,
    max_tokens: int,
    max_source_chars: int,
    attempts: int,
    resume: bool,
    worker: ModelClient,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = _load_translation_cache(cache_path) if resume else {}
    pending: list[tuple[str, Mapping[str, str]]] = []
    results: dict[str, dict[str, Any]] = {}
    for _, segments, _ in records:
        cached = cache.get(segments.record_key)
        if cached and cached.get("status") == "complete" and cached.get("translations"):
            results[segments.record_key] = cached
        else:
            pending.append((segments.record_key, dict(segments.values)))
    LOGGER.info("Translation requests: %d pending, %d cached", len(pending), len(results))

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="translate") as executor:
            futures = {
                executor.submit(
                    _translate_one,
                    record_key,
                    segment_values,
                    model_alias=model_alias,
                    max_tokens=max_tokens,
                    max_source_chars=max_source_chars,
                    worker=worker,
                    attempts=attempts,
                ): record_key
                for record_key, segment_values in pending
            }
            for future in as_completed(futures):
                record_key = futures[future]
                try:
                    item = future.result()
                except Exception as exc:  # defensive: _translate_one already catches
                    item = {
                        "record_key": record_key,
                        "status": "error",
                        "translations": {},
                        "error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_TEXT],
                    }
                results[record_key] = item
                _append_jsonl(cache_path, item)

    output: list[dict[str, Any]] = []
    error_count = 0
    for raw_record, segments, warnings in records:
        translation = results.get(segments.record_key, {})
        translations = translation.get("translations") if isinstance(translation, dict) else {}
        translations = translations if isinstance(translations, dict) else {}
        output_record = _apply_translations(raw_record, translations)
        record_warnings = list(warnings)
        if translation.get("status") != "complete":
            error_count += 1
            record_warnings.append(f"translation_failed:{translation.get('error', 'unknown error')}")
        if record_warnings:
            output_record["record_status"] = "incomplete"
            output_record["warnings"] = sorted(set(record_warnings))
        else:
            output_record["record_status"] = "complete"
        output.append(output_record)
    return output, {"pending": len(pending), "cached": len(results) - len(pending), "errors": error_count}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--translation-model-alias",
        default=os.environ.get("ANNOTATION_TRANSLATION_MODEL_ALIAS", DEFAULT_TRANSLATION_MODEL),
    )
    parser.add_argument("--translation-workers", type=int, default=DEFAULT_TRANSLATION_WORKERS)
    parser.add_argument("--translation-max-tokens", type=int, default=DEFAULT_TRANSLATION_MAX_TOKENS)
    parser.add_argument("--translation-source-max-chars", type=int, default=DEFAULT_TRANSLATION_SOURCE_MAX_CHARS)
    parser.add_argument("--translation-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples (for smoke tests).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore the translation/upload caches.")
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Testing-only: emit English fields with empty zh fields without calling an LLM.",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--oss-user", default=os.environ.get("COS_USERID", "deepresearch-benchmark"))
    parser.add_argument("--oss-mode", default="annotation_jsonl")
    parser.add_argument("--oss-retries", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None, *, model_worker: ModelClient | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    vqa_dir = args.vqa_dir.resolve()
    graph_dir = args.graph_dir.resolve()
    output_jsonl = args.output_jsonl.resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    questions, samples = _load_vqa(vqa_dir, args.limit)
    LOGGER.info("Loaded %d questions and %d samples", len(questions), len(samples))
    graph = GraphLookup.load(graph_dir, samples)
    _write_json(output_jsonl.with_name(output_jsonl.stem + ".graph_lookup_report.json"), graph.report)

    cache_path = output_jsonl.with_name(output_jsonl.stem + ".translation_cache.jsonl")
    upload_cache_path = output_jsonl.with_name(output_jsonl.stem + ".image_upload_cache.jsonl")
    uploader = None
    if LocalImageUploader.configured():
        uploader = LocalImageUploader(
            upload_cache_path,
            user=args.oss_user,
            mode=args.oss_mode,
            retries=args.oss_retries,
            resume=not args.no_resume,
        )
    else:
        LOGGER.info("OSS is not configured; local image paths will be reported as incomplete")
    path_roots = (vqa_dir, graph_dir, Path.cwd())
    raw_records: list[tuple[dict[str, Any], TextSegments, list[str]]] = []
    build_errors: list[dict[str, Any]] = []
    for question, sample in zip(questions, samples):
        try:
            raw_records.append(
                _build_record(
                    question,
                    sample,
                    graph=graph,
                    uploader=uploader,
                    path_roots=path_roots,
                )
            )
        except Exception as exc:
            key = f"{question.get('question_id', '')}::{_as_dict(sample.get('path')).get('path_id', '')}"
            build_errors.append({"record_key": key, "error": f"{type(exc).__name__}: {exc}"})
            LOGGER.exception("Failed to build %s", key)

    if args.skip_translation:
        output_records = []
        for raw_record, _, warnings in raw_records:
            output_record = _apply_translations(raw_record, {})
            combined = list(warnings) + ["translation_skipped"]
            output_record["record_status"] = "incomplete"
            output_record["warnings"] = sorted(set(combined))
            output_records.append(output_record)
        translation_report = {"skipped": True, "pending": 0, "cached": 0, "errors": 0}
    else:
        output_records, translation_report = _translate_records(
            raw_records,
            cache_path=cache_path,
            model_alias=args.translation_model_alias,
            workers=args.translation_workers,
            max_tokens=args.translation_max_tokens,
            max_source_chars=args.translation_source_max_chars,
            attempts=args.translation_attempts,
            resume=not args.no_resume,
            worker=model_worker or LLM_WORKER,
        )

    _write_jsonl(output_jsonl, output_records)
    incomplete_count = sum(item.get("record_status") != "complete" for item in output_records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not build_errors and incomplete_count == 0 else "incomplete",
        "created_at": _utc_now(),
        "vqa_dir": str(vqa_dir),
        "graph_dir": str(graph_dir),
        "output_jsonl": str(output_jsonl),
        "question_count": len(questions),
        "emitted_count": len(output_records),
        "build_error_count": len(build_errors),
        "build_errors": build_errors[:100],
        "incomplete_count": incomplete_count,
        "translation": {
            "model_alias": args.translation_model_alias,
            "workers": args.translation_workers,
            **translation_report,
        },
        "paths": {
            "translation_cache": str(cache_path),
            "image_upload_cache": str(upload_cache_path),
            "graph_lookup_report": str(output_jsonl.with_name(output_jsonl.stem + ".graph_lookup_report.json")),
        },
    }
    _write_json(output_jsonl.with_name(output_jsonl.stem + ".report.json"), report)
    LOGGER.info("Wrote %d records to %s (%d incomplete)", len(output_records), output_jsonl, incomplete_count)
    if args.fail_on_incomplete and (build_errors or incomplete_count):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
