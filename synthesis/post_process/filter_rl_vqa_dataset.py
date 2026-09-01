#!/usr/bin/env python3
"""Filter and balance VQA tasks for online agentic RL.

The script intentionally reuses the existing OpenSearch-VL inference stack:

1. ``run_infer_no_tools`` is run four times.  Tasks with no-tools pass@4 are
   treated as shortcuts and removed.
2. ``run_infer`` is run four times on the remaining tasks.  The existing
   ``eval_infer_with_llm`` judge evaluates completed trajectories.  Incomplete
   trajectories, including max-turn cases, are retained in the compact audit
   manifest; they are not directly filtered by trajectory length.
3. A separate LLM judge audits question/evidence consistency and emits
   accept/review/reject. Only explicit rejects are removed.
4. The remaining tasks are bucketed into easy/medium/hard and sampled with a
   default 20/50/30 target ratio.

The pipeline is resumable by default.  Completed sample stages, trajectory
evaluations, and quality-judge records are reused.  The scheduler is sample
level: one worker advances one sample through all of its no-tools repeats,
tool repeats, and quality judging, then immediately takes another sample.
With the API backend, ``--parallel-workers`` multiplied by
``--repeat-workers`` is the number of sample pipelines allowed concurrently.

The raw VQA directory is never modified.  The output directory is itself a
VQA directory: its ``questions.jsonl`` and ``samples.jsonl`` contain the
selected subset, while inference artifacts and compact audit files live
alongside them.  Input images are materialised once under ``images/`` and the
subset JSONL points to those local files, so the directory can be sent directly
to the existing VQA-to-RL converter.

Example::

    python synthesis/post_process/filter_rl_vqa_dataset.py \
        --vqa-dir runs/.../vqa/0803_batch_1 \
        --output-dir runs/.../vqa/0803_batch_1/rl_filter_8b \
        --model 8b \
        --backend api \
        --base-url http://localhost:8001/v1 \
        --served-model-name Qwen3-VL-8B-Instruct \
        --quality-judge-model-alias gpt54_internal_azure \
        --checkpoint /path/to/checkpoint \
        --gpus 0 \
        --parallel-workers 8 \
        --repeat-workers 2

Then convert the filtered VQA directory with the existing converter::

    python RL/rllm/vision_deepresearch_async_workflow/data_prepare/convert_vqa_run_batch.py \
        --run_dir runs/.../vqa/0803_batch_1/rl_filter_8b \
        --output_jsonl /tmp/rl_selected.jsonl \
        --image_mode download
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
OPENSEARCH_VL_ROOT = ROOT / "opensearch_vl"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("filter_rl_vqa_dataset")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _jsonl_write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _write_parquet_via_local_tmp(dataframe: Any, destination: Path) -> None:
    """Write a parquet file locally, then move it to the requested path.

    The project directory can be an HDFS-FUSE mount.  PyArrow's parquet writer
    performs local-file operations (including truncate/random writes) that are
    not reliably supported by that mount and may fail with ``EBUSY``.  Writing
    to a real local filesystem first avoids that limitation.  ``shutil.move``
    falls back to a byte-wise copy when /tmp and the destination are on
    different filesystems, which is the expected case here.
    """

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="filter_rl_vqa_", dir="/tmp"))
    local_path = tmp_dir / destination.name
    try:
        dataframe.to_parquet(local_path, index=False)
        logger.info("Moving completed local parquet %s -> %s", local_path, destination)
        shutil.move(str(local_path), str(destination))
        if not destination.is_file():
            raise OSError(f"Parquet move completed without a destination file: {destination}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _question_text(record: dict[str, Any]) -> str:
    return _first_text(
        record,
        ("final_question", "question", "enhanced_question", "polished_question", "draft_question"),
    )


def _safe_id(record: dict[str, Any], source_index: int) -> str:
    value = _first_text(record, ("question_id", "id", "sample_id"))
    return value or f"q_{source_index + 1:06d}"


def _sample_by_id(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("sample_id")): item
        for item in samples
        if item.get("sample_id") is not None
    }


def _image_values(question: dict[str, Any], sample: dict[str, Any]) -> list[Any]:
    for key in ("images", "image_urls", "image_url"):
        value = question.get(key)
        if value:
            return value if isinstance(value, list) else [value]

    value = sample.get("input_image_url") if isinstance(sample, dict) else None
    if value:
        return value if isinstance(value, list) else [value]
    return []


def _mime_for_path(path: Path) -> str:
    guessed = mimetypes.guess_type(str(path))[0]
    return guessed if guessed and guessed.startswith("image/") else "image/png"


def _image_entry_for_inference(value: Any) -> dict[str, Any] | None:
    """Normalize an image to the shape understood by ``_bootstrap_images``."""

    if isinstance(value, dict):
        url = value.get("url") or value.get("image_url") or value.get("cos_url")
        payload = value.get("bytes")
        if url or payload:
            return {"url": url, "bytes": payload}
        return None

    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return {"url": value, "bytes": None}
    if value.startswith("data:image/"):
        try:
            _, encoded = value.split(",", 1)
            return {"url": None, "bytes": base64.b64decode(encoded)}
        except Exception:
            return None

    path = Path(value).expanduser()
    if path.exists() and path.is_file():
        # Keep local paths as paths instead of embedding bytes in the staging
        # parquet.  The local runner can open the path directly and the API
        # runner will inline it only at request construction time.  In both
        # cases the original remote URL is no longer fetched.
        return {"url": str(path.resolve()), "bytes": None}
    return None


def _image_reference_key(value: Any) -> str | None:
    """Return a stable cache key for a URL/path-like image value."""

    if isinstance(value, dict):
        value = value.get("url") or value.get("image_url") or value.get("cos_url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _cached_image_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _oss_accelerated_url(reference: str) -> str:
    """Use the Aliyun OSS accelerate endpoint for Beijing OSS URLs.

    The original reference remains the cache key and audit value.  Only the
    network request URL is rewritten, so output provenance and deduplication
    stay stable across runs.
    """

    parsed = urlsplit(reference)
    host = parsed.hostname or ""
    suffix = ".oss-cn-beijing.aliyuncs.com"
    if not host.endswith(suffix):
        return reference
    accelerated_host = f"{host[: -len(suffix)]}.oss-accelerate.aliyuncs.com"
    netloc = accelerated_host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(parsed._replace(netloc=netloc))


def _image_suffix(image_format: str | None, content_type: str | None, reference: str) -> str:
    format_suffixes = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }
    if image_format:
        suffix = format_suffixes.get(image_format.upper())
        if suffix:
            return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    guessed = Path(reference.split("?", 1)[0]).suffix.lower()
    return guessed if guessed in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"} else ".img"


def _write_cached_bytes(
    payload: bytes,
    *,
    reference: str,
    cache_dir: Path,
    digest: str,
    content_type: str | None = None,
) -> str:
    """Validate and atomically write one image to the shared cache."""

    with Image.open(io.BytesIO(payload)) as image:
        image_format = image.format
        image.verify()
    destination = cache_dir / f"{digest}{_image_suffix(image_format, content_type, reference)}"
    if destination.exists() and _cached_image_is_valid(destination):
        return str(destination.resolve())

    temporary = destination.with_name(f".{destination.name}.part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return str(destination.resolve())


def _materialize_image_reference(
    reference: str,
    *,
    cache_dir: Path,
    timeout: float,
    retries: int,
) -> tuple[str, str]:
    """Resolve one image reference to a reusable local cache file.

    The URL hash is shared by all repeats and all inference stages.  Existing
    valid files are reused without a network request.  The second return
    value is the cache status used for the audit manifest.
    """

    digest = hashlib.sha1(reference.encode("utf-8")).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)

    for existing in sorted(cache_dir.glob(f"{digest}.*")):
        if existing.name.endswith(".part"):
            continue
        if _cached_image_is_valid(existing):
            return str(existing.resolve()), "reused"

    if reference.startswith("data:image/"):
        try:
            _, encoded = reference.split(",", 1)
            payload = base64.b64decode(encoded)
            return _write_cached_bytes(
                payload,
                reference=reference,
                cache_dir=cache_dir,
                digest=digest,
                content_type=reference.split(";", 1)[0][5:],
            ), "downloaded"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Invalid data URI image: {exc}") from exc

    local_path = Path(reference).expanduser()
    if local_path.exists() and local_path.is_file():
        # Copy local inputs into the output VQA directory as well.  This keeps
        # the filtered directory portable instead of leaving references to the
        # original run directory.
        try:
            payload = local_path.read_bytes()
            cached = _write_cached_bytes(
                payload,
                reference=reference,
                cache_dir=cache_dir,
                digest=digest,
                content_type=_mime_for_path(local_path),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Local image is not decodable: {reference}: {exc}") from exc
        return cached, "copied"
    if local_path.is_absolute() or "://" not in reference:
        raise FileNotFoundError(f"Image path does not exist: {reference}")

    last_error: Exception | None = None
    request_url = _oss_accelerated_url(reference)
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                request_url,
                timeout=(timeout, timeout),
                headers={"User-Agent": "OpenSearch-VL-RL-filter/1.0"},
            )
            response.raise_for_status()
            payload = response.content
            content_type = response.headers.get("content-type")
            response.close()
            return _write_cached_bytes(
                payload,
                reference=reference,
                cache_dir=cache_dir,
                digest=digest,
                content_type=content_type,
            ), "downloaded"
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0**attempt, 8.0))
    raise RuntimeError(
        f"Failed to materialize image after {retries + 1} attempts: {reference}: {last_error}"
    ) from last_error


def _prepare_image_cache(
    records: list[dict[str, Any]],
    *,
    cache_dir: Path,
    manifest_path: Path | None = None,
    workers: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    """Materialize all candidate input images once and rewrite records.

    The returned report is intentionally small; the URL-to-path mapping is
    persisted separately as ``image_cache_manifest.jsonl`` for traceability.
    """

    unique: dict[str, str] = {}
    for record in records:
        for value in record.get("images") or []:
            reference = _image_reference_key(value)
            if reference:
                unique.setdefault(reference, reference)

    resolved: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def materialize(reference: str) -> tuple[str, str, str]:
        local, status = _materialize_image_reference(
            reference,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )
        return reference, local, status

    items = list(unique.values())
    if workers == 1:
        for reference in tqdm(items, desc="Image cache", unit="image"):
            try:
                _, local, status = materialize(reference)
                resolved[reference] = local
                manifest.append({"source": reference, "local_path": local, "status": status})
            except Exception as exc:  # noqa: BLE001
                failures.append({"source": reference, "error": str(exc)})
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(materialize, reference): reference for reference in items}
            with tqdm(total=len(pending), desc="Image cache", unit="image") as progress:
                for future in as_completed(pending):
                    reference = pending[future]
                    try:
                        _, local, status = future.result()
                        resolved[reference] = local
                        manifest.append({"source": reference, "local_path": local, "status": status})
                    except Exception as exc:  # noqa: BLE001
                        failures.append({"source": reference, "error": str(exc)})
                    progress.update(1)

    _jsonl_write(
        manifest_path or (cache_dir.parent / "image_cache_manifest.jsonl"),
        manifest + failures,
    )
    failed_references = {item["source"] for item in failures}
    image_failed_records: list[dict[str, Any]] = []
    kept_records: list[dict[str, Any]] = []
    for record in records:
        record_references = {
            reference
            for value in record.get("images") or []
            if (reference := _image_reference_key(value))
        }
        failed_for_record = sorted(record_references & failed_references)
        if failed_for_record:
            image_failed_records.append(
                {
                    **record,
                    "rejected_reasons": ["image_materialization_failed"],
                    "image_materialization_failures": [
                        item for item in failures if item["source"] in failed_for_record
                    ],
                }
            )
            continue
        kept_records.append(record)

    # A transient/unreachable image must not abort the whole batch.  Samples
    # depending on that image are excluded and retained in the audit output;
    # all other samples continue through the normal pipeline.
    records[:] = kept_records
    for record in records:
        rewritten: list[Any] = []
        for value in record.get("images") or []:
            reference = _image_reference_key(value)
            rewritten.append(resolved.get(reference, value) if reference else value)
        record["images"] = rewritten

    downloaded = sum(item["status"] == "downloaded" for item in manifest)
    reused = sum(item["status"] == "reused" for item in manifest)
    copied = sum(item["status"] == "copied" for item in manifest)
    return {
        "unique_references": len(items),
        "downloaded": downloaded,
        "reused": reused,
        "copied": copied,
        "failed_references": len(failures),
        "failed_samples": image_failed_records,
        "filtered_sample_count": len(image_failed_records),
        "cache_dir": str(cache_dir.resolve()),
    }


def _compact_hop_chain(sample: dict[str, Any], max_hops: int = 12) -> list[dict[str, Any]]:
    chain = sample.get("question_hop_chain") or sample.get("hop_chain") or []
    if not isinstance(chain, list):
        return []
    supporting_facts_by_hop: dict[int, list[str]] = {}
    compose = sample.get("compose") or {}
    payload = compose.get("payload") if isinstance(compose, dict) else {}
    for merged_hop in (payload or {}).get("merged_hops") or []:
        if not isinstance(merged_hop, dict):
            continue
        hop_index = merged_hop.get("hop_index")
        facts = merged_hop.get("supporting_facts") or []
        if isinstance(hop_index, int) and isinstance(facts, list):
            supporting_facts_by_hop[hop_index] = [str(fact)[:1200] for fact in facts[:6]]
    compact: list[dict[str, Any]] = []
    for item in chain[:max_hops]:
        if not isinstance(item, dict):
            continue
        compact_item: dict[str, Any] = {}
        for key in ("hop_index", "source", "target", "relation", "statement", "retrieval_query"):
            value = item.get(key)
            if value is not None:
                compact_item[key] = str(value)[:800]
        hop_index = item.get("hop_index")
        if isinstance(hop_index, int) and supporting_facts_by_hop.get(hop_index):
            compact_item["supporting_facts"] = supporting_facts_by_hop[hop_index]
        compact.append(compact_item)
    return compact


def _path_features(question: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    path = sample.get("path") or {}
    trajectory = path.get("trajectory") or {}
    node_types = list(path.get("node_types") or [])
    image_positions = [
        "start" if index == 0 else "end" if index == len(node_types) - 1 else "middle"
        for index, node_type in enumerate(node_types)
        if str(node_type).lower() == "image"
    ]
    images = _image_values(question, sample)
    return {
        "image_count": len(images),
        "image_node_count": int(trajectory.get("image_node_count") or sum(1 for x in node_types if str(x).lower() == "image")),
        "image_node_positions": image_positions,
        "node_types": node_types,
        "modality_sequence": list(trajectory.get("modality_sequence") or []),
        "modality_switch_count": int(trajectory.get("modality_switch_count") or 0),
        "hop_count": int(trajectory.get("hop_count") or len(path.get("edge_ids") or [])),
        "core_signature": path.get("core_signature"),
        "exact_signature": path.get("exact_signature"),
    }


def _compact_task_record(
    question: dict[str, Any],
    sample: dict[str, Any],
    source_index: int,
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "id": _safe_id(question, source_index),
        "question_id": question.get("question_id"),
        "sample_id": question.get("sample_id"),
        "path_id": question.get("path_id"),
        "status": question.get("status"),
        "question": _question_text(question),
        "answer": str(question.get("answer") or "").strip(),
        "images": _image_values(question, sample),
        "path_features": _path_features(question, sample),
    }


def _build_inference_dataframe(records: list[dict[str, Any]]):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for record in records:
        inference_images = [
            image_entry
            for value in record.get("images") or []
            if (image_entry := _image_entry_for_inference(value)) is not None
        ]
        rows.append(
            {
                "data_id": record["id"],
                "id": record["id"],
                "question_id": record.get("question_id"),
                "sample_id": record.get("sample_id"),
                "path_id": record.get("path_id"),
                "question": record["question"],
                "answer": record["answer"],
                "images": inference_images,
            }
        )
    return pd.DataFrame(rows)


def _image_field_value(images: list[Any]) -> Any:
    """Use the scalar shape of the source VQA files when possible."""

    if not images:
        return None
    return images[0] if len(images) == 1 else images


def _write_filtered_vqa_dir(
    output_dir: Path,
    *,
    source_questions: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    final_records: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    """Write the selected subset using the original VQA run-batch schema.

    ``questions.jsonl`` and ``samples.jsonl`` intentionally remain source-like
    files.  Only their image reference is rewritten to a local cached path;
    the original URL is retained in a small side field where one existed.
    Detailed filtering information remains in ``manifest.jsonl`` and the
    stage-specific artifacts rather than bloating every question record.
    """

    selected_questions: list[dict[str, Any]] = []
    selected_samples: list[dict[str, Any]] = []
    selected_ids = {record["id"] for record in final_records}

    for record in sorted(final_records, key=lambda item: int(item["source_index"])):
        source_index = int(record["source_index"])
        question = dict(source_questions[source_index])
        local_images = list(record.get("images") or [])
        original_images = next(
            (
                question.get(key)
                for key in ("images", "image_urls", "image_url")
                if question.get(key)
            ),
            None,
        )
        if original_images is not None and local_images:
            question["source_image_urls"] = original_images
        for key in ("images", "image_urls", "image_url"):
            question.pop(key, None)
        if local_images:
            question["image_url"] = _image_field_value(local_images)
        selected_questions.append(question)

        sample_id = record.get("sample_id")
        if sample_id is not None:
            sample = samples_by_id.get(str(sample_id))
            if sample is not None:
                sample_copy = dict(sample)
                original_input_image = sample_copy.get("input_image_url")
                if original_input_image is not None and local_images:
                    sample_copy["source_input_image_urls"] = original_input_image
                sample_copy.pop("input_image_url", None)
                if local_images:
                    sample_copy["input_image_url"] = _image_field_value(local_images)
                selected_samples.append(sample_copy)

    _jsonl_write(output_dir / "questions.jsonl", selected_questions)
    _jsonl_write(output_dir / "samples.jsonl", selected_samples)
    _jsonl_write(
        output_dir / "selection_manifest.jsonl",
        [item for item in manifest if item.get("id") in selected_ids],
    )
    _json_write(output_dir / "selection_report.json", report)
    with (output_dir / "source_indices.txt").open("w", encoding="utf-8") as handle:
        for record in sorted(final_records, key=lambda item: int(item["source_index"])):
            handle.write(f"{record['source_index']}\n")


def _load_inference_modules():
    """Import the existing entrypoints after runtime env knobs are set."""

    if str(OPENSEARCH_VL_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENSEARCH_VL_ROOT))
    from eval_infer_with_llm import run_eval
    from run_infer import main as run_infer_main
    from run_infer_no_tools import main as run_no_tools_main

    return run_eval, run_infer_main, run_no_tools_main


def _common_infer_args(args: argparse.Namespace, data_path: Path, output_dir: Path) -> list[str]:
    values = [
        "--model", args.model,
        "--backend", args.backend,
        "--gpus", args.gpus,
        "--dtype", args.dtype,
        "--data-path", str(data_path),
        "--output-dir", str(output_dir),
        "--dataset", "train",
        "--parallel-workers", str(args.parallel_workers),
        "--api-timeout", str(args.api_timeout),
        "--api-max-retries", str(args.api_max_retries),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--log-level", args.log_level,
    ]
    if args.checkpoint:
        values.extend(["--checkpoint", args.checkpoint])
    if args.backend == "api":
        if args.base_url:
            values.extend(["--base-url", args.base_url])
        if args.api_key:
            values.extend(["--api-key", args.api_key])
        if args.served_model_name:
            values.extend(["--served-model-name", args.served_model_name])
    return values


def _eval_details_path(output_dir: Path) -> Path:
    return output_dir / "llm_eval_report_details.jsonl"


def _load_details(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("case_id")): item
        for item in _jsonl_load(path)
        if item.get("case_id") is not None
    }


def _trajectory_summary(
    output_dir: Path,
    case_id: str,
    eval_details: dict[str, dict[str, Any]],
    *,
    no_tools: bool,
    max_turns: int,
) -> dict[str, Any]:
    completed_path = output_dir / f"{case_id}_trajectory.json"
    failure_path = output_dir / "failures" / f"{case_id}_failure.json"
    path = completed_path if completed_path.exists() else failure_path
    try:
        trajectory: dict[str, Any] = _read_json(path) if path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read trajectory artifact %s: %s", path, exc)
        trajectory = {
            "status": "failed",
            "failure_kind": "corrupt_artifact",
            "failure_reason": str(exc),
        }
    turns = trajectory.get("turns") or []
    failure_kind = trajectory.get("failure_kind")
    status = trajectory.get("status") or ("completed" if completed_path.exists() else "missing")
    detail = eval_details.get(case_id) or {}
    inference_failed = any(
        isinstance(turn, dict) and turn.get("error")
        for turn in turns
    )
    if inference_failed and no_tools:
        status = "failed"
        failure_kind = failure_kind or "inference_error"
    evaluation_available = bool(detail) and not bool(detail.get("error")) and not inference_failed
    correct = bool(detail.get("acc")) if evaluation_available else False
    tool_call_count = int(
        trajectory.get("tool_call_count")
        or sum(1 for turn in turns if isinstance(turn, dict) and turn.get("tool_call"))
    )
    return {
        "case_id": case_id,
        "status": status,
        "failure_kind": failure_kind,
        "turn_count": len(turns),
        "tool_call_count": 0 if no_tools else tool_call_count,
        "max_turn_reached": failure_kind == "max_turns" or (
            status != "completed" and len(turns) >= max_turns and failure_kind is not None
        ),
        "correct": correct,
        "evaluation_available": evaluation_available,
        "judge_reasoning": str(detail.get("reasoning") or "")[:500],
        "evaluation_error": detail.get("error"),
    }


def _trajectory_has_inference_error(trajectory_path: Path) -> bool:
    try:
        trajectory = _read_json(trajectory_path)
    except (OSError, json.JSONDecodeError):
        return True
    return any(
        isinstance(turn, dict) and turn.get("error")
        for turn in trajectory.get("turns") or []
    )


def _inference_case_complete(
    repeat_dir: Path,
    case_id: str,
    *,
    no_tools: bool,
) -> bool:
    trajectory_path = repeat_dir / f"{case_id}_trajectory.json"
    if not trajectory_path.exists() or _trajectory_has_inference_error(trajectory_path):
        return False
    try:
        trajectory = _read_json(trajectory_path)
    except (OSError, json.JSONDecodeError):
        return False
    if no_tools:
        return "final_response_text" in trajectory
    return trajectory.get("status") == "completed"


def _inference_repeat_complete(
    repeat_dir: Path,
    records: list[dict[str, Any]],
    *,
    no_tools: bool,
) -> bool:
    return bool(records) and all(
        _inference_case_complete(repeat_dir, record["id"], no_tools=no_tools)
        for record in records
    )


def _quarantine_no_tools_errors(repeat_dir: Path, records: list[dict[str, Any]]) -> int:
    """Move legacy no-tools error trajectories out of the runner's resume set."""

    failure_dir = repeat_dir / "failures"
    moved = 0
    for record in records:
        trajectory_path = repeat_dir / f"{record['id']}_trajectory.json"
        if not trajectory_path.exists() or not _trajectory_has_inference_error(trajectory_path):
            continue
        failure_dir.mkdir(parents=True, exist_ok=True)
        target = failure_dir / f"{record['id']}_failure.json"
        if target.exists():
            target = failure_dir / f"{record['id']}_no_tools_failure.json"
        shutil.move(str(trajectory_path), str(target))
        moved += 1
    return moved


def _stage_signature(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    no_tools: bool,
) -> str:
    return _sha256(
        {
            "stage": "no_tools" if no_tools else "tools",
            "case_records": [
                {
                    "id": record["id"],
                    "source_index": record["source_index"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "sample_id": record.get("sample_id"),
                    "images": record.get("images"),
                }
                for record in records
            ],
            "model": args.model,
            "checkpoint": args.checkpoint,
            "backend": args.backend,
            "base_url": args.base_url,
            "served_model_name": args.served_model_name,
            "dtype": args.dtype,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "max_tool_calls": args.max_tool_calls,
            "repetitions": args.repetitions,
        }
    )


def _prepare_stage_state(
    stage_root: Path,
    *,
    signature: str,
    case_count: int,
    repetitions: int,
    resume: bool,
) -> None:
    state_path = stage_root / "stage_state.json"
    previous = _read_json(state_path)
    if resume and previous.get("signature") and previous["signature"] != signature:
        raise RuntimeError(
            f"Resume signature mismatch in {stage_root}. Use a new --output-dir "
            "when changing the model, data, or inference parameters."
        )
    if resume and not previous and any(stage_root.iterdir()):
        logger.warning(
            "Resuming legacy stage without stage_state.json: %s. "
            "Existing trajectory artifacts will be inspected.",
            stage_root,
        )
    _json_write(
        state_path,
        {
            "signature": signature,
            "status": "running",
            "case_count": case_count,
            "repetitions": repetitions,
            "updated_at": _utc_now(),
        },
    )


def _run_one_repeat(
    *,
    args: argparse.Namespace,
    data_path: Path,
    answer_path: Path,
    records: list[dict[str, Any]],
    stage_root: Path,
    repeat: int,
    no_tools: bool,
    run_eval: Any,
    run_infer_main: Any,
    run_no_tools_main: Any,
) -> tuple[int, list[dict[str, Any]]]:
    repeat_dir = stage_root / f"repeat_{repeat:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    repeat_state_path = repeat_dir / "repeat_state.json"
    previous_state = _read_json(repeat_state_path)
    repeat_signature = _sha256(
        {
            "stage_signature": _stage_signature(args, records, no_tools=no_tools),
            "repeat": repeat,
        }
    )
    if args.resume and previous_state.get("signature") and previous_state["signature"] != repeat_signature:
        raise RuntimeError(
            f"Resume signature mismatch in {repeat_state_path}. Use a new --output-dir."
        )

    reusable_attempt = (
        args.resume
        and not args.retry_failures
        and previous_state.get("status") == "completed"
    )
    if args.resume and not reusable_attempt and no_tools:
        moved = _quarantine_no_tools_errors(repeat_dir, records)
        if moved:
            logger.info("Quarantined %d legacy no-tools failure artifact(s) in %s", moved, repeat_dir)
    if not args.resume or args.retry_failures:
        inference_reused = False
    elif reusable_attempt:
        inference_reused = True
    else:
        inference_reused = _inference_repeat_complete(
            repeat_dir, records, no_tools=no_tools
        )

    _json_write(
        repeat_state_path,
        {
            "signature": repeat_signature,
            "repeat": repeat,
            "status": "running",
            "inference_reused": inference_reused,
            "started_at": _utc_now(),
        },
    )

    return_code: int | None = None
    launcher_error: str | None = None
    if not inference_reused:
        argv = _common_infer_args(args, data_path, repeat_dir)
        try:
            if no_tools:
                return_code = run_no_tools_main(argv)
            else:
                return_code = run_infer_main(argv)
            if isinstance(return_code, int) and return_code != 0:
                _json_write(repeat_dir / "launcher_error.json", {"return_code": return_code})
        except Exception as exc:  # noqa: BLE001
            launcher_error = str(exc)
            logger.exception("Inference stage failed for repeat %d", repeat)
            _json_write(repeat_dir / "launcher_error.json", {"error": launcher_error})

    details_path = _eval_details_path(repeat_dir)
    eval_error: str | None = None
    try:
        run_eval(
            traj_dir=str(repeat_dir),
            answer_file=str(answer_path),
            output_path=str(repeat_dir / "llm_eval_report.json"),
            max_workers=args.judge_workers,
            judge_model_alias=args.answer_judge_model_alias,
            judge_max_tokens=args.answer_judge_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        eval_error = str(exc)
        logger.warning("Trajectory evaluation unavailable for %s: %s", repeat_dir, exc)

    details = _load_details(details_path)
    summaries = [
        _trajectory_summary(
            repeat_dir,
            record["id"],
            details,
            no_tools=no_tools,
            max_turns=args.max_turns,
        )
        for record in records
    ]
    inference_complete_count = sum(
        int(_inference_case_complete(repeat_dir, record["id"], no_tools=no_tools))
        for record in records
    )
    inference_finished = (
        launcher_error is None
        and (return_code is None or return_code == 0)
        and inference_complete_count == len(records)
    )
    _json_write(
        repeat_state_path,
        {
            "signature": repeat_signature,
            "repeat": repeat,
            "status": "completed" if inference_finished else "partial",
            "inference_reused": inference_reused,
            "return_code": return_code,
            "launcher_error": launcher_error,
            "evaluation_error": eval_error,
            "inference_complete_count": inference_complete_count,
            "trajectory_count": sum(item["status"] != "missing" for item in summaries),
            "evaluation_available_count": sum(item["evaluation_available"] for item in summaries),
            "finished_at": _utc_now(),
        },
    )
    return repeat, summaries


def _run_one_stage(
    *,
    args: argparse.Namespace,
    data_path: Path,
    answer_path: Path,
    records: list[dict[str, Any]],
    stage_root: Path,
    no_tools: bool,
    run_eval: Any,
    run_infer_main: Any,
    run_no_tools_main: Any,
) -> list[dict[str, Any]]:
    stage_root.mkdir(parents=True, exist_ok=True)
    signature = _stage_signature(args, records, no_tools=no_tools)
    _prepare_stage_state(
        stage_root,
        signature=signature,
        case_count=len(records),
        repetitions=args.repetitions,
        resume=args.resume,
    )
    summaries_by_id: dict[str, list[dict[str, Any]]] = {record["id"]: [] for record in records}

    repeat_workers = max(1, args.repeat_workers)
    if args.backend == "local" and repeat_workers > 1:
        logger.warning(
            "--repeat-workers=%d is unsafe with --backend local; forcing 1 to avoid "
            "loading multiple models onto the same GPU set.",
            repeat_workers,
        )
        repeat_workers = 1

    def _run_repeat(repeat: int) -> tuple[int, list[dict[str, Any]]]:
        return _run_one_repeat(
            args=args,
            data_path=data_path,
            answer_path=answer_path,
            records=records,
            stage_root=stage_root,
            repeat=repeat,
            no_tools=no_tools,
            run_eval=run_eval,
            run_infer_main=run_infer_main,
            run_no_tools_main=run_no_tools_main,
        )

    repeat_description = "No-tools repeats" if no_tools else "Tools repeats"
    if repeat_workers == 1:
        repeat_results = []
        for repeat in tqdm(range(args.repetitions), desc=repeat_description, unit="repeat"):
            repeat_results.append(_run_repeat(repeat))
    else:
        with ThreadPoolExecutor(max_workers=repeat_workers) as executor:
            futures = [executor.submit(_run_repeat, repeat) for repeat in range(args.repetitions)]
            repeat_results = []
            with tqdm(total=len(futures), desc=repeat_description, unit="repeat") as progress:
                for future in as_completed(futures):
                    repeat_results.append(future.result())
                    progress.update(1)

    for _, summaries in sorted(repeat_results, key=lambda item: item[0]):
        for summary in summaries:
            summaries_by_id[summary["case_id"]].append(summary)

    results = [
        {
            "case_id": record["id"],
            "repetitions": summaries_by_id[record["id"]],
            "success_count": sum(int(item["correct"]) for item in summaries_by_id[record["id"]]),
            "evaluation_available_count": sum(
                int(item["evaluation_available"])
                for item in summaries_by_id[record["id"]]
            ),
            "evaluation_complete": all(
                item["evaluation_available"]
                for item in summaries_by_id[record["id"]]
            ),
            "pass_at_4": all(
                item["evaluation_available"]
                for item in summaries_by_id[record["id"]]
            ) and any(item["correct"] for item in summaries_by_id[record["id"]]),
            "max_turn_count": sum(int(item["max_turn_reached"]) for item in summaries_by_id[record["id"]]),
            "evaluation_error_count": sum(
                int(bool(item.get("evaluation_error")))
                for item in summaries_by_id[record["id"]]
            ),
        }
        for record in records
    ]
    _json_write(
        stage_root / "stage_summary.json",
        {
            "stage": "no_tools" if no_tools else "tools",
            "case_count": len(records),
            "repetitions": args.repetitions,
            "evaluation_complete_count": sum(
                int(item["evaluation_complete"]) for item in results
            ),
            "results": results,
        },
    )
    _json_write(
        stage_root / "stage_state.json",
        {
            "signature": signature,
            "status": "completed",
            "case_count": len(records),
            "repetitions": args.repetitions,
            "repeat_workers": repeat_workers,
            "evaluation_complete_count": sum(
                int(item["evaluation_complete"]) for item in results
            ),
            "finished_at": _utc_now(),
        },
    )
    return results


QUALITY_SYSTEM_PROMPT_BASE = """You are an evidence-consistency auditor for multi-hop deep-research questions.

You receive a question, its intended construction hops, and source-derived supporting facts. Your sole task is to decide whether a *core premise of the question* directly contradicts, reverses, or materially distorts the supplied evidence. This is not a general difficulty, style, or answerability review.

Important scope rule: the hops are navigation breadcrumbs, not a complete proof of the final answer. A terminal question is deliberately expected to require a fresh web lookup after the last hop. Some hops may also have no supporting-fact excerpt because the provenance extractor did not retain one. Missing support is therefore UNKNOWN, not an error. Never reject because a final answer, a later fact, an image detail, or a hop's evidence excerpt is absent from this packet.

Audit each hop and the way the question paraphrases it. Pay special attention to subject/object or active/passive reversal; temporal reversal (before/after/later/earlier) or anachronistic historical-to-modern identity mapping; entity-type or identity errors; strengthening a source claim (for example, "played a role" becoming "prosecuted"); and a final question that relies on a relationship absent from the supplied evidence.

Decision policy:
1. `reject` ONLY when you can point to a specific core claim in the question and cite supplied evidence that directly contradicts it, or where the question visibly strengthens/changes a supplied statement into a different relationship. Use one or more of: `relation_reversal`, `temporal_error`, `entity_error`, `unsupported_strengthening`, `answer_leakage`. Do not use `unsupported_relation` merely because evidence is missing.
2. `review` for a real but non-fatal issue: imprecise wording, an historical/modern mapping that needs clearer wording, or a potential ambiguity that is not demonstrated by two concrete incompatible readings. Review items are retained by the pipeline.
3. `accept` when the evidence supports the question's essential chain.

Do NOT reject merely because the question is difficult, needs additional web retrieval, has several hops, omits facts that a solver must search for, has a missing supporting-fact excerpt, or has an intermediate entity that is easy to identify. Do NOT invent facts beyond the supplied material. Do NOT treat an alternative interpretation as ambiguity unless you name two concrete interpretations that produce different answers. If an image would be needed to establish ambiguity but no image evidence is supplied, do not reject for visual ambiguity.

Return JSON only, with no Markdown or extra text:
{"decision":"accept"|"review"|"reject","reject_categories":["..."],"reasons":["For each issue: question claim; supporting or contradicting evidence; why it matters."],"difficulty_hint":"easy"|"medium"|"hard"}
"""


# Few-shot examples are temporarily disabled.  Keep this constant so they can
# be reintroduced later without changing the request-building interface.
QUALITY_FEW_SHOT = ""
QUALITY_SYSTEM_PROMPT = QUALITY_SYSTEM_PROMPT_BASE.rstrip()


def _quality_prompt(record: dict[str, Any], sample: dict[str, Any]) -> str:
    path = sample.get("path") or {}
    trajectory = path.get("trajectory") or {}
    hop_chain = _compact_hop_chain(sample)
    statement_lines: list[str] = []
    for position, hop in enumerate(hop_chain):
        statement = hop.get("statement") or "(no statement provided)"
        source = hop.get("source") or ""
        relation = hop.get("relation") or ""
        target = hop.get("target") or ""
        statement_lines.append(
            "Hop {position}:\n"
            "  source: {source}\n"
            "  relation: {relation}\n"
            "  statement: {statement}\n"
            "  target: {target}\n"
            "  supporting_facts: {supporting_facts}".format(
                position=position,
                source=source,
                relation=relation,
                statement=statement,
                target=target,
                supporting_facts=json.dumps(hop.get("supporting_facts") or [], ensure_ascii=False),
            )
        )
    if not statement_lines:
        statement_lines.append("(construction hop chain is missing)")

    path_summary = {
        "node_types": path.get("node_types") or [],
        "hop_count": trajectory.get("hop_count"),
        "modality_sequence": trajectory.get("modality_sequence") or [],
    }
    return (
        "Audit the following candidate question against its construction evidence.\n"
        "Apply the system decision policy exactly; this task is about factual and "
        "relational consistency, not generic question difficulty.\n\n"
        "[Question]\n"
        f"{record['question']}\n\n"
        "[Construction hop chain and intermediate statements]\n"
        + "\n\n".join(statement_lines)
        + "\n\n[Path summary]\n"
        + json.dumps(path_summary, ensure_ascii=False, indent=2)
    )


def _quality_audit_input(
    record: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Any]:
    """Compact input context saved beside every quality-judge decision."""

    path = sample.get("path") or {}
    trajectory = path.get("trajectory") or {}
    return {
        "question": record.get("question", ""),
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "hop_chain": _compact_hop_chain(sample),
        "path_summary": {
            "node_types": path.get("node_types") or [],
            "hop_count": trajectory.get("hop_count"),
            "modality_sequence": trajectory.get("modality_sequence") or [],
        },
    }


def _parse_quality_judge(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    parsed: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        return {
            "parse_ok": False,
            "decision": "reject",
            "reject_categories": ["invalid_output"],
            "reasons": ["quality_judge_output_not_json"],
            "difficulty_hint": "hard",
            "raw": raw[:3000],
        }

    decision = str(parsed.get("decision") or "").strip().lower()
    # Accept the old ``keep`` spelling when inspecting legacy judge artifacts,
    # but all new prompts and outputs use the explicit accept/reject protocol.
    if decision == "keep":
        decision = "accept"
    if decision not in {"accept", "review", "reject"}:
        return {
            "parse_ok": False,
            "decision": "reject",
            "reject_categories": ["invalid_decision"],
            "reasons": ["quality_judge_decision_must_be_accept_review_or_reject"],
            "difficulty_hint": "hard",
            "raw": raw[:3000],
        }
    reject_categories = parsed.get("reject_categories") or []
    if isinstance(reject_categories, str):
        reject_categories = [reject_categories]
    if not isinstance(reject_categories, list):
        reject_categories = [str(reject_categories)]
    reasons = parsed.get("reasons") or parsed.get("reason") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    difficulty_hint = str(parsed.get("difficulty_hint") or "medium").lower()
    if difficulty_hint not in {"easy", "medium", "hard"}:
        difficulty_hint = "medium"
    return {
        "parse_ok": True,
        "decision": decision,
        "reject_categories": [str(item)[:120] for item in reject_categories[:5]],
        "reasons": [str(item)[:300] for item in reasons[:5]],
        "difficulty_hint": difficulty_hint,
        "raw": raw[:3000],
    }


def _quality_judge_one(
    record: dict[str, Any],
    sample: dict[str, Any],
    *,
    model_alias: str,
    max_tokens: int,
) -> dict[str, Any]:
    from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest

    prompt = _quality_prompt(record, sample)
    response = LLM_WORKER.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=QUALITY_SYSTEM_PROMPT),
                ModelMessage(role="user", content=prompt),
            ],
            max_tokens=max_tokens,
            metadata={"trace_label": "rl_vqa_quality_judge"},
        )
    )
    parsed = _parse_quality_judge(str(response.content or ""))
    parsed["case_id"] = record["id"]
    parsed["source_index"] = record["source_index"]
    parsed["audit_input"] = _quality_audit_input(record, sample)
    parsed["judge_model_alias"] = model_alias
    parsed["system_prompt_sha256"] = _sha256(QUALITY_SYSTEM_PROMPT)
    return parsed


def _load_quality_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("Ignoring incomplete quality-cache line %d in %s", line_number, path)
                continue
            if isinstance(item, dict) and item.get("case_id") is not None:
                cached[str(item["case_id"])] = item
    return cached


def _run_quality_judge(
    records: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    model_alias: str,
    max_tokens: int,
    workers: int,
    resume: bool,
    retry_failures: bool,
) -> dict[str, dict[str, Any]]:
    cached = _load_quality_cache(output_path) if resume else {}
    for record in records:
        cached_result = cached.get(record["id"])
        if cached_result is not None:
            # Backfill audit context for cache files produced by an older
            # version without issuing another LLM request.
            cached_result.setdefault(
                "audit_input",
                _quality_audit_input(
                    record,
                    samples_by_id.get(str(record.get("sample_id")), {}),
                ),
            )
            cached_result.setdefault("judge_model_alias", model_alias)
            cached_result.setdefault("system_prompt_sha256", _sha256(QUALITY_SYSTEM_PROMPT))
    pending = [
        record
        for record in records
        if record["id"] not in cached
        or (
            retry_failures
            and (
                bool(cached[record["id"]].get("error"))
                or not bool(cached[record["id"]].get("parse_ok"))
            )
        )
    ]
    if pending:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(
                        _quality_judge_one,
                        record,
                        samples_by_id.get(str(record.get("sample_id")), {}),
                        model_alias=model_alias,
                        max_tokens=max_tokens,
                    ): record
                    for record in pending
                }
                with tqdm(total=len(futures), desc="Quality judge", unit="sample") as progress:
                    for future in as_completed(futures):
                        record = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "case_id": record["id"],
                                "source_index": record["source_index"],
                                "parse_ok": False,
                                "decision": "reject",
                                "reject_categories": ["judge_error"],
                                "reasons": [f"quality_judge_error:{type(exc).__name__}"],
                                "difficulty_hint": "hard",
                                "raw": "",
                                "error": str(exc),
                                "audit_input": _quality_audit_input(
                                    record,
                                    samples_by_id.get(str(record.get("sample_id")), {}),
                                ),
                                "judge_model_alias": model_alias,
                                "system_prompt_sha256": _sha256(QUALITY_SYSTEM_PROMPT),
                            }
                        cached[result["case_id"]] = result
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                        progress.update(1)
    return cached


def _load_pipeline_inference_context(args: argparse.Namespace) -> dict[str, Any]:
    """Build one shared inference runner for the sample-level pipeline.

    Calling ``run_infer.main`` once per sample would reload the model for every
    task.  The pipeline scheduler therefore reuses the same runner instance and
    calls the existing per-case pipeline functions directly.  This preserves
    the original inference/tool logic while allowing independent samples to
    progress through different stages concurrently.
    """

    if str(OPENSEARCH_VL_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENSEARCH_VL_ROOT))
    from eval_infer_with_llm import process_single_trajectory, run_eval
    from opensearch_infer import no_tools_pipeline, pipeline
    from opensearch_infer.runners import InferenceConfig, build_runner

    runner = build_runner(
        model_name=args.model,
        checkpoint=args.checkpoint,
        gpus=args.gpus,
        dtype=args.dtype,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        served_model_name=args.served_model_name,
        timeout=args.api_timeout,
        max_retries=args.api_max_retries,
    )
    logger.info("Loading shared pipeline runner: %s (backend=%s)", runner.display_name, args.backend)
    runner.load()
    return {
        "runner": runner,
        "pipeline": pipeline,
        "no_tools_pipeline": no_tools_pipeline,
        "inference_config": InferenceConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        "process_single_trajectory": process_single_trajectory,
        "run_eval": run_eval,
    }


def _missing_pipeline_summary(case_id: str, *, no_tools: bool, failure_kind: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "missing",
        "failure_kind": failure_kind,
        "turn_count": 0,
        "tool_call_count": 0 if no_tools else 0,
        "max_turn_reached": False,
        "correct": False,
        "evaluation_available": False,
        "judge_reasoning": "",
        "evaluation_error": failure_kind,
    }


def _aggregate_pipeline_repetitions(
    case_id: str,
    repetitions: list[dict[str, Any]],
    *,
    expected_repetitions: int,
) -> dict[str, Any]:
    ordered = sorted(repetitions, key=lambda item: int(item.get("repeat", 0)))
    evaluation_complete = (
        len(ordered) == expected_repetitions
        and all(item.get("evaluation_available") for item in ordered)
    )
    return {
        "case_id": case_id,
        "repetitions": ordered,
        "success_count": sum(int(item.get("correct", False)) for item in ordered),
        "evaluation_available_count": sum(
            int(item.get("evaluation_available", False)) for item in ordered
        ),
        "evaluation_complete": evaluation_complete,
        "pass_at_4": evaluation_complete and any(
            item.get("correct", False) for item in ordered
        ),
        "max_turn_count": sum(int(item.get("max_turn_reached", False)) for item in ordered),
        "evaluation_error_count": sum(
            int(bool(item.get("evaluation_error"))) for item in ordered
        ),
    }


def _replace_pipeline_repeat(
    repetitions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    repeat = int(summary.get("repeat", 0))
    result = [item for item in repetitions if int(item.get("repeat", -1)) != repeat]
    result.append(summary)
    return sorted(result, key=lambda item: int(item.get("repeat", 0)))


def _evaluate_pipeline_trajectory(
    trajectory_path: Path,
    *,
    details_path: Path,
    cache: dict[str, dict[str, Any]],
    cache_lock: threading.Lock,
    process_single_trajectory: Any,
    judge_model_alias: str,
    judge_max_tokens: int,
    judge_semaphore: threading.Semaphore,
) -> dict[str, Any]:
    """Evaluate one trajectory using the existing LLM judge implementation."""

    case_id = trajectory_path.name.removesuffix("_trajectory.json")
    with cache_lock:
        cached = cache.get(case_id)
        if cached is not None and not cached.get("error"):
            return cached

    try:
        with judge_semaphore:
            result = process_single_trajectory(
                trajectory_path,
                {},
                judge_model_alias,
                judge_max_tokens,
            )
    except Exception as exc:  # noqa: BLE001
        result = {
            "case_id": case_id,
            "acc": 0,
            "reasoning": "",
            "raw_judge": "",
            "error": str(exc),
        }

    with cache_lock:
        existing = cache.get(case_id)
        if existing is not None and not existing.get("error"):
            return existing
        cache[case_id] = result
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    return result


def _quality_pipeline_result(
    record: dict[str, Any],
    sample: dict[str, Any],
    *,
    cache: dict[str, dict[str, Any]],
    cache_lock: threading.Lock,
    output_path: Path,
    model_alias: str,
    max_tokens: int,
    retry_failures: bool,
    judge_semaphore: threading.Semaphore,
) -> dict[str, Any]:
    """Get one quality decision, reusing the append-only quality cache."""

    case_id = record["id"]
    with cache_lock:
        cached = cache.get(case_id)
        should_retry = retry_failures and cached is not None and (
            bool(cached.get("error")) or not bool(cached.get("parse_ok"))
        )
        if cached is not None and not should_retry:
            cached.setdefault("audit_input", _quality_audit_input(record, sample))
            cached.setdefault("judge_model_alias", model_alias)
            cached.setdefault("system_prompt_sha256", _sha256(QUALITY_SYSTEM_PROMPT))
            return cached

    try:
        with judge_semaphore:
            result = _quality_judge_one(
                record,
                sample,
                model_alias=model_alias,
                max_tokens=max_tokens,
            )
    except Exception as exc:  # noqa: BLE001
        result = {
            "case_id": case_id,
            "source_index": record["source_index"],
            "parse_ok": False,
            "decision": "reject",
            "reject_categories": ["judge_error"],
            "reasons": [f"quality_judge_error:{type(exc).__name__}"],
            "difficulty_hint": "hard",
            "raw": "",
            "error": str(exc),
            "audit_input": _quality_audit_input(record, sample),
            "judge_model_alias": model_alias,
            "system_prompt_sha256": _sha256(QUALITY_SYSTEM_PROMPT),
        }

    with cache_lock:
        existing = cache.get(case_id)
        if existing is not None and not retry_failures:
            return existing
        cache[case_id] = result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    return result


def _pipeline_sample_state_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "pipeline" / "sample_states" / f"{_sha256(case_id)[:24]}.json"


def _write_pipeline_sample_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _json_write(path, state)


def _run_pipeline_sample(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    sample: dict[str, Any],
    row: Any,
    output_dir: Path,
    pipeline_signature: str,
    context: dict[str, Any],
    eval_caches: dict[str, dict[str, dict[str, Any]]],
    eval_locks: dict[str, threading.Lock],
    quality_cache: dict[str, dict[str, Any]],
    quality_lock: threading.Lock,
    judge_semaphore: threading.Semaphore,
    quality_semaphore: threading.Semaphore,
) -> dict[str, Any]:
    """Run one sample through the complete filter pipeline.

    This function is the unit scheduled by the outer ThreadPoolExecutor.  A
    worker stays with one sample while it completes all no-tools repetitions,
    the shortcut decision, all required tool repetitions, and the quality
    judge.  When it returns, the executor immediately assigns another sample.
    """

    case_id = record["id"]
    state_path = _pipeline_sample_state_path(output_dir, case_id)
    state = _read_json(state_path) if args.resume else {}
    if args.resume and state.get("signature") and state["signature"] != pipeline_signature:
        raise RuntimeError(f"Pipeline sample signature mismatch for {case_id}; use --force-rerun")
    state.update(
        {
            "signature": pipeline_signature,
            "case_id": case_id,
            "source_index": record["source_index"],
            "updated_at": _utc_now(),
        }
    )

    runner = context["runner"]
    inference_config = context["inference_config"]
    eval_fn = context["process_single_trajectory"]

    def run_attempt(*, no_tools: bool, repeat: int) -> dict[str, Any]:
        stage_name = "no_tools" if no_tools else "tools"
        stage_root = output_dir / stage_name / f"repeat_{repeat:02d}"
        stage_root.mkdir(parents=True, exist_ok=True)
        trajectory_path = stage_root / f"{case_id}_trajectory.json"

        reusable = args.resume and _inference_case_complete(
            stage_root,
            case_id,
            no_tools=no_tools,
        )
        if not reusable and args.resume and no_tools:
            _quarantine_no_tools_errors(stage_root, [record])

        if not reusable:
            module = context["no_tools_pipeline"] if no_tools else context["pipeline"]
            try:
                module.process_single_case(
                    row=row,
                    runner=runner,
                    output_dir=str(stage_root),
                    case_idx=int(record["source_index"]),
                    dataset_type="train",
                    inference_cfg=inference_config,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Pipeline %s inference failed for %s repeat=%d: %s",
                    stage_name,
                    case_id,
                    repeat,
                    exc,
                )

        eval_cache = eval_caches.setdefault(str(stage_root), _load_details(_eval_details_path(stage_root)))
        eval_lock = eval_locks.setdefault(str(stage_root), threading.Lock())
        if trajectory_path.exists() and not _trajectory_has_inference_error(trajectory_path):
            _evaluate_pipeline_trajectory(
                trajectory_path,
                details_path=_eval_details_path(stage_root),
                cache=eval_cache,
                cache_lock=eval_lock,
                process_single_trajectory=eval_fn,
                judge_model_alias=args.answer_judge_model_alias,
                judge_max_tokens=args.answer_judge_max_tokens,
                judge_semaphore=judge_semaphore,
            )
        summary = _trajectory_summary(
            stage_root,
            case_id,
            eval_cache,
            no_tools=no_tools,
            max_turns=args.max_turns,
        )
        summary["repeat"] = repeat
        return summary

    no_tools_repetitions: list[dict[str, Any]] = []
    for repeat in range(args.repetitions):
        summary = run_attempt(no_tools=True, repeat=repeat)
        no_tools_repetitions = _replace_pipeline_repeat(no_tools_repetitions, summary)
        state["no_tools_repetitions"] = no_tools_repetitions
        state["updated_at"] = _utc_now()
        _write_pipeline_sample_state(state_path, state)

    no_tools_result = _aggregate_pipeline_repetitions(
        case_id,
        no_tools_repetitions,
        expected_repetitions=args.repetitions,
    )
    state["no_tools_result"] = no_tools_result
    if not no_tools_result["evaluation_complete"]:
        state.update({"status": "completed", "decision": "no_tools_evaluation_incomplete"})
        _write_pipeline_sample_state(state_path, state)
        return {"case_id": case_id, "no_tools": no_tools_result, "tools": None, "quality_judge": None,
                "decision": "no_tools_evaluation_incomplete"}
    if no_tools_result["pass_at_4"]:
        state.update({"status": "completed", "decision": "no_tools_pass_at_4_shortcut"})
        _write_pipeline_sample_state(state_path, state)
        return {"case_id": case_id, "no_tools": no_tools_result, "tools": None, "quality_judge": None,
                "decision": "no_tools_pass_at_4_shortcut"}

    tools_repetitions: list[dict[str, Any]] = []
    for repeat in range(args.repetitions):
        summary = run_attempt(no_tools=False, repeat=repeat)
        tools_repetitions = _replace_pipeline_repeat(tools_repetitions, summary)
        state["tools_repetitions"] = tools_repetitions
        state["updated_at"] = _utc_now()
        _write_pipeline_sample_state(state_path, state)

    tools_result = _aggregate_pipeline_repetitions(
        case_id,
        tools_repetitions,
        expected_repetitions=args.repetitions,
    )
    state["tools_result"] = tools_result
    valid_attempts = len(tools_repetitions) - sum(
        int(item.get("failure_kind") in {"inference_error", "tool_execution_error"})
        for item in tools_repetitions
    ) - sum(int(item.get("status") == "missing") for item in tools_repetitions)
    if valid_attempts <= 0:
        state.update({"status": "completed", "decision": "tool_evaluation_unavailable"})
        _write_pipeline_sample_state(state_path, state)
        return {"case_id": case_id, "no_tools": no_tools_result, "tools": tools_result,
                "quality_judge": None, "decision": "tool_evaluation_unavailable"}

    quality_result = _quality_pipeline_result(
        record,
        sample,
        cache=quality_cache,
        cache_lock=quality_lock,
        output_path=output_dir / "quality_judge_details.jsonl",
        model_alias=args.quality_judge_model_alias,
        max_tokens=args.quality_judge_max_tokens,
        retry_failures=args.retry_failures,
        judge_semaphore=quality_semaphore,
    )
    state.update({"quality_judge": quality_result, "status": "completed", "decision": quality_result.get("decision")})
    _write_pipeline_sample_state(state_path, state)
    return {"case_id": case_id, "no_tools": no_tools_result, "tools": tools_result,
            "quality_judge": quality_result, "decision": quality_result.get("decision")}


def _run_pipeline_async(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    task_df: Any,
    output_dir: Path,
    pipeline_signature: str,
    quality_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Run sample pipelines with dynamic scheduling across all stages."""

    context = _load_pipeline_inference_context(args)
    rows_by_id = {
        str(row.get("data_id")): row
        for _, row in task_df.iterrows()
    }
    eval_caches: dict[str, dict[str, dict[str, Any]]] = {}
    eval_locks: dict[str, threading.Lock] = {}
    quality_lock = threading.Lock()
    judge_semaphore = threading.BoundedSemaphore(max(1, args.judge_workers))
    quality_semaphore = threading.BoundedSemaphore(max(1, args.quality_judge_workers))

    pipeline_workers = max(1, args.parallel_workers)
    if args.backend == "api":
        pipeline_workers *= max(1, args.repeat_workers)
    elif args.repeat_workers > 1:
        logger.warning(
            "--repeat-workers is only used to increase sample pipeline slots with the API backend; "
            "local backend keeps %d pipeline worker(s).",
            pipeline_workers,
        )
    logger.info(
        "Starting sample-level async pipeline: workers=%d, repetitions=%d",
        pipeline_workers,
        args.repetitions,
    )

    results_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=pipeline_workers) as executor:
        futures = {
            executor.submit(
                _run_pipeline_sample,
                args=args,
                record=record,
                sample=samples_by_id.get(str(record.get("sample_id")), {}),
                row=rows_by_id[record["id"]],
                output_dir=output_dir,
                pipeline_signature=pipeline_signature,
                context=context,
                eval_caches=eval_caches,
                eval_locks=eval_locks,
                quality_cache=quality_cache,
                quality_lock=quality_lock,
                judge_semaphore=judge_semaphore,
                quality_semaphore=quality_semaphore,
            ): record
            for record in records
        }
        with tqdm(total=len(futures), desc="Sample pipeline", unit="sample") as progress:
            for future in as_completed(futures):
                record = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Sample pipeline failed for %s", record["id"])
                    result = {
                        "case_id": record["id"],
                        "no_tools": _aggregate_pipeline_repetitions(
                            record["id"], [], expected_repetitions=args.repetitions
                        ),
                        "tools": None,
                        "quality_judge": None,
                        "decision": "pipeline_error",
                        "error": str(exc),
                    }
                results_by_id[record["id"]] = result
                progress.update(1)

    no_tools_by_id = {
        case_id: result["no_tools"]
        for case_id, result in results_by_id.items()
        if result.get("no_tools") is not None
    }
    tool_by_id = {
        case_id: result["tools"]
        for case_id, result in results_by_id.items()
        if result.get("tools") is not None
    }
    quality_by_id = {
        case_id: result["quality_judge"]
        for case_id, result in results_by_id.items()
        if result.get("quality_judge") is not None
    }
    pipeline_items = [results_by_id[record["id"]] for record in records]
    _jsonl_write(output_dir / "pipeline_manifest.jsonl", pipeline_items)

    for stage_name, stage_results in (("no_tools", no_tools_by_id), ("tools", tool_by_id)):
        stage_root = output_dir / stage_name
        stage_root.mkdir(parents=True, exist_ok=True)
        values = [stage_results[record["id"]] for record in records if record["id"] in stage_results]
        _json_write(
            stage_root / "stage_summary.json",
            {
                "stage": stage_name,
                "scheduler": "sample_pipeline_async",
                "case_count": len(values),
                "repetitions": args.repetitions,
                "evaluation_complete_count": sum(int(item["evaluation_complete"]) for item in values),
                "results": values,
            },
        )
        _json_write(
            stage_root / "stage_state.json",
            {
                "status": "completed",
                "scheduler": "sample_pipeline_async",
                "case_count": len(values),
                "repetitions": args.repetitions,
                "finished_at": _utc_now(),
            },
        )

    # The per-case judge calls above reuse eval_infer_with_llm directly.  Run
    # its normal report aggregation afterwards so existing report files keep
    # their original schema.
    for stage_name in ("no_tools", "tools"):
        for repeat in range(args.repetitions):
            repeat_dir = output_dir / stage_name / f"repeat_{repeat:02d}"
            try:
                context["run_eval"](
                    traj_dir=str(repeat_dir),
                    output_path=str(repeat_dir / "llm_eval_report.json"),
                    max_workers=1,
                    judge_model_alias=args.answer_judge_model_alias,
                    judge_max_tokens=args.answer_judge_max_tokens,
                )
            except FileNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not write aggregate judge report for %s: %s", repeat_dir, exc)

    return no_tools_by_id, tool_by_id, quality_by_id, pipeline_items


def _difficulty_level(
    record: dict[str, Any],
    tool_eval: dict[str, Any],
    quality: dict[str, Any],
    max_turns: int,
) -> tuple[str, dict[str, Any]]:
    repetitions = tool_eval.get("repetitions") or []
    # Difficulty is deliberately determined only by answer correctness.  A
    # long trajectory, many tool calls, image complexity, or an incomplete
    # rollout is not itself evidence that the question is difficult.
    evaluated = [
        item for item in repetitions
        if item.get("status") != "missing" and item.get("evaluation_available")
    ]
    correct_count = sum(bool(item.get("correct")) for item in evaluated)
    evaluated_count = len(evaluated)
    if correct_count == 0:
        level = "hard"
    elif correct_count >= 3:
        level = "easy"
    else:
        level = "medium"

    # Keep rollout/path statistics for auditing, but none of these fields
    # participates in the level assignment above.
    turns = [int(item.get("turn_count") or 0) for item in repetitions if item.get("status") != "missing"]
    tool_calls = [int(item.get("tool_call_count") or 0) for item in repetitions if item.get("status") != "missing"]
    max_turn_count = int(tool_eval.get("max_turn_count") or 0)
    success_count = int(tool_eval.get("success_count") or 0)
    features = record.get("path_features") or {}
    image_positions = list(features.get("image_node_positions") or [])
    image_count = int(features.get("image_count") or 0)
    image_node_count = int(features.get("image_node_count") or 0)
    hop_count = int(features.get("hop_count") or 0)
    features_out = {
        "tool_success_count": success_count,
        "answer_correct_count": correct_count,
        "answer_evaluated_count": evaluated_count,
        "tool_pass_at_4": bool(tool_eval.get("pass_at_4")),
        "max_turn_count": max_turn_count,
        "turn_counts": turns,
        "tool_call_counts": tool_calls,
        "hop_count": hop_count,
        "image_count": image_count,
        "image_node_count": image_node_count,
        "image_node_positions": image_positions,
        "quality_hint": quality.get("difficulty_hint"),
    }
    return level, features_out


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in ratios.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _select_balanced(
    records: list[dict[str, Any]],
    *,
    target_count: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_level: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for record in records:
        by_level.setdefault(record["difficulty_level"], []).append(record)
    for values in by_level.values():
        rng.shuffle(values)

    target_total = min(target_count, len(records)) if target_count is not None else len(records)
    targets = _target_counts(target_total, {"easy": 0.2, "medium": 0.5, "hard": 0.3})
    selected: list[dict[str, Any]] = []
    shortfall: dict[str, int] = {}
    for level in ("easy", "medium", "hard"):
        take = min(targets[level], len(by_level[level]))
        selected.extend(by_level[level][:take])
        shortfall[level] = targets[level] - take

    selected_ids = {record["id"] for record in selected}
    leftovers = [record for record in records if record["id"] not in selected_ids]
    rng.shuffle(leftovers)
    if len(selected) < target_total:
        selected.extend(leftovers[: target_total - len(selected)])

    rng.shuffle(selected)
    actual_counts = Counter(record["difficulty_level"] for record in selected)
    report = {
        "target_count": target_total,
        "target_ratios": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
        "target_counts": targets,
        "available_counts": {key: len(value) for key, value in by_level.items()},
        "shortfall_counts": shortfall,
        "selected_counts": dict(actual_counts),
    }
    return selected, report


def _pipeline_signature(
    args: argparse.Namespace,
    *,
    vqa_dir: Path,
) -> str:
    questions_path = vqa_dir / "questions.jsonl"
    try:
        source_identity = {
            "path": str(questions_path),
            "size": questions_path.stat().st_size,
            "mtime_ns": questions_path.stat().st_mtime_ns,
        }
    except FileNotFoundError:
        source_identity = {"path": str(questions_path)}
    return _sha256(
        {
            "vqa_dir": str(vqa_dir),
            # Deliberately independent of offset/limit. Per-sample artifacts
            # are keyed by case ID, while this identifies the immutable input.
            "source_identity": source_identity,
            "backend": args.backend,
            "dtype": args.dtype,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "max_tool_calls": args.max_tool_calls,
            "repetitions": args.repetitions,
            "answer_judge_max_tokens": args.answer_judge_max_tokens,
            "quality_judge_max_tokens": args.quality_judge_max_tokens,
            "image_cache_dir": str(args.image_cache_dir) if args.image_cache_dir else None,
            "image_timeout": args.image_timeout,
            "image_retries": args.image_retries,
        }
    )


def _prepare_pipeline_state(
    output_dir: Path,
    *,
    signature: str,
    resume: bool,
) -> None:
    state_path = output_dir / "pipeline_state.json"
    previous = _read_json(state_path)
    if resume and previous.get("signature") and previous["signature"] != signature:
        raise RuntimeError(
            f"Pipeline resume signature mismatch in {output_dir}. Use a new "
            "--output-dir when changing data or processing parameters."
        )
    _json_write(
        state_path,
        {
            "signature": signature,
            "status": "running",
            "updated_at": _utc_now(),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="8b")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--backend", default="local", choices=("local", "api"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=os.environ.get("AGENT_API_KEY", "EMPTY"))
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of sample pipelines allowed concurrently.",
    )
    parser.add_argument(
        "--repeat-workers",
        type=int,
        default=1,
        help=(
            "Additional sample-pipeline concurrency multiplier for --backend api. "
            "The local backend does not multiply pipeline workers."
        ),
    )
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--quality-judge-workers", type=int, default=8)
    parser.add_argument("--answer-judge-model-alias", default=os.environ.get("JUDGE_MODEL_ALIAS", "gpt54_internal_azure"))
    parser.add_argument("--quality-judge-model-alias", default=os.environ.get("RL_QUALITY_JUDGE_MODEL_ALIAS", "gpt54_internal_azure"))
    parser.add_argument("--answer-judge-max-tokens", type=int, default=1024)
    parser.add_argument("--quality-judge-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--quality-judge-only",
        action="store_true",
        help=(
            "Run exactly one evidence-consistency LLM audit per candidate question, "
            "then exit without materializing images or running no-tools/tool rollouts."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-timeout", type=int, default=600)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("AGENT_MAX_TURNS", "50")))
    parser.add_argument("--max-tool-calls", type=int, default=int(os.environ.get("AGENT_MAX_TOOL_CALLS", "45")))
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=None,
        help="Persistent image cache directory (default: <output-dir>/images).",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=8,
        help="Concurrent one-time image materialization workers.",
    )
    parser.add_argument(
        "--image-timeout",
        type=float,
        default=60.0,
        help="Connect/read timeout in seconds for one image download.",
    )
    parser.add_argument(
        "--image-retries",
        type=int,
        default=3,
        help="Additional retries for one image download.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed repeat/evaluation/quality artifacts (default: true).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Delete the entire --output-dir before starting and rerun every stage "
            "from scratch. This overrides --resume."
        ),
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="On resume, rerun inference repeats even if repeat_state says completed.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.repetitions != 4:
        logger.warning("The requested design uses four repetitions; got %d", args.repetitions)
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError("--offset and --limit must be non-negative")
    for name in (
        "parallel_workers",
        "repeat_workers",
        "judge_workers",
        "quality_judge_workers",
        "image_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.image_timeout <= 0:
        raise ValueError("--image-timeout must be positive")
    if args.image_retries < 0:
        raise ValueError("--image-retries must be non-negative")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
    )
    os.environ["AGENT_MAX_TURNS"] = str(args.max_turns)
    os.environ["AGENT_MAX_TOOL_CALLS"] = str(args.max_tool_calls)

    vqa_dir = args.vqa_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == vqa_dir:
        raise ValueError("--output-dir must be different from --vqa-dir")
    if args.force_rerun:
        if output_dir.is_symlink():
            raise ValueError("Refusing to force-rerun through a symlink output directory")
        if output_dir in {Path("/"), ROOT}:
            raise ValueError(f"Refusing to delete unsafe output directory: {output_dir}")
        if output_dir.exists():
            logger.warning("Force rerun: deleting output directory %s", output_dir)
            shutil.rmtree(output_dir)
        args.resume = False
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing {questions_path}")

    questions = _jsonl_load(questions_path)
    samples = _jsonl_load(samples_path) if samples_path.exists() else []
    samples_by_id = _sample_by_id(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = (
        args.image_cache_dir.expanduser().resolve()
        if args.image_cache_dir is not None
        else (output_dir / "images").resolve()
    )

    start = min(args.offset, len(questions))
    end = len(questions) if args.limit is None else min(len(questions), start + args.limit)
    selected_source = questions[start:end]

    candidate_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for source_index, question in enumerate(selected_source, start=start):
        sample = samples_by_id.get(str(question.get("sample_id")), {})
        compact = _compact_task_record(question, sample, source_index)
        reasons: list[str] = []
        if question.get("status") != "verified":
            reasons.append("source_status_not_verified")
        if not compact["question"]:
            reasons.append("empty_question")
        if not compact["answer"]:
            reasons.append("empty_answer")
        if compact["id"] in seen_ids:
            reasons.append("duplicate_id")
        normalized_question = re.sub(r"\s+", " ", compact["question"].casefold()).strip()
        if normalized_question and normalized_question in seen_questions:
            reasons.append("duplicate_question")
        if reasons:
            rejected_records.append({**compact, "rejected_reasons": reasons})
            continue
        seen_ids.add(compact["id"])
        seen_questions.add(normalized_question)
        candidate_records.append(compact)

    if args.quality_judge_only:
        quality_path = output_dir / "quality_judge_details.jsonl"
        quality_by_id = _run_quality_judge(
            candidate_records,
            samples_by_id,
            quality_path,
            model_alias=args.quality_judge_model_alias,
            max_tokens=args.quality_judge_max_tokens,
            workers=args.quality_judge_workers,
            resume=args.resume,
            retry_failures=args.retry_failures,
        )
        decisions: dict[str, int] = {"accept": 0, "review": 0, "reject": 0, "error": 0}
        categories: Counter[str] = Counter()
        audit_rows: list[dict[str, Any]] = []
        for record in candidate_records:
            result = quality_by_id.get(record["id"])
            if result is None or not result.get("parse_ok"):
                decisions["error"] += 1
                decision = "error"
            else:
                decision = str(result.get("decision") or "reject")
                decisions[decision] = decisions.get(decision, 0) + 1
            if result:
                categories.update(str(item) for item in result.get("reject_categories") or [])
            audit_rows.append({**record, "quality_judge": result, "quality_decision": decision})
        _jsonl_write(output_dir / "quality_judge_audit.jsonl", audit_rows)
        _json_write(
            output_dir / "quality_judge_report.json",
            {
                "mode": "quality_judge_only",
                "vqa_dir": str(vqa_dir),
                "source_range": [start, end],
                "candidate_count": len(candidate_records),
                "pre_rejected_count": len(rejected_records),
                "decisions": decisions,
                "reject_category_counts": dict(categories),
                "model_alias": args.quality_judge_model_alias,
                "system_prompt_sha256": _sha256(QUALITY_SYSTEM_PROMPT),
                "output": str(quality_path),
            },
        )
        logger.info(
            "Quality-only audit complete: accept=%d review=%d reject=%d error=%d",
            decisions["accept"], decisions["review"], decisions["reject"], decisions["error"],
        )
        return 0

    pipeline_signature = _pipeline_signature(
        args,
        vqa_dir=vqa_dir,
    )
    _prepare_pipeline_state(
        output_dir,
        signature=pipeline_signature,
        resume=args.resume,
    )

    _json_write(
        output_dir / "input_summary.json",
        {
            "vqa_dir": str(vqa_dir),
            "questions_total": len(questions),
            "range": [start, end],
            "candidate_count": len(candidate_records),
            "pre_rejected_count": len(rejected_records),
            "repetitions": args.repetitions,
            "parallel_workers": args.parallel_workers,
            "repeat_workers": args.repeat_workers,
            "scheduler": "sample_pipeline_async",
            "pipeline_workers": args.parallel_workers * (
                args.repeat_workers if args.backend == "api" else 1
            ),
            "image_cache_dir": str(image_cache_dir),
            "image_workers": args.image_workers,
            "resume": args.resume,
            "pipeline_signature": pipeline_signature,
        },
    )

    if not candidate_records:
        _jsonl_write(output_dir / "rejected_questions.jsonl", rejected_records)
        _jsonl_write(output_dir / "accepted_questions.jsonl", [])
        empty_report = {
            "vqa_dir": str(vqa_dir),
            "output_dir": str(output_dir),
            "candidate_count": 0,
            "pre_rejected_count": len(rejected_records),
        }
        _json_write(output_dir / "report.json", empty_report)
        _write_filtered_vqa_dir(
            output_dir,
            source_questions=questions,
            samples_by_id=samples_by_id,
            final_records=[],
            manifest=[],
            report=empty_report,
        )
        _json_write(
            output_dir / "pipeline_state.json",
            {
                "signature": pipeline_signature,
                "status": "completed",
                "candidate_count": 0,
                "finished_at": _utc_now(),
            },
        )
        return 0

    image_cache_report = _prepare_image_cache(
        candidate_records,
        cache_dir=image_cache_dir,
        manifest_path=output_dir / "image_cache_manifest.jsonl",
        workers=args.image_workers,
        timeout=args.image_timeout,
        retries=args.image_retries,
    )
    image_failed_records = image_cache_report.get("failed_samples", [])
    if image_failed_records:
        rejected_records.extend(image_failed_records)
        logger.warning(
            "Filtering %d sample(s) because %d image reference(s) could not be "
            "materialized after retries; continuing with the remaining samples.",
            len(image_failed_records),
            image_cache_report.get("failed_references", 0),
        )
    task_df = _build_inference_dataframe(candidate_records)
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    task_path = staging_dir / "tasks.parquet"
    # Do not let PyArrow write directly to the HDFS-FUSE project mount.  The
    # completed local file is moved back before the inference stages consume it.
    _write_parquet_via_local_tmp(task_df, task_path)

    # The sample-level scheduler calls the existing per-case inference and
    # evaluation functions directly. The runner is initialized once and the
    # worker that owns a sample advances it through all required stages.
    quality_cache_path = output_dir / "quality_judge_details.jsonl"
    quality_signature = _sha256(
        {
            # The quality cache is per sample and must remain reusable when
            # the requested offset/limit is expanded or the judge alias is
            # changed.
            "max_tokens": args.quality_judge_max_tokens,
            "system_prompt": QUALITY_SYSTEM_PROMPT,
            "few_shot": QUALITY_FEW_SHOT,
        }
    )
    quality_state_path = output_dir / "quality_judge_state.json"
    previous_quality_state = _read_json(quality_state_path)
    if (
        args.resume
        and previous_quality_state.get("signature")
        and previous_quality_state["signature"] != quality_signature
    ):
        raise RuntimeError(
            "Quality-judge resume signature mismatch. Use a new --output-dir "
            "when changing the quality judge or its input records."
        )
    _json_write(
        quality_state_path,
        {
            "signature": quality_signature,
            "status": "running",
            "record_count": len(candidate_records),
            "scheduler": "sample_pipeline_async",
            "updated_at": _utc_now(),
        },
    )
    quality_cache = _load_quality_cache(quality_cache_path) if args.resume else {}
    no_tools_by_id, tool_by_id, quality_by_id, pipeline_items = _run_pipeline_async(
        args=args,
        records=candidate_records,
        samples_by_id=samples_by_id,
        task_df=task_df,
        output_dir=output_dir,
        pipeline_signature=pipeline_signature,
        quality_cache=quality_cache,
    )

    non_shortcut: list[dict[str, Any]] = []
    for record in candidate_records:
        no_tools = no_tools_by_id.get(
            record["id"],
            _aggregate_pipeline_repetitions(
                record["id"], [], expected_repetitions=args.repetitions
            ),
        )
        if not no_tools["evaluation_complete"]:
            rejected_records.append(
                {
                    **record,
                    "rejected_reasons": ["no_tools_evaluation_incomplete"],
                    "no_tools_evaluation": no_tools,
                }
            )
        elif no_tools["pass_at_4"]:
            rejected_records.append(
                {
                    **record,
                    "rejected_reasons": ["no_tools_pass_at_4_shortcut"],
                    "no_tools_evaluation": no_tools,
                }
            )
        else:
            non_shortcut.append(record)

    quality_records: list[dict[str, Any]] = []
    for record in non_shortcut:
        tool_result = tool_by_id.get(record["id"])
        if tool_result is None:
            rejected_records.append(
                {**record, "rejected_reasons": ["tool_evaluation_unavailable"]}
            )
            continue
        reps = tool_result.get("repetitions") or []
        infra_errors = sum(
            int(item.get("failure_kind") in {"inference_error", "tool_execution_error"})
            for item in reps
        )
        valid_attempts = len(reps) - infra_errors - sum(
            int(item.get("status") == "missing") for item in reps
        )
        if valid_attempts <= 0:
            rejected_records.append(
                {
                    **record,
                    "rejected_reasons": ["tool_evaluation_unavailable"],
                    "tool_evaluation": tool_result,
                }
            )
            continue
        quality_records.append(record)

    _json_write(
        quality_state_path,
        {
            "signature": quality_signature,
            "status": "completed",
            "record_count": len(quality_records),
            "cached_result_count": sum(
                int(record["id"] in quality_by_id) for record in quality_records
            ),
            "scheduler": "sample_pipeline_async",
            "finished_at": _utc_now(),
        },
    )

    accepted_before_balance: list[dict[str, Any]] = []
    for record in quality_records:
        quality = quality_by_id.get(record["id"], {})
        reasons: list[str] = []
        if not quality.get("parse_ok"):
            reasons.append("quality_judge_parse_error")
        if str(quality.get("decision") or "reject").lower() == "reject":
            reasons.append("quality_judge_reject")
        if reasons:
            rejected_records.append(
                {
                    **record,
                    "rejected_reasons": reasons,
                    "quality_judge": {
                        key: quality.get(key)
                        for key in (
                            "decision",
                            "reject_categories",
                            "reasons",
                            "difficulty_hint",
                            "parse_ok",
                            "error",
                        )
                        if key in quality
                    },
                    "tool_evaluation": tool_by_id[record["id"]],
                }
            )
            continue
        level, difficulty_features = _difficulty_level(
            record,
            tool_by_id[record["id"]],
            quality,
            args.max_turns,
        )
        accepted_before_balance.append(
            {
                **record,
                "quality_judge": {
                    key: quality.get(key)
                    for key in ("decision", "reject_categories", "reasons", "difficulty_hint")
                    if key in quality
                },
                "difficulty_level": level,
                "difficulty_features": difficulty_features,
                "no_tools_pass_at_4": no_tools_by_id[record["id"]]["pass_at_4"],
                "tool_pass_at_4": tool_by_id[record["id"]]["pass_at_4"],
            }
        )

    final_records, balance_report = _select_balanced(
        accepted_before_balance,
        target_count=args.target_count,
        seed=args.seed,
    )
    final_ids = {record["id"] for record in final_records}
    for record in accepted_before_balance:
        if record["id"] not in final_ids:
            rejected_records.append(
                {
                    **record,
                    "rejected_reasons": ["difficulty_balance_not_selected"],
                }
            )

    manifest: list[dict[str, Any]] = []
    final_records_by_id = {record["id"]: record for record in final_records}
    for record in candidate_records:
        case_id = record["id"]
        item = {
            "source_index": record["source_index"],
            "id": case_id,
            "question_id": record.get("question_id"),
            "sample_id": record.get("sample_id"),
            "path_id": record.get("path_id"),
            "selected": case_id in final_records_by_id,
            "rejected_reasons": [],
            "path_features": record.get("path_features"),
            "no_tools": no_tools_by_id.get(case_id),
            "tools": tool_by_id.get(case_id),
            "quality_judge": quality_by_id.get(case_id),
        }
        if not item["selected"]:
            rejected = next((x for x in rejected_records if x.get("id") == case_id), None)
            item["rejected_reasons"] = list((rejected or {}).get("rejected_reasons") or [])
        else:
            item["difficulty_level"] = final_records_by_id[case_id].get("difficulty_level")
            item["difficulty_features"] = final_records_by_id[case_id].get("difficulty_features")
        manifest.append(item)

    _jsonl_write(output_dir / "accepted_questions.jsonl", final_records)
    _jsonl_write(output_dir / "rejected_questions.jsonl", rejected_records)
    _jsonl_write(output_dir / "manifest.jsonl", manifest)
    _jsonl_write(
        output_dir / "quality_judge_details.jsonl",
        [quality_by_id[record["id"]] for record in quality_records if record["id"] in quality_by_id],
    )
    with (output_dir / "selected_indices.txt").open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(f"{record['source_index']}\n")

    report = {
        "vqa_dir": str(vqa_dir),
        "output_dir": str(output_dir),
        "source_range": [start, end],
        "input_count": len(selected_source),
        "pre_candidate_count": len(candidate_records),
        "no_tools_shortcut_count": sum(
            1 for item in rejected_records if "no_tools_pass_at_4_shortcut" in item.get("rejected_reasons", [])
        ),
        "quality_kept_before_balance": len(accepted_before_balance),
        "final_selected_count": len(final_records),
        "rejected_count": len(rejected_records),
        "repetitions": args.repetitions,
        "resume": args.resume,
        "retry_failures": args.retry_failures,
        "parallel_workers": args.parallel_workers,
        "repeat_workers": args.repeat_workers if args.backend == "api" else 1,
        "scheduler": "sample_pipeline_async",
        "pipeline_workers": args.parallel_workers * (
            args.repeat_workers if args.backend == "api" else 1
        ),
        "image_cache": image_cache_report,
        "balance": balance_report,
        "paths": {
            "task_parquet": str(task_path),
            "vqa_dir": str(output_dir),
            "vqa_questions": str(output_dir / "questions.jsonl"),
            "vqa_samples": str(output_dir / "samples.jsonl"),
            "vqa_images": str(image_cache_dir),
            "image_cache_manifest": str(output_dir / "image_cache_manifest.jsonl"),
            "selected_indices": str(output_dir / "selected_indices.txt"),
            "manifest": str(output_dir / "manifest.jsonl"),
            "accepted": str(output_dir / "accepted_questions.jsonl"),
            "rejected": str(output_dir / "rejected_questions.jsonl"),
        },
    }
    _write_filtered_vqa_dir(
        output_dir,
        source_questions=questions,
        samples_by_id=samples_by_id,
        final_records=final_records,
        manifest=manifest,
        report=report,
    )
    _json_write(output_dir / "report.json", report)
    _json_write(
        output_dir / "pipeline_state.json",
        {
            "signature": pipeline_signature,
            "status": "completed",
            "candidate_count": len(candidate_records),
            "final_selected_count": len(final_records),
            "finished_at": _utc_now(),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
