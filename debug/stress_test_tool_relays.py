#!/usr/bin/env python3
"""Concurrency benchmark for the project's tool backends and relays.

This script deliberately calls the same Python tool implementations used by
the SFT agent.  It does *not* call ``google.serper.dev`` directly, so the
configured Serper relay, key handling, retry policy, and debug instrumentation
are all included in the measurement.

Currently implemented:

* ``--backend serper --tool t2t_search``: read ``final_question`` from a
  question JSONL and use a random contiguous 2--5 word phrase as each query;
* ``--backend serper --tool i2i_search``: list images below the
  ``vision_deepresearch/`` OSS prefix with the authenticated OSS API and send
  their public URLs to the project's reverse-image-search tool.
* ``--backend oss --tool oss_upload``: read local dataset images and call the
  project's existing OSS uploader, including its ``OSS_RELAY_URL`` path.

``t2i_search`` is intentionally rejected for now.  The command-line shape is
kept stable so more backends (Firecrawl, HTTP, OSS, Enhanced Reader) can be
added without replacing the benchmark driver.

Examples::

    python debug/stress_test_tool_relays.py \
      --backend serper --tool t2t_search \
      --question-list runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/questions.jsonl \
      --concurrencies 16,32,64,128,256 --requests-per-level 256 \
      --output /tmp/serper_t2t_stress.json

    python debug/stress_test_tool_relays.py \
      --backend serper --tool i2i_search \
      --oss-prefix vision_deepresearch/ --max-images 10000 \
      --concurrencies 16,32,64,128,256 --requests-per-level 256 \
      --output /tmp/serper_i2i_stress.json

    python debug/stress_test_tool_relays.py \
      --backend oss --tool oss_upload \
      --image-dir data/sharegpt_dataset_refined_v3_rewritten_full_24k_complete/images \
      --max-images 10000 \
      --concurrencies 16,32,64,128,256 --requests-per-level 256 \
      --output /tmp/oss_upload_stress.json

The direct OSS listing requires ``oss2`` and the usual credentials:
``OSS_ACCESS_KEY_ID``, ``OSS_ACCESS_KEY_SECRET``, ``OSS_ENDPOINT``, and
``OSS_BUCKET_NAME``.  ``OSS_PUBLIC_ENDPOINT`` can be set separately when the
endpoint used for authenticated listing differs from the public URL endpoint
(for example, an OSS acceleration endpoint).  A private bucket can be listed,
but its generated URLs must be publicly fetchable by Serper Lens for i2i
requests to return matches.

The OSS upload mode uses ``OSS_RELAY_URL`` when configured; otherwise it uses
the project's direct OSS credentials and endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_QUESTION_LIST = PROJECT_ROOT / "runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/question.jsonl"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/sharegpt_dataset_refined_v3_rewritten_full_24k_complete/images"
DEFAULT_OSS_PREFIX = "vision_deepresearch/"
DEFAULT_CONCURRENCIES = (16, 32, 64, 128, 256)
DEFAULT_REQUESTS_PER_LEVEL = 256
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}

# Keep URLs such as ``World Cup 2026`` and contractions such as ``Messi's``
# intact while making the phrase sampler independent of punctuation.
_WORD_RE = re.compile(r"[\w]+(?:['’\-][\w]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class ImageObject:
    key: str
    url: str


def _resolve_question_list(path: Path) -> Path:
    """Resolve the historical singular ``question.jsonl`` name gracefully."""

    if path.exists():
        return path
    if path.name == "question.jsonl":
        plural = path.with_name("questions.jsonl")
        if plural.exists():
            print(
                f"[input] {path} was not found; using existing {plural}",
                file=sys.stderr,
                flush=True,
            )
            return plural
    raise FileNotFoundError(f"question JSONL does not exist: {path}")


def _read_questions(path: Path, field: str) -> tuple[list[str], dict[str, int]]:
    questions: list[str] = []
    stats = {"lines": 0, "valid": 0, "missing_field": 0, "invalid_json": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue
            value = record.get(field) if isinstance(record, dict) else None
            if not isinstance(value, str) or not value.strip():
                # A few historical question files used ``question`` rather
                # than ``final_question``.  Keep the requested field primary,
                # but do not silently throw away usable rows.
                if field != "question":
                    value = record.get("question") if isinstance(record, dict) else None
            if not isinstance(value, str) or not value.strip():
                stats["missing_field"] += 1
                continue
            questions.append(" ".join(value.split()))
            stats["valid"] += 1
    if not questions:
        raise ValueError(
            f"no usable questions found in {path} (field={field!r}, stats={stats})"
        )
    return questions, stats


def _sample_phrase(question: str, rng: random.Random, *, min_words: int = 2, max_words: int = 5) -> str:
    words = _WORD_RE.findall(question)
    if not words:
        return question.strip()
    if len(words) < min_words:
        return " ".join(words)
    phrase_size = rng.randint(min_words, min(max_words, len(words)))
    start = rng.randint(0, len(words) - phrase_size)
    return " ".join(words[start : start + phrase_size])


def _normalise_endpoint(endpoint: str) -> tuple[str, str]:
    value = endpoint.strip()
    if not value:
        raise ValueError("OSS endpoint is empty")
    if "://" in value:
        scheme, host = value.split("://", 1)
    else:
        scheme, host = "https", value
    return scheme.lower(), host.rstrip("/")


def _oss_public_url(*, bucket_name: str, endpoint: str, key: str) -> str:
    scheme, host = _normalise_endpoint(endpoint)
    # OSS virtual-hosted style is also the form used by the existing uploader.
    encoded_key = quote(key, safe="/~!$&'()*+,;=:@")
    return f"{scheme}://{bucket_name}.{host}/{encoded_key}"


def _list_oss_images(
    *,
    prefix: str,
    max_images: int,
    seed: int,
    include_non_images: bool = False,
) -> list[ImageObject]:
    """List a bounded, shuffled image sample from OSS.

    This is a read-only operation.  The full bucket is not downloaded: only
    object metadata/keys are iterated, which keeps memory bounded by
    ``max_images``.  The current implementation intentionally uses direct
    ``oss2`` credentials because ``debug/oss_relay.py`` exposes upload only.
    """

    try:
        import oss2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "oss2 is required for i2i OSS listing; install it on the benchmark host"
        ) from exc

    access_key_id = os.environ.get("OSS_ACCESS_KEY_ID", "").strip()
    access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET", "").strip()
    endpoint = os.environ.get("OSS_ENDPOINT", "").strip()
    bucket_name = os.environ.get("OSS_BUCKET_NAME", "").strip()
    missing = [
        name
        for name, value in (
            ("OSS_ACCESS_KEY_ID", access_key_id),
            ("OSS_ACCESS_KEY_SECRET", access_key_secret),
            ("OSS_ENDPOINT", endpoint),
            ("OSS_BUCKET_NAME", bucket_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "missing OSS environment variables for i2i listing: " + ", ".join(missing)
        )

    public_endpoint = (
        os.environ.get("OSS_PUBLIC_ENDPOINT", "").strip()
        or os.environ.get("OSS_ENDPOINT_ACCELERATE", "").strip()
        or endpoint
    )
    auth = oss2.Auth(access_key_id, access_key_secret)
    scheme, endpoint_host = _normalise_endpoint(endpoint)
    bucket = oss2.Bucket(
        auth,
        f"{scheme}://{endpoint_host}",
        bucket_name,
        connect_timeout=60,
    )

    # Reservoir sampling avoids systematically selecting only the oldest
    # objects when a prefix contains millions of files.  It still performs one
    # metadata listing pass, but never downloads object bodies.
    rng = random.Random(seed)
    selected: list[str] = []
    scanned = 0
    for item in oss2.ObjectIterator(bucket, prefix=prefix):
        key = str(getattr(item, "key", "") or "")
        if not key or key.endswith("/"):
            continue
        suffix = Path(key).suffix.lower()
        if not include_non_images and suffix not in IMAGE_SUFFIXES:
            continue
        scanned += 1
        if len(selected) < max_images:
            selected.append(key)
            continue
        replacement = rng.randrange(scanned)
        if replacement < max_images:
            selected[replacement] = key
    rng.shuffle(selected)
    objects = [
        ImageObject(
            key=key,
            url=_oss_public_url(bucket_name=bucket_name, endpoint=public_endpoint, key=key),
        )
        for key in selected
    ]
    if not objects:
        raise RuntimeError(f"no image objects found under OSS prefix {prefix!r}")
    print(
        f"[oss-list] bucket={bucket_name!r} prefix={prefix!r} scanned={scanned} "
        f"selected={len(objects)} public_endpoint={public_endpoint!r}",
        file=sys.stderr,
        flush=True,
    )
    return objects


def _list_local_images(*, directory: Path, max_images: int, seed: int) -> list[Path]:
    """Return a shuffled bounded list of local image files.

    ``images`` in ShareGPT datasets is often a symlink to a shared HDFS image
    directory, so resolve it before recursively walking it.
    """

    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"local image directory does not exist: {directory}")
    rng = random.Random(seed)
    selected: list[Path] = []
    scanned = 0
    # Use os.walk rather than Path.rglob: the image directory can be an HDFS
    # symlink containing tens of thousands of small files, where constructing
    # one Path object per entry is unnecessarily expensive.
    for root, _directories, filenames in os.walk(resolved):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            scanned += 1
            if len(selected) < max_images:
                selected.append(path)
                continue
            replacement = rng.randrange(scanned)
            if replacement < max_images:
                selected[replacement] = path
    if not selected:
        raise RuntimeError(f"no image files found under {directory}")
    rng.shuffle(selected)
    print(
        f"[local-images] directory={directory!r} resolved={str(resolved)!r} "
        f"found={scanned} selected={len(selected)}",
        file=sys.stderr,
        flush=True,
    )
    return selected


def _parse_concurrencies(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        number = int(raw)
        if number <= 0:
            raise ValueError("concurrency must be positive")
        values.append(number)
    if not values:
        raise ValueError("at least one concurrency level is required")
    return tuple(dict.fromkeys(values))


def _input_hash(value: str) -> str:
    return sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _result_count(tool: str, result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    if tool == "t2t_search":
        value = result.get("results")
    else:
        value = result.get("matches")
    return len(value) if isinstance(value, list) else (1 if value else 0)


def _invoke(tool: str, value: str, *, include_input: bool) -> dict[str, Any]:
    """Invoke one real tool and return only compact benchmark metadata."""

    started = time.perf_counter()
    payload: dict[str, Any] = {
        "tool": tool,
        "input_hash": _input_hash(value),
    }
    if include_input:
        payload["input"] = value
    try:
        # Import lazily, so --help and argument validation work on hosts that
        # do not have the full SFT optional dependency set installed.
        if tool == "oss_upload":
            from opensearch_vl.opensearch_infer.upload import upload_cos

            image_path = Path(value)
            input_bytes = image_path.stat().st_size
            suffix = image_path.suffix.lower() or ".bin"
            filename = f"relay_stress_{_input_hash(str(image_path))}{suffix}"
            object_key, public_url = upload_cos(
                str(image_path),
                filename,
                time.strftime("%Y-%m-%d", time.gmtime()),
                "relay_stress_oss",
                "relay-stress",
                use_direct_url=True,
            )
            uploaded = bool(object_key or public_url)
            payload.update(
                {
                    "transport_ok": True,
                    "semantic_ok": uploaded,
                    "result_count": 1 if uploaded else 0,
                    "backend_error": None if uploaded else "empty_upload_result",
                    "input_bytes": input_bytes,
                }
            )
            if include_input and public_url:
                payload["uploaded_url"] = public_url
            if include_input and object_key:
                payload["object_key"] = object_key
            payload["elapsed_s"] = time.perf_counter() - started
            return payload

        from synthesis.sft.tools import i2i_search, t2t_search

        if tool == "t2t_search":
            result = t2t_search(query=value)
        elif tool == "i2i_search":
            result = i2i_search(image_url=value)
        else:  # defensive; argparse rejects this before execution
            raise ValueError(f"unsupported tool: {tool}")
        count = _result_count(tool, result)
        transport_ok = True
        semantic_ok = bool(isinstance(result, dict) and result.get("ok") and count > 0)
        payload.update(
            {
                "transport_ok": transport_ok,
                "semantic_ok": semantic_ok,
                "result_count": count,
                # Keep backend errors compact; do not store full raw results.
                "backend_error": (
                    str(result.get("error"))[:500]
                    if isinstance(result, dict) and result.get("error")
                    else None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on live relays
        payload.update(
            {
                "transport_ok": False,
                "semantic_ok": False,
                "result_count": 0,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )
    payload["elapsed_s"] = time.perf_counter() - started
    return payload


def _make_inputs(
    *,
    tool: str,
    count: int,
    questions: Sequence[str] | None,
    images: Sequence[ImageObject] | None,
    local_images: Sequence[Path] | None,
    rng: random.Random,
) -> list[str]:
    if tool == "t2t_search":
        if not questions:
            raise ValueError("t2t_search requires a non-empty question list")
        return [_sample_phrase(rng.choice(questions), rng) for _ in range(count)]
    if tool == "i2i_search":
        if not images:
            raise ValueError("i2i_search requires a non-empty OSS image list")
        order = list(range(len(images)))
        rng.shuffle(order)
        values: list[str] = []
        for index in range(count):
            values.append(images[order[index % len(order)]].url)
            if (index + 1) % len(order) == 0:
                rng.shuffle(order)
        return values
    if tool == "oss_upload":
        if not local_images:
            raise ValueError("oss_upload requires a non-empty local image list")
        order = list(range(len(local_images)))
        rng.shuffle(order)
        values = []
        for index in range(count):
            values.append(str(local_images[order[index % len(order)]]))
            if (index + 1) % len(order) == 0:
                rng.shuffle(order)
        return values
    raise ValueError(f"unsupported tool: {tool}")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _failure_label(detail: dict[str, Any]) -> str:
    """Collapse variable request text into a useful failure category."""

    raw = str(detail.get("error") or detail.get("backend_error") or "").strip()
    lowered = raw.lower()
    if "http 429" in lowered or "status code: 429" in lowered:
        return "HTTP 429"
    if "http 401" in lowered or "status code: 401" in lowered:
        return "HTTP 401"
    if "http 403" in lowered or "status code: 403" in lowered:
        return "HTTP 403"
    if re.search(r"http 5\d\d|status code: 5\d\d", lowered):
        return "HTTP 5xx"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "connection" in lowered or "connecterror" in lowered:
        return "connection_error"
    if detail.get("error_type"):
        return str(detail["error_type"])
    if raw:
        return "backend_error"
    return "empty_or_not_ok"


def _failure_diagnostics(details: Sequence[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    failed = [item for item in details if not item.get("transport_ok")]
    counts = Counter(_failure_label(item) for item in failed)
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in failed:
        label = _failure_label(item)
        if label in seen:
            continue
        seen.add(label)
        sample = {
            "reason": label,
            "input_hash": item.get("input_hash"),
            "error_type": item.get("error_type"),
            "error": str(item.get("error") or item.get("backend_error") or "")[:1000],
        }
        if "input" in item:
            sample["input"] = item["input"]
        samples.append(sample)
        if len(samples) >= 10:
            break
    return dict(counts), samples


def _run_level(
    *,
    tool: str,
    concurrency: int,
    inputs: Sequence[str],
    include_inputs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    level_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="relay-test") as executor:
        futures = [
            executor.submit(_invoke, tool, value, include_input=include_inputs)
            for value in inputs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_s = time.perf_counter() - level_started
    elapsed = [float(item["elapsed_s"]) for item in results]
    transport_ok = sum(bool(item.get("transport_ok")) for item in results)
    semantic_ok = sum(bool(item.get("semantic_ok")) for item in results)
    summary = {
        "concurrency": concurrency,
        "requests": len(results),
        "transport_ok": transport_ok,
        "transport_failed": len(results) - transport_ok,
        "semantic_ok": semantic_ok,
        "semantic_failed_or_empty": len(results) - semantic_ok,
        "transport_success_rate": transport_ok / len(results) if results else 0.0,
        "semantic_success_rate": semantic_ok / len(results) if results else 0.0,
        "wall_s": wall_s,
        "throughput_rps": len(results) / wall_s if wall_s > 0 else 0.0,
        "latency_s": {
            "mean": mean(elapsed) if elapsed else None,
            "p50": _percentile(elapsed, 0.50),
            "p95": _percentile(elapsed, 0.95),
            "p99": _percentile(elapsed, 0.99),
            "max": max(elapsed) if elapsed else None,
        },
    }
    total_input_bytes = sum(int(item.get("input_bytes", 0) or 0) for item in results)
    if total_input_bytes:
        summary["input_bytes_total"] = total_input_bytes
        summary["input_mib_per_s"] = (
            total_input_bytes / (1024 * 1024) / wall_s if wall_s > 0 else 0.0
        )
    failure_reasons, failure_samples = _failure_diagnostics(results)
    summary["failure_reason_counts"] = failure_reasons
    summary["failure_samples"] = failure_samples
    return summary, results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("serper", "oss"), default="serper")
    parser.add_argument(
        "--tool",
        choices=("t2t_search", "t2i_search", "i2i_search", "oss_upload"),
        required=True,
        help="Tool to invoke through the project's implementation/relay.",
    )
    parser.add_argument(
        "--question-list",
        type=Path,
        default=DEFAULT_QUESTION_LIST,
        help="JSONL containing final_question values for t2t_search.",
    )
    parser.add_argument("--question-field", default="final_question")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Local dataset image directory for --backend oss --tool oss_upload.",
    )
    parser.add_argument(
        "--concurrencies",
        default=",".join(str(value) for value in DEFAULT_CONCURRENCIES),
        help="Comma-separated concurrency levels, e.g. 16,32,64,128,256.",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        help="Total calls at each concurrency level (default: 256).",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output containing summaries and compact per-call results.",
    )
    parser.add_argument(
        "--include-inputs",
        action="store_true",
        help="Include generated queries/image URLs in per-call output (can be large).",
    )
    parser.add_argument("--oss-prefix", default=DEFAULT_OSS_PREFIX)
    parser.add_argument(
        "--max-images",
        type=int,
        default=10000,
        help="Maximum image keys retained from OSS listing (default: 10000).",
    )
    parser.add_argument(
        "--include-non-images",
        action="store_true",
        help="Include every object under the prefix instead of image suffixes only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.tool == "t2i_search":
        parser.error("t2i_search 压测暂未实现；当前 Serper 支持 t2t_search/i2i_search，OSS 支持 oss_upload")
    if args.backend == "serper" and args.tool == "oss_upload":
        parser.error("oss_upload must be used with --backend oss")
    if args.backend == "oss" and args.tool != "oss_upload":
        parser.error("--backend oss currently only supports --tool oss_upload")
    if args.requests_per_level <= 0:
        parser.error("--requests-per-level must be positive")
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    try:
        concurrencies = _parse_concurrencies(args.concurrencies)
    except ValueError as exc:
        parser.error(str(exc))

    rng = random.Random(args.seed)
    questions: list[str] | None = None
    images: list[ImageObject] | None = None
    local_images: list[Path] | None = None
    input_stats: dict[str, Any] = {}
    if args.tool == "t2t_search":
        question_path = _resolve_question_list(args.question_list)
        questions, question_stats = _read_questions(question_path, args.question_field)
        input_stats = {
            "question_list": str(question_path),
            "question_field": args.question_field,
            "question_stats": question_stats,
        }
        print(
            f"[input] t2t questions={len(questions)} path={question_path}",
            file=sys.stderr,
            flush=True,
        )
    elif args.tool == "i2i_search":
        try:
            images = _list_oss_images(
                prefix=args.oss_prefix,
                max_images=args.max_images,
                seed=args.seed,
                include_non_images=args.include_non_images,
            )
        except Exception as exc:
            parser.error(str(exc))
        input_stats = {
            "oss_prefix": args.oss_prefix,
            "oss_images": len(images or []),
        }
    else:
        try:
            local_images = _list_local_images(
                directory=args.image_dir,
                max_images=args.max_images,
                seed=args.seed,
            )
        except Exception as exc:
            parser.error(str(exc))
        input_stats = {
            "image_dir": str(args.image_dir),
            "local_images": len(local_images or []),
        }

    report: dict[str, Any] = {
        "backend": args.backend,
        "tool": args.tool,
        "seed": args.seed,
        "requests_per_level": args.requests_per_level,
        "concurrencies": list(concurrencies),
        "input": input_stats,
        "levels": [],
    }
    all_details: list[dict[str, Any]] = []
    for level_index, concurrency in enumerate(concurrencies, start=1):
        # Separate RNG streams keep the request order independent of the
        # number of workers used in a preceding level.
        level_rng = random.Random(args.seed + level_index * 1_000_003)
        inputs = _make_inputs(
            tool=args.tool,
            count=args.requests_per_level,
            questions=questions,
            images=images,
            local_images=local_images,
            rng=level_rng,
        )
        print(
            f"[stress] level={level_index}/{len(concurrencies)} tool={args.tool} "
            f"concurrency={concurrency} requests={len(inputs)}",
            file=sys.stderr,
            flush=True,
        )
        summary, details = _run_level(
            tool=args.tool,
            concurrency=concurrency,
            inputs=inputs,
            include_inputs=args.include_inputs,
        )
        summary["level"] = level_index
        report["levels"].append(summary)
        for detail in details:
            detail["level"] = level_index
            detail["concurrency"] = concurrency
        all_details.extend(details)
        latency = summary["latency_s"]
        print(
            f"[stress] concurrency={concurrency} "
            f"transport={summary['transport_ok']}/{summary['requests']} "
            f"semantic={summary['semantic_ok']}/{summary['requests']} "
            f"wall_s={summary['wall_s']:.3f} rps={summary['throughput_rps']:.2f} "
            f"latency_mean_s={latency['mean']:.3f} p95_s={latency['p95']:.3f}",
            file=sys.stderr,
            flush=True,
        )
        for reason, count in sorted(
            summary.get("failure_reason_counts", {}).items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(
                f"[stress-error] concurrency={concurrency} count={count} reason={reason}",
                file=sys.stderr,
                flush=True,
            )
        for sample in summary.get("failure_samples", []):
            message = str(sample.get("error") or "")
            print(
                f"[stress-error-sample] concurrency={concurrency} "
                f"reason={sample.get('reason')} input_hash={sample.get('input_hash')} "
                f"error={message[:500]}",
                file=sys.stderr,
                flush=True,
            )
    report["results"] = all_details
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(f"[stress] report={args.output}", file=sys.stderr, flush=True)
    print(json.dumps({key: report[key] for key in ("backend", "tool", "input", "levels")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
