#!/usr/bin/env python3
"""Stress-test a vLLM Qwen3-VL server with long Wikipedia Markdown prompts.

The test has one input phase and one inference phase:

1. Read long Wikipedia ``description`` fields directly from the existing
   ``nodes.jsonl`` file.
2. Submit all descriptions to vLLM at once and keep the server queue full.

No Enhanced Reader or Wikipedia network discovery is used.  This makes a
vLLM failure attributable to the inference server instead of to reader or
discovery latency.

Example:

    python -u debug/stress_vllm_wikipedia.py \
      --nodes-jsonl runs/0712_multi_seed_visual_test_8192_6/nodes.jsonl \
      --vllm-base-url http://10.124.138.0:8000/v1 \
      --model Qwen3-VL-32B \
      --count 1024 \
      --request-timeout 3600 \
      --vllm-log /tmp/vllm.log

The script reads the repository's existing
``runs/0712_multi_seed_visual_test_8192_6/nodes.jsonl`` and keeps the longest
unique Wikipedia text-node descriptions above the configured threshold.  With
``--image``, one random local image is added to each multimodal request.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from dataclasses import dataclass, asdict
import heapq
import json
import mimetypes
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3-VL-32B"
DEFAULT_RESULTS = Path("/tmp/vllm_wikipedia_stress_results.jsonl")
DEFAULT_NODES_JSONL = PROJECT_ROOT / "runs/0712_multi_seed_visual_test_8192_6/nodes.jsonl"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/sharegpt_dataset_refined_v3_rewritten_full_24k_complete/images"
ERROR_PATTERNS = re.compile(
    r"illegal memory|illegal instruction|TMA descriptor|CUDA out of memory|"
    r"EngineCore failed|EngineCore encountered a fatal error|worker.*died",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Page:
    index: int
    url: str
    content: str
    content_chars: int


@dataclass(slots=True)
class VLLMResult:
    index: int
    url: str
    content_chars: int
    started_at: float
    elapsed_s: float
    ok: bool
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_preview: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--health-url", default="", help="Override vLLM health URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--nodes-jsonl",
        type=Path,
        default=DEFAULT_NODES_JSONL,
        help="nodes.jsonl containing Wikipedia text-node descriptions.",
    )
    parser.add_argument(
        "--min-description-chars",
        type=int,
        default=20000,
        help="Keep text nodes whose description has at least this many characters.",
    )
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument(
        "--image",
        action="store_true",
        help="Add one randomly selected local image to every vLLM request.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Image directory used by --image; images are searched recursively.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3600.0,
        help="Per-vLLM-request timeout. Long enough for the 1024-request backlog.",
    )
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--health-interval", type=float, default=2.0)
    parser.add_argument("--vllm-log", type=Path, default=None)
    parser.add_argument(
        "--client-connections",
        type=int,
        default=1024,
        help="Maximum HTTP connections used by the client; keep >= count for a burst test.",
    )
    return parser.parse_args()


def collect_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not paths:
        raise RuntimeError(f"no supported images found under: {image_dir}")
    print(f"[images] directory={image_dir} available={len(paths)}", flush=True)
    return paths


def encode_image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_node_pages(nodes_path: Path, *, count: int, min_description_chars: int) -> list[Page]:
    """Extract the longest unique Wikipedia descriptions without network I/O."""

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes file does not exist: {nodes_path}")
    if count <= 0:
        raise ValueError("--count must be positive")

    # Keep only the top ``count`` full descriptions in memory while streaming
    # the potentially large nodes.jsonl file.
    candidates: list[tuple[int, int, str, str]] = []
    seen_urls: set[str] = set()
    total_text_nodes = 0
    eligible_nodes = 0
    serial = 0
    with nodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                node = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[nodes][skip_invalid_json] line={line_number} error={exc}", flush=True)
                continue
            if node.get("node_type") != "text":
                continue
            total_text_nodes += 1
            description = node.get("description") or ""
            if not isinstance(description, str) or len(description) < min_description_chars:
                continue
            metadata = node.get("metadata") or {}
            url = metadata.get("source_url") or node.get("source_page_url")
            if not isinstance(url, str) or "wikipedia.org/wiki/" not in url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            eligible_nodes += 1
            item = (len(description), serial, url, description)
            serial += 1
            if len(candidates) < count:
                heapq.heappush(candidates, item)
            elif item[0] > candidates[0][0]:
                heapq.heapreplace(candidates, item)

    candidates.sort(key=lambda item: (-item[0], item[2]))
    pages = [
        Page(index=index, url=url, content=description, content_chars=content_chars)
        for index, (content_chars, _, url, description) in enumerate(candidates)
    ]
    print(
        f"[nodes] path={nodes_path} text_nodes={total_text_nodes} "
        f"eligible_unique_urls={eligible_nodes} selected={len(pages)} "
        f"min_description_chars={min_description_chars}",
        flush=True,
    )
    return pages


def health_url(args: argparse.Namespace) -> str:
    if args.health_url:
        return args.health_url
    base = args.vllm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/health"


async def health_monitor(
    args: argparse.Namespace,
    stop: asyncio.Event,
    health_events: list[dict[str, Any]],
) -> None:
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while not stop.is_set():
            started = time.perf_counter()
            try:
                response = await client.get(health_url(args))
                event = {
                    "time": time.time(),
                    "ok": response.is_success,
                    "status_code": response.status_code,
                    "elapsed_s": time.perf_counter() - started,
                }
            except Exception as exc:
                event = {
                    "time": time.time(),
                    "ok": False,
                    "status_code": None,
                    "elapsed_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            health_events.append(event)
            if not event["ok"]:
                print(f"[vllm-health] {json.dumps(event, ensure_ascii=False)}", flush=True)
            await asyncio.sleep(max(0.1, args.health_interval))


async def vllm_log_monitor(
    path: Path | None,
    stop: asyncio.Event,
    events: list[dict[str, Any]],
) -> None:
    if path is None:
        return
    offset = 0
    while not stop.is_set():
        try:
            if path.exists():
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    for line in handle:
                        offset = handle.tell()
                        if ERROR_PATTERNS.search(line):
                            event = {"time": time.time(), "line": line.rstrip("\n")}
                            events.append(event)
                            print(f"[vllm-log-event] {json.dumps(event, ensure_ascii=False)}", flush=True)
        except Exception as exc:
            print(f"[vllm-log-monitor][error] {exc!r}", flush=True)
        await asyncio.sleep(1.0)


async def send_one(
    client: httpx.AsyncClient,
    *,
    args: argparse.Namespace,
    page: Page,
    image_paths: list[Path],
) -> VLLMResult:
    prompt = (
        "Compress the following Wikipedia article into a concise factual summary. "
        "Preserve important entities, dates, numbers, events, and causal relations. "
        "Return only the summary.\n\n"
        f"SOURCE URL: {page.url}\n\n"
        "ARTICLE MARKDOWN:\n"
        f"{page.content}"
    )
    message_content: str | list[dict[str, Any]] = prompt
    if image_paths:
        image_path = random.choice(image_paths)
        image_data_url = await asyncio.to_thread(encode_image_data_url, image_path)
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
        "stream": False,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    started_at = time.time()
    started = time.perf_counter()
    endpoint = args.vllm_base_url.rstrip("/") + "/chat/completions"
    try:
        response = await client.post(endpoint, json=payload)
        elapsed = time.perf_counter() - started
        body = response.text[:4000]
        data: dict[str, Any] = {}
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            pass
        usage = data.get("usage") or {}
        error_message = data.get("error")
        ok = response.is_success and not ERROR_PATTERNS.search(body)
        return VLLMResult(
            index=page.index,
            url=page.url,
            content_chars=page.content_chars,
            started_at=started_at,
            elapsed_s=elapsed,
            ok=ok,
            status_code=response.status_code,
            error_type=None if ok else "http_or_server_error",
            error=str(error_message or body[:1000]) if not ok else None,
            prompt_tokens=int(usage.get("prompt_tokens")) if usage.get("prompt_tokens") is not None else None,
            completion_tokens=(
                int(usage.get("completion_tokens"))
                if usage.get("completion_tokens") is not None
                else None
            ),
            response_preview=body[:500] if not ok else None,
        )
    except Exception as exc:
        return VLLMResult(
            index=page.index,
            url=page.url,
            content_chars=page.content_chars,
            started_at=started_at,
            elapsed_s=time.perf_counter() - started,
            ok=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )


async def run_vllm_burst(
    args: argparse.Namespace,
    pages: list[Page],
    image_paths: list[Path],
) -> tuple[list[VLLMResult], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_pages = [page for page in pages if page.content_chars > 0 and page.content]
    if not valid_pages:
        raise RuntimeError("No valid node descriptions are available for the vLLM phase")

    limits = httpx.Limits(
        max_connections=max(1, args.client_connections),
        max_keepalive_connections=max(1, min(args.client_connections, 256)),
    )
    timeout = httpx.Timeout(args.request_timeout, connect=60.0, read=args.request_timeout, write=300.0)
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    stop = asyncio.Event()
    health_events: list[dict[str, Any]] = []
    log_events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:
        health_task = asyncio.create_task(health_monitor(args, stop, health_events))
        log_task = asyncio.create_task(vllm_log_monitor(args.vllm_log, stop, log_events))
        started = time.perf_counter()
        print(
            f"[vllm] submitting={len(valid_pages)} requests at once "
            f"client_connections={args.client_connections} endpoint={args.vllm_base_url}",
            flush=True,
        )
        tasks = [
            asyncio.create_task(
                send_one(client, args=args, page=page, image_paths=image_paths)
            )
            for page in valid_pages
        ]
        results: list[VLLMResult] = []
        for done_count, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            results.append(result)
            if not result.ok:
                print(f"[vllm-error] {json.dumps(asdict(result), ensure_ascii=False)}", flush=True)
            if done_count % 25 == 0 or done_count == len(tasks):
                ok = sum(item.ok for item in results)
                elapsed = time.perf_counter() - started
                print(
                    f"[vllm] completed={done_count}/{len(tasks)} success={ok} "
                    f"failed={done_count - ok} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
        stop.set()
        await asyncio.gather(health_task, log_task, return_exceptions=True)
    results.sort(key=lambda item: item.index)
    return results, health_events, log_events


def write_results(
    args: argparse.Namespace,
    pages: list[Page],
    results: list[VLLMResult],
    health_events: list[dict[str, Any]],
    log_events: list[dict[str, Any]],
) -> None:
    args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.results_jsonl.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({"type": "request", **asdict(result)}, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"type": "health_events", "events": health_events}, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"type": "vllm_log_events", "events": log_events}, ensure_ascii=False) + "\n")
        handle.write(
            json.dumps(
                {
                    "type": "config",
                    "model": args.model,
                    "vllm_base_url": args.vllm_base_url,
                    "nodes_jsonl": str(args.nodes_jsonl),
                    "min_description_chars": args.min_description_chars,
                    "image_enabled": bool(args.image),
                    "image_dir": str(args.image_dir) if args.image else None,
                    "requested_count": args.count,
                    "node_pages": sum(page.content_chars > 0 for page in pages),
                    "vllm_submitted": len(results),
                    "vllm_success": sum(result.ok for result in results),
                    "vllm_failed": sum(not result.ok for result in results),
                    "max_output_tokens": args.max_output_tokens,
                    "temperature": args.temperature,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def print_summary(pages: list[Page], results: list[VLLMResult], health_events: list[dict[str, Any]], log_events: list[dict[str, Any]]) -> None:
    successful_pages = [page for page in pages if page.content_chars > 0]
    successful_requests = [result for result in results if result.ok]
    failed_requests = [result for result in results if not result.ok]
    latencies = sorted(result.elapsed_s for result in results)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    print("=== vLLM Wikipedia stress summary ===", flush=True)
    print(f"node_pages: {len(pages)}", flush=True)
    print(f"node_pages_with_content: {len(successful_pages)}", flush=True)
    print(f"vllm_submitted: {len(results)}", flush=True)
    print(f"vllm_success: {len(successful_requests)}", flush=True)
    print(f"vllm_failed: {len(failed_requests)}", flush=True)
    print(f"latency_s: p50={p50:.3f} p95={p95:.3f} max={(max(latencies) if latencies else 0.0):.3f}", flush=True)
    print(f"health_failures: {sum(not event.get('ok') for event in health_events)}", flush=True)
    print(f"vllm_log_error_events: {len(log_events)}", flush=True)
    if failed_requests:
        print("failure_types:", dict(Counter(result.error_type or "unknown" for result in failed_requests)), flush=True)
        for result in failed_requests[:10]:
            print(f"failure: {json.dumps(asdict(result), ensure_ascii=False)}", flush=True)


async def async_main(args: argparse.Namespace) -> int:
    pages = extract_node_pages(
        args.nodes_jsonl,
        count=args.count,
        min_description_chars=args.min_description_chars,
    )
    if len(pages) < args.count:
        raise RuntimeError(
            f"Only collected {len(pages)} unique long descriptions, but --count={args.count}. "
            "Lower --min-description-chars or provide a larger nodes.jsonl."
        )
    print(f"[nodes] selected={len(pages)} descriptions; submitting them directly", flush=True)
    image_paths = collect_image_paths(args.image_dir) if args.image else []
    if image_paths:
        print("[images] one random image will be inserted into every request", flush=True)
    results, health_events, log_events = await run_vllm_burst(args, pages, image_paths)
    write_results(args, pages, results, health_events, log_events)
    print_summary(pages, results, health_events, log_events)
    print(f"[results] {args.results_jsonl}", flush=True)
    return 0 if results and all(result.ok for result in results) else 2


def main() -> int:
    args = parse_args()
    if args.client_connections < 1:
        raise SystemExit("--client-connections must be positive")
    if args.max_output_tokens < 1 or args.count < 1:
        raise SystemExit("--count and --max-output-tokens must be positive")

    def stop_handler(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; waiting for current requests to finish", flush=True)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
