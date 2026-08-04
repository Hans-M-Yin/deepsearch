#!/usr/bin/env python3
"""Keep a vLLM OpenAI-compatible server busy with configurable synthetic load.

The program is intentionally a client-side load generator: it never changes the
server configuration and it is safe to stop with Ctrl-C.  Each request starts
with a unique nonce and then contains a different synthetic prompt.  This is
important when vLLM prefix caching is enabled: varying only a suffix can still
reuse the cached prefill for a long common prefix.

Examples:
  # Use the server configured by opensearch_vl/serve_vllm.sh
  python utils/vllm_gpu_keepalive.py --model OpenSearch-VL-8B

  # A bounded, higher-throughput run
  python utils/vllm_gpu_keepalive.py --base-url http://host:8000/v1 \\
      --model Qwen3-VL-8B --concurrency 16 --prompt-tokens 2048 \\
      --max-tokens 512 --duration 30m
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import secrets
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


DEFAULT_WORDS = (
    "amber basin cedar delta ember forest granite harbor ivory juniper "
    "keystone lantern meadow north orbit prairie quartz river summit "
    "tundra umber valley willow xenon yarrow zephyr"
).split()


def parse_duration(value: str) -> float:
    """Parse a positive duration such as ``90``, ``15m`` or ``2h`` in seconds."""
    units = {"s": 1, "m": 60, "h": 3600}
    value = value.strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("duration cannot be empty")
    suffix = value[-1]
    try:
        seconds = float(value[:-1]) * units[suffix] if suffix in units else float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use seconds, or a number ending in s, m, or h") from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return seconds


def random_prompt(token_count: int, request_id: int) -> str:
    """Build a request-unique prompt whose variation begins before token block 1."""
    rng = random.Random(secrets.randbits(128))
    nonce = secrets.token_urlsafe(18)
    words = [f"request_nonce_{nonce}", f"sequence_{request_id}"]
    words.extend(f"{rng.choice(DEFAULT_WORDS)}_{rng.randrange(1_000_000_000)}" for _ in range(token_count))
    return (
        " ".join(words)
        + "\nSynthetic inference workload. Produce a concise independent analysis "
        "of the synthetic sequence."
    )


def build_payload(args: argparse.Namespace, request_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": random_prompt(args.prompt_tokens, request_id)}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.seed is not None:
        # Vary server-side sampling too; this does not replace prompt variation.
        payload["seed"] = args.seed + request_id
    if args.extra_body:
        try:
            extra = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--extra-body must be a JSON object: {exc}") from exc
        if not isinstance(extra, dict):
            raise ValueError("--extra-body must be a JSON object")
        payload.update(extra)
    return payload


def post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None) -> tuple[int, int]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status, len(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection error: {exc.reason}") from exc


@dataclass
class Stats:
    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    total_latency: float = 0.0
    errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, latency: float, error: str | None = None) -> None:
        with self.lock:
            if error is None:
                self.succeeded += 1
                self.total_latency += latency
            else:
                self.failed += 1
                if len(self.errors) < 5:
                    self.errors.append(error)


def run_request(args: argparse.Namespace, request_id: int, stats: Stats) -> None:
    started = time.monotonic()
    try:
        post_json(args.endpoint, build_payload(args, request_id), args.timeout, args.api_key)
    except (RuntimeError, ValueError) as exc:
        stats.record(time.monotonic() - started, str(exc))
    else:
        stats.record(time.monotonic() - started)


def print_status(stats: Stats, started: float) -> None:
    with stats.lock:
        elapsed = max(time.monotonic() - started, 0.001)
        mean_latency = stats.total_latency / stats.succeeded if stats.succeeded else 0.0
        print(
            f"[keepalive] elapsed={elapsed:.0f}s submitted={stats.submitted} "
            f"ok={stats.succeeded} failed={stats.failed} "
            f"req/s={stats.succeeded / elapsed:.2f} mean_latency={mean_latency:.2f}s",
            flush=True,
        )
        for error in stats.errors:
            print(f"[keepalive] recent error: {error}", flush=True)
        stats.errors.clear()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:6657/v1", help="OpenAI-compatible vLLM base URL")
    parser.add_argument("--endpoint", help="Full endpoint override (defaults to BASE_URL/chat/completions)")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--api-key", help="Optional Bearer token")
    parser.add_argument("--concurrency", type=int, default=4, help="In-flight requests (default: 4)")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Approximate random prompt words per request")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens per request")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--seed", type=int, help="Optional base seed; each request gets seed + request id")
    parser.add_argument("--duration", type=parse_duration, help="Stop after e.g. 30m; default is until Ctrl-C")
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds")
    parser.add_argument("--report-interval", type=float, default=10, help="Status interval in seconds")
    parser.add_argument("--extra-body", help="JSON object merged into every OpenAI request body")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.concurrency < 1 or args.prompt_tokens < 1 or args.max_tokens < 1 or args.timeout <= 0:
        raise SystemExit("--concurrency, --prompt-tokens, --max-tokens and --timeout must be positive")
    args.endpoint = args.endpoint or args.base_url.rstrip("/") + "/chat/completions"
    try:
        build_payload(args, 0)  # Validate --extra-body before load begins.
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    stats, started = Stats(), time.monotonic()
    deadline = started + args.duration if args.duration else None
    print(f"[keepalive] endpoint={args.endpoint} model={args.model} concurrency={args.concurrency}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending: set[concurrent.futures.Future[None]] = set()
        request_id, next_report = 0, started + args.report_interval
        while not stopped.is_set() and (deadline is None or time.monotonic() < deadline):
            while len(pending) < args.concurrency and not stopped.is_set() and (deadline is None or time.monotonic() < deadline):
                with stats.lock:
                    stats.submitted += 1
                pending.add(executor.submit(run_request, args, request_id, stats))
                request_id += 1
            timeout = max(0.05, min(1.0, next_report - time.monotonic()))
            done, pending = concurrent.futures.wait(pending, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            if time.monotonic() >= next_report:
                print_status(stats, started)
                next_report = time.monotonic() + args.report_interval
    print_status(stats, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
