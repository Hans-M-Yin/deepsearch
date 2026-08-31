#!/usr/bin/env python3
"""Continuous long-context concurrency stress test for a vLLM server.

This intentionally sends text-only chat-completion requests with random
8k--12k-token prompts.  It is meant to run beside a vLLM process while the
operator watches GPU utilization, KV-cache usage, throughput, and server
logs for CUDA errors such as ``illegal memory access``.

Example:

    python debug/stress_vllm_long_context.py \
        --base-url http://localhost:6658/v1 \
        --model Qwen3-VL-32B \
        --workers 32 \
        --tokenizer /path/to/Qwen3-VL-32B

If ``--tokenizer`` is omitted, the script uses random words and reports an
estimated prompt length.  Supplying the tokenizer gives substantially more
accurate 8k--12k token prompts, at the cost of local tokenization CPU time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import signal
import string
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


DEFAULT_BASE_URL = "http://localhost:6658/v1"
DEFAULT_MODEL = "Qwen3-VL-32B"


@dataclass
class Stats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: int = 0
    completed: int = 0
    failed: int = 0
    illegal_memory: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    status_codes: dict[int, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    def record_start(self) -> None:
        with self.lock:
            self.started += 1

    def record_result(
        self,
        *,
        ok: bool,
        prompt_tokens: int,
        output_tokens: int,
        latency: float,
        status_code: Optional[int] = None,
        illegal_memory: bool = False,
    ) -> None:
        with self.lock:
            self.completed += int(ok)
            self.failed += int(not ok)
            self.illegal_memory += int(illegal_memory)
            self.prompt_tokens += prompt_tokens
            self.output_tokens += output_tokens
            self.latencies.append(latency)
            if status_code is not None:
                self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = max(time.monotonic() - self.started_at, 1e-6)
            latencies = sorted(self.latencies[-1000:])
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95)) if latencies else 0
            p95 = latencies[p95_index] if latencies else 0.0
            return {
                "started": self.started,
                "completed": self.completed,
                "failed": self.failed,
                "illegal_memory": self.illegal_memory,
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "request_per_sec": self.completed / elapsed,
                "prompt_tokens_per_sec": self.prompt_tokens / elapsed,
                "output_tokens_per_sec": self.output_tokens / elapsed,
                "p50_latency": p50,
                "p95_latency": p95,
                "status_codes": dict(self.status_codes),
            }


class PromptBuilder:
    def __init__(self, tokenizer: Any, seed: int) -> None:
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer

    def build(self, target_tokens: int) -> tuple[str, int]:
        if self.tokenizer is not None:
            vocab_size = int(self.tokenizer.vocab_size)
            low = min(100, max(0, vocab_size - 1))
            high = max(low + 1, vocab_size - 100)
            ids = [self.rng.randrange(low, high) for _ in range(target_tokens)]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            # Account for normalization introduced by decode.  This estimate
            # is also useful in logs even when the server uses a different
            # chat-template wrapper.
            measured = len(self.tokenizer.encode(text, add_special_tokens=False))
            return text, measured

        # Rough fallback: random ASCII words are usually close to one token
        # per 3--5 characters, but the server's tokenizer remains authoritative.
        alphabet = string.ascii_letters + string.digits
        words = []
        estimated = 0
        while estimated < target_tokens:
            word = "".join(self.rng.choice(alphabet) for _ in range(self.rng.randint(2, 12)))
            words.append(word)
            estimated += max(1, (len(word) + 3) // 4)
        return " ".join(words), estimated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent long-context request workers (default: 32).",
    )
    parser.add_argument("--min-tokens", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--report-every", type=float, default=10.0)
    parser.add_argument("--tokenizer", default=None, help="Local/HF tokenizer path for accurate prompt lengths.")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--stop-on-illegal-memory",
        action="store_true",
        help="Stop all workers after the first response containing illegal memory access.",
    )
    return parser


def one_worker(
    worker_id: int,
    args: argparse.Namespace,
    stop: threading.Event,
    stats: Stats,
    tokenizer: Any,
) -> None:
    builder = PromptBuilder(tokenizer, args.seed + worker_id)
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
            "Connection": "keep-alive",
        }
    )
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    while not stop.is_set():
        target = builder.rng.randint(args.min_tokens, args.max_tokens)
        prompt, estimated_tokens = builder.build(target)
        request_id = f"w{worker_id}-{uuid.uuid4().hex[:10]}"
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Stress-test request. Read the random payload and return "
                        "only the word OK. Random payload follows:\n\n" + prompt
                    ),
                }
            ],
            "temperature": 0.0,
            "max_tokens": args.output_tokens,
            "stream": False,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }

        stats.record_start()
        started = time.monotonic()
        try:
            response = session.post(endpoint, json=payload, timeout=args.timeout)
            latency = time.monotonic() - started
            body = response.text[:4000]
            lower_body = body.lower()
            illegal = "illegal memory" in lower_body or "illegal memory access" in lower_body
            usage = {}
            try:
                data = response.json()
                usage = data.get("usage") or {}
            except (ValueError, TypeError):
                data = {}

            prompt_tokens = int(usage.get("prompt_tokens") or estimated_tokens)
            output_tokens = int(usage.get("completion_tokens") or 0)
            ok = response.ok and not illegal
            stats.record_result(
                ok=ok,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                latency=latency,
                status_code=response.status_code,
                illegal_memory=illegal,
            )

            if not response.ok or illegal:
                print(
                    f"[worker={worker_id}][request={request_id}] "
                    f"HTTP {response.status_code}, latency={latency:.2f}s, "
                    f"illegal_memory={illegal}: {body!r}",
                    flush=True,
                )
                if illegal and args.stop_on_illegal_memory:
                    stop.set()
        except Exception as exc:
            latency = time.monotonic() - started
            stats.record_result(
                ok=False,
                prompt_tokens=estimated_tokens,
                output_tokens=0,
                latency=latency,
            )
            print(
                f"[worker={worker_id}][request={request_id}] "
                f"exception after {latency:.2f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            # Avoid turning a dead server into a tight retry loop.
            stop.wait(1.0)


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.min_tokens < 1 or args.max_tokens < args.min_tokens:
        raise SystemExit("Invalid workers/token range")

    stop = threading.Event()
    stats = Stats()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer once from {args.tokenizer} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
            use_fast=True,
        )

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}; stopping workers...", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        json.dumps(
            {
                "endpoint": args.base_url,
                "model": args.model,
                "workers": args.workers,
                "prompt_tokens": [args.min_tokens, args.max_tokens],
                "output_tokens": args.output_tokens,
                "tokenizer": args.tokenizer or "fallback-estimate",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(one_worker, worker_id, args, stop, stats, tokenizer)
            for worker_id in range(args.workers)
        ]
        try:
            while not stop.wait(args.report_every):
                print("[stats] " + json.dumps(stats.snapshot(), ensure_ascii=False), flush=True)
        finally:
            stop.set()
            for future in futures:
                try:
                    future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    pass
                except Exception:
                    traceback.print_exc()

    print("[final] " + json.dumps(stats.snapshot(), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
