#!/usr/bin/env python3
"""Poll router load and run cancellable synthetic requests on idle SGLang workers.

Requires aiohttp and prometheus-client. All real inference traffic must pass
through the monitored router; synthetic traffic MUST go directly to workers.
See sglang_idle_keepalive.md for configuration and monitoring limitations.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid
from dataclasses import dataclass

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

LOG = logging.getLogger("sglang-idle")
ACTIVE_METRICS = (
    "sgl_router_running_requests",
    "smg_worker_requests_active",
    "sgl_router_worker_inflight_requests",
)
WORDS = "amber cedar forest river valley matrix orbit crystal meadow summit".split()


def metric_values(body: str, name: str) -> list[float]:
    """Read total series only: priority='' is the total in newer SGLang."""
    values = [
        float(s.value)
        for family in text_string_to_metric_families(body)
        for s in family.samples
        if s.name == name and s.labels.get("priority", "") == ""
    ]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"missing or invalid metric {name}")
    return values


def metric_total(body: str, name: str) -> float:
    return sum(metric_values(body, name))


@dataclass
class Background:
    worker: str
    task: asyncio.Task | None = None
    terminal: bool = False


class Controller:
    def __init__(self, args):
        self.args = args
        self.records: dict[str, Background] = {}
        self.stop = asyncio.Event()
        self.session = None
        self.inference_session = None
        self.last_busy = time.monotonic()
        self.clean_since = None
        self.retry_after = 0.0
        self.active_metric = None if args.router_active_metric == "auto" else args.router_active_metric
        self.previous_counter = None
        self.mode = None
        self.next_report = 0.0
        self.completed = 0
        self.failed = 0
        self.draining = False

    def save(self):
        path = Path(self.args.state_file)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({rid: {"worker": r.worker} for rid, r in self.records.items()}))
        os.replace(temporary, path)

    def load(self):
        path = Path(self.args.state_file)
        if path.exists():
            data = json.loads(path.read_text())
            for rid, record in data.items():
                if not rid.startswith("idle-fill-") or record["worker"] not in self.args.worker_url:
                    raise ValueError("state file contains unknown requests/workers; keep original worker URLs for cleanup")
                self.records[rid] = Background(record["worker"])
            if self.records:
                LOG.warning("recovering %d recorded requests before generating new load", len(self.records))

    async def get_metrics(self, url):
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.text()

    async def snapshot(self):
        urls = [self.args.router_metrics_url] + [w + "/metrics" for w in self.args.worker_url]
        results = await asyncio.gather(*(self.get_metrics(u) for u in urls), return_exceptions=True)
        errors = [f"{u}: {r}" for u, r in zip(urls, results) if isinstance(r, Exception)]
        if errors:
            raise RuntimeError("; ".join(errors))
        router = results[0]
        if self.active_metric is None:
            for name in ACTIVE_METRICS:
                try:
                    metric_total(router, name)
                except ValueError:
                    continue
                self.active_metric = name
                LOG.info("router active metric: %s", name)
                break
            if self.active_metric is None:
                raise ValueError("no known router in-flight gauge; set --router-active-metric to an actual gauge")
        active = metric_total(router, self.active_metric)
        activity = active > 0
        if self.args.router_arrival_metric:
            counter = metric_total(router, self.args.router_arrival_metric)
            # Reset/membership changes also postpone load. Initial value is a baseline.
            activity |= self.previous_counter is not None and counter != self.previous_counter
            self.previous_counter = counter
        workers = {}
        for worker, body in zip(self.args.worker_url, results[1:]):
            running = metric_total(body, "sglang:num_running_reqs")
            queued = metric_total(body, "sglang:num_queue_reqs")
            throughput = None
            try:
                throughput = metric_total(body, "sglang:gen_throughput")
            except ValueError:
                pass
            workers[worker] = (running, queued, throughput)
        return active, activity, workers

    async def generate(self, rid, record):
        payload = {
            "rid": rid,
            "text": uuid.uuid4().hex + " " + " ".join(random.choices(WORDS, k=self.args.prompt_words))
                    + "\nExplain patterns in this sequence in detail.",
            "stream": True,
            "sampling_params": {"temperature": self.args.temperature, "max_new_tokens": self.args.max_tokens},
        }
        if self.args.background_priority is not None:
            payload["priority"] = self.args.background_priority
        try:
            async with self.inference_session.post(record.worker + "/generate", json=payload) as response:
                response.raise_for_status()
                # Consume SSE without retaining output. Only terminal events confirm completion.
                async for raw in response.content:
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        record.terminal = True
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    if event.get("meta_info", {}).get("finish_reason") is not None:
                        record.terminal = True
                if not record.terminal:
                    raise RuntimeError("stream closed without a terminal event")
            self.completed += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failed += 1
            self.retry_after = time.monotonic() + self.args.error_backoff
            LOG.warning("background %s failed: %s; retaining ID for cancellation", rid, exc)

    async def abort_one(self, rid, record):
        try:
            async with self.session.post(record.worker + "/abort_request", json={"rid": rid, "abort_all": False}) as response:
                response.raise_for_status()
                await response.read()
            return True
        except Exception as exc:
            LOG.warning("abort %s failed: %s", rid, exc)
            return False

    async def abort_pending(self):
        # Keep streams open until terminal response; repeat aborts to handle submit/abort races.
        results = await asyncio.gather(*(self.abort_one(rid, r) for rid, r in self.records.items()))
        return all(results)

    def reap(self):
        remove = [rid for rid, r in self.records.items() if r.terminal and r.task is not None and r.task.done()]
        for rid in remove:
            self.records.pop(rid)
        if remove:
            self.save()

    async def step(self):
        self.reap()
        now = time.monotonic()
        try:
            active, activity, workers = await self.snapshot()
        except Exception as exc:
            self.last_busy = time.monotonic()
            self.clean_since = None
            self.draining = bool(self.records)
            self.set_mode("monitor-error")
            LOG.warning("monitoring failed; background load paused: %s", exc)
            await self.abort_pending()
            return
        now = time.monotonic()
        uncertain = any(r.task is None or (r.task.done() and not r.terminal) for r in self.records.values())
        # Worker metrics include synthetic requests; never subtract local task counts.
        backend_busy = any(running > 0 or queued > 0 for running, queued, _ in workers.values())
        untracked_busy = not self.records and backend_busy
        self.draining = self.draining and bool(self.records)
        paused = activity or uncertain or untracked_busy or now < self.retry_after or self.draining
        if paused:
            self.draining = bool(self.records)
            self.last_busy = now
            self.set_mode("business-busy" if activity else "cleanup-or-backoff")
            aborted = await self.abort_pending()
            # Recovery/disconnected streams have no live response to confirm termination.
            # Repeated successful aborts + an extended all-worker idle window clear them.
            if uncertain and not backend_busy and aborted and not activity:
                if self.clean_since is None:
                    self.clean_since = time.monotonic()
                if time.monotonic() - self.clean_since >= self.args.idle_grace:
                    for rid, r in list(self.records.items()):
                        if r.task is None or r.task.done():
                            del self.records[rid]
                    self.save()
                    self.clean_since = None
            else:
                self.clean_since = None
        elif now - self.last_busy < self.args.idle_grace:
            self.set_mode("idle-grace")
        else:
            self.set_mode("filling" if not self.args.dry_run else "dry-run-idle")
            if not self.args.dry_run and not self.stop.is_set():
                for worker in self.args.worker_url:
                    count = sum(r.worker == worker for r in self.records.values())
                    for _ in range(max(0, self.args.concurrency_per_worker - count)):
                        rid = "idle-fill-" + uuid.uuid4().hex
                        record = Background(worker)
                        self.records[rid] = record
                        self.save()  # Persist BEFORE POST, including crash during submission.
                        record.task = asyncio.create_task(self.generate(rid, record))
        if now >= self.next_report:
            LOG.info("mode=%s real_active=%g background=%d completed=%d failed=%d workers=%s",
                     self.mode, active, len(self.records), self.completed, self.failed, workers)
            self.next_report = now + self.args.report_interval

    def set_mode(self, mode):
        if mode != self.mode:
            LOG.info("mode: %s -> %s", self.mode, mode)
            self.mode = mode

    async def run(self):
        self.load()
        headers = {"Authorization": "Bearer " + os.environ[self.args.api_key_env]} if self.args.api_key_env else {}
        admin_headers = dict(headers)
        if self.args.admin_key_env:
            admin_headers["Authorization"] = "Bearer " + os.environ[self.args.admin_key_env]
        # Independent connections ensure generation cannot starve monitoring/abort.
        async with aiohttp.ClientSession(headers=admin_headers, timeout=aiohttp.ClientTimeout(total=self.args.http_timeout)) as monitor:
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=self.args.request_timeout)) as inference:
                self.session, self.inference_session = monitor, inference
                try:
                    while not self.stop.is_set():
                        await self.step()
                        try:
                            await asyncio.wait_for(self.stop.wait(), timeout=self.args.poll_interval)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    LOG.info("stopping; cancelling %d background requests", len(self.records))
                    for _ in range(3):
                        await self.abort_pending()
                        await asyncio.sleep(0.2)
                        self.reap()
                        if not self.records:
                            break
                    tasks = [r.task for r in self.records.values() if r.task is not None]
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self.reap()
                    self.save()
                    if self.records:
                        LOG.warning("%d unconfirmed IDs preserved for next startup", len(self.records))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-metrics-url", required=True, help="Router Prometheus URL, e.g. http://127.0.0.1:6661/metrics")
    parser.add_argument("--worker-url", action="append", required=True, help="Direct worker URL; repeat for every replica (never use router URL)")
    parser.add_argument("--router-active-metric", default="auto", help="In-flight GAUGE; auto recognizes common router versions")
    parser.add_argument("--router-arrival-metric", help="Optional arrival counter; use only if it increments when inference requests ARRIVE")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--idle-grace", type=float, default=20.0)
    parser.add_argument("--concurrency-per-worker", type=int, default=2)
    parser.add_argument("--prompt-words", type=int, default=64, help="Random words, not a token count")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--background-priority", type=int, help="Optional priority; requires worker priority scheduling configuration")
    parser.add_argument("--http-timeout", type=float, default=3.0, help="Monitoring and abort timeout")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--error-backoff", type=float, default=30.0)
    parser.add_argument("--report-interval", type=float, default=10.0)
    parser.add_argument("--state-file", default=".sglang_idle_keepalive_state.json")
    parser.add_argument("--api-key-env", help="Environment variable containing inference Bearer token")
    parser.add_argument("--admin-key-env", help="Environment variable containing metrics/abort Bearer token (defaults to inference key)")
    parser.add_argument("--dry-run", action="store_true", help="Monitor without submitting new requests; recorded requests still get cleaned up")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    for field in ("poll_interval", "idle_grace", "concurrency_per_worker", "prompt_words", "max_tokens",
                  "http_timeout", "request_timeout", "error_backoff", "report_interval"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    args.worker_url = list(dict.fromkeys(w.rstrip("/") for w in args.worker_url))
    for url in [args.router_metrics_url] + args.worker_url:
        if not url.startswith(("http://", "https://")):
            parser.error("URLs must start with http:// or https://")
    for env in (args.api_key_env, args.admin_key_env):
        if env and not os.environ.get(env):
            parser.error(f"environment variable {env} is empty/unset")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Lock is held for process lifetime. Do not delete the lock file while running.
    with open(args.state_file + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another controller is using this state file")
        async def entry():
            controller = Controller(args)
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, controller.stop.set)
            await controller.run()
        asyncio.run(entry())


if __name__ == "__main__":
    main()
