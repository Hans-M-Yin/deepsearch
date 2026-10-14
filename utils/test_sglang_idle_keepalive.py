"""HTTP lifecycle tests; no GPU/model required."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

import aiohttp
from aiohttp import web

from utils.sglang_idle_keepalive import Controller, make_parser, metric_total


class MetricsTests(unittest.TestCase):
    def test_priority_totals_are_not_double_counted(self):
        self.assertEqual(metric_total('sglang:num_running_reqs{priority=""} 3\n'
                                     'sglang:num_running_reqs{priority="0"} 2\n'
                                     'sglang:num_running_reqs{priority="-1"} 1\n',
                                     "sglang:num_running_reqs"), 3)

    def test_invalid_metrics_are_not_idle(self):
        for body in ("", "load NaN\n", "load -1\n"):
            with self.assertRaises(ValueError):
                metric_total(body, "load")


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.real_active = 0
        self.running = {}
        self.generated = []
        self.aborted = []
        self.broken = False
        self.abort_broken = False
        self.admit = asyncio.Event()
        self.admit.set()
        app = web.Application()
        app.router.add_get("/router/metrics", self.router_metrics)
        app.router.add_get("/metrics", self.worker_metrics)
        app.router.add_post("/generate", self.generate)
        app.router.add_post("/abort_request", self.abort)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
        args = make_parser().parse_args([
            "--router-metrics-url", self.url + "/router/metrics", "--worker-url", self.url,
            "--state-file", str(Path(self.tmp.name) / "state.json"),
            "--idle-grace", "0.01", "--concurrency-per-worker", "1",
        ])
        self.c = Controller(args)
        self.c.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1))
        self.c.inference_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2))

    async def asyncTearDown(self):
        self.admit.set()
        await self.c.abort_pending()
        for event in self.running.values():
            event.set()
        tasks = [r.task for r in self.c.records.values() if r.task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.c.session.close()
        await self.c.inference_session.close()
        await self.runner.cleanup()
        self.tmp.cleanup()

    async def router_metrics(self, request):
        if self.broken:
            return web.Response(text="# no load gauge\n")
        return web.Response(text=f"sgl_router_running_requests {self.real_active}\n")

    async def worker_metrics(self, request):
        count = sum(not e.is_set() for e in self.running.values())
        return web.Response(text=f"sglang:num_running_reqs {count}\nsglang:num_queue_reqs 0\n")

    async def generate(self, request):
        body = await request.json()
        rid = body["rid"]
        self.generated.append(rid)
        await self.admit.wait()
        event = self.running[rid] = asyncio.Event()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"meta_info": {"finish_reason": null}}\n\n')
        await event.wait()
        try:
            await response.write(b'data: {"meta_info": {"finish_reason": {"type": "abort"}}}\n\ndata: [DONE]\n\n')
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    async def abort(self, request):
        body = await request.json()
        self.assertFalse(body["abort_all"])
        self.aborted.append(body["rid"])
        if self.abort_broken:
            return web.Response(status=503)
        if body["rid"] in self.running:
            self.running[body["rid"]].set()
        return web.Response()

    async def fill(self):
        self.c.last_busy = time.monotonic() - 1
        await self.c.step()
        for _ in range(100):
            if self.generated:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(len(self.generated), 1)

    async def test_idle_fill_business_cancel_and_resume(self):
        self.real_active = 1
        await self.c.step()
        self.assertFalse(self.generated)
        self.real_active = 0
        await self.fill()
        rid = self.generated[0]
        self.assertIn(rid, json.loads(Path(self.c.args.state_file).read_text()))
        self.real_active = 2
        await self.c.step()
        self.assertIn(rid, self.aborted)
        await asyncio.sleep(0.01)
        await self.c.step()
        self.assertFalse(self.c.records)
        self.real_active = 0
        await self.c.step()
        self.assertEqual(len(self.generated), 1)
        await asyncio.sleep(0.02)
        await self.c.step()
        await asyncio.sleep(0.01)
        self.assertEqual(len(self.generated), 2)

    async def test_missing_metric_cancels_existing_load(self):
        await self.fill()
        self.broken = True
        await self.c.step()
        self.assertEqual(self.c.mode, "monitor-error")
        self.assertIn(self.generated[0], self.aborted)
        await self.c.step()
        self.assertEqual(len(self.generated), 1)

    async def test_late_submission_is_aborted_after_business_finishes(self):
        self.admit.clear()
        await self.fill()
        self.real_active = 1
        await self.c.step()  # abort arrives before worker admits request
        self.real_active = 0
        self.admit.set()
        await asyncio.sleep(0.01)
        await self.c.step()  # must keep cancelling even though real_active is now zero
        self.assertGreaterEqual(self.aborted.count(self.generated[0]), 2)
        await asyncio.sleep(0.01)
        self.c.reap()
        self.assertFalse(self.c.records)

    async def test_state_recovery_cleans_before_fill(self):
        rid = "idle-fill-recovered"
        Path(self.c.args.state_file).write_text(json.dumps({rid: {"worker": self.url}}))
        self.c.load()
        await self.c.step()
        self.assertFalse(self.generated)
        self.assertIn(rid, self.aborted)
        await asyncio.sleep(0.02)
        await self.c.step()
        self.assertFalse(self.c.records)

    async def test_dry_run_does_not_submit(self):
        self.c.args.dry_run = True
        self.c.last_busy = time.monotonic() - 1
        await self.c.step()
        self.assertEqual(self.c.mode, "dry-run-idle")
        self.assertFalse(self.generated)

    async def test_abort_failure_preserves_id_and_blocks_refill(self):
        await self.fill()
        self.abort_broken = True
        self.real_active = 1
        await self.c.step()
        self.real_active = 0
        self.c.last_busy = time.monotonic() - 1
        await self.c.step()
        self.assertEqual(len(self.generated), 1)
        self.assertIn(self.generated[0], self.c.records)
        self.abort_broken = False
        await self.c.step()
        await asyncio.sleep(0.01)
        self.c.reap()
        self.assertFalse(self.c.records)

    async def test_run_shutdown_cancels_and_saves_empty_state(self):
        controller = Controller(self.c.args)
        controller.args.poll_interval = 0.005
        controller.last_busy = time.monotonic() - 1
        task = asyncio.create_task(controller.run())
        try:
            for _ in range(100):
                if self.generated:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(len(self.generated), 1)
        finally:
            controller.stop.set()
            await asyncio.wait_for(task, timeout=2)
        self.assertIn(self.generated[0], self.aborted)
        self.assertEqual(json.loads(Path(controller.args.state_file).read_text()), {})


if __name__ == "__main__":
    unittest.main()
