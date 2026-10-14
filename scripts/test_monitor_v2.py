"""Real HTTP/process lifecycle tests using CPU-only dummy holders."""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from scripts.monitor_v2 import HolderManager, live_group_members, make_handler, resolve_gpus


class GpuTests(unittest.TestCase):
    def test_visibility_and_validation(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            self.assertEqual(resolve_gpus(None, "unused"), ["2", "3"])
            self.assertEqual(resolve_gpus("0,1", "unused"), ["0", "1"])
        for value in ("", "0,0", "-1", "0;echo foo"):
            with self.assertRaises(ValueError):
                resolve_gpus(value, "unused")


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script = self.root / "holder.py"
        self.script.write_text(
            "import json, os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "Path('started.json').write_text(json.dumps({'child': child.pid, 'gpus': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
            "time.sleep(120)\n"
        )
        self.manager = HolderManager(self.script, sys.executable, ["0", "2"], self.root / "holder.log", 0.1)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.manager, "test-token"))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.manager.close()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, path, method="GET", token="test-token"):
        req = urllib.request.Request(self.url + path, method=method,
                                     headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    def wait_started(self):
        marker = self.root / "started.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                try:
                    return json.loads(marker.read_text())
                except json.JSONDecodeError:
                    pass
            time.sleep(0.01)
        self.fail("dummy holder did not start")

    def test_http_start_stop_and_foreign_process_survives(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            self.assertFalse(self.request("/status")["running"])
            started = self.request("/start", "POST")
            info = self.wait_started()
            self.assertEqual(info["gpus"], "0,2")
            self.assertEqual(self.request("/start", "POST")["pid"], started["pid"])
            self.assertIn(info["child"], live_group_members(started["pid"]))
            stopped = self.request("/stop", "POST")
            self.assertFalse(stopped["running"])
            self.assertFalse(live_group_members(started["pid"]))
            self.assertIsNone(foreign.poll())
            self.assertEqual(self.request("/stop", "POST")["action"], "already_stopped")
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)

    def test_concurrent_start_creates_one_holder(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.request("/start", "POST"), range(8)))
        self.assertEqual(len({r["pid"] for r in results}), 1)
        self.assertEqual(sum(r["action"] == "started" for r in results), 1)
        self.wait_started()

    def test_authorization_and_unknown_routes(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/start", "POST", token="wrong")
        self.assertEqual(error.exception.code, 401)
        self.assertFalse(self.manager.status()["running"])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/unknown")
        self.assertEqual(error.exception.code, 404)

    def test_parent_crash_cleanup_and_relaunch(self):
        self.script.write_text(self.script.read_text().replace("time.sleep(120)\n", "sys.exit(7)\n"))
        started = self.request("/start", "POST")
        self.wait_started()
        self.manager.process.wait(timeout=5)
        self.assertFalse(self.request("/status")["running"])
        self.assertTrue(live_group_members(started["pid"]))
        restarted = self.request("/start", "POST")
        self.assertNotEqual(started["pid"], restarted["pid"])
        self.assertFalse(live_group_members(started["pid"]))

    def test_close_stops_children_and_rejects_future_start(self):
        started = self.request("/start", "POST")
        self.wait_started()
        self.manager.close()
        self.assertFalse(live_group_members(started["pid"]))
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/start", "POST")
        self.assertEqual(error.exception.code, 500)

    def test_sigkill_fallback(self):
        self.script.write_text(
            "import signal, time\nfrom pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "Path('started.json').write_text('{}')\n"
            "time.sleep(120)\n"
        )
        started = self.request("/start", "POST")
        self.wait_started()
        self.request("/stop", "POST")
        self.assertFalse(live_group_members(started["pid"]))
        self.assertEqual(self.manager.last_exit_code, -9)


if __name__ == "__main__":
    unittest.main()
