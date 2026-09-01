"""Tests for process-local retry monitoring."""

from __future__ import annotations

from contextvars import copy_context
import threading
import tempfile
import time
import unittest
from urllib.error import HTTPError
from pathlib import Path

from synthesis.retry_monitor import (
    RetryMonitor,
    case_context,
    install_monitor,
    retry_reason_from_exception,
    tracked_sleep,
)


class RetryMonitorTests(unittest.TestCase):
    def test_classifies_urllib_http_error_status(self) -> None:
        error = HTTPError("https://example.com", 502, "Bad Gateway", {}, None)
        self.assertEqual(
            retry_reason_from_exception(error, default="reader_error"),
            "http_502",
        )
        self.assertEqual(
            retry_reason_from_exception(
                RuntimeError("HTTP Error 502: Bad Gateway"),
                default="reader_error",
            ),
            "http_502",
        )

    def test_tracks_active_retry_and_case_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = RetryMonitor(
                interval_s=10.0,
                error_log_path=f"{temp_dir}/retry_errors.txt",
            )
            install_monitor(monitor)
            try:
                with case_context("case-17", 17):
                    context = copy_context()
                    worker = threading.Thread(
                        target=context.run,
                        args=(
                            lambda: tracked_sleep(
                                0.15,
                                reason="http_502",
                                tool="llm",
                                url="https://example.com/api",
                                error_type="HTTPError",
                                error=RuntimeError(
                                    "HTTP 502 Bad Gateway\n"
                                    "response_body:\nupstream unavailable"
                                ),
                            ),
                        ),
                    )
                    worker.start()
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        active = monitor.snapshot()["active"]
                        if active:
                            break
                        time.sleep(0.005)
                    active = monitor.snapshot()["active"]
                    self.assertEqual(len(active), 1)
                    self.assertEqual(active[0].case_id, "case-17")
                    self.assertEqual(active[0].case_idx, 17)
                    self.assertEqual(active[0].reason, "http_502")
                    self.assertEqual(active[0].tool, "llm")
                    worker.join(timeout=1.0)
                    self.assertFalse(worker.is_alive())

                snapshot = monitor.snapshot()
                self.assertEqual(snapshot["active"], [])
                self.assertEqual(snapshot["total_retry_events"], 1)
                self.assertGreaterEqual(snapshot["total_retry_sleep_s"], 0.1)
                error_log = Path(temp_dir, "retry_errors.txt").read_text(encoding="utf-8")
                self.assertIn("url='https://example.com/api'", error_log)
                self.assertIn("error_type='HTTPError'", error_log)
                self.assertIn("upstream unavailable", error_log)
            finally:
                install_monitor(None)


if __name__ == "__main__":
    unittest.main()
