"""Timing includes failures and distinguishes this run from old checkpoints."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from threading import Event
from concurrent.futures import ThreadPoolExecutor

from synthesis.post_process.fix.storage import Cache, read_json, write_json
from synthesis.post_process.fix.timing import Timer, timed_call, AsyncLogWriter
from synthesis.post_process.fix.run import main
from synthesis.post_process.fix.tests.test_pipeline import fixture


class TimingTests(unittest.TestCase):
    def test_slow_sink_does_not_block_producers_and_drains_in_order(self):
        blocked, release = Event(), Event()
        lines = []

        class SlowHandle:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def write(self, text):
                blocked.set()
                if not release.wait(3):
                    raise TimeoutError("test sink was not released")
                lines.append(text)

            def flush(self):
                pass

        with patch.object(Path, "open", return_value=SlowHandle()) as opened:
            writer = AsyncLogWriter(self.root / "slow.jsonl", batch_size=1)
            writer.emit({"producer": -1, "seq": 0})
            try:
                self.assertTrue(blocked.wait(1))

                def produce(index):
                    for seq in range(50):
                        writer.emit({"producer": index, "seq": seq})

                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(produce, i) for i in range(8)]
                    for future in futures:
                        future.result(timeout=1)
            finally:
                release.set()
                writer.close()
            self.assertEqual(opened.call_count, 1)
        rows = [json.loads(line) for line in "".join(lines).splitlines()]
        self.assertEqual(len(rows), 401)
        for i in range(8):
            self.assertEqual(
                [r["seq"] for r in rows if r["producer"] == i], list(range(50))
            )

    def test_background_write_failure_reported_on_close(self):
        with patch.object(Path, "open", side_effect=OSError("disk failure")):
            writer = AsyncLogWriter(self.root / "failed.jsonl")
            with self.assertRaisesRegex(RuntimeError, "Timing log write failed"):
                writer.close()

    def test_multiple_timers_share_writer_and_flush_on_close(self):
        path = self.root / "shared.jsonl"
        timers = [Timer(path, index=i) for i in range(5)]
        for timer in timers:
            with timer.measure("work"):
                pass
        timers[0].close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["index"] for r in rows}, set(range(5)))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_stage_duration_uses_monotonic_clock(self):
        with patch(
            "synthesis.post_process.fix.timing.perf_counter", side_effect=[0, 1, 4, 6]
        ):
            timer = Timer()
            with timer.activate():
                self.assertEqual(timed_call("stage", lambda: 42), 42)
            stats = timer.summary()
        self.assertEqual(stats["elapsed_s"], 6)
        self.assertEqual(stats["stages"]["stage"]["elapsed_s"], 3)

    def test_cache_hit_miss_and_failure_recorded(self):
        timer = Timer(self.root / "timing.jsonl", index=2)
        cache = Cache(self.root / "cache")
        with timer.activate():
            cache.get_or_compute("llm", "a", lambda: {"ok": True})
            cache.get_or_compute("llm", "a", lambda: self.fail("Should use cache"))

            def fail():
                raise TimeoutError("not persisted as sensitive text")

            with self.assertRaises(TimeoutError):
                cache.get_or_compute("llm", "b", fail)
        stats = timer.summary()["stages"]["cache.llm"]
        self.assertEqual(
            (
                stats["calls"],
                stats["cache_hits"],
                stats["cache_misses"],
                stats["errors"],
            ),
            (3, 1, 2, 1),
        )
        timer.close()
        rows = [json.loads(line) for line in timer.path.read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["error_type"], "TimeoutError")
        self.assertNotIn("sensitive text", timer.path.read_text())

    def test_resume_preserves_historical_timing_but_reports_current_cache_reuse(self):
        record, _, _ = fixture(self.root)
        source, work, output = (
            self.root / "input.json",
            self.root / "work",
            self.root / "output.json",
        )
        write_json(source, [record])
        argv = [
            "--input",
            str(source),
            "--workdir",
            str(work),
            "--output",
            str(output),
            "--mode",
            "augment",
        ]
        with (
            patch(
                "synthesis.post_process.fix.pipeline.process_record",
                return_value=(record, {"status": "rewritten", "windows": []}),
            ) as process,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(argv)
            first = read_json(work / "records/0.json")["audit"]["timing"]
            main(argv)
            self.assertEqual(process.call_count, 1)
        second = read_json(work / "summary.json")["timing"]
        self.assertTrue(second["records"][0]["reused_checkpoint"])
        self.assertNotIn("process_record", second["records"][0]["stages"])
        self.assertEqual(first, read_json(work / "records/0.json")["audit"]["timing"])
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(list((work / "timing_runs").glob("*.json"))), 2)

    def test_failed_record_retains_elapsed_and_failed_stage(self):
        record, _, _ = fixture(self.root)
        source, work = self.root / "input.json", self.root / "work"
        write_json(source, [record])
        with (
            patch(
                "synthesis.post_process.fix.pipeline.process_record",
                side_effect=RuntimeError("failed"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    "--input",
                    str(source),
                    "--workdir",
                    str(work),
                    "--output",
                    str(self.root / "output.json"),
                    "--mode",
                    "augment",
                ]
            )
        audit = read_json(work / "records/0.json")["audit"]
        self.assertEqual(audit["status"], "quarantined")
        self.assertEqual(audit["timing"]["stages"]["process_record"]["errors"], 1)

    def test_pre_fix_code_requires_new_workdir(self):
        source, work = self.root / "input.json", self.root / "work"
        write_json(source, [])
        argv = ["--input", str(source), "--workdir", str(work), "--mode", "index"]
        with contextlib.redirect_stdout(io.StringIO()):
            main(argv)
            manifest = read_json(work / "manifest.json")
            manifest["code_digest"] = (
                "3d52a426a365d432c9eefe7e7b960e20ad03ffc96968dd3f93c4e00e351aab5f"
            )
            write_json(work / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                main(argv)
            manifest["code_digest"] = "unknown_algorithm_change"
            write_json(work / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                main(argv)


if __name__ == "__main__":
    unittest.main()
