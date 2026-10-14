"""Bounded concurrency, failure isolation and telemetry propagation."""

import json
import tempfile
import unittest
from pathlib import Path
from threading import Barrier, Lock, BoundedSemaphore
from types import SimpleNamespace
from unittest.mock import patch

from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.timing import Timer
from synthesis.post_process.fix.visual_evidence import verify_pool
from synthesis.post_process.fix.storage import Cache
from concurrent.futures import ThreadPoolExecutor


class ParallelChecksTests(unittest.TestCase):
    def test_parallel_failure_isolation_and_timer(self):
        barrier = Barrier(2)
        lock = Lock()
        active = peak = 0

        def verify(candidate, *args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
                if candidate.key == "0":
                    raise ValueError("unreadable")
                candidate.category, candidate.verdict = "SIMILAR", "NO"
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as root:
            timer = Timer(Path(root) / "timing.jsonl", index=7)
            candidates = [Candidate(str(i), {}) for i in range(4)]
            with (
                timer.activate(),
                patch(
                    "synthesis.post_process.fix.visual_evidence.verify",
                    side_effect=verify,
                ),
            ):
                failures = verify_pool(
                    candidates,
                    "source",
                    SimpleNamespace(question="q", call={"arguments": {}}),
                    {"target": "t", "entity": "e"},
                    None,
                    None,
                    Config(visual_check_workers=2),
                )
            self.assertEqual(peak, 2)
            self.assertEqual(failures, [{"candidate": "0", "error": "unreadable"}])
            self.assertEqual(candidates[0].category, "UNRESOLVED")
            self.assertTrue(all(c.verdict == "NO" for c in candidates[1:]))
            timer.close()
            events = [json.loads(line) for line in timer.path.read_text().splitlines()]
            self.assertEqual(len(events), 16)
            self.assertTrue(all(e["index"] == 7 for e in events))
            stats = timer.summary()["stages"]["visual_candidate"]
            self.assertEqual((stats["calls"], stats["errors"]), (4, 1))

    def test_parallel_cache_computes_identical_key_once(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Cache(root)
            count = 0
            barrier = Barrier(4)

            def compute():
                nonlocal count
                count += 1
                return {"value": 1}

            def task(_):
                barrier.wait(timeout=5)
                return cache.get_or_compute("test", {"same": True}, compute)

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(task, range(4)))
            self.assertEqual(count, 1)
            self.assertEqual(results, [{"value": 1}] * 4)

    def test_workers_validation(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Config(visual_check_workers=value).validate()

    def test_parallel_records_checkpoint_failure_and_order(self):
        from synthesis.post_process.fix.run import main
        from synthesis.post_process.fix.storage import write_json, read_json
        import contextlib
        import io

        barrier = Barrier(2)

        def process(record, *args):
            barrier.wait(timeout=5)
            if record["id"] == "0":
                raise ValueError("isolated failure")
            return record, {"status": "rewritten", "windows": []}

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output, work = (
                root / "input.json",
                root / "output.json",
                root / "work",
            )
            write_json(
                source,
                [{"id": str(i), "conversations": [], "images": []} for i in range(2)],
            )
            argv = [
                "--input",
                str(source),
                "--output",
                str(output),
                "--workdir",
                str(work),
                "--mode",
                "augment",
                "--selected-only",
                "--record-workers",
                "2",
            ]
            with (
                patch(
                    "synthesis.post_process.fix.pipeline.process_record",
                    side_effect=process,
                ) as mocked,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(argv)
                main(argv)
                self.assertEqual(mocked.call_count, 2)
            self.assertEqual([r["id"] for r in read_json(output)], ["1"])
            self.assertEqual(
                [a["index"] for a in read_json(work / "retry_pending.json")], [0]
            )
            self.assertEqual(
                read_json(work / "records/0.json")["audit"]["status"], "quarantined"
            )
            self.assertEqual(
                read_json(work / "records/1.json")["audit"]["status"], "rewritten"
            )
            self.assertTrue(
                all(
                    r["reused_checkpoint"]
                    for r in read_json(work / "summary.json")["timing"]["records"]
                )
            )
            for line in (work / "timing.jsonl").read_text().splitlines():
                json.loads(line)

    def test_global_visual_limit_shared_by_records(self):
        barrier, lock = Barrier(2), Lock()
        active = peak = 0

        def verify(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            with lock:
                active -= 1

        backend = SimpleNamespace(visual_slots=BoundedSemaphore(2))

        def record(_):
            return verify_pool(
                [Candidate(str(i), {}) for i in range(4)],
                "source",
                SimpleNamespace(question="q", call={"arguments": {}}),
                {"target": "t", "entity": "e"},
                backend,
                None,
                Config(visual_check_workers=4),
            )

        with (
            patch(
                "synthesis.post_process.fix.visual_evidence.verify", side_effect=verify
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(record, range(2)))
        self.assertEqual(results, [[], []])
        self.assertEqual(peak, 2)
