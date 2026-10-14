import unittest
from threading import Barrier, Lock
from types import SimpleNamespace

from synthesis.post_process.fix.rewrite import write_trace
from synthesis.post_process.fix.schema import Config
from synthesis.post_process.fix.tests.writer_fixture import observation


class ParallelWriterTests(unittest.TestCase):
    def run_writer(self, fail=False):
        barrier = Barrier(3, timeout=5)
        lock = Lock()
        active = peak = 0

        class Model:
            def ask(self, alias, prompt, payload, images):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(active, peak)
                try:
                    barrier.wait()
                    key = payload["selected_card"]["image_id"]
                    if fail and key == "1":
                        raise ValueError("writer failure")
                    return observation("Observed " + key)
                finally:
                    with lock:
                        active -= 1

        attempts = [
            SimpleNamespace(
                key=str(i),
                public={"image_id": str(i)},
                image_paths=[],
                verdict="YES",
                category="CONSISTENT",
                evidence=["match"],
            )
            for i in range(3)
        ]
        plan = SimpleNamespace(
            first=attempts,
            attempts=attempts,
            branch="contrast",
            metadata={"selection_reasoning": {str(i): "Inspect" for i in range(3)}},
        )
        steps = write_trace(
            {},
            SimpleNamespace(question="q"),
            plan,
            {"target": "target"},
            "source",
            Model(),
            Config(),
        )
        self.assertEqual(peak, 3)
        self.assertEqual(
            [s["after"] for s in steps], ["Observed 0", "Observed 1", "Observed 2"]
        )

    def test_parallel_requests_preserve_order(self):
        self.run_writer()

    def test_failure_propagates(self):
        with self.assertRaisesRegex(ValueError, "writer failure"):
            self.run_writer(fail=True)

    def test_invalid_worker_count(self):
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                Config(writer_workers=count).validate()
