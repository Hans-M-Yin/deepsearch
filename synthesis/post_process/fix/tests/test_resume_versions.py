import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthesis.post_process.fix.resume_version import resume_manifest
from synthesis.post_process.fix.run import main
from synthesis.post_process.fix.storage import read_json, write_json


class ResumeVersionTests(unittest.TestCase):
    def test_concurrency_changes_only_on_resume(self):
        old = dict(version="v", input="file", input_digest="data", code_digest="same",
                   config={"record_workers": 16, "visual_check_workers": 16,
                           "writer_workers": 16, "global_visual_workers": 128,
                           "model": "model_a", "seed": 42})
        config = {**old["config"], "record_workers": 32, "visual_check_workers": 8,
                  "writer_workers": 8, "global_visual_workers": 256}
        new = {**old, "config": config}
        opts = dict(resume=True, allow_code_change=False, run_id="r", started_at="now")
        result = resume_manifest(old, new, **opts)
        self.assertEqual(result["config"], config)
        self.assertEqual(len(result["config_history"][0]["changes"]), 4)
        self.assertEqual(resume_manifest(result, new, **opts), result)
        self.assertNotIn("config_history", old)
        with self.assertRaises(ValueError):
            resume_manifest(old, new, **{**opts, "resume": False})
        for key, value in [("model", "model_b"), ("seed", 43), ("unknown", 1), ("record_workers", 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume_manifest(old, {**new, "config": {**config, key: value}}, **opts)

    def test_only_code_may_change_with_explicit_opt_in(self):
        old = dict(
            version="v",
            input="file",
            input_digest="data",
            config={"workers": 1},
            code_digest="old",
        )
        new = {**old, "code_digest": "new"}
        opts = dict(resume=True, allow_code_change=True, run_id="r", started_at="now")
        migrated = resume_manifest(old, new, **opts)
        self.assertEqual(migrated["legacy_code_digest"], "old")
        self.assertEqual(len(migrated["code_history"]), 1)
        self.assertEqual(resume_manifest(migrated, new, **opts), migrated)
        again = resume_manifest(migrated, {**new, "code_digest": "newer"}, **opts)
        self.assertEqual(again["legacy_code_digest"], "old")
        self.assertEqual(len(again["code_history"]), 2)
        for key, value in [
            ("version", "v2"),
            ("input", "other"),
            ("input_digest", "changed"),
            ("config", {}),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume_manifest(old, {**new, key: value}, **opts)
        for resume, allow in [(True, False), (False, True), (False, False)]:
            with self.assertRaises(ValueError):
                resume_manifest(
                    old, new, **{**opts, "resume": resume, "allow_code_change": allow}
                )

    def test_resume_keeps_old_records_tracks_provenance_and_isolates_llm_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work = root / "input.json", root / "work"
            write_json(
                source,
                [{"id": str(i), "conversations": [], "images": []} for i in range(3)],
            )
            argv = [
                "--input",
                str(source),
                "--workdir",
                str(work),
                "--output",
                str(root / "output.json"),
                "--mode",
                "augment",
                "--record-workers",
                "1",
            ]
            calls, caches = [], []

            def process(
                record, dataset_root, record_root, config, backend, model, cache
            ):
                calls.append(record["id"])
                caches.append(model.cache.root)
                return record, {"status": "rewritten", "windows": []}

            with (
                patch(
                    "synthesis.post_process.fix.pipeline.process_record",
                    side_effect=process,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with patch("synthesis.post_process.fix.run.digest", return_value="old"):
                    main(argv + ["--limit", "1"])
                # Simulate a v34 checkpoint without per-record version metadata.
                original = read_json(work / "records/0.json")
                original.pop("code_digest")
                write_json(work / "records/0.json", original)
                with patch("synthesis.post_process.fix.run.digest", return_value="new"):
                    with self.assertRaisesRegex(ValueError, "code differs"):
                        main(argv + ["--resume", "--limit", "2"])
                    main(
                        argv
                        + ["--resume", "--resume-allow-code-change", "--limit", "2", "--record-workers", "2"]
                    )
                    self.assertEqual(read_json(work / "manifest.json")["config"]["record_workers"], 2)
                    main(argv + ["--resume", "--limit", "2"])
                self.assertEqual(calls, ["0", "1"])
                saved = read_json(work / "records/0.json")
                self.assertEqual(saved.pop("code_digest"), "old")
                self.assertEqual(saved, original)
                self.assertEqual(
                    read_json(work / "records/1.json")["code_digest"], "new"
                )
                self.assertNotEqual(caches[0], caches[1])
                self.assertEqual(
                    read_json(work / "summary.json")["record_code_digests"],
                    {"old": 1, "new": 1},
                )
                with patch(
                    "synthesis.post_process.fix.run.digest", return_value="newer"
                ):
                    main(argv + ["--resume", "--resume-allow-code-change"])
                self.assertEqual(calls, ["0", "1", "2"])
                self.assertEqual(
                    len(read_json(work / "manifest.json")["code_history"]), 2
                )
                self.assertEqual(len(read_json(root / "output.json")), 3)

    def test_flag_requires_resume(self):
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            main(
                [
                    "--input",
                    "unused",
                    "--workdir",
                    "unused",
                    "--resume-allow-code-change",
                ]
            )
