"""Semantic review must see downstream image evidence, not just its text."""

import tempfile
import unittest
from pathlib import Path

from synthesis.post_process.fix.review_images import build_review_images
from synthesis.post_process.fix.schema import Candidate
from synthesis.post_process.fix.tests.test_pipeline import fixture


class ReviewImagesTests(unittest.TestCase):
    def test_downstream_relative_image_and_repeated_reference_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source, target = fixture(root, True)
            record = {
                "images": [str(source), Path(target).name, Path(target).name],
                "conversations": [
                    {"from": "human", "value": "Question <image>"},
                    {
                        "from": "observation",
                        "value": '{"image_id":"podium"}\nThe image is shown below:\n<image>',
                    },
                    {
                        "from": "observation",
                        "value": '{"image_id":"podium_again"}\n<image>',
                    },
                ],
            }
            candidate = Candidate("checked", {}, image_paths=[str(source)])
            paths, mapping = build_review_images(record, str(source), [candidate], root)
            self.assertEqual(len(paths), 2)
            self.assertEqual(paths[1], str(Path(target).resolve()))
            self.assertEqual(mapping["candidate_checks"][0]["image_indices"], [0])
            self.assertEqual(
                mapping["trajectory_images"][1],
                {
                    "turn_index": 1,
                    "placeholder_index": 0,
                    "resource_image_id": "podium",
                    "image_index": 1,
                },
            )
            self.assertEqual(mapping["trajectory_images"][2]["image_index"], 1)

    def test_missing_downstream_image_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source, _ = fixture(root, True)
            record = {
                "images": ["missing.jpg"],
                "conversations": [
                    {"from": "observation", "value": '{"image_id":"podium"}\n<image>'}
                ],
            }
            with self.assertRaises(FileNotFoundError):
                build_review_images(record, str(source), [], root)
