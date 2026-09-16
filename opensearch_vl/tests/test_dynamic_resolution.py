"""Tests for optional image downscaling before model inference."""

from __future__ import annotations

import base64
import io
import unittest

from PIL import Image

from opensearch_vl.opensearch_infer import pipeline


class DynamicResolutionTests(unittest.TestCase):
    def test_inline_image_long_side_is_limited(self) -> None:
        source = Image.new("RGB", (3200, 2000), (20, 40, 60))
        raw = io.BytesIO()
        source.save(raw, format="PNG")

        image_paths, parts = pipeline._bootstrap_images(
            {"images": [{"bytes": raw.getvalue()}]},
            "case-1",
            0,
            "fvqa_train",
            None,
            dynamic_resolution=True,
        )

        self.assertEqual(list(image_paths), ["img_1"])
        payload = base64.b64decode(parts[0]["inline_data"]["data"])
        with Image.open(io.BytesIO(payload)) as resized:
            self.assertEqual(resized.size, (1600, 1000))

    def test_disabled_mode_keeps_original_dimensions(self) -> None:
        source = Image.new("RGB", (3200, 2000), (20, 40, 60))
        raw = io.BytesIO()
        source.save(raw, format="PNG")

        _, parts = pipeline._bootstrap_images(
            {"images": [{"bytes": raw.getvalue()}]},
            "case-1",
            0,
            "fvqa_train",
            None,
        )

        payload = base64.b64decode(parts[0]["inline_data"]["data"])
        with Image.open(io.BytesIO(payload)) as original:
            self.assertEqual(original.size, (3200, 2000))


if __name__ == "__main__":
    unittest.main()
