"""Pixel-level regression coverage for inference reverse-image search crops."""

import tempfile
import unittest
from unittest import mock

from PIL import Image

from opensearch_vl.opensearch_infer import tools


class I2IRegionCoordinatesTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (600, 400), "white")
        self.source.paste("red", (0, 280, 180, 400))

    def run_search(self, source, region):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(tools.image_io, "ensure_image_local", return_value=(source, None)), \
                mock.patch.object(tools.cos_upload, "upload_pil_image", return_value="https://example.com/crop.png") as upload, \
                mock.patch.object(tools.sft_tools, "i2i_search", return_value={"ok": True, "results": []}) as search:
            message, images = tools.execute_tool(
                {"name": "i2i_search", "parameters": {"region": region}},
                {"img_1": "https://example.com/source.png"},
                "case", 0, 1, directory,
            )
            if upload.called:
                crop = upload.call_args.args[0].copy()
                with Image.open(next(iter(images.values()))) as saved:
                    self.assertEqual(saved.size, crop.size)
                    self.assertEqual(saved.tobytes(), crop.tobytes())
                search.assert_called_once()
                self.assertEqual(search.call_args.kwargs["image_url"], "https://example.com/crop.png")
            else:
                crop = None
                search.assert_not_called()
                self.assertEqual(images, {})
            return message, crop

    def test_full_image_preserves_pixels_and_actual_dimensions(self):
        for size in [(600, 400), (1600, 1200), (400, 600)]:
            with self.subTest(size=size):
                source = Image.new("RGB", size, "orange")
                _, crop = self.run_search(source, [0, 0, 1000, 1000])
                self.assertEqual(crop.size, size)
                self.assertEqual(crop.tobytes(), source.tobytes())

    def test_previously_black_crop_uses_normalized_region(self):
        _, crop = self.run_search(self.source, [0, 700, 300, 1000])
        self.assertEqual(crop.size, (180, 120))
        self.assertEqual(crop.tobytes(), Image.new("RGB", (180, 120), "red").tobytes())

    def test_x_and_y_scale_independently(self):
        _, crop = self.run_search(self.source, [250, 250, 750, 750])
        expected = self.source.crop((150, 100, 450, 300))
        self.assertEqual(crop.size, (300, 200))
        self.assertEqual(crop.tobytes(), expected.tobytes())

    def test_fractional_coordinates_keep_precision_until_scaling(self):
        source = Image.new("RGB", (2000, 400), "orange")
        source.paste("red", (251, 100, 1751, 300))
        _, crop = self.run_search(source, [125.5, 250, 875.5, 750])
        self.assertEqual(crop.size, (1500, 200))
        self.assertEqual(crop.tobytes(), Image.new("RGB", (1500, 200), "red").tobytes())

    def test_out_of_bounds_is_clipped_without_black_padding(self):
        _, crop = self.run_search(self.source, [-100, 700, 1100, 1100])
        expected = self.source.crop((0, 280, 600, 400))
        self.assertEqual(crop.size, (600, 120))
        self.assertEqual(crop.tobytes(), expected.tobytes())

    def test_invalid_or_empty_regions_never_reach_search(self):
        regions = [
            [500, 0, 400, 1000], [0, 0, 0, 1000],
            [1100, 0, 1200, 1000], [-200, 0, -100, 1000],
            [0, 0, 0.01, 0.01], [0, 0, float("nan"), 1000],
            [0, 0, float("inf"), 1000], [0, 0, "oops", 1000],
            [0, 0, 10**1000, 1000],
            [0, 0, 1000], {"x": 0, "y": 0, "width": 0, "height": 10},
        ]
        for region in regions:
            with self.subTest(region=region):
                message, crop = self.run_search(self.source, region)
                self.assertIn("Tool execution error:", message)
                self.assertIsNone(crop)

    def test_legacy_pixel_dictionary_is_preserved(self):
        _, crop = self.run_search(self.source, {"x": 0, "y": 280, "width": 180, "height": 120})
        self.assertEqual(crop.size, (180, 120))
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))

    def test_missing_image_does_not_upload_or_search(self):
        with mock.patch.object(tools.cos_upload, "upload_pil_image") as upload, \
                mock.patch.object(tools.sft_tools, "i2i_search") as search:
            message, images = tools.execute_tool(
                {"name": "i2i_search", "parameters": {"region": [0, 0, 1000, 1000]}},
                {}, "case", 0, 1, "/tmp",
            )
        self.assertIn("requires an 'image' reference", message)
        self.assertEqual(images, {})
        upload.assert_not_called()
        search.assert_not_called()

    def test_omitted_region_searches_original_url(self):
        with mock.patch.object(tools.cos_upload, "upload_pil_image") as upload, \
                mock.patch.object(tools.sft_tools, "i2i_search", return_value={"ok": True, "results": []}) as search:
            _, images = tools.execute_tool(
                {"name": "i2i_search", "parameters": {}},
                {"img_1": "https://example.com/original.png"}, "case", 0, 1, "/tmp",
            )
        self.assertEqual(search.call_args.kwargs["image_url"], "https://example.com/original.png")
        self.assertEqual(images, {})
        upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
