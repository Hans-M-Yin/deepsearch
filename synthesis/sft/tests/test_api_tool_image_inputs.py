"""Tests for raster validation before images are submitted to a model."""

from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from synthesis.sft.api_tools import (
    ToolRuntimeContext,
    _image_source_to_model_url,
    _persist_pil_image,
    _try_upload_pil_image,
)


class ApiToolImageInputTests(unittest.TestCase):
    @staticmethod
    def _jpeg_bytes() -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (4, 3), "red").save(output, format="JPEG")
        return output.getvalue()

    def test_local_image_is_reencoded_to_valid_png_even_with_wrong_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_really_a_png.png"
            path.write_bytes(self._jpeg_bytes())
            context = ToolRuntimeContext(working_dir=directory)
            data_url = _image_source_to_model_url(str(path), context)

        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        image_bytes = base64.b64decode(data_url.split("base64,", 1)[1])
        self.assertTrue(image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_invalid_image_bytes_are_rejected_before_model_request(self) -> None:
        context = ToolRuntimeContext(working_dir=tempfile.gettempdir())
        with self.assertRaisesRegex(ValueError, "not a decodable raster image"):
            _image_source_to_model_url(b"<html>blocked</html>", context)

    def test_cos_upload_retries_transient_failure(self) -> None:
        from opensearch_vl.opensearch_infer import cos_upload

        image = Image.new("RGB", (4, 3), "red")
        context = ToolRuntimeContext(working_dir=tempfile.gettempdir())
        with (
            mock.patch.object(
                cos_upload,
                "upload_pil_image",
                side_effect=[RuntimeError("temporary failure"), "https://cdn.example/crop.png"],
            ) as upload,
            mock.patch("synthesis.sft.api_tools.time.sleep") as sleep,
        ):
            result = _try_upload_pil_image(image, context, "i2i_region_test")

        self.assertEqual(result, "https://cdn.example/crop.png")
        self.assertEqual(upload.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_cmyk_images_are_converted_before_cos_upload_and_local_save(self) -> None:
        from opensearch_vl.opensearch_infer import cos_upload

        image = Image.new("CMYK", (4, 3), (0, 128, 128, 0))
        with tempfile.TemporaryDirectory() as directory:
            context = ToolRuntimeContext(working_dir=directory)
            with mock.patch.object(cos_upload, "upload_pil_image", return_value="https://cdn.example/crop.png") as upload:
                result = _try_upload_pil_image(image, context, "i2i_region_test")

            self.assertEqual(result, "https://cdn.example/crop.png")
            uploaded_image = upload.call_args.args[0]
            self.assertEqual(uploaded_image.mode, "RGB")

            _, saved_path = _persist_pil_image(image, context, "i2i_region_test")
            with Image.open(saved_path) as saved_image:
                self.assertEqual(saved_image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
