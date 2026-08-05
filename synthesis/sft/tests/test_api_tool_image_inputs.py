"""Tests for raster validation before images are submitted to a model."""

from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from synthesis.sft.api_tools import ToolRuntimeContext, _image_source_to_model_url


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


if __name__ == "__main__":
    unittest.main()
