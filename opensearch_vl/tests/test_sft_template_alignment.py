"""Regression tests for the SFT-compatible multimodal message layout."""

from __future__ import annotations

import base64
import io

from PIL import Image

from opensearch_vl.opensearch_infer import messages
from synthesis.sft.qwen3_vl_template import (
    add_sft_image_placeholders,
    interleave_sft_image_parts,
    render_sft_qwen3_vl_text,
)


def _inline_png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_sft_image_placeholder_is_interleaved_at_its_text_position() -> None:
    assert add_sft_image_placeholders("Question?", 2) == "Question?\n<image>\n<image>"
    assert interleave_sft_image_parts(
        [("text", "before\n<image>\nafter"), ("image", "img_1")]
    ) == [("text", "before\n"), ("image", "img_1"), ("text", "\nafter")]


def test_run_infer_qwen_conversion_matches_sft_image_order() -> None:
    contents = [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": _inline_png()}},
                {"text": "Question?\n<image>"},
            ],
        }
    ]

    converted = messages.to_qwen3vl_messages(contents)
    assert [part["type"] for part in converted[0]["content"]] == ["text", "image"]
    assert "<image>" not in converted[0]["content"][0]["text"]


def test_sft_observation_wrapper_and_generation_prompt() -> None:
    rendered = render_sft_qwen3_vl_text(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<thinking>plan</thinking><tool_call>{}</tool_call>"},
            {"role": "tool", "content": '{"image_id":"img_2"}\nThe image is shown below:\n<image>'},
        ],
        add_generation_prompt=True,
    )
    assert "<|im_start|>system\nsystem<|im_end|>\n" in rendered
    assert "<|im_start|>user\n<tool_response>\n" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")
