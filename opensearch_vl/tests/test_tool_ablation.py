"""Offline checks for tool removal and execution blocking."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from opensearch_vl.opensearch_infer import tools
from opensearch_vl.opensearch_infer.prompts import build_system_prompt
from opensearch_vl.opensearch_infer import config, pipeline
import run_infer


def test_t2i_ablation_removes_schema_and_prompt_mentions():
    baseline = tools.get_tools_definition()
    schema = tools.get_tools_definition(disabled_tools=("t2i_search",))
    names = lambda value: {tool["function"]["name"] for tool in json.loads(value)}
    assert "t2i_search" in names(baseline)
    assert names(schema) == names(baseline) - {"t2i_search"}
    prompt = build_system_prompt(schema, disabled_tools=("t2i_search",))
    assert "t2i_search" not in prompt
    assert "four tools" not in prompt
    for name in ("t2t_search", "i2i_search", "read_url"):
        assert name in prompt
    assert "t2i_search" in build_system_prompt(baseline)
    assert "You cannot download or view new external images" in prompt
    assert "image will be downloaded for you" not in prompt


def test_t2i_ablation_blocks_calls_before_dispatch():
    with patch.object(tools.sft_tools, "t2i_search") as search:
        message, images = tools.execute_tool(
            {"name": "t2i_search", "parameters": {"query": "test"}},
            {}, "case", 0, 1, "/tmp", disabled_tools=("t2i_search",),
        )
        search.assert_not_called()
    assert "disabled" in message
    assert images == {}


def test_default_dispatch_remains_available():
    with patch.object(tools.sft_tools, "t2i_search", return_value={"ok": False}) as search:
        tools.execute_tool(
            {"name": "t2i_search", "parameters": {"query": "test"}},
            {}, "case", 0, 1, "/tmp",
        )
        search.assert_called_once()


def test_cli_parses_ablation():
    args = run_infer._build_arg_parser().parse_args([
        "--data-path", "test.parquet", "--output-dir", "/tmp/test",
        "--ablation", "t2i_search",
    ])
    assert args.ablation == ["t2i_search"]


def test_cli_parses_multiple_and_repeated_ablation_flags():
    for flags in (["--ablation", "t2i_search", "i2i_search"],
                  ["--ablation", "t2i_search", "--ablation", "i2i_search"]):
        args = run_infer._build_arg_parser().parse_args([
            "--data-path", "test.parquet", "--output-dir", "/tmp/test", *flags,
        ])
        assert set(args.ablation) == {"t2i_search", "i2i_search"}


def test_combined_ablation_removes_examples_and_blocks_both_tools():
    disabled = ("t2i_search", "i2i_search")
    schema = tools.get_tools_definition(disabled_tools=disabled)
    prompt = build_system_prompt(schema, disabled_tools=disabled)
    assert "t2i_search" not in prompt
    assert "i2i_search" not in prompt
    assert "t2t_search" in prompt
    assert "read_url" in prompt
    with patch.object(tools.sft_tools, "t2i_search") as t2i, patch.object(
        tools.sft_tools, "i2i_search"
    ) as i2i:
        for name in disabled:
            message, images = tools.execute_tool(
                {"name": name, "parameters": {}}, {}, "case", 0, 1, "/tmp",
                disabled_tools=disabled,
            )
            assert "disabled" in message
            assert images == {}
        t2i.assert_not_called()
        i2i.assert_not_called()


def test_pipeline_ablation_blocks_model_call_and_records_mode():
    class Runner:
        display_name = "fake"
        prompts = []

        def infer(self, *, contents, system_instruction, cfg):
            self.prompts.append(system_instruction)
            text = ('<thinking>Search.</thinking><tool_call>'
                    '{"name":"t2i_search","arguments":{"query":"test"}}</tool_call>'
                    if len(self.prompts) == 1 else
                    '<thinking>Done.</thinking><answer>test</answer>')
            return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    runner = Runner()
    with tempfile.TemporaryDirectory() as output_dir, patch.object(config, "MAX_TURNS", 4), patch.object(
        tools.sft_tools, "t2i_search"
    ) as search:
        trajectory = pipeline.process_single_case(
            {"id": "ablation-test", "question": "Test?", "images": []},
            runner, output_dir, 0, disabled_tools=("t2i_search",),
        )
        search.assert_not_called()
    assert all("t2i_search" not in prompt for prompt in runner.prompts)
    assert trajectory["disabled_tools"] == ["t2i_search"]
    assert trajectory["status"] == "completed"


def test_read_url_ablation_blocks_all_image_download_candidates_before_fetch():
    sft = tools.sft_tools
    image_resource = sft.UrlResource(
        primary_url="https://example.com/media/123", kind="image",
        thumbnail_url="https://example.com/thumb.jpg",
        source_page_url="https://example.com/page",
    )
    targets = [
        ("https://example.com/photo.jpg", None),
        ("https://commons.wikimedia.org/wiki/File:Painting.jpg", None),
        (image_resource.primary_url, image_resource),
    ]
    with patch.object(sft, "_download_image_payload_with_cache") as cache, patch.object(
        sft, "_source_page_image_candidates"
    ) as fallback:
        for url, resource in targets:
            result = sft.read_url(url, resource=resource, allow_images=False)
            assert not result["ok"]
            assert result["error_code"] == "image_download_disabled"
            assert "local_path" not in result
        cache.assert_not_called()
        fallback.assert_not_called()


def test_read_url_dispatch_applies_image_policy_and_rejects_unexpected_image():
    with patch.object(tools.sft_tools, "read_url", return_value={
        "ok": True, "local_path": "/tmp/unexpected.png",
    }) as read:
        message, images = tools.execute_tool(
            {"name": "read_url", "parameters": {"url": "https://example.com/photo.png"}},
            {}, "case", 0, 1, "/tmp", disabled_tools=("t2i_search",),
        )
        assert read.call_args.kwargs["allow_images"] is False
        assert images == {}
        assert "disabled" in message


def test_text_reading_remains_available_without_image_downloads():
    sft = tools.sft_tools
    with patch.object(sft, "_read_document_with_retry", return_value={
        "title": "Page", "content": "Useful caption text.",
    }), patch.object(sft, "summarize_with_qwen", return_value="Useful caption text."):
        result = sft.read_url("https://example.com/page", allow_images=False)
    assert result["ok"]
    assert result["content"] == "Useful caption text."
    assert "local_path" not in result


def test_default_read_url_image_delivery_is_unchanged():
    with patch.object(tools.sft_tools, "read_url", return_value={
        "ok": True, "local_path": "/tmp/photo.png",
    }) as read:
        _, images = tools.execute_tool(
            {"name": "read_url", "parameters": {"url": "https://example.com/photo.png"}},
            {}, "case", 0, 1, "/tmp",
        )
    assert "allow_images" not in read.call_args.kwargs
    assert images == {"img_1": "/tmp/photo.png"}


def test_old_image_enabled_ablation_results_cannot_be_resumed_under_new_policy():
    with tempfile.TemporaryDirectory() as output_dir:
        (Path(output_dir) / "run_config.json").write_text(json.dumps({
            "arguments": {"ablation": ["t2i_search"]},
        }))
        with pytest.raises(SystemExit) as error:
            run_infer.main([
                "--data-path", "test.parquet", "--output-dir", output_dir,
                "--ablation", "t2i_search",
            ])
        assert error.value.code == 2
