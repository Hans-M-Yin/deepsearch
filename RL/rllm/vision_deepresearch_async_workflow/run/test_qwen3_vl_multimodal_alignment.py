#!/usr/bin/env python3
"""Targeted Qwen3-VL multimodal RL smoke test.

This script does not start Ray or SGLang.  It checks the two pieces that are
easy to confuse in the current DeepResearch RL path:

1. ``processor`` expansion: one logical ``<image>`` marker becomes the
   correct number of visual tokens and produces ``pixel_values``/
   ``image_grid_thw``.
2. trajectory-to-DataProto conversion: the current custom workflow rebuilds
   trajectories from chat messages and may drop ``multi_modal_inputs``.  The
   script exercises both that current fallback path and a control path where
   ``Step.model_output`` is preserved.

The control path is intentionally included so that a failure is easy to
interpret: if the current path has empty multimodal inputs while the control
path has real visual inputs, the loss happens between rollout and training
batch construction.

Example:

    PYTHONPATH=.:RL/rllm python \
      RL/rllm/vision_deepresearch_async_workflow/run/test_qwen3_vl_multimodal_alignment.py \
      --model-path /path/to/qwen3-vl-8b --case all --strict

The script uses synthetic PIL images by default, so no network image fetch is
needed.  Use ``--image`` one or more times to test real local images instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# Make the script runnable from the repository root or from RL/rllm.  The
# parser imports synthesis.sft.qwen3_vl_template, while rllm/ and the bundled
# verl/ package live below RL/rllm.
SCRIPT_PATH = Path(__file__).resolve()
RLLM_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = SCRIPT_PATH.parents[4]
for import_root in (str(PROJECT_ROOT), str(RLLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import ModelOutput
from rllm.parser import ChatTemplateParser
from rllm.workflows.workflow import TerminationReason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local Qwen3-VL model/processor path. Defaults to MODEL_PATH.",
    )
    parser.add_argument(
        "--case",
        choices=("one", "multi", "tool", "all"),
        default="all",
        help="Message case to test. 'tool' includes an image returned by a tool.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Local image path; repeat this option to provide multiple images.",
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=16384
    )
    parser.add_argument(
        "--max-response-length", type=int, default=512
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Only check parser/processor expansion.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when the current workflow drops image inputs.",
    )
    return parser.parse_args()


def synthetic_images() -> list[Image.Image]:
    """Use different aspect ratios so dynamic image token counts are tested."""

    return [
        Image.new("RGB", (640, 480), (180, 40, 40)),
        Image.new("RGB", (1280, 720), (40, 180, 40)),
        Image.new("RGB", (512, 1024), (40, 40, 180)),
    ]


def load_images(paths: list[str]) -> list[Image.Image]:
    if not paths:
        return synthetic_images()

    images: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        images.append(image)
    return images


def image_token_id(processor: Any, tokenizer: Any) -> int:
    for owner in (processor, tokenizer):
        value = getattr(owner, "image_token_id", None)
        if value is not None:
            return int(value)

    value = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if value is None or value == tokenizer.unk_token_id:
        raise RuntimeError("Cannot find Qwen3-VL image token id (<|image_pad|>)")
    return int(value)


def tensor_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(x) for x in value.shape]
    return None


def count_token(ids: Any, token_id: int) -> int:
    if isinstance(ids, torch.Tensor):
        return int((ids == token_id).sum().item())
    return sum(int(token) == token_id for token in ids)


def make_messages(images: list[Image.Image], case: str) -> list[dict[str, Any]]:
    if case == "one":
        images = images[:1]
    elif case == "multi":
        images = images[:2]
    elif case == "tool":
        images = images[:2]
    else:
        raise ValueError(f"Unsupported case: {case}")

    initial_content = "\n".join(["<image>"] * len(images))
    initial_content += "\nCompare the visual evidence and answer the question."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a visual research assistant."},
        {"role": "user", "content": initial_content, "images": images},
    ]

    if case == "tool":
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        "<thinking>Need more visual evidence.</thinking>\n"
                        "<tool_call>{\"name\":\"read_url\",\"arguments\":{}}"
                        "</tool_call>"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "<tool_response>\n"
                        "The tool returned an additional image.\n"
                        "The image is shown below:\n"
                        "<image>\n"
                        "</tool_response>"
                    ),
                    "images": [images[0] if len(images) == 1 else synthetic_images()[2]],
                },
            ]
        )

    messages.append(
        {
            "role": "assistant",
            "content": "<thinking>The evidence is sufficient.</thinking>\n<answer>smoke-test</answer>",
        }
    )
    return messages


def make_config(max_prompt_length: int, max_response_length: int) -> Any:
    return SimpleNamespace(
        data=SimpleNamespace(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        ),
        rllm=SimpleNamespace(
            stepwise_advantage=SimpleNamespace(enable=False),
            compact_filtering=SimpleNamespace(enable=False),
        ),
    )


def make_engine(tokenizer: Any, processor: Any, parser: Any, config: Any) -> Any:
    """Build only the conversion part; no executor, Ray, or server is started."""

    engine = AgentWorkflowEngine.__new__(AgentWorkflowEngine)
    engine.config = config
    engine.rollout_engine = SimpleNamespace(
        tokenizer=tokenizer,
        processor=processor,
        chat_parser=parser,
    )
    return engine


def make_episode(
    messages: list[dict[str, Any]],
    model_output: ModelOutput | None = None,
) -> Episode:
    final_assistant = messages[-1]
    step = Step(
        chat_completions=messages,
        model_response=final_assistant.get("content", ""),
        model_output=model_output,
        reward=1.0,
        done=True,
    )
    trajectory = Trajectory(
        name="multimodal_alignment_smoke",
        task={"question": "synthetic smoke test"},
        steps=[step],
        reward=1.0,
    )
    return Episode(
        id="multimodal-alignment-smoke",
        task={"question": "synthetic smoke test"},
        termination_reason=TerminationReason.ENV_DONE,
        is_correct=True,
        trajectories=[trajectory],
    )


def processor_check(
    tokenizer: Any,
    processor: Any,
    parser: Any,
    messages: list[dict[str, Any]],
    token_id: int,
) -> dict[str, Any]:
    prompt = parser.parse(
        messages,
        add_generation_prompt=True,
        is_first_msg=True,
        accumulate_reasoning=False,
    )
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ordered_images = [
        image
        for message in messages
        for image in (message.get("images") or [])
    ]
    encoded = processor(
        text=[prompt],
        images=ordered_images or None,
        return_tensors="pt",
    )
    expanded_ids = encoded["input_ids"][0]
    grid = encoded.get("image_grid_thw")
    pixel_values = encoded.get("pixel_values")

    grid_count = int(grid.shape[0]) if grid is not None else 0
    merge_size = int(getattr(processor.image_processor, "merge_size", 1))
    expected_feature_tokens = None
    if grid is not None:
        expected_feature_tokens = int(
            sum(
                int(t) * (int(h) // merge_size) * (int(w) // merge_size)
                for t, h, w in grid.tolist()
            )
        )

    result = {
        "message_images": len(ordered_images),
        "raw_prompt_length": len(raw_ids),
        "raw_image_token_count": count_token(raw_ids, token_id),
        "expanded_prompt_length": int(expanded_ids.shape[-1]),
        "expanded_image_token_count": count_token(expanded_ids, token_id),
        "image_grid_thw_shape": tensor_shape(grid),
        "pixel_values_shape": tensor_shape(pixel_values),
        "expected_feature_tokens_from_grid": expected_feature_tokens,
        "rendered_prompt_tail": prompt[-500:],
    }
    result["grid_matches_images"] = grid_count == len(ordered_images)
    result["raw_tokens_match_logical_images"] = (
        result["raw_image_token_count"] == len(ordered_images)
    )
    result["expanded_tokens_match_features"] = (
        expected_feature_tokens is not None
        and result["expanded_image_token_count"] == expected_feature_tokens
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return {
        "prompt": prompt,
        "ordered_images": ordered_images,
        "encoded": encoded,
        "result": result,
    }


def summarize_dataproto(data: Any, token_id: int, label: str) -> dict[str, Any]:
    input_ids = data.batch["input_ids"]
    position_ids = data.batch["position_ids"]
    attention_mask = data.batch["attention_mask"]
    mm = data.non_tensor_batch.get("multi_modal_inputs")

    entries: list[dict[str, Any]] = []
    if mm is not None:
        for item in list(mm):
            if not isinstance(item, dict):
                entries.append({"type": type(item).__name__})
                continue
            entries.append(
                {
                    "keys": sorted(item.keys()),
                    "pixel_values_shape": tensor_shape(item.get("pixel_values")),
                    "image_grid_thw_shape": tensor_shape(item.get("image_grid_thw")),
                }
            )

    summary = {
        "label": label,
        "input_ids_shape": tensor_shape(input_ids),
        "attention_mask_shape": tensor_shape(attention_mask),
        "position_ids_shape": tensor_shape(position_ids),
        "image_token_counts": [count_token(row, token_id) for row in input_ids],
        "has_multi_modal_inputs_field": mm is not None,
        "multi_modal_entries": entries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def transform_check(
    tokenizer: Any,
    processor: Any,
    parser: Any,
    messages: list[dict[str, Any]],
    encoded: Any,
    config: Any,
    token_id: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    engine = make_engine(tokenizer, processor, parser, config)
    task_ids = np.array(["multimodal-alignment-smoke"], dtype=object)

    # This reproduces the current DeepResearchWorkflow behavior, which sets
    # step.model_output = None before conversion.
    current_episode = make_episode(messages, model_output=None)
    current_summary: dict[str, Any] | None = None
    try:
        current_data = engine.transform_results_for_verl([current_episode], task_ids)
        current_summary = summarize_dataproto(
            current_data, token_id, "current_workflow_fallback"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[CURRENT PATH ERROR] {type(exc).__name__}: {exc}")

    # Control path: preserve exactly the processor output that VerlEngine puts
    # into ModelOutput.  This is the expected shape for a visual training batch.
    mm_inputs = dict(encoded)
    mm_inputs.pop("input_ids", None)
    mm_inputs.pop("attention_mask", None)
    completion_ids = tokenizer.encode(
        "<answer>smoke-test</answer>", add_special_tokens=False
    )
    preserved_output = ModelOutput(
        prompt_ids=encoded["input_ids"][0].tolist(),
        completion_ids=completion_ids,
        multi_modal_inputs=mm_inputs,
    )
    control_summary: dict[str, Any] | None = None
    try:
        control_episode = make_episode(messages, model_output=preserved_output)
        control_data = engine.transform_results_for_verl([control_episode], task_ids)
        control_summary = summarize_dataproto(
            control_data, token_id, "preserved_model_output_control"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[CONTROL PATH ERROR] {type(exc).__name__}: {exc}")

    return current_summary, control_summary


def main() -> int:
    args = parse_args()
    model_path = args.model_path
    if not model_path:
        import os

        model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("Missing --model-path and MODEL_PATH", file=sys.stderr)
        return 2

    print(f"Loading tokenizer and processor from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    parser = ChatTemplateParser.get_parser(tokenizer, processor=processor)
    token_id = image_token_id(processor, tokenizer)
    print(f"image_token_id={token_id}")

    all_images = load_images(args.image)
    cases = ["one", "multi", "tool"] if args.case == "all" else [args.case]
    config = make_config(args.max_prompt_length, args.max_response_length)

    processor_failures = 0
    current_drops = 0
    control_failures = 0

    for case in cases:
        print(f"\n===== CASE: {case} =====")
        messages = make_messages(all_images, case)
        try:
            processor_result = processor_check(
                tokenizer, processor, parser, messages, token_id
            )
        except Exception as exc:  # noqa: BLE001
            processor_failures += 1
            print(f"[PROCESSOR ERROR] {type(exc).__name__}: {exc}")
            continue

        if not processor_result["result"]["grid_matches_images"]:
            processor_failures += 1
        if not processor_result["result"]["raw_tokens_match_logical_images"]:
            processor_failures += 1
        if not processor_result["result"]["expanded_tokens_match_features"]:
            processor_failures += 1

        if args.skip_transform:
            continue

        current_summary, control_summary = transform_check(
            tokenizer,
            processor,
            parser,
            messages,
            processor_result["encoded"],
            config,
            token_id,
        )

        if current_summary is not None:
            has_visual_payload = any(
                entry.get("pixel_values_shape") or entry.get("image_grid_thw_shape")
                for entry in current_summary["multi_modal_entries"]
                if isinstance(entry, dict)
            )
            if not has_visual_payload:
                current_drops += 1
        else:
            # A transform exception is also a failed current-path check.  The
            # printed exception above explains whether it is a missing
            # multimodal payload or an incompatible processor/position-id
            # implementation.
            current_drops += 1

        if control_summary is None:
            control_failures += 1
        else:
            control_has_visual_payload = any(
                entry.get("pixel_values_shape") or entry.get("image_grid_thw_shape")
                for entry in control_summary["multi_modal_entries"]
                if isinstance(entry, dict)
            )
            if not control_has_visual_payload:
                control_failures += 1

    print("\n===== SUMMARY =====")
    summary = {
        "cases": cases,
        "processor_failures": processor_failures,
        "current_workflow_visual_payload_drops": current_drops,
        "control_path_failures": control_failures,
        "interpretation": (
            "The current custom workflow drops visual payloads if current_workflow_visual_payload_drops > 0. "
            "The preserved ModelOutput control should retain pixel_values/image_grid_thw."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict and (
        processor_failures or current_drops or control_failures
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
