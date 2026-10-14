"""Single-turn inference pipeline without tool use."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config
from .no_tools_prompts import SYSTEM_PROMPT
from .pipeline import (
    _bootstrap_images,
    _first_present,
    _format_inference_error,
    _prompt_from_row,
    _row_to_dict,
    _save_failure_trajectory,
    _strip_base64_payloads,
    InferenceTiming,
)
from .runners import BaseRunner, InferenceConfig
from synthesis.sft.qwen3_vl_template import add_sft_image_placeholders


logger = logging.getLogger(__name__)


def process_single_case(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    inference_cfg: Optional[InferenceConfig] = None,
    dynamic_resolution: bool = False,
    timing: Optional[InferenceTiming] = None,
) -> Dict[str, Any]:
    """Run one benchmark example through a no-tools single-turn baseline."""

    filename_prefix = "fvqa_test" if dataset_type == "test" else "fvqa_train"
    row_dict = _row_to_dict(row)

    if row_dict:
        case_id = _first_present(
            row_dict, ("data_id", "id", "idx", "_id"), f"case_{case_idx}"
        )
        category = row_dict.get("category", "unknown")
        data_source = row_dict.get("data_source", row_dict.get("source", "unknown"))
        prompt_list, prompt_text = _prompt_from_row(row_dict)
    else:
        case_id = f"case_{case_idx}"
        category = "unknown"
        data_source = "unknown"
        prompt_list, prompt_text = [], ""
    case_id = str(case_id)

    logger.info(
        "Processing no-tools case %d (%s, category=%s, source=%s)",
        case_idx + 1,
        case_id,
        category,
        data_source,
    )

    image_paths_dict, initial_parts = _bootstrap_images(
        row,
        case_id,
        case_idx,
        filename_prefix,
        image_urls_dict,
        dynamic_resolution=dynamic_resolution,
    )
    initial_image_count = sum(1 for part in initial_parts if "image_url" in part or "inline_data" in part)
    prompt_text = add_sft_image_placeholders(prompt_text, initial_image_count)
    initial_parts.append({"text": prompt_text})

    gemini_contents: List[Dict[str, Any]] = [{"role": "user", "parts": initial_parts}]
    trajectory: Dict[str, Any] = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": category,
        "data_source": data_source,
        "prompt": prompt_list,
        "original_data": row_dict,
        "turns": [],
        "timestamp": datetime.now().isoformat(),
        "mode": "no_tools",
    }

    cfg = inference_cfg or InferenceConfig()

    def fail(kind: str, reason: str) -> None:
        try:
            _save_failure_trajectory(
                trajectory,
                output_dir,
                image_paths_dict,
                {},
                failure_kind=kind,
                failure_reason=reason,
                failed_turn=0,
                tool_call_count=0,
                max_turns=1,
                max_tool_calls=0,
            )
        except Exception as save_exc:
            logger.error("Could not save failure trajectory: %s", save_exc, exc_info=True)

    llm_started_at = time.perf_counter()
    llm_status = "exception"
    if config.TRACE_LOGS:
        logger.info(
            "[inference-trace] LLM_START case_id=%s case_idx=%s turn=0 model=%s mode=no_tools",
            case_id,
            case_idx,
            runner.display_name,
        )
    try:
        response = runner.infer(
            contents=gemini_contents,
            system_instruction=SYSTEM_PROMPT,
            cfg=cfg,
        )
        llm_status = "error_response" if response.get("inference_error") else "ok"
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        trajectory["turns"].append({"turn": 0, "error": str(exc)})
        fail("inference_error", str(exc))
        raise RuntimeError(f"No-tools inference failed: {exc}") from exc
    else:
        response_text = ""
        for cand in response.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    response_text += part["text"]

        trajectory["turns"].append(
            {
                "turn": 0,
                "response": response,
                "response_text": response_text,
            }
        )
        trajectory["final_response_text"] = response_text
        if response.get("inference_error"):
            reason = _format_inference_error(response["inference_error"], 0)
            fail("inference_error", reason)
            raise RuntimeError(reason)
        if not response_text.strip():
            llm_status = "invalid_response"
            reason = "No-tools inference returned an empty response."
            fail("invalid_response", reason)
            raise RuntimeError(reason)
    finally:
        if timing is not None:
            timing.record("llm", time.perf_counter() - llm_started_at)
        if config.TRACE_LOGS:
            logger.info(
                "[inference-trace] LLM_END case_id=%s case_idx=%s turn=0 "
                "model=%s mode=no_tools status=%s elapsed_s=%.3f",
                case_id,
                case_idx,
                runner.display_name,
                llm_status,
                time.perf_counter() - llm_started_at,
            )

    trajectory["status"] = "completed"

    image_urls = {
        img_id: data
        for img_id, data in image_paths_dict.items()
        if isinstance(data, str) and data.startswith(("http://", "https://"))
    }
    serialised = _strip_base64_payloads(trajectory, image_urls)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}_trajectory.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(serialised, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("No-tools trajectory saved to %s", out_path)
    return trajectory
