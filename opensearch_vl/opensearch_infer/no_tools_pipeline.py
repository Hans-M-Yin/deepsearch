"""Single-turn inference pipeline without tool use."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .no_tools_prompts import SYSTEM_PROMPT
from .pipeline import (
    _bootstrap_images,
    _first_present,
    _prompt_from_row,
    _row_to_dict,
    _strip_base64_payloads,
)
from .runners import BaseRunner, InferenceConfig


logger = logging.getLogger(__name__)


def process_single_case(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    inference_cfg: Optional[InferenceConfig] = None,
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
        row, case_id, case_idx, filename_prefix, image_urls_dict
    )
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

    try:
        response = runner.infer(
            contents=gemini_contents,
            system_instruction=SYSTEM_PROMPT,
            cfg=cfg,
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        trajectory["turns"].append({"turn": 0, "error": str(exc)})
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

    if "final_response_text" not in trajectory:
        trajectory["final_response_text"] = "\n\n".join(
            turn.get("response_text", "")
            for turn in trajectory["turns"]
            if turn.get("response_text")
        )

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
