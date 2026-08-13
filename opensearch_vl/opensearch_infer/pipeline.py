"""Per-case inference pipeline (multi-turn agent loop)."""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from . import config, image_io, tools
from .runners import BaseRunner, InferenceConfig
from .prompts import build_system_prompt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory serialisation
# ---------------------------------------------------------------------------


def _strip_base64_payloads(obj: Any, image_urls: Dict[str, str]) -> Any:
    """Remove image payloads before writing trajectories to JSON."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<image bytes omitted: {len(obj)} bytes>"
    if isinstance(obj, list):
        return [_strip_base64_payloads(item, image_urls) for item in obj]
    if isinstance(obj, dict):
        replacement_url = next(iter(image_urls.values()), None)
        if "source" in obj and isinstance(obj["source"], dict):
            data = obj["source"].get("data", "")
            if isinstance(data, str) and len(data) > 100:
                if replacement_url:
                    return {"type": "image_url", "image_url": {"url": replacement_url}}
                return {"type": "image", "source": "<image payload omitted>"}
        if "inline_data" in obj and isinstance(obj["inline_data"], dict):
            data = obj["inline_data"].get("data", "")
            if isinstance(data, str) and len(data) > 100:
                if replacement_url:
                    return {"type": "image_url", "image_url": {"url": replacement_url}}
                return {
                    "inline_data": {
                        "mime_type": obj["inline_data"].get("mime_type", "image/*"),
                        "data": f"<image payload omitted: {len(data)} chars>",
                    }
                }
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > 1000:
                if any(key in obj for key in ("type", "media_type", "mime_type")):
                    out[k] = f"<image payload omitted: {len(v)} chars>"
                    continue
            out[k] = _strip_base64_payloads(v, image_urls)
        return out
    try:
        json.dumps(obj)
    except TypeError:
        return str(obj)
    return obj


def _save_failure_trajectory(
    trajectory: Dict[str, Any],
    output_dir: str,
    image_paths_dict: Dict[str, Any],
    *,
    failure_kind: str,
    failure_reason: str,
    failed_turn: int,
    tool_call_count: int,
) -> str:
    """Persist an incomplete trajectory for post-mortem inspection.

    Failure files live outside the normal ``*_trajectory.json`` namespace so
    ``run_infer`` will retry them on a later invocation rather than treating
    them as completed cases.
    """

    trajectory["status"] = "failed"
    trajectory["failure_kind"] = failure_kind
    trajectory["failure_reason"] = failure_reason
    trajectory["failed_turn"] = failed_turn
    trajectory["tool_call_count"] = tool_call_count
    trajectory["max_turns"] = config.MAX_TURNS
    trajectory["max_tool_calls"] = config.MAX_TOOL_CALLS
    trajectory["final_response_text"] = "\n\n".join(
        turn.get("response_text", "")
        for turn in trajectory.get("turns", [])
        if turn.get("response_text")
    )

    image_urls = {
        img_id: data
        for img_id, data in image_paths_dict.items()
        if isinstance(data, str) and data.startswith(("http://", "https://"))
    }
    serialised = _strip_base64_payloads(trajectory, image_urls)

    failure_dir = os.path.join(output_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)
    case_id = str(trajectory.get("case_id", "unknown_case"))
    out_path = os.path.join(failure_dir, f"{case_id}_failure.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(serialised, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("Failure trajectory saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Image bootstrap helpers
# ---------------------------------------------------------------------------


def _add_image_url(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    url: str,
) -> None:
    image_id = f"img_{len(image_paths_dict) + 1}"
    image_paths_dict[image_id] = url
    initial_parts.append({"image_url": {"url": url}})


def _add_inline_image(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    payload: bytes | str,
    raw_storage: Any,
) -> None:
    encoded = image_io.image_to_base64(payload) if isinstance(payload, bytes) else payload
    if not encoded:
        return
    image_id = f"img_{len(image_paths_dict) + 1}"
    image_paths_dict[image_id] = raw_storage
    initial_parts.append(
        {
            "inline_data": {
                "mime_type": image_io.detect_image_format(encoded),
                "data": encoded,
            }
        }
    )


def _bootstrap_images(
    row: Any,
    case_id: str,
    case_idx: int,
    filename_prefix: str,
    image_urls_dict: Optional[Dict[str, List[Any]]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the initial (image_paths_dict, parts) pair from row data.

    Resolution order:

    1. URLs supplied via ``image_urls_dict`` (cheapest, no network).
    2. The configured ``FVQA_IMAGE_DIR`` (loads bytes from disk).
    3. The ``images`` column on the row (parquet payloads).
    """

    image_paths_dict: Dict[str, Any] = {}
    initial_parts: List[Dict[str, Any]] = []

    def _normalize_image_entries(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return [value]

    if image_urls_dict and case_id in image_urls_dict:
        for entry in image_urls_dict[case_id]:
            if isinstance(entry, dict):
                url = entry.get("cos_url") or entry.get("image_url") or entry.get("url")
            else:
                url = str(entry)
            if url:
                _add_image_url(image_paths_dict, initial_parts, url)

    if not initial_parts and config.FVQA_IMAGE_DIR:
        local = image_io._try_local_image(case_id)
        if local:
            try:
                with open(local, "rb") as fh:
                    payload = fh.read()
                _add_inline_image(image_paths_dict, initial_parts, payload, local)
            except OSError as exc:
                logger.warning("Failed to read local image %s: %s", local, exc)

    if not initial_parts:
        images = row.get("images", None) if hasattr(row, "get") else None
        for entry in _normalize_image_entries(images):
            if entry is None:
                continue
            url: Optional[str] = None
            payload: Optional[bytes] = None
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("image_url") or entry.get("cos_url")
                payload = entry.get("bytes")
            elif isinstance(entry, str) and entry.startswith(("http://", "https://")):
                url = entry
            elif isinstance(entry, bytes):
                payload = entry

            if url:
                _add_image_url(image_paths_dict, initial_parts, url)
                continue
            if payload:
                # For benchmark/parquet inputs, inline the bytes directly.
                # External upload is only needed later for search tools that
                # require a publicly reachable URL.
                _add_inline_image(
                    image_paths_dict, initial_parts, payload, payload
                )
    return image_paths_dict, initial_parts


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if isinstance(row, dict):
        return dict(row)
    return {}


def _first_present(
    row: Dict[str, Any], keys: Iterable[str], default: Any = None
) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def _prompt_from_row(row: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
    prompt_list = row.get("prompt", [])
    if isinstance(prompt_list, list) and prompt_list:
        first = prompt_list[0]
        prompt_text = (
            first.get("content", "") if isinstance(first, dict) else str(first)
        )
        return prompt_list, prompt_text

    question = row.get("question") or row.get("query") or ""
    prompt_text = str(question) if question is not None else ""
    if prompt_text:
        return [{"role": "user", "content": prompt_text}], prompt_text
    return [], ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_single_case(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    """Drive one benchmark example through the agent."""

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
        "Processing case %d (%s, category=%s, source=%s)",
        case_idx + 1,
        case_id,
        category,
        data_source,
    )

    image_paths_dict, initial_parts = _bootstrap_images(
        row, case_id, case_idx, filename_prefix, image_urls_dict
    )
    url_registry: dict[str, Any] = {}

    tools_schema = tools.get_tools_definition()
    system_prompt = build_system_prompt(tools_schema)
    initial_parts.append({"text": prompt_text})

    gemini_contents: List[Dict[str, Any]] = [
        {"role": "user", "parts": initial_parts}
    ]

    trajectory: Dict[str, Any] = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": category,
        "data_source": data_source,
        "prompt": prompt_list,
        "original_data": row_dict,
        "turns": [],
        "timestamp": datetime.now().isoformat(),
    }

    intermediate_dir = os.path.join(output_dir, "intermediate")
    cfg = inference_cfg or InferenceConfig()
    completed = False
    tool_call_count = 0

    def persist_failure(kind: str, reason: str, failed_turn: int) -> None:
        try:
            _save_failure_trajectory(
                trajectory,
                output_dir,
                image_paths_dict,
                failure_kind=kind,
                failure_reason=reason,
                failed_turn=failed_turn,
                tool_call_count=tool_call_count,
            )
        except Exception as save_exc:
            # Preserve the original inference failure even if serialising the
            # diagnostic artifact itself fails.
            logger.error("Could not save failure trajectory: %s", save_exc, exc_info=True)

    for turn_num in range(config.MAX_TURNS):
        try:
            response = runner.infer(
                contents=gemini_contents,
                system_instruction=system_prompt,
                cfg=cfg,
            )
        except Exception as exc:
            logger.error("Inference failed on turn %d: %s", turn_num, exc, exc_info=True)
            persist_failure("inference_error", str(exc), turn_num)
            raise RuntimeError(f"Inference failed on turn {turn_num}: {exc}") from exc

        response_text = ""
        for cand in response.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    response_text += part["text"]

        turn_record: Dict[str, Any] = {
            "turn": turn_num,
            "response": response,
            "response_text": response_text,
        }
        trajectory["turns"].append(turn_record)

        candidates = response.get("candidates", []) or []
        if candidates:
            gemini_contents.append(
                {
                    "role": "model",
                    "parts": candidates[0].get("content", {}).get("parts", []),
                }
            )

        if tools.has_response_tag(response_text):
            completed = True
            break

        tool_call_json = tools.extract_tool_call(response_text)
        if not tool_call_json:
            persist_failure(
                "invalid_response",
                f"Inference ended on turn {turn_num} without a complete <answer> response.",
                turn_num,
            )
            raise RuntimeError(
                f"Inference ended on turn {turn_num} without a complete <answer> response."
            )

        tool_call_count += 1
        turn_record["tool_call"] = tool_call_json
        turn_record["tool_call_index"] = tool_call_count
        if tool_call_count > config.MAX_TOOL_CALLS:
            reason = (
                f"Inference attempted more than {config.MAX_TOOL_CALLS} tool calls "
                f"on turn {turn_num}."
            )
            persist_failure("tool_call_limit", reason, turn_num)
            raise RuntimeError(reason)

        os.makedirs(intermediate_dir, exist_ok=True)
        try:
            tool_message, new_images = tools.execute_tool(
                tool_call_json,
                image_paths_dict,
                case_id,
                case_idx,
                turn_num,
                intermediate_dir,
                filename_prefix=filename_prefix,
                visual_lookup=visual_lookup,
                url_registry=url_registry,
            )
        except Exception as exc:
            persist_failure("tool_execution_error", str(exc), turn_num)
            raise
        observation_text = f"<tool_response>\n{tool_message}\n</tool_response>"
        turn_record["tool_output"] = observation_text

        if new_images:
            for new_id, payload in new_images.items():
                image_paths_dict[new_id] = payload
                if isinstance(payload, str) and payload.startswith(
                    ("http://", "https://")
                ):
                    gemini_contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": observation_text},
                                {"image_url": {"url": payload}},
                            ],
                        }
                    )
                elif isinstance(payload, str) and os.path.exists(payload):
                    try:
                        with open(payload, "rb") as fh:
                            data = fh.read()
                        encoded = image_io.image_to_base64(data) or ""
                        gemini_contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {"text": observation_text},
                                    {
                                        "inline_data": {
                                            "mime_type": image_io.detect_image_format(
                                                encoded
                                            ),
                                            "data": encoded,
                                        }
                                    },
                                ],
                            }
                        )
                    except OSError as exc:
                        logger.warning("Cannot read intermediate image %s: %s", payload, exc)
                        gemini_contents.append(
                            {"role": "user", "parts": [{"text": observation_text}]}
                        )
                elif isinstance(payload, bytes):
                    encoded = image_io.image_to_base64(payload) or ""
                    gemini_contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": observation_text},
                                {
                                    "inline_data": {
                                        "mime_type": image_io.detect_image_format(
                                            encoded
                                        ),
                                        "data": encoded,
                                    }
                                },
                            ],
                        }
                    )
                else:
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": observation_text}]}
                    )
        else:
            gemini_contents.append(
                {"role": "user", "parts": [{"text": observation_text}]}
            )

    if not completed:
        reason = (
            f"Inference reached the {config.MAX_TURNS}-turn limit "
            "without a complete <answer> response."
        )
        persist_failure("max_turns", reason, config.MAX_TURNS - 1)
        raise RuntimeError(reason)

    trajectory["status"] = "completed"
    trajectory["tool_call_count"] = tool_call_count
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
    logger.info("Trajectory saved to %s", out_path)
    return trajectory
