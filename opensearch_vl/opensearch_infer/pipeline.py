"""Per-case inference pipeline (multi-turn agent loop)."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from . import config, image_io, tools
from .runners import BaseRunner, InferenceConfig
from .prompts import build_system_prompt
from synthesis.sft.qwen3_vl_template import add_sft_image_placeholders


logger = logging.getLogger(__name__)


_MAX_TURNS_FINALIZATION_PROMPT = (
    "已达到搜索上限，不能再调用任何工具。这是最后一次回答机会。"
    "请根据现有线索和你的知识直接给出原问题的最终答案；即使证据不完整，"
    "也必须给出最可能的答案，不能要求继续搜索或调用工具。"
    "请直接根据历史搜索记录以及输入问题进行回答。"
    "你的完整回复必须且只能是"
    "<thinking>...</thinking><answer>...</answer>。"
    "严禁输出<tool_call>、任何工具调用 JSON、工具名或工具参数；"
    "除这两个标签外，不得输出任何文本。"
)


class InferenceTiming:
    """Collect per-sample wall-clock timings for INFO-level diagnostics."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self._modules: Dict[str, Dict[str, float | int]] = {}

    def record(self, module: str, elapsed_s: float) -> None:
        stats = self._modules.setdefault(module, {"calls": 0, "total_s": 0.0})
        stats["calls"] = int(stats["calls"]) + 1
        stats["total_s"] = float(stats["total_s"]) + elapsed_s

    def emit(self, *, case_id: str, case_idx: int, status: str) -> None:
        """Write one summary plus one line per module at INFO level."""

        if not config.TRACE_LOGS or not logger.isEnabledFor(logging.INFO):
            return

        elapsed_s = time.perf_counter() - self.started_at
        logger.info(
            "[inference-timing] case_id=%s case_idx=%s status=%s "
            "total_elapsed_s=%.3f modules=%d",
            case_id,
            case_idx,
            status,
            elapsed_s,
            len(self._modules),
        )
        for module, stats in self._modules.items():
            calls = int(stats["calls"])
            total_s = float(stats["total_s"])
            average_s = total_s / calls if calls else 0.0
            logger.info(
                "[inference-timing] case_id=%s module=%s calls=%d "
                "total_s=%.3f average_s=%.3f",
                case_id,
                module,
                calls,
                total_s,
                average_s,
            )


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


def _serialise_url_resources(url_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the per-case URL registry into JSON-safe provenance data.

    ``url_registry`` intentionally contains aliases as well as resource IDs:
    ``read_url`` may be called with either form.  Keep both pieces of
    information so a saved trajectory can be used to reconstruct exactly
    which URL a resource ID resolved to.
    """

    resources: Dict[str, Dict[str, Any]] = {}
    aliases: Dict[str, str] = {}
    collisions: List[Dict[str, Any]] = []

    for registry_key, resource in url_registry.items():
        if hasattr(resource, "to_dict"):
            payload = resource.to_dict()
        elif isinstance(resource, dict):
            payload = dict(resource)
        else:
            payload = {"value": str(resource)}

        resource_id = str(payload.get("resource_id") or registry_key)
        payload["resource_id"] = resource_id
        registry_key = str(registry_key)

        existing = resources.get(resource_id)
        if existing is not None:
            existing_url = existing.get("primary_url")
            current_url = payload.get("primary_url")
            if existing_url != current_url:
                collisions.append(
                    {
                        "resource_id": resource_id,
                        "existing_primary_url": existing_url,
                        "new_primary_url": current_url,
                        "registry_key": registry_key,
                    }
                )
        else:
            resources[resource_id] = payload

        aliases[registry_key] = resource_id

    return {
        "schema_version": 1,
        "resource_count": len(resources),
        "resources": resources,
        "aliases": aliases,
        "collisions": collisions,
    }


def _save_failure_trajectory(
    trajectory: Dict[str, Any],
    output_dir: str,
    image_paths_dict: Dict[str, Any],
    url_registry: Dict[str, Any],
    *,
    failure_kind: str,
    failure_reason: str,
    failed_turn: int,
    tool_call_count: int,
    max_turns: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
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
    trajectory["max_turns"] = config.MAX_TURNS if max_turns is None else max_turns
    trajectory["max_tool_calls"] = config.MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls
    trajectory["final_response_text"] = "\n\n".join(
        turn.get("response_text", "")
        for turn in trajectory.get("turns", [])
        if turn.get("response_text")
    )
    trajectory["url_resources"] = _serialise_url_resources(url_registry)

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


DYNAMIC_RESOLUTION_MAX_SIDE = 1600


def _resize_image_bytes(
    payload: bytes,
    max_side: int = DYNAMIC_RESOLUTION_MAX_SIDE,
) -> bytes:
    """Downscale an image so its longest side is at most ``max_side``."""

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if max(source.size) <= max_side:
                return payload

            image = source.copy()
            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail((max_side, max_side), resampling.LANCZOS)
            output_format = (
                source.format
                if source.format in {"JPEG", "PNG", "WEBP"}
                else "PNG"
            )
            if output_format == "JPEG" and image.mode not in {"L", "RGB"}:
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format=output_format)
            return output.getvalue()
    except Exception as exc:
        logger.warning(
            "Could not resize image for dynamic resolution; using original: %s",
            exc,
        )
        return payload


def _model_image_part(
    value: Any,
    dynamic_resolution: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convert an image reference into the content part sent to the model."""

    payload: Optional[bytes] = None
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        if value.startswith(("http://", "https://")):
            if dynamic_resolution:
                payload = image_io.download_image_bytes(value)
                if payload is None:
                    logger.warning(
                        "Dynamic resolution could not download %s; keeping URL input",
                        value[:120],
                    )
            if payload is None:
                return {"image_url": {"url": value}}
        else:
            local_path = image_io.local_image_path_from_reference(value)
            if local_path and os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as fh:
                        payload = fh.read()
                except OSError as exc:
                    logger.warning("Could not read image %s: %s", local_path, exc)

    if payload is None:
        return None
    if dynamic_resolution:
        payload = _resize_image_bytes(payload)
    encoded = image_io.image_to_base64(payload)
    if not encoded:
        return None
    return {
        "inline_data": {
            "mime_type": image_io.detect_image_format(encoded),
            "data": encoded,
        }
    }


def _add_image_url(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    url: str,
    dynamic_resolution: bool = False,
) -> None:
    image_id = f"img_{len(image_paths_dict) + 1}"
    image_paths_dict[image_id] = url
    image_part = _model_image_part(url, dynamic_resolution)
    if image_part is not None:
        initial_parts.append(image_part)


def _add_inline_image(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    payload: bytes | str,
    raw_storage: Any,
    dynamic_resolution: bool = False,
) -> None:
    image_part = _model_image_part(payload, dynamic_resolution)
    if image_part is None:
        return
    image_id = f"img_{len(image_paths_dict) + 1}"
    image_paths_dict[image_id] = raw_storage
    initial_parts.append(image_part)


def _bootstrap_images(
    row: Any,
    case_id: str,
    case_idx: int,
    filename_prefix: str,
    image_urls_dict: Optional[Dict[str, List[Any]]],
    dynamic_resolution: bool = False,
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
                _add_image_url(
                    image_paths_dict,
                    initial_parts,
                    url,
                    dynamic_resolution=dynamic_resolution,
                )

    if not initial_parts and config.FVQA_IMAGE_DIR:
        local = image_io._try_local_image(case_id)
        if local:
            try:
                with open(local, "rb") as fh:
                    payload = fh.read()
                _add_inline_image(
                    image_paths_dict,
                    initial_parts,
                    payload,
                    local,
                    dynamic_resolution=dynamic_resolution,
                )
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
                _add_image_url(
                    image_paths_dict,
                    initial_parts,
                    url,
                    dynamic_resolution=dynamic_resolution,
                )
                continue
            if payload:
                # For benchmark/parquet inputs, inline the bytes directly.
                # External upload is only needed later for search tools that
                # require a publicly reachable URL.
                _add_inline_image(
                    image_paths_dict,
                    initial_parts,
                    payload,
                    payload,
                    dynamic_resolution=dynamic_resolution,
                )
    return image_paths_dict, initial_parts


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if isinstance(row, dict):
        return dict(row)
    return {}


def _format_inference_error(details: Any, turn_num: int) -> str:
    """Make a runner-level error actionable in logs and failure artifacts."""

    if not isinstance(details, dict):
        return f"Model request failed on turn {turn_num}: {details}"

    status_code = details.get("status_code")
    url = details.get("url") or "<unknown>"
    attempt = details.get("attempt")
    max_attempts = details.get("max_attempts")
    response_body = details.get("response_body") or "<empty>"
    error_type = details.get("error_type") or "<unknown>"
    message = details.get("message") or "<empty>"
    request_summary = details.get("request_summary") or {}
    return (
        f"Model request failed on turn {turn_num}: "
        f"url={url} status_code={status_code} error_type={error_type} "
        f"attempt={attempt}/{max_attempts} response_body={response_body} "
        f"request_summary={request_summary} error={message}"
    )


def _tool_name(tool_call: Any) -> str:
    """Extract a stable tool label for timing output."""

    call = tool_call
    if isinstance(tool_call, str):
        try:
            call = json.loads(tool_call)
        except (TypeError, ValueError):
            return "unknown"
    if isinstance(call, dict):
        name = call.get("name")
        if name:
            return str(name)
    return "unknown"


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


def _format_error_answer_text(text: str) -> str:
    """Normalize public final text only; never turn unfinished thinking into an answer."""
    # A final answer may have only an opening tag. Earlier turns must not be
    # included here: their unclosed tags could capture subsequent tool history.
    text = re.sub(
        r'<?(?:thinking|think|thought)\s*>.*?(?:</(?:thinking|think|thought)\s*>|\Z)',
        '', text, flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if re.search(r'<tool_call\b', text, re.IGNORECASE):
        return ''
    match = re.search(r'<answer\s*>(.*?)(?:</answer\s*>|\Z)', text,
                      flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def process_single_case(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
    timing: Optional[InferenceTiming] = None,
    dynamic_resolution: bool = False,
    disabled_tools: tuple[str, ...] = (),
    retain_format_errors: bool = False,
) -> Dict[str, Any]:
    """Drive one example; strict by default, optionally retain format-only errors."""

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
        row,
        case_id,
        case_idx,
        filename_prefix,
        image_urls_dict,
        dynamic_resolution=dynamic_resolution,
    )
    url_registry: dict[str, Any] = {}

    tools_schema = tools.get_tools_definition(disabled_tools=disabled_tools)
    system_prompt = build_system_prompt(tools_schema, disabled_tools=disabled_tools)
    initial_image_count = sum(1 for part in initial_parts if "image_url" in part or "inline_data" in part)
    prompt_text = add_sft_image_placeholders(prompt_text, initial_image_count)
    # Keep the logical message in the same form as SFT data: question text
    # followed by one <image> marker per supplied image.  The runner-specific
    # converter places the actual image payload at those marker positions.
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
    if disabled_tools:
        trajectory["disabled_tools"] = list(disabled_tools)
        trajectory["image_download_disabled"] = "t2i_search" in disabled_tools

    intermediate_dir = os.path.join(output_dir, "intermediate")
    cfg = inference_cfg or InferenceConfig()
    completed = False
    tool_call_count = 0
    terminal_format_error: Optional[Dict[str, Any]] = None

    def persist_failure(kind: str, reason: str, failed_turn: int) -> None:
        try:
            _save_failure_trajectory(
                trajectory,
                output_dir,
                image_paths_dict,
                url_registry,
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
        llm_started_at = time.perf_counter()
        llm_status = "exception"
        response: Dict[str, Any] = {}
        if config.TRACE_LOGS:
            logger.info(
                "[inference-trace] LLM_START case_id=%s case_idx=%s turn=%d model=%s",
                case_id,
                case_idx,
                turn_num,
                runner.display_name,
            )
        try:
            response = runner.infer(
                contents=gemini_contents,
                system_instruction=system_prompt,
                cfg=cfg,
            )
            llm_status = (
                "error_response"
                if response.get("inference_error")
                else "ok"
            )
        except Exception as exc:
            logger.error("Inference failed on turn %d: %s", turn_num, exc, exc_info=True)
            persist_failure("inference_error", str(exc), turn_num)
            raise RuntimeError(f"Inference failed on turn {turn_num}: {exc}") from exc
        finally:
            if config.TRACE_LOGS:
                logger.info(
                    "[inference-trace] LLM_END case_id=%s case_idx=%s turn=%d "
                    "model=%s status=%s elapsed_s=%.3f",
                    case_id,
                    case_idx,
                    turn_num,
                    runner.display_name,
                    llm_status,
                    time.perf_counter() - llm_started_at,
                )
            if timing is not None:
                timing.record("llm", time.perf_counter() - llm_started_at)

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

        # OpenAI-compatible runners return a synthetic ERROR response after
        # exhausting retries so that the common Gemini-shaped contract is
        # preserved.  Detect that marker before parsing the text; otherwise a
        # backend HTTP 4xx/5xx is misleadingly reported as a missing answer.
        if response.get("inference_error"):
            reason = _format_inference_error(
                response["inference_error"], turn_num
            )
            logger.error(
                "Model request failed: case_id=%s case_idx=%s turn=%d details=%s",
                case_id,
                case_idx,
                turn_num,
                reason,
            )
            persist_failure("inference_error", reason, turn_num)
            raise RuntimeError(reason)

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

        format_diagnostics: dict = {}
        tool_call_json = tools.extract_tool_call(response_text, diagnostics=format_diagnostics)
        if format_diagnostics:
            turn_record["tool_call_format"] = format_diagnostics
        if not tool_call_json:
            if "error" in format_diagnostics:
                observation_text = (
                    '<tool_response>\nError format\n</tool_response>\n'
                    'Correct tool call format:\n'
                    '<tool_call>{"name": "t2t_search", "arguments": '
                    '{"query": "your search query"}}</tool_call>\n'
                    'Use a tool name and arguments from the available tool schema. '
                    'The payload must be valid JSON: keys need double quotes, '
                    'use a colon between each key and value, and arguments must '
                    'be an object. No tool was executed. Correct your tool call '
                    'or provide a complete <answer>...</answer>.'
                )
                if turn_num >= config.MAX_TURNS - 2:
                    observation_text += "\n" + _MAX_TURNS_FINALIZATION_PROMPT
                turn_record["tool_output"] = observation_text
                turn_record["tool_output_type"] = "error_format"
                gemini_contents.append(
                    {"role": "user", "parts": [{"text": observation_text}]}
                )
                logger.warning(
                    "Tool format error: case_id=%s turn=%d details=%s",
                    case_id, turn_num, format_diagnostics["error"],
                )
                continue
            reason = f"Inference ended on turn {turn_num} without a complete <answer> response."
            if retain_format_errors:
                terminal_format_error = {
                    "kind": "invalid_response", "turn": turn_num, "reason": reason,
                }
                logger.warning("[react-format-retained] case_id=%s turn=%d: %s",
                               case_id, turn_num, reason)
                completed = True
                break
            persist_failure(
                "invalid_response",
                reason,
                turn_num,
            )
            print(response_text)
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

        # Reserve the final model turn for an answer.  If the penultimate
        # turn still asks for a tool, do not execute it: feed back a synthetic
        # tool response telling the agent to stop searching and answer from
        # the evidence already collected.  With MAX_TURNS=46 this happens on
        # the 45th (zero-based turn 44), leaving turn 45 for the final answer.
        if config.MAX_TURNS >= 2 and turn_num == config.MAX_TURNS - 2:
            skipped_tool = _tool_name(tool_call_json)
            observation_text = (
                f"<tool_response>\n{_MAX_TURNS_FINALIZATION_PROMPT}"
                "\n</tool_response>"
            )
            turn_record["tool_output"] = observation_text
            turn_record["tool_output_type"] = "max_turns_finalization"
            gemini_contents.append(
                {"role": "user", "parts": [{"text": observation_text}]}
            )
            if config.TRACE_LOGS:
                logger.info(
                    "Reached the penultimate turn for case_id=%s case_idx=%s; "
                    "skipped tool=%s and replaced its output with finalization prompt",
                    case_id,
                    case_idx,
                    skipped_tool,
                )
            continue

        os.makedirs(intermediate_dir, exist_ok=True)
        tool_module = _tool_name(tool_call_json)
        tool_started_at = time.perf_counter()
        tool_status = "exception"
        tool_message = ""
        new_images: Dict[str, Any] = {}
        if config.TRACE_LOGS:
            logger.info(
                "[inference-trace] TOOL_START case_id=%s case_idx=%s turn=%d "
                "tool_call_index=%d tool=%s",
                case_id,
                case_idx,
                turn_num,
                tool_call_count,
                tool_module,
            )
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
                disabled_tools=disabled_tools,
            )
            tool_status = (
                "error_result"
                if str(tool_message).startswith("Tool execution error:")
                else "ok"
            )
        except Exception as exc:
            persist_failure("tool_execution_error", str(exc), turn_num)
            raise
        finally:
            if config.TRACE_LOGS:
                logger.info(
                    "[inference-trace] TOOL_END case_id=%s case_idx=%s turn=%d "
                    "tool_call_index=%d tool=%s status=%s elapsed_s=%.3f "
                    "new_images=%d",
                    case_id,
                    case_idx,
                    turn_num,
                    tool_call_count,
                    tool_module,
                    tool_status,
                    time.perf_counter() - tool_started_at,
                    len(new_images),
                )
            if timing is not None:
                timing.record(tool_module, time.perf_counter() - tool_started_at)
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
        if (retain_format_errors and trajectory["turns"]
                and trajectory["turns"][-1].get("tool_output_type") == "error_format"):
            terminal_format_error = {
                "kind": "invalid_tool_call", "turn": config.MAX_TURNS - 1,
                "reason": reason,
            }
            logger.warning("[react-format-retained] case_id=%s: %s", case_id, reason)
        else:
            persist_failure("max_turns", reason, config.MAX_TURNS - 1)
            raise RuntimeError(reason)

    trajectory["status"] = "completed"
    trajectory["tool_call_count"] = tool_call_count
    trajectory["final_response_text"] = "\n\n".join(
        turn.get("response_text", "")
        for turn in trajectory["turns"]
        if turn.get("response_text")
    )
    if retain_format_errors:
        trajectory["format_error_policy"] = "retain_for_evaluation"
        trajectory["final_response_text_scope"] = "final_turn_only"
        final_text = trajectory["turns"][-1].get("response_text", "")
        if terminal_format_error:
            answer = _format_error_answer_text(final_text)
            trajectory["format_valid"] = False
            trajectory["format_error"] = terminal_format_error
            trajectory["answer_missing"] = not bool(answer)
            # Empty answer tags keep this attempted sample in evaluation/resume
            # without letting the evaluator fall back to previous search text.
            trajectory["final_response_text"] = f"<answer>{answer}</answer>"
        else:
            trajectory["final_response_text"] = final_text
    trajectory["url_resources"] = _serialise_url_resources(url_registry)

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
