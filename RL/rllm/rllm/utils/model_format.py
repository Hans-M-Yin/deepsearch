"""Shared model protocol validation and failure classification (CPU only)."""

import json
import re
from dataclasses import dataclass
from typing import Any


MODEL_ERROR_TYPES = frozenset({"model_format_error", "tool_argument_error"})
MODEL_FAILURE_REWARD = -0.1
_TAG = re.compile(r"</?(?:thinking|tool_call|answer|tool_response)\b[^>]*>")
_ENVELOPE = re.compile(
    r"\A<thinking>(.*?)</thinking>\s*<(tool_call|answer)>(.*?)</\2>\s*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class FormatCheck:
    action: str = ""
    payload: Any = None
    error_type: str = ""
    detail: str = ""

    @property
    def valid(self) -> bool:
        return not self.error_type


def _schema_error(value: Any, schema: dict, path: str = "arguments") -> str:
    """Validate the types/required fields/array bounds used by the tool schemas."""
    kind = schema.get("type")
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if kind in matches and not matches[kind]:
        return f"{path} must be {kind}."
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}."
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key} is required."
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                error = _schema_error(item, properties[key], f"{path}.{key}")
                if error:
                    return error
            elif schema.get("additionalProperties") is False:
                return f"{path}.{key} is not allowed."
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            return f"{path} has an invalid number of items."
        for index, item in enumerate(value):
            error = _schema_error(item, schema.get("items", {}), f"{path}[{index}]")
            if error:
                return error
    return ""


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def validate_response(content: str, tool_schemas: dict | None = None) -> FormatCheck:
    """Require one thinking block and exactly one action, with strict JSON.

    Protocol tags are reserved, including inside reasoning and JSON strings.
    Other markup, such as HTML in evidence or mathematical comparisons, is
    allowed. Tool registry/schema checks are optional for reward-only callers.
    """
    if not isinstance(content, str):
        return FormatCheck(error_type="model_format_error", detail="Response must be text.")
    content = content.strip()
    match = _ENVELOPE.fullmatch(content)
    tags = _TAG.findall(content)
    if match is None or tags != ["<thinking>", "</thinking>", f"<{match[2]}>", f"</{match[2]}>"]:
        counts = "; ".join(
            f"{tag}: opening={tags.count(f'<{tag}>')}, closing={tags.count(f'</{tag}>')}"
            for tag in ("thinking", "tool_call", "answer")
        )
        return FormatCheck(
            error_type="model_format_error",
            detail=f"Tag counts ({counts}). Use exactly one complete <thinking> block followed by exactly one complete <tool_call> or <answer> block; no extra protocol tags or text outside these blocks.",
        )
    action, body = match[2], match[3]
    if action == "answer":
        if not body.strip():
            return FormatCheck(error_type="model_format_error", detail="The answer block must not be empty.")
        return FormatCheck(action=action, payload=body.strip())
    try:
        payload = json.loads(body, object_pairs_hook=_json_object, parse_constant=_invalid_json_constant)
    except (ValueError, TypeError) as exc:
        detail = f"Invalid JSON: {exc}."
        if re.search(r'"name"\s*,', body):
            detail += ' Use a colon after "name": "name":"t2t_search".'
        return FormatCheck(error_type="model_format_error", detail=detail)
    if not isinstance(payload, dict):
        return FormatCheck(error_type="model_format_error", detail="Tool call must be a JSON object.")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return FormatCheck(error_type="model_format_error", detail='Tool call requires a nonempty "name" string; do not omit the tool name.')
    if "arguments" not in payload or not isinstance(payload["arguments"], dict):
        return FormatCheck(error_type="model_format_error", detail='Tool call requires an "arguments" JSON object.')
    if tool_schemas is not None:
        if name not in tool_schemas:
            return FormatCheck(error_type="tool_argument_error", detail=f"Unknown tool {name!r}. Available tools: {', '.join(tool_schemas)}. Use <answer> for the final answer.")
        error = _schema_error(payload["arguments"], tool_schemas[name])
        if error:
            return FormatCheck(error_type="tool_argument_error", detail=error)
    return FormatCheck(action=action, payload=payload)


def format_feedback(detail: str) -> str:
    return (
        f"Error format: {detail}\n"
        "Correct tool-call format:\n"
        '<thinking>I will search for evidence.</thinking><tool_call>{"name":"t2t_search","arguments":{"query":"your query"}}</tool_call>\n'
        "Correct final-answer format:\n"
        "<thinking>I have enough evidence.</thinking><answer>your answer</answer>\n"
        "Fix the call structure or the indicated arguments; changing only the query will not fix a structural error."
    )


def classify_tool_failure(detail: str) -> str:
    """Only explicit input/reference failures are attributed to model arguments."""
    lower = detail.lower()
    argument_markers = (
        "query is required", "url is required", "resource_id not found",
        "unknown tool:", "error: image reference", "python code is required",
        "invalid region", "invalid bounding box",
    )
    return "tool_argument_error" if any(marker in lower for marker in argument_markers) else "external_tool_error"


def step_error_type(step) -> str:
    """Prefer structured metadata; retain conservative support for old episodes."""
    info = step.info or {}
    if "error_type" in info:
        return info["error_type"] or ""
    # Old error-only records without a generated response do not provide
    # enough evidence to attribute the failure to the model.
    if not step.model_response and info.get("step_error"):
        observation = step.observation if isinstance(step.observation, str) else ""
        return classify_tool_failure(observation)
    if not validate_response(step.model_response).valid:
        return "model_format_error"
    observation = step.observation if isinstance(step.observation, str) else ""
    if "[Json Parse Error]" in observation or "Error: Invalid content format" in observation:
        return "model_format_error"
    if info.get("step_error"):
        return classify_tool_failure(observation)
    return ""


def fatal_requires_protection(is_fatal: bool, is_model_failure: bool) -> bool:
    return bool(is_fatal and not is_model_failure)


def clamp_external_fatal_advantages(batch) -> None:
    """Protect infrastructure failures without removing model-failure gradients."""
    fatal = batch.non_tensor_batch.get("is_fatal", ())
    model_failure = batch.non_tensor_batch.get("is_model_failure")
    for index, is_fatal in enumerate(fatal):
        if fatal_requires_protection(is_fatal, model_failure[index] if model_failure is not None else False):
            batch.batch["advantages"][index].clamp_(min=0.0)
            batch.batch["returns"][index].clamp_(min=0.0)


def model_failure_advantage_metrics(batch) -> dict[str, float]:
    """Report the actual signed advantages on learnable model-failure tokens."""
    flags = batch.non_tensor_batch.get("is_model_failure", ())
    valid = batch.non_tensor_batch.get("is_valid")
    padded = batch.non_tensor_batch.get("is_pad_step")
    rows = tokens = negative = 0
    advantage_sum = 0.0
    for index, flag in enumerate(flags):
        if not flag or (valid is not None and not valid[index]) or (padded is not None and padded[index]):
            continue
        values = batch.batch["advantages"][index][batch.batch["response_mask"][index].bool()]
        rows += 1
        tokens += values.numel()
        negative += int((values < 0).sum().item())
        advantage_sum += float(values.sum().item())
    return {
        "format_learning/model_failure_rows": float(rows),
        "format_learning/model_failure_tokens": float(tokens),
        "format_learning/model_failure_negative_token_fraction": negative / tokens if tokens else 0.0,
        "format_learning/model_failure_advantage_mean": advantage_sum / tokens if tokens else 0.0,
    }
