"""Interfaces for model-worker backed generation.

The concrete client can call a vLLM/OpenAI-compatible HTTP endpoint, a local
server, or a mocked worker in tests. Pipeline components should depend on this
small protocol instead of a specific serving stack.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Protocol


# #### START Response 0720 ####
VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
# #### END Response 0720 ####
FIXED_TT_LOGID = "3200636808"
# A request is attempted once initially, then retried at most this many times.
LLM_RETRY_COUNT = 50
_ADAPTIVE_QPM_ENV = "SYNTHESIS_ADAPTIVE_QPM_ENABLED"
_ADAPTIVE_QPM_ERROR_WINDOW_S = 60.0
_ADAPTIVE_QPM_RECOVERY_INTERVAL_S = 60.0
_ADAPTIVE_QPM_RECOVERY_UTILIZATION = 0.8
_ADAPTIVE_QPM_COOLDOWN_S = 30.0
_RAW_INPUT_ENV = "LLM_WORKER_PRINT_RAW_INPUT"
_RAW_OUTPUT_ENV = "LLM_WORKER_PRINT_RAW_OUTPUT"


def _adaptive_qpm_enabled() -> bool:
    """Return whether process-local adaptive QPM limiting is enabled."""

    return str(os.environ.get(_ADAPTIVE_QPM_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_env_enabled(name: str) -> bool:
    """Return whether one of the opt-in, potentially verbose worker logs is enabled."""

    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _redact_base64_images_for_debug(value: Any, *, image_context: bool = False) -> Any:
    """Copy a request payload while replacing inline image bytes with a marker.

    Remote image URLs remain visible because they are useful for debugging
    multimodal routing.  Inline ``data:image/...;base64,...`` URLs and bare
    image ``data``/``base64`` fields are deliberately not emitted.
    """

    if isinstance(value, dict):
        block_type = str(value.get("type") or "").strip().lower()
        is_image_block = image_context or block_type in {"image", "image_url", "input_image"}
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if is_image_block and key_text in {"data", "base64", "b64_json"} and isinstance(item, str):
                redacted[key] = f"<omitted base64 image bytes: {len(item)} chars>"
            else:
                redacted[key] = _redact_base64_images_for_debug(item, image_context=is_image_block)
        return redacted
    if isinstance(value, list):
        return [_redact_base64_images_for_debug(item, image_context=image_context) for item in value]
    if isinstance(value, tuple):
        return [_redact_base64_images_for_debug(item, image_context=image_context) for item in value]
    if isinstance(value, str) and value.lstrip().lower().startswith("data:image/"):
        return f"<omitted inline base64 image: {len(value)} chars>"
    return value


def _print_raw_model_input(*, alias: str, request: ModelRequest | ResponsesModelRequest) -> None:
    """Emit the routed request exactly as seen by the provider client when enabled.

    This intentionally includes multimodal message blocks and tool definitions,
    but strips inline base64 image bytes.  It is therefore still opt-in: inputs
    can be large and may contain signed URLs or other task data that should not
    normally be placed in a shared log.
    """

    if not _debug_env_enabled(_RAW_INPUT_ENV):
        return
    print(
        f"[llm-raw-input] alias={alias} model={request.model or ''}\n"
        "--- begin routed request ---\n"
        f"{json.dumps(_redact_base64_images_for_debug(request.to_dict()), ensure_ascii=False, indent=2, default=str)}\n"
        "--- end routed request ---",
        file=sys.stderr,
        flush=True,
    )


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(slots=True)
class ModelMessage:
    role: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class ModelRequest:
    messages: list[ModelMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class ModelResponse:
    content: str
    raw_response: dict[str, Any] | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


# #### START Response 0720 ####
@dataclass(slots=True)
class ResponsesModelRequest:
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    max_output_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    store: bool | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class ResponsesModelResponse:
    raw_response: dict[str, Any]
    content: str = ""
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summaries: list[str] = field(default_factory=list)
    response_id: str | None = None
    status: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))
# #### END Response 0720 ####


@dataclass(slots=True)
class _AdaptiveQpmState:
    """Per-alias process-local state for adaptive QPM limiting."""

    effective_qpm: int
    rate_limit_times: deque[float] = field(default_factory=deque)
    cooldown_until: float = 0.0
    last_recovery_at: float = 0.0


class ModelWorkerClient(Protocol):
    """Minimal generation interface for task-specific planners."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Run one model generation request."""


def _trace_timing_enabled() -> bool:
    return os.environ.get("SYNTHESIS_TRACE_TIMING", "0") != "0"


def _trace_model_call(
    *,
    phase: str,
    label: str,
    model: str,
    base_url: str | None,
    elapsed_s: float | None = None,
    message_count: int | None = None,
    max_tokens: int | None = None,
) -> None:
    if not _trace_timing_enabled():
        return
    parts = [
        "[trace][llm]",
        f"phase={phase}",
        f"label={label!r}",
        f"model={model!r}",
        f"base_url={base_url!r}",
    ]
    if message_count is not None:
        parts.append(f"messages={message_count}")
    if max_tokens is not None:
        parts.append(f"max_tokens={max_tokens}")
    if elapsed_s is not None:
        parts.append(f"elapsed_s={elapsed_s:.3f}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _is_gpt_model_name(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("gpt")


def _normalize_reasoning_effort(value: Any, *, default: str = "medium") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_REASONING_EFFORTS:
        return normalized
    return default


def _usage_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# #### START Response 0720 ####
def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(value)
# #### END Response 0720 ####


def _normalize_extra_body(extra_body: Any) -> dict[str, Any] | None:
    if not isinstance(extra_body, dict):
        base: dict[str, Any] = {}
    else:
        base = dict(extra_body)
    return base


def _request_extra_headers(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    metadata = dict(metadata or {})
    headers: dict[str, str] = {}
    extra_payload: dict[str, str] = {}
    for key in ("session_id", "prompt_cache_key", "user_cache_key", "user_id"):
        value_text = str(metadata.get(key) or "").strip()
        if value_text:
            extra_payload[key] = value_text
    if extra_payload:
        headers["extra"] = json.dumps(extra_payload, ensure_ascii=False, separators=(",", ":"))
    logid = (
        metadata.get("x_tt_logid")
        or metadata.get("X-TT-LOGID")
        or metadata.get("X-TT-logid")
    )
    logid_text = str(logid or "").strip()
    if logid_text:
        headers["X-TT-LOGID"] = logid_text
    return headers or None


# #### START Response 0720 ####
def _extract_responses_content_and_function_calls(raw_response: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    function_calls: list[dict[str, Any]] = []
    for item in raw_response.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content_item in item.get("content") or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if text:
                        text_parts.append(str(text))
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            function_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )
    return "\n".join(part for part in text_parts if part).strip(), function_calls


def _extract_responses_reasoning_summaries(raw_response: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    for item in raw_response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for summary_item in item.get("summary") or []:
            if not isinstance(summary_item, dict):
                continue
            text = summary_item.get("text") or summary_item.get("summary_text")
            if text:
                summaries.append(str(text).strip())
    return [item for item in summaries if item]


def _responses_create_kwargs(request: ResponsesModelRequest, *, default_model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model or default_model,
        "input": request.input,
    }
    if request.tools:
        kwargs["tools"] = request.tools
    if request.instructions:
        kwargs["instructions"] = request.instructions
    if request.previous_response_id:
        kwargs["previous_response_id"] = request.previous_response_id
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.reasoning:
        kwargs["reasoning"] = request.reasoning
    if request.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = request.parallel_tool_calls
    if request.store is not None:
        kwargs["store"] = request.store
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    extra_body = _normalize_extra_body(request.metadata.get("extra_body"))
    if isinstance(extra_body, dict) and extra_body:
        kwargs["extra_body"] = extra_body
    extra_headers = _request_extra_headers(request.metadata)
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    return kwargs


def _responses_model_response_from_raw(
    raw_response: dict[str, Any],
    *,
    default_model: str,
    base_url: str | None,
    api_version: str | None = None,
) -> ResponsesModelResponse:
    content, function_calls = _extract_responses_content_and_function_calls(raw_response)
    usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
    metadata = {"base_url": base_url}
    if api_version:
        metadata["api_version"] = api_version
    return ResponsesModelResponse(
        raw_response=raw_response,
        content=content,
        function_calls=function_calls,
        reasoning_summaries=_extract_responses_reasoning_summaries(raw_response),
        response_id=raw_response.get("id"),
        status=raw_response.get("status"),
        model=raw_response.get("model") or default_model,
        usage=usage,
        metadata=metadata,
    )
# #### END Response 0720 ####


def _get_usage_value(payload: Any, *path: str) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


class OpenAIModelWorkerClient:
    """OpenAI-compatible model worker.

    This client works with both commercial OpenAI endpoints and vLLM servers
    exposing the OpenAI-compatible `/v1/chat/completions` API. Configure one
    instance per endpoint/base URL, and use `ModelRequest.model` to override
    the default model when needed.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIModelWorkerClient requires the `openai` package. "
                "Install it or use a different ModelWorkerClient implementation."
            ) from exc

        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout_s = timeout_s
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout_s,
            default_headers=default_headers,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        label = str(request.metadata.get("trace_label") or request.model or self.model)
        _trace_model_call(
            phase="start",
            label=label,
            model=request.model or self.model,
            base_url=self.base_url,
            message_count=len(request.messages),
            max_tokens=request.max_tokens,
        )
        started_at = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.to_dict() for message in request.messages],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            # Newer GPT deployments (including GPT-5.x behind Azure/OpenAI
            # compatible gateways) use ``max_completion_tokens``.  Some
            # gateways silently accept the legacy ``max_tokens`` field but do
            # not enforce it, which can make a short JSON task generate many
            # thousands of tokens.  Keep the legacy field for non-GPT models
            # such as vLLM/Qwen endpoints.
            token_key = "max_completion_tokens" if _is_gpt_model_name(request.model or self.model) else "max_tokens"
            kwargs[token_key] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        reasoning_effort = request.metadata.get("reasoning_effort")
        if reasoning_effort is not None and _is_gpt_model_name(request.model or self.model):
            kwargs["reasoning_effort"] = _normalize_reasoning_effort(reasoning_effort)
        stop = request.metadata.get("stop")
        if isinstance(stop, list) and stop:
            kwargs["stop"] = stop
        elif isinstance(stop, str) and stop:
            kwargs["stop"] = [stop]
        extra_body = _normalize_extra_body(request.metadata.get("extra_body"))
        if isinstance(extra_body, dict):
            kwargs["extra_body"] = extra_body
        extra_headers = _request_extra_headers(request.metadata)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        completion = self.client.chat.completions.create(**kwargs)
        elapsed_s = time.perf_counter() - started_at
        choice = completion.choices[0]
        content = choice.message.content or ""
        _trace_model_call(
            phase="done",
            label=label,
            model=getattr(completion, "model", None) or kwargs["model"],
            base_url=self.base_url,
            elapsed_s=elapsed_s,
            message_count=len(request.messages),
            max_tokens=request.max_tokens,
        )

        raw_response = completion.model_dump() if hasattr(completion, "model_dump") else None
        usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
        return ModelResponse(
            content=content,
            raw_response=raw_response,
            model=getattr(completion, "model", None) or kwargs["model"],
            usage=usage,
            metadata={
                "finish_reason": getattr(choice, "finish_reason", None),
                "base_url": self.base_url,
            },
        )

    def generate_json(self, request: ModelRequest) -> dict[str, Any]:
        """Generate and parse a JSON object response."""

        response = self.generate(request)
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response is not valid JSON: {response.content[:500]}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON response must be an object.")
        return parsed

    # #### START Response 0720 ####
    def responses_generate(self, request: ResponsesModelRequest) -> ResponsesModelResponse:
        label = str(request.metadata.get("trace_label") or request.model or self.model)
        _trace_model_call(
            phase="start",
            label=label,
            model=request.model or self.model,
            base_url=self.base_url,
            message_count=len(request.input),
            max_tokens=request.max_output_tokens,
        )
        started_at = time.perf_counter()
        kwargs = _responses_create_kwargs(request, default_model=self.model)
        response = self.client.responses.create(**kwargs)
        elapsed_s = time.perf_counter() - started_at
        raw_response = response.model_dump() if hasattr(response, "model_dump") else {"repr": repr(response)}
        _trace_model_call(
            phase="done",
            label=label,
            model=raw_response.get("model") or kwargs["model"],
            base_url=self.base_url,
            elapsed_s=elapsed_s,
            message_count=len(request.input),
            max_tokens=request.max_output_tokens,
        )
        return _responses_model_response_from_raw(raw_response, default_model=kwargs["model"], base_url=self.base_url)
    # #### END Response 0720 ####


class AzureOpenAIModelWorkerClient:
    """Azure OpenAI-compatible model worker.

    This client is used for endpoints that expect the AzureOpenAI SDK shape,
    including `azure_endpoint` and `api_version`.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        azure_endpoint: str,
        api_version: str,
        timeout_s: float | None = None,
        default_headers: dict[str, str] | None = None,
        generate_tt_logid: bool = False,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "AzureOpenAIModelWorkerClient requires the `openai` package. "
                "Install it or use a different ModelWorkerClient implementation."
            ) from exc

        self.model = model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout_s = timeout_s
        self.generate_tt_logid = generate_tt_logid
        self._azure_client_cls = AzureOpenAI
        self._default_headers = dict(default_headers or {})
        self.client = self._build_client()

    def _build_headers(self) -> dict[str, str] | None:
        headers = dict(self._default_headers)
        if self.generate_tt_logid and "X-TT-LOGID" not in headers:
            headers["X-TT-LOGID"] = FIXED_TT_LOGID
        return headers or None

    def _build_client(self) -> Any:
        return self._azure_client_cls(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            timeout=self.timeout_s,
            default_headers=self._build_headers(),
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        label = str(request.metadata.get("trace_label") or request.model or self.model)
        _trace_model_call(
            phase="start",
            label=label,
            model=request.model or self.model,
            base_url=self.azure_endpoint,
            message_count=len(request.messages),
            max_tokens=request.max_tokens,
        )
        started_at = time.perf_counter()
        # Rebuild client per request when dynamic TT logid is enabled.
        if self.generate_tt_logid:
            self.client = self._build_client()
        kwargs: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.to_dict() for message in request.messages],
            "stream": False,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            # See OpenAIModelWorkerClient.generate: GPT-5.x expects the
            # modern completion-budget field, while non-GPT compatible APIs
            # generally still expect max_tokens.
            token_key = "max_completion_tokens" if _is_gpt_model_name(request.model or self.model) else "max_tokens"
            kwargs[token_key] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        reasoning_effort = request.metadata.get("reasoning_effort")
        if reasoning_effort is not None and _is_gpt_model_name(request.model or self.model):
            kwargs["reasoning_effort"] = _normalize_reasoning_effort(reasoning_effort)
        stop = request.metadata.get("stop")
        if isinstance(stop, list) and stop:
            kwargs["stop"] = stop
        elif isinstance(stop, str) and stop:
            kwargs["stop"] = [stop]
        extra_body = _normalize_extra_body(request.metadata.get("extra_body"))
        if isinstance(extra_body, dict):
            kwargs["extra_body"] = extra_body
        extra_headers = _request_extra_headers(request.metadata)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        completion = self.client.chat.completions.create(**kwargs)
        elapsed_s = time.perf_counter() - started_at
        choice = completion.choices[0]
        content = choice.message.content or ""
        _trace_model_call(
            phase="done",
            label=label,
            model=getattr(completion, "model", None) or kwargs["model"],
            base_url=self.azure_endpoint,
            elapsed_s=elapsed_s,
            message_count=len(request.messages),
            max_tokens=request.max_tokens,
        )

        raw_response = completion.model_dump() if hasattr(completion, "model_dump") else None
        usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
        return ModelResponse(
            content=content,
            raw_response=raw_response,
            model=getattr(completion, "model", None) or kwargs["model"],
            usage=usage,
            metadata={
                "finish_reason": getattr(choice, "finish_reason", None),
                "base_url": self.azure_endpoint,
                "api_version": self.api_version,
            },
        )

    # #### START Response 0720 ####
    def responses_generate(self, request: ResponsesModelRequest) -> ResponsesModelResponse:
        label = str(request.metadata.get("trace_label") or request.model or self.model)
        _trace_model_call(
            phase="start",
            label=label,
            model=request.model or self.model,
            base_url=self.azure_endpoint,
            message_count=len(request.input),
            max_tokens=request.max_output_tokens,
        )
        started_at = time.perf_counter()
        if self.generate_tt_logid:
            self.client = self._build_client()
        kwargs = _responses_create_kwargs(request, default_model=self.model)
        response = self.client.responses.create(**kwargs)
        elapsed_s = time.perf_counter() - started_at
        raw_response = response.model_dump() if hasattr(response, "model_dump") else {"repr": repr(response)}
        _trace_model_call(
            phase="done",
            label=label,
            model=raw_response.get("model") or kwargs["model"],
            base_url=self.azure_endpoint,
            elapsed_s=elapsed_s,
            message_count=len(request.input),
            max_tokens=request.max_output_tokens,
        )
        return _responses_model_response_from_raw(
            raw_response,
            default_model=kwargs["model"],
            base_url=self.azure_endpoint,
            api_version=self.api_version,
        )
    # #### END Response 0720 ####


class ModelRouterWorkerClient:
    """Config-driven router for OpenAI-compatible and AzureOpenAI endpoints."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self._configs: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, Any] = {}
        self._qpm_lock = threading.Lock()
        self._qpm_windows: dict[str, deque[float]] = {}
        self._adaptive_qpm_states: dict[str, _AdaptiveQpmState] = {}
        self._token_totals_lock = threading.Lock()
        self._token_totals: dict[str, dict[str, int | float]] = {}
        if self.config_path is not None and self.config_path.exists():
            self.load_config(self.config_path)

    @classmethod
    def from_env(cls) -> "ModelRouterWorkerClient":
        config_path = os.environ.get("SYNTHESIS_MODEL_CONFIG")
        if config_path:
            return cls(cls._resolve_config_path(config_path))
        default_path = Path(__file__).with_name("models.json")
        return cls(default_path if default_path.exists() else None)

    @staticmethod
    def _resolve_config_path(config_path: str | Path) -> Path:
        path = Path(config_path)
        if path.is_absolute() or path.exists():
            return path

        project_relative = Path(__file__).resolve().parents[1] / path
        if project_relative.exists():
            return project_relative

        synthesis_relative = Path(__file__).resolve().parent / path
        if synthesis_relative.exists():
            return synthesis_relative

        return path

    def load_config(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self.config_path = Path(config_path)
        if self.config_path is None:
            raise ValueError("No model config path is set.")

        with self.config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        models = payload.get("models", payload)
        if not isinstance(models, dict):
            raise ValueError("Model config must contain an object at key 'models'.")

        normalized: dict[str, dict[str, Any]] = {}
        for alias, config in models.items():
            if not isinstance(config, dict):
                raise ValueError(f"Model config for {alias!r} must be an object.")
            if not config.get("served_model"):
                raise ValueError(f"Model config for {alias!r} is missing 'served_model'.")
            client_type = str(config.get("client_type") or "openai")
            if client_type not in {"openai", "azure_openai"}:
                raise ValueError(f"Unsupported client_type for {alias!r}: {client_type}")
            if client_type == "azure_openai":
                if not config.get("azure_endpoint"):
                    raise ValueError(f"Azure model config for {alias!r} is missing 'azure_endpoint'.")
                if not config.get("api_version"):
                    raise ValueError(f"Azure model config for {alias!r} is missing 'api_version'.")
            qpm = config.get("qpm")
            if qpm is not None:
                try:
                    qpm_value = int(qpm)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Model config for {alias!r} has invalid qpm: {qpm!r}") from exc
                if qpm_value <= 0:
                    raise ValueError(f"Model config for {alias!r} must set qpm > 0 when configured.")
            sampling_params = dict(config.get("sampling_params") or {})
            if "reasoning_effort" in sampling_params:
                effort = _normalize_reasoning_effort(sampling_params.get("reasoning_effort"), default="")
                if effort not in VALID_REASONING_EFFORTS:
                    raise ValueError(
                        f"Model config for {alias!r} has invalid reasoning_effort: {sampling_params.get('reasoning_effort')!r}"
                    )
            normalized_config = dict(config)
            if qpm is not None:
                normalized_config["qpm"] = qpm_value
            normalized[alias] = normalized_config

        self._configs = normalized
        self._clients.clear()
        with self._qpm_lock:
            self._qpm_windows.clear()
            self._adaptive_qpm_states.clear()

    def reload(self) -> None:
        self.load_config(self.config_path)

    def get_model(self, alias: str) -> dict[str, Any] | None:
        config = self._configs.get(alias)
        return dict(config) if config is not None else None

    def list_models(self) -> dict[str, dict[str, Any]]:
        return {alias: dict(config) for alias, config in self._configs.items()}

    def clear(self) -> None:
        self._configs.clear()
        self._clients.clear()
        with self._qpm_lock:
            self._qpm_windows.clear()
            self._adaptive_qpm_states.clear()

    def generate(self, request: ModelRequest) -> ModelResponse:
        alias = request.model
        if not alias:
            if len(self._configs) == 1:
                alias = next(iter(self._configs))
            else:
                raise ValueError("ModelRequest.model must be a registered alias.")

        config = self._configs.get(alias)
        if config is None:
            raise KeyError(f"Model alias is not registered: {alias}")

        client = self._client_for(alias, config)
        sampling_params = dict(config.get("sampling_params") or {})
        extra_body = dict(request.metadata.get("extra_body") or {})
        reasoning_effort = request.metadata.get("reasoning_effort")
        routed_temperature = sampling_params.pop("temperature", request.temperature)
        # Completion budgets are an alias-level policy.  Do not let an
        # individual caller silently override it: if the alias has no budget,
        # omit the provider parameter entirely and let that deployment decide.
        out_seq_length = sampling_params.pop("out_seq_length", None)
        max_tokens = sampling_params.pop("max_tokens", None)
        chosen_max_tokens = out_seq_length if out_seq_length is not None else max_tokens
        routed_max_tokens = int(chosen_max_tokens) if chosen_max_tokens is not None else None
        if reasoning_effort is None and _is_gpt_model_name(config.get("served_model")):
            reasoning_effort = sampling_params.pop("reasoning_effort", "medium")
        else:
            sampling_params.pop("reasoning_effort", None)
        extra_body = {**sampling_params, **extra_body} if sampling_params or extra_body else {}
        routed_metadata = dict(request.metadata or {})
        if reasoning_effort is not None and _is_gpt_model_name(config.get("served_model")):
            routed_metadata["reasoning_effort"] = _normalize_reasoning_effort(reasoning_effort)
        if extra_body:
            routed_metadata["extra_body"] = extra_body
        routed_request = ModelRequest(
            messages=request.messages,
            model=config["served_model"],
            temperature=routed_temperature,
            max_tokens=routed_max_tokens,
            response_format=request.response_format,
            metadata=routed_metadata,
        )
        _print_raw_model_input(alias=alias, request=routed_request)
        started_at = time.perf_counter()
        response = self._generate_with_retry(
            alias=alias,
            config=config,
            client=client,
            request=routed_request,
        )
        # This is end-to-end worker latency: it includes local QPM waiting and
        # retry backoff as well as the upstream model request.  That is the
        # useful number when diagnosing throughput of a post-processing run.
        response.metadata["worker_elapsed_s"] = time.perf_counter() - started_at
        response.metadata.update(
            {
                "model_alias": alias,
                "served_model": config["served_model"],
                "base_url": config.get("base_url") or config.get("azure_endpoint"),
                "sampling_params": config.get("sampling_params"),
                "qpm": config.get("qpm"),
            }
        )
        self._update_and_print_token_totals(alias=alias, response=response)
        return response

    def generate_json(self, request: ModelRequest) -> dict[str, Any]:
        response = self.generate(request)
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response is not valid JSON: {response.content[:500]}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON response must be an object.")
        return parsed

    # #### START Response 0720 ####
    def responses_generate(self, request: ResponsesModelRequest) -> ResponsesModelResponse:
        alias = request.model
        if not alias:
            if len(self._configs) == 1:
                alias = next(iter(self._configs))
            else:
                raise ValueError("ResponsesModelRequest.model must be a registered alias.")

        config = self._configs.get(alias)
        if config is None:
            raise KeyError(f"Model alias is not registered: {alias}")

        client = self._client_for(alias, config)
        sampling_params = dict(config.get("sampling_params") or {})
        extra_body = dict(request.metadata.get("extra_body") or {})

        routed_reasoning = dict(sampling_params.pop("reasoning", {}) or {})
        if request.reasoning:
            routed_reasoning.update(request.reasoning)
        reasoning_effort = sampling_params.pop("reasoning_effort", None)
        if reasoning_effort is not None and "effort" not in routed_reasoning:
            routed_reasoning["effort"] = _normalize_reasoning_effort(reasoning_effort)

        # Same alias-only completion-budget rule as chat completions above.
        # In particular, do not forward a caller-local ``max_output_tokens``.
        routed_max_output_tokens = None
        for key in ("max_output_tokens", "out_seq_length", "max_tokens"):
            value = sampling_params.pop(key, None)
            if value is not None:
                routed_max_output_tokens = int(value)
                break

        routed_temperature = sampling_params.pop("temperature", request.temperature)
        routed_parallel_tool_calls = request.parallel_tool_calls
        if routed_parallel_tool_calls is None and "parallel_tool_calls" in sampling_params:
            # #### START Response 0720 ####
            routed_parallel_tool_calls = _coerce_optional_bool(sampling_params.pop("parallel_tool_calls"))
            # #### END Response 0720 ####
        routed_store = request.store
        if routed_store is None and "store" in sampling_params:
            # #### START Response 0720 ####
            routed_store = _coerce_optional_bool(sampling_params.pop("store"))
            # #### END Response 0720 ####

        extra_body = {**sampling_params, **extra_body} if sampling_params or extra_body else {}
        routed_metadata = dict(request.metadata or {})
        if extra_body:
            routed_metadata["extra_body"] = extra_body

        routed_request = ResponsesModelRequest(
            model=config["served_model"],
            input=request.input,
            tools=request.tools,
            instructions=request.instructions,
            previous_response_id=request.previous_response_id,
            max_output_tokens=routed_max_output_tokens,
            reasoning=routed_reasoning or None,
            parallel_tool_calls=routed_parallel_tool_calls,
            store=routed_store,
            temperature=routed_temperature,
            metadata=routed_metadata,
        )
        _print_raw_model_input(alias=alias, request=routed_request)
        started_at = time.perf_counter()
        response = self._responses_generate_with_retry(
            alias=alias,
            config=config,
            client=client,
            request=routed_request,
        )
        response.metadata["worker_elapsed_s"] = time.perf_counter() - started_at
        response.metadata.update(
            {
                "model_alias": alias,
                "served_model": config["served_model"],
                "base_url": config.get("base_url") or config.get("azure_endpoint"),
                "sampling_params": config.get("sampling_params"),
                "qpm": config.get("qpm"),
            }
        )
        self._update_and_print_token_totals(alias=alias, response=response)
        return response
    # #### END Response 0720 ####

    def _client_for(self, alias: str, config: dict[str, Any]) -> Any:
        client = self._clients.get(alias)
        if client is None:
            client_type = str(config.get("client_type") or "openai")
            if client_type == "azure_openai":
                client = AzureOpenAIModelWorkerClient(
                    model=config["served_model"],
                    api_key=config.get("api_key"),
                    azure_endpoint=config["azure_endpoint"],
                    api_version=config["api_version"],
                    timeout_s=config.get("timeout_s"),
                    default_headers=config.get("default_headers"),
                    generate_tt_logid=bool(config.get("generate_tt_logid")),
                )
            else:
                client = OpenAIModelWorkerClient(
                    model=config["served_model"],
                    api_key=config.get("api_key"),
                    base_url=config.get("base_url"),
                    timeout_s=config.get("timeout_s"),
                    default_headers=config.get("default_headers"),
                )
            self._clients[alias] = client
        return client

    def _generate_with_retry(
        self,
        *,
        alias: str,
        config: dict[str, Any],
        client: Any,
        request: ModelRequest,
    ) -> ModelResponse:
        served_model = str(config.get("served_model") or request.model or alias)
        last_error: Exception | None = None
        for attempt_index in range(LLM_RETRY_COUNT + 1):
            dispatch_qpm = self._wait_for_qpm_slot(alias=alias, config=config, served_model=served_model)
            try:
                response = client.generate(request)
                response.metadata["dispatch_qpm"] = dispatch_qpm
                self._record_success_and_maybe_recover_qpm(alias=alias, config=config, served_model=served_model)
                response.metadata["adaptive_qpm_limit"] = self._effective_qpm_limit(alias=alias, config=config)
                return response
            except Exception as exc:
                last_error = exc
                self._record_capacity_rate_limit(
                    alias=alias,
                    config=config,
                    served_model=served_model,
                    error=exc,
                )
                if self._is_non_retryable_request_error(exc):
                    self._print_non_retryable_error(
                        alias=alias,
                        served_model=served_model,
                        attempt=attempt_index + 1,
                        error=exc,
                    )
                    break
                # print(request)
                if attempt_index >= LLM_RETRY_COUNT:
                    break
                print(
                    "[llm-retry]"
                    f" alias={alias}"
                    f" served_model={served_model}"
                    f" attempt={attempt_index + 1}"
                    f" error_type={exc.__class__.__name__}"
                    f" error={str(exc)!r}"
                    " sleep_seconds=30",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(30)
        if last_error is None:
            raise RuntimeError(f"Model request failed without a captured exception for alias={alias!r}")
        raise last_error

    # #### START Response 0720 ####
    def _responses_generate_with_retry(
        self,
        *,
        alias: str,
        config: dict[str, Any],
        client: Any,
        request: ResponsesModelRequest,
    ) -> ResponsesModelResponse:
        served_model = str(config.get("served_model") or request.model or alias)
        last_error: Exception | None = None
        for attempt_index in range(LLM_RETRY_COUNT + 1):
            dispatch_qpm = self._wait_for_qpm_slot(alias=alias, config=config, served_model=served_model)
            try:
                response = client.responses_generate(request)
                response.metadata["dispatch_qpm"] = dispatch_qpm
                self._record_success_and_maybe_recover_qpm(alias=alias, config=config, served_model=served_model)
                response.metadata["adaptive_qpm_limit"] = self._effective_qpm_limit(alias=alias, config=config)
                return response
            except Exception as exc:
                last_error = exc
                self._record_capacity_rate_limit(
                    alias=alias,
                    config=config,
                    served_model=served_model,
                    error=exc,
                )
                if self._is_non_retryable_request_error(exc):
                    self._print_non_retryable_error(
                        alias=alias,
                        served_model=served_model,
                        attempt=attempt_index + 1,
                        error=exc,
                        api_mode="responses",
                    )
                    break
                if attempt_index >= LLM_RETRY_COUNT:
                    break
                print(
                    "[llm-retry]"
                    f" alias={alias}"
                    f" served_model={served_model}"
                    f" api_mode=responses"
                    f" attempt={attempt_index + 1}"
                    f" error_type={exc.__class__.__name__}"
                    f" error={str(exc)!r}"
                    " sleep_seconds=30",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(30)
        if last_error is None:
            raise RuntimeError(f"Responses request failed without a captured exception for alias={alias!r}")
        raise last_error
    # #### END Response 0720 ####

    def _wait_for_qpm_slot(self, *, alias: str, config: dict[str, Any], served_model: str) -> int:
        qpm = config.get("qpm")
        if qpm is None:
            return self._record_qpm_dispatch(alias=alias)
        while True:
            now = time.monotonic()
            with self._qpm_lock:
                qpm_limit = self._effective_qpm_limit_locked(alias=alias, config=config)
                window = self._qpm_windows.setdefault(alias, deque())
                cutoff = now - 60.0
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) < qpm_limit:
                    window.append(now)
                    return len(window)
                current_qpm = len(window)
            print(
                "[llm-qpm]"
                f" alias={alias}"
                f" served_model={served_model}"
                f" qpm_limit={qpm_limit}"
                f" current_window_calls={current_qpm}"
                " sleep_seconds=30",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(30)

    def _effective_qpm_limit(self, *, alias: str, config: dict[str, Any]) -> int | None:
        """Return the currently active QPM cap, never exceeding configured qpm."""

        if config.get("qpm") is None:
            return None
        with self._qpm_lock:
            return self._effective_qpm_limit_locked(alias=alias, config=config)

    def _effective_qpm_limit_locked(self, *, alias: str, config: dict[str, Any]) -> int:
        configured_qpm = int(config["qpm"])
        if not _adaptive_qpm_enabled():
            return configured_qpm
        state = self._adaptive_qpm_states.get(alias)
        if state is None:
            state = _AdaptiveQpmState(effective_qpm=configured_qpm)
            self._adaptive_qpm_states[alias] = state
        state.effective_qpm = min(configured_qpm, max(1, state.effective_qpm))
        return state.effective_qpm

    @staticmethod
    def _is_non_retryable_request_error(error: Exception) -> bool:
        """Return true for deterministic client-side/request-validation failures."""

        message = str(error).lower()
        return (
            error.__class__.__name__ == "BadRequestError"
            or "error code: 400" in message
            or "invalid_request_error" in message
            or "invalid_value" in message
        )

    @staticmethod
    def _print_non_retryable_error(
        *,
        alias: str,
        served_model: str,
        attempt: int,
        error: Exception,
        api_mode: str | None = None,
    ) -> None:
        """Log a rejected request once instead of sleeping through retry budget."""

        mode = f" api_mode={api_mode}" if api_mode else ""
        print(
            "[llm-non-retryable]"
            f" alias={alias}"
            f" served_model={served_model}"
            f"{mode}"
            f" attempt={attempt}"
            f" error_type={error.__class__.__name__}"
            f" error={str(error)!r}",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _is_capacity_rate_limit(error: Exception) -> bool:
        """Recognize rate-limit errors that indicate dynamically reduced capacity."""

        message = str(error).lower()
        return (
            error.__class__.__name__ == "RateLimitError"
            or "error code: 429" in message
            or "http 429" in message
            or "too many requests" in message
            or "'code': '-2004'" in message
            or '"code": "-2004"' in message
            or "资源不足" in str(error)
        )

    def _record_capacity_rate_limit(
        self,
        *,
        alias: str,
        config: dict[str, Any],
        served_model: str,
        error: Exception,
    ) -> None:
        """Decrease the local QPM cap quickly when the upstream reports capacity pressure."""

        if not _adaptive_qpm_enabled() or config.get("qpm") is None or not self._is_capacity_rate_limit(error):
            return
        now = time.monotonic()
        configured_qpm = int(config["qpm"])
        with self._qpm_lock:
            state = self._adaptive_qpm_states.setdefault(
                alias,
                _AdaptiveQpmState(effective_qpm=configured_qpm),
            )
            cutoff = now - _ADAPTIVE_QPM_ERROR_WINDOW_S
            while state.rate_limit_times and state.rate_limit_times[0] <= cutoff:
                state.rate_limit_times.popleft()
            state.rate_limit_times.append(now)
            error_count = len(state.rate_limit_times)
            before = state.effective_qpm
            if error_count == 1:
                # A lone 429 may be transient. Reduce only one slot instead of
                # collapsing the cap to the current instantaneous traffic rate.
                after = before - 1
            elif error_count == 2:
                after = int(before * 0.8)
            else:
                after = int(before * 0.65)
                state.cooldown_until = max(state.cooldown_until, now + _ADAPTIVE_QPM_COOLDOWN_S)
            state.effective_qpm = min(configured_qpm, max(1, after))
            state.last_recovery_at = now
            cooldown_s = max(0.0, state.cooldown_until - now)
        print(
            "[llm-adaptive-qpm]"
            f" alias={alias}"
            f" served_model={served_model}"
            " event=capacity_rate_limit"
            f" effective_qpm_before={before}"
            f" effective_qpm_after={state.effective_qpm}"
            f" configured_qpm={configured_qpm}"
            f" rate_limit_errors_60s={error_count}"
            f" cooldown_seconds={cooldown_s:.0f}",
            file=sys.stderr,
            flush=True,
        )

    def _record_success_and_maybe_recover_qpm(self, *, alias: str, config: dict[str, Any], served_model: str) -> None:
        """Slowly probe upward after a saturated, error-free recovery interval."""

        if not _adaptive_qpm_enabled() or config.get("qpm") is None:
            return
        now = time.monotonic()
        configured_qpm = int(config["qpm"])
        with self._qpm_lock:
            state = self._adaptive_qpm_states.setdefault(
                alias,
                _AdaptiveQpmState(effective_qpm=configured_qpm),
            )
            cutoff = now - _ADAPTIVE_QPM_ERROR_WINDOW_S
            while state.rate_limit_times and state.rate_limit_times[0] <= cutoff:
                state.rate_limit_times.popleft()
            if (
                state.effective_qpm >= configured_qpm
                or state.rate_limit_times
                or now < state.cooldown_until
                or now < state.last_recovery_at + _ADAPTIVE_QPM_RECOVERY_INTERVAL_S
            ):
                return
            current_qpm = self._current_qpm_locked(alias=alias, now=now)
            required_qpm = int(state.effective_qpm * _ADAPTIVE_QPM_RECOVERY_UTILIZATION + 0.999)
            if current_qpm < required_qpm:
                return
            before = state.effective_qpm
            state.effective_qpm = min(configured_qpm, before + 1)
            state.last_recovery_at = now
            after = state.effective_qpm
        print(
            "[llm-adaptive-qpm]"
            f" alias={alias}"
            f" served_model={served_model}"
            " event=healthy_recovery"
            f" effective_qpm_before={before}"
            f" effective_qpm_after={after}"
            f" configured_qpm={configured_qpm}"
            f" current_window_calls={current_qpm}",
            file=sys.stderr,
            flush=True,
        )

    def _current_qpm_locked(self, *, alias: str, now: float) -> int:
        window = self._qpm_windows.get(alias)
        if window is None:
            return 0
        cutoff = now - 60.0
        while window and window[0] <= cutoff:
            window.popleft()
        return len(window)

    def _record_qpm_dispatch(self, *, alias: str) -> int:
        """Record every outbound request, including aliases without a QPM limit."""
        now = time.monotonic()
        with self._qpm_lock:
            window = self._qpm_windows.setdefault(alias, deque())
            cutoff = now - 60.0
            while window and window[0] <= cutoff:
                window.popleft()
            window.append(now)
            return len(window)

    def _current_qpm(self, *, alias: str) -> int:
        """Return requests dispatched for one alias in the rolling 60-second window."""
        now = time.monotonic()
        with self._qpm_lock:
            window = self._qpm_windows.get(alias)
            if window is None:
                return 0
            cutoff = now - 60.0
            while window and window[0] <= cutoff:
                window.popleft()
            return len(window)

    def _update_and_print_token_totals(self, *, alias: str, response: ModelResponse) -> None:
        usage = dict(response.usage or {})
        # #### START Response 0720 ####
        prompt_tokens = _usage_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion_tokens = _usage_int(usage.get("completion_tokens") or usage.get("output_tokens"))
        total_tokens = _usage_int(usage.get("total_tokens"))
        reasoning_tokens = _usage_int(
            _get_usage_value({"usage": usage}, "usage", "reasoning_tokens")
            or _get_usage_value({"usage": usage}, "usage", "output_tokens_details", "reasoning_tokens")
        )
        cached_tokens = _usage_int(
            _get_usage_value({"usage": usage}, "usage", "prompt_tokens_details", "cached_tokens")
            or _get_usage_value({"usage": usage}, "usage", "input_tokens_details", "cached_tokens")
        )
        worker_elapsed_s = float(response.metadata.get("worker_elapsed_s") or 0.0)
        # #### END Response 0720 ####
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        with self._token_totals_lock:
            totals = self._token_totals.setdefault(
                alias,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "elapsed_s": 0.0,
                },
            )
            totals["calls"] += 1
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens
            totals["total_tokens"] += total_tokens
            totals["reasoning_tokens"] += reasoning_tokens
            totals["cached_tokens"] += cached_tokens
            totals["elapsed_s"] += worker_elapsed_s
            snapshot = dict(totals)
        print(
            "[llm-usage]"
            f" alias={alias}"
            f" served_model={response.metadata.get('served_model') or response.model}"
            f" current_qpm={response.metadata.get('dispatch_qpm', self._current_qpm(alias=alias))}"
            f" qpm_limit={response.metadata.get('qpm')}"
            f" adaptive_qpm_limit={response.metadata.get('adaptive_qpm_limit', self._effective_qpm_limit(alias=alias, config=self._configs.get(alias) or {}))}"
            f" call_prompt_tokens={prompt_tokens}"
            f" call_completion_tokens={completion_tokens}"
            f" call_total_tokens={total_tokens}"
            f" call_reasoning_tokens={reasoning_tokens}"
            f" call_cached_tokens={cached_tokens}"
            f" cumulative_calls={snapshot['calls']}"
            f" cumulative_prompt_tokens={snapshot['prompt_tokens']}"
            f" cumulative_completion_tokens={snapshot['completion_tokens']}"
            f" cumulative_total_tokens={snapshot['total_tokens']}"
            f" cumulative_reasoning_tokens={snapshot['reasoning_tokens']}"
            f" cumulative_cached_tokens={snapshot['cached_tokens']}"
            f" call_elapsed_s={worker_elapsed_s:.3f}"
            f" cumulative_elapsed_s={float(snapshot.get('elapsed_s', 0.0)):.3f}",
            file=sys.stderr,
            flush=True,
        )
        if _debug_env_enabled(_RAW_OUTPUT_ENV):
            print(
                f"[llm-raw-output] alias={alias} model={response.model or ''}\n"
                "--- begin response.content ---\n"
                f"{response.content}\n"
                "--- end response.content ---",
                file=sys.stderr,
                flush=True,
            )
        # Empty content is never a usable writer response.  Unlike normal raw
        # output logging, always surface the provider payload for this rare
        # failure: it distinguishes a genuinely empty completion from a parser
        # incompatibility or a provider-side reasoning/refusal response.
        if not str(response.content or "").strip():
            print(
                f"[llm-empty-output] alias={alias} model={response.model or ''}\n"
                "--- begin provider raw_response ---\n"
                f"{json.dumps(response.raw_response, ensure_ascii=False, indent=2, default=str)}\n"
                "--- end provider raw_response ---",
                file=sys.stderr,
                flush=True,
            )


LLM_WORKER = ModelRouterWorkerClient.from_env()


def _smoke_test() -> None:
    request = ModelRequest(
        model="text_process",
        messages=[ModelMessage(role="user", content="hello")],
    )
    assert request.to_dict()["messages"][0]["content"] == "hello"

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "models.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": {
                        "text_process": {
                            "served_model": "dummy-model",
                            "base_url": "http://127.0.0.1:8000/v1",
                            "api_key": "EMPTY",
                            "qpm": 12,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        router = ModelRouterWorkerClient(config_path)
        assert router.get_model("text_process")["served_model"] == "dummy-model"
        assert router.get_model("text_process")["qpm"] == 12
        assert "text_process" in router.list_models()
    print("model_worker smoke test passed")


if __name__ == "__main__":
    _smoke_test()
