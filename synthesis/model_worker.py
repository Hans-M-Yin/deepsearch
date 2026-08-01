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
            kwargs["max_tokens"] = request.max_tokens
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
            kwargs["max_tokens"] = request.max_tokens
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
        self._token_totals_lock = threading.Lock()
        self._token_totals: dict[str, dict[str, int]] = {}
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
        routed_max_tokens = request.max_tokens
        if routed_max_tokens is None:
            out_seq_length = sampling_params.pop("out_seq_length", None)
            max_tokens = sampling_params.pop("max_tokens", None)
            chosen_max_tokens = out_seq_length if out_seq_length is not None else max_tokens
            if chosen_max_tokens is not None:
                routed_max_tokens = int(chosen_max_tokens)
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
        response = self._generate_with_retry(
            alias=alias,
            config=config,
            client=client,
            request=routed_request,
        )
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

        routed_max_output_tokens = request.max_output_tokens
        if routed_max_output_tokens is None:
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
        response = self._responses_generate_with_retry(
            alias=alias,
            config=config,
            client=client,
            request=routed_request,
        )
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
        for attempt_index in range(2):
            self._wait_for_qpm_slot(alias=alias, config=config, served_model=served_model)
            try:
                return client.generate(request)
            except Exception as exc:
                last_error = exc
                # print(request)
                if attempt_index >= 1:
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
        for attempt_index in range(2):
            self._wait_for_qpm_slot(alias=alias, config=config, served_model=served_model)
            try:
                return client.responses_generate(request)
            except Exception as exc:
                last_error = exc
                if attempt_index >= 1:
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

    def _wait_for_qpm_slot(self, *, alias: str, config: dict[str, Any], served_model: str) -> None:
        qpm = config.get("qpm")
        if qpm is None:
            return
        qpm_limit = int(qpm)
        while True:
            now = time.monotonic()
            with self._qpm_lock:
                window = self._qpm_windows.setdefault(alias, deque())
                cutoff = now - 60.0
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) < qpm_limit:
                    window.append(now)
                    return
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
                },
            )
            totals["calls"] += 1
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens
            totals["total_tokens"] += total_tokens
            totals["reasoning_tokens"] += reasoning_tokens
            totals["cached_tokens"] += cached_tokens
            snapshot = dict(totals)
        print(
            "[llm-usage]"
            f" alias={alias}"
            f" served_model={response.metadata.get('served_model') or response.model}"
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
            f" cumulative_cached_tokens={snapshot['cached_tokens']}",
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
