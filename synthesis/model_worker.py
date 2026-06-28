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


VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}


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
        # print(kwargs['messages'])
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        reasoning_effort = request.metadata.get("reasoning_effort")
        if reasoning_effort is not None and _is_gpt_model_name(request.model or self.model):
            kwargs["reasoning_effort"] = _normalize_reasoning_effort(reasoning_effort)

        extra_body = request.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            kwargs["extra_body"] = extra_body

        completion = self.client.chat.completions.create(**kwargs)
        elapsed_s = time.perf_counter() - started_at
        # print(completion)
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
            try:
                import logid as tt_logid

                headers["X-TT-LOGID"] = tt_logid.generate_v2()
            except Exception:
                pass
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

        extra_body = request.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            kwargs["extra_body"] = extra_body
        # Rebuild client per request when dynamic TT logid is enabled.
        if self.generate_tt_logid:
            self.client = self._build_client()
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
        prompt_tokens = _usage_int(usage.get("prompt_tokens"))
        completion_tokens = _usage_int(usage.get("completion_tokens"))
        total_tokens = _usage_int(usage.get("total_tokens"))
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
                },
            )
            totals["calls"] += 1
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens
            totals["total_tokens"] += total_tokens
            snapshot = dict(totals)
        ##### DEBUG #####
        print(
            "[llm-usage]"
            f" alias={alias}"
            f" served_model={response.metadata.get('served_model') or response.model}"
            f" call_prompt_tokens={prompt_tokens}"
            f" call_completion_tokens={completion_tokens}"
            f" call_total_tokens={total_tokens}"
            f" cumulative_calls={snapshot['calls']}"
            f" cumulative_prompt_tokens={snapshot['prompt_tokens']}"
            f" cumulative_completion_tokens={snapshot['completion_tokens']}"
            f" cumulative_total_tokens={snapshot['total_tokens']}",
            file=sys.stderr,
            flush=True,
        )
        ##### END #####


LLM_WORKER = ModelRouterWorkerClient.from_env()


def _smoke_test() -> None:
    request = ModelRequest(
        model="text_process",
        messages=[ModelMessage(role="user", content="hello")],
        temperature=0.0,
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
