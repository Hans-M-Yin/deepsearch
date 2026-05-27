"""Interfaces for model-worker backed generation.

The concrete client can call a vLLM/OpenAI-compatible HTTP endpoint, a local
server, or a mocked worker in tests. Pipeline components should depend on this
small protocol instead of a specific serving stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol


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
    temperature: float = 0.0
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
        kwargs: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.to_dict() for message in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        extra_body = request.metadata.get("extra_body")
        if isinstance(extra_body, dict):
            kwargs["extra_body"] = extra_body

        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        content = choice.message.content or ""

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


class ModelRouterWorkerClient:
    """Config-driven router for OpenAI-compatible model endpoints."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self._configs: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, OpenAIModelWorkerClient] = {}
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
            normalized[alias] = dict(config)

        self._configs = normalized
        self._clients.clear()

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
        routed_request = ModelRequest(
            messages=request.messages,
            model=config["served_model"],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format=request.response_format,
            metadata=request.metadata,
        )
        response = client.generate(routed_request)
        response.metadata.update(
            {
                "model_alias": alias,
                "served_model": config["served_model"],
                "base_url": config.get("base_url"),
            }
        )
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

    def _client_for(self, alias: str, config: dict[str, Any]) -> OpenAIModelWorkerClient:
        client = self._clients.get(alias)
        if client is None:
            client = OpenAIModelWorkerClient(
                model=config["served_model"],
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                timeout_s=config.get("timeout_s"),
                default_headers=config.get("default_headers"),
            )
            self._clients[alias] = client
        return client


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
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        router = ModelRouterWorkerClient(config_path)
        assert router.get_model("text_process")["served_model"] == "dummy-model"
        assert "text_process" in router.list_models()
    print("model_worker smoke test passed")


if __name__ == "__main__":
    _smoke_test()
