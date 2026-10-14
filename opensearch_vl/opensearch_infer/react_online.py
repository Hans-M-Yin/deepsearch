"""Online-only text ReAct transport, isolated from the original runners."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, Iterable, Optional

import requests

from synthesis.retry_monitor import retry_reason_from_exception, tracked_sleep
from . import config, messages
from .runners import (
    InferenceConfig, OpenAICompatibleRunner, _empty_response,
    _openai_request_summary, _truncate_for_log,
)

logger = logging.getLogger(__name__)
SAMPLING_PARAMETERS = (
    "temperature", "max_tokens", "top_p", "top_k",
    "repetition_penalty", "presence_penalty",
)


@dataclass
class OnlineInferenceConfig(InferenceConfig):
    """None means use the server default; do not send the field."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class OnlineReActRunner(OpenAICompatibleRunner):
    """Same response/error contract as the legacy HTTP runner, without Qwen defaults."""

    def __init__(self, *args, model_alias=None, client_type="openai", api_version=None,
                 default_headers=None, extra_body=None, use_sdk=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_alias = model_alias
        self.client_type = client_type
        self.api_version = api_version
        self.default_headers = deepcopy(default_headers or {})
        self.extra_body = deepcopy(extra_body or {})
        self.use_sdk = use_sdk
        self.client = None

    def load(self) -> None:
        if not self.use_sdk or self.client is not None:
            return
        from openai import AzureOpenAI, OpenAI
        common = {"api_key": self.api_key, "timeout": self.timeout, "max_retries": 0,
                  "default_headers": self.default_headers or None}
        if self.client_type == "azure_openai":
            self.client = AzureOpenAI(azure_endpoint=self.base_url, api_version=self.api_version, **common)
        else:
            self.client = OpenAI(base_url=self.base_url, **common)

    def configuration(self) -> dict[str, Any]:
        return {"model_alias": self.model_alias, "served_model": self.model,
                "client_type": self.client_type, "base_url": self.base_url,
                "api_version": self.api_version, "extra_body": deepcopy(self.extra_body),
                "transport": "openai_sdk" if self.use_sdk else "requests",
                "sampling_policy": "cli_only"}

    def _sdk_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.load()
        standard = {"model", "messages", "temperature", "max_tokens", "top_p", "presence_penalty"}
        kwargs = {key: value for key, value in payload.items() if key in standard}
        # top_k/repetition_penalty are not SDK keyword arguments, but a gateway
        # may accept them in the JSON body when explicitly requested by the user.
        extra = {**self.extra_body, **{key: value for key, value in payload.items() if key not in standard}}
        if extra:
            kwargs["extra_body"] = extra
        response = self.client.chat.completions.create(**kwargs)
        return response if isinstance(response, dict) else response.model_dump(exclude_none=True)

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        cfg = cfg or OnlineInferenceConfig()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages.to_openai_messages(contents, system_instruction),
        }
        for name in SAMPLING_PARAMETERS:
            value = getattr(cfg, name, None)
            if value is not None:  # Preserve explicit zero/1.0; never test truthiness.
                payload[name] = value

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None
        last_error_details: Optional[Dict[str, Any]] = None
        request_summary = _openai_request_summary(payload)
        for attempt in range(1, self.max_retries + 1):
            response = None
            try:
                if self.use_sdk:
                    response_json = self._sdk_chat(payload)
                else:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )
                    logger.debug(
                        "OpenAI-compatible raw response body (status_code=%s): %s",
                        response.status_code, response.text,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                choice = (response_json.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                logger.debug(
                    "OpenAI-compatible model raw output "
                    "(finish_reason=%s, chars=%d):\n"
                    "----- BEGIN MODEL OUTPUT -----\n%s\n"
                    "----- END MODEL OUTPUT -----",
                    choice.get("finish_reason"),
                    len(text),
                    text,
                )
                usage = response_json.get("usage") or {}
                return {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": text}],
                            },
                            "finishReason": choice.get("finish_reason", "STOP"),
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": usage.get("prompt_tokens", 0),
                        "candidatesTokenCount": usage.get("completion_tokens", 0),
                        "totalTokenCount": usage.get("total_tokens", 0),
                    },
                    "modelVersion": self.display_name,
                    "raw_response": response_json,
                }
            except Exception as exc:
                last_error = exc
                error_response = getattr(exc, "response", None) or response
                response_body = None
                response_headers: Dict[str, str] = {}
                status_code = None
                if error_response is not None:
                    status_code = getattr(error_response, "status_code", None)
                    response_body = _truncate_for_log(
                        getattr(error_response, "text", "") or ""
                    )
                    headers = getattr(error_response, "headers", {}) or {}
                    for header_name in (
                        "x-request-id",
                        "request-id",
                        "trace-id",
                        "retry-after",
                        "content-type",
                    ):
                        header_value = headers.get(header_name)
                        if header_value:
                            response_headers[header_name] = str(header_value)
                last_error_details = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "url": url,
                    "attempt": attempt,
                    "max_attempts": self.max_retries,
                    "status_code": status_code,
                    "response_body": response_body,
                    "response_headers": response_headers,
                    "request_summary": request_summary,
                }
                if attempt < self.max_retries:
                    tracked_sleep(
                        min(2 ** (attempt - 1), 8),
                        reason=retry_reason_from_exception(exc, default="llm_error"),
                        tool="llm",
                        error=exc,
                        url=url,
                        error_type=type(exc).__name__,
                        fallback_sleep=time.sleep,
                    )

        assert last_error is not None
        details = last_error_details or {
            "error_type": type(last_error).__name__,
            "message": str(last_error),
            "url": url,
            "attempt": self.max_retries,
            "max_attempts": self.max_retries,
            "status_code": None,
            "response_body": None,
            "response_headers": {},
            "request_summary": request_summary,
        }
        logger.error(
            "OpenAI-compatible inference failed after %d attempt(s): "
            "url=%s model=%s status_code=%s error_type=%s "
            "response_body=%s response_headers=%s request_summary=%s error=%s",
            self.max_retries,
            url,
            self.model,
            details.get("status_code"),
            details.get("error_type"),
            details.get("response_body") or "<empty>",
            details.get("response_headers") or "{}",
            details.get("request_summary") or "{}",
            details.get("message") or str(last_error),
        )
        return _empty_response(
            self.display_name,
            str(last_error),
            error_details=details,
        )



def build_runner(
    model_name: Optional[str], checkpoint=None, gpus="0", dtype="bfloat16",
    backend="api", base_url=None, api_key=None, served_model_name=None,
    timeout=600, max_retries=3, model_alias=None, client_type=None,
    azure_endpoint=None, api_version=None,
) -> OnlineReActRunner:
    if backend != "api":
        raise ValueError("Online ReAct supports --backend api only.")
    if checkpoint:
        raise ValueError("Online ReAct does not load checkpoints.")
    alias_config = {}
    if model_alias:
        if model_name or served_model_name:
            raise ValueError("Use --model-alias or a direct model name, not both.")
        from synthesis.model_worker import LLM_WORKER
        alias_config = LLM_WORKER.get_model(model_alias) or {}
        if not alias_config:
            raise ValueError(f"Unknown synthesis model alias: {model_alias}")
    model = alias_config.get("served_model") or served_model_name or model_name
    if not model:
        raise ValueError("Specify --model-alias, --model or --served-model-name.")
    if not model_alias and not served_model_name and model in config.MODEL_REGISTRY:
        model = config.MODEL_REGISTRY[model].display_name
    client_type = client_type or alias_config.get("client_type") or "openai"
    if client_type not in {"openai", "azure_openai"}:
        raise ValueError(f"Unsupported online client_type: {client_type}")
    api_version = api_version or alias_config.get("api_version") or os.environ.get("OPENAI_API_VERSION")
    if client_type == "azure_openai":
        endpoint = azure_endpoint or base_url or alias_config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint or not api_version:
            raise ValueError("Azure requires an endpoint and API version (CLI, alias or environment).")
    else:
        endpoint = base_url or alias_config.get("base_url") or os.environ.get("AGENT_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not endpoint:
        raise ValueError("--base-url or AGENT_BASE_URL / OPENAI_BASE_URL is required.")
    api_key = (api_key or alias_config.get("api_key")
               or (os.environ.get("AZURE_OPENAI_API_KEY") if client_type == "azure_openai" else None)
               or os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    # Standard sampling stays CLI-only. Preserve configured vendor controls
    # (e.g. Gemini thinking), not alias defaults for temperature/token budgets.
    excluded = set(SAMPLING_PARAMETERS) | {"out_seq_length", "max_output_tokens", "max_completion_tokens", "chat_template_kwargs"}
    extra_body = {key: deepcopy(value) for key, value in (alias_config.get("sampling_params") or {}).items()
                  if key not in excluded}
    default_headers = deepcopy(alias_config.get("default_headers") or {})
    if alias_config.get("generate_tt_logid"):
        from synthesis.model_worker import FIXED_TT_LOGID
        default_headers.setdefault("X-TT-LOGID", FIXED_TT_LOGID)
    spec = config.ModelSpec(name=model, family="online", display_name=model)
    return OnlineReActRunner(
        spec, model, endpoint,
        api_key=api_key,
        timeout=timeout, max_retries=max_retries,
        model_alias=model_alias, client_type=client_type, api_version=api_version,
        default_headers=default_headers, extra_body=extra_body,
        use_sdk=bool(model_alias) or client_type == "azure_openai",
    )
