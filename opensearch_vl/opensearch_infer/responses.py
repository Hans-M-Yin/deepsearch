"""Native Responses transport and per-case tool loop.

Each case owns its history and previous_response_id. A shared runner contains
only immutable API configuration and a thread-safe HTTP SDK client.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from synthesis.retry_monitor import tracked_sleep

from . import config, messages, pipeline, tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a multimodal research agent and a VQA and world-knowledge expert.
Answer the user's question accurately using the supplied images, relevant knowledge,
and the available search and reading tools when needed. Keep the target entity and
the exact attribute, date, event or version requested by the question in mind.
Inspect search evidence before pursuing a candidate. If a search path fails, reconsider
the target or search strategy. Stop once you have sufficient evidence to answer.

Call tools using the API's native function calls. Do not write <tool_call>, <action>,
<tool_response>, or tool-call JSON in ordinary text. You may give a brief public
progress update before a call; do not expose private chain-of-thought or fabricate
<thinking> blocks. Use assistant phase commentary for intermediate updates and
final_answer for the final answer when the model supports phase.

Search returns compact source_page_id and image_id references. Copy resource IDs
exactly; never invent them. Search-result images are not visible until read_url
successfully returns the actual image. Use read_url(resource_id, goal) to inspect
a result. i2i_search region is [x1,y1,x2,y2] in normalized 0-1000 coordinates on the
current image. A failed read is not evidence; change sources if retries do not help.
Request at most one function per response. Your final response should answer the
original question directly in ordinary text, without XML tags or tool calls.
"""

FINALIZATION_PROMPT = (
    "The search budget is exhausted. Answer the original question now using the "
    "input images and the evidence already gathered. Give your best supported "
    "answer directly in ordinary text; do not request any more tools or output "
    "tool-call JSON. This is the final response opportunity."
)


def add_responses_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = "OpenSearch-VL benchmark inference with native Responses API tools."
    parser.set_defaults(responses_api=True)
    parser.add_argument("--model-alias", help="Registered synthesis/models.json alias; CLI overrides alias defaults.")
    parser.add_argument("--client-type", choices=("openai", "azure_openai"), default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--max-turns", type=int, default=config.MAX_TURNS)
    parser.add_argument("--max-tool-calls", type=int, default=config.MAX_TOOL_CALLS)
    parser.add_argument("--reasoning-effort", "--responses-reasoning-effort", dest="reasoning_effort", default=None)
    parser.add_argument("--reasoning-summary", "--responses-reasoning-summary", dest="reasoning_summary",
                        choices=("auto", "concise", "detailed"), default=None)
    parser.add_argument("--responses-state-mode", choices=("replay", "previous-response-id"), default="replay",
                        help="replay keeps full native output including reasoning; previous-response-id requires storage.")
    parser.add_argument("--responses-store", choices=("true", "false"), default=None)
    parser.add_argument("--no-encrypted-reasoning", action="store_true",
                        help="Do not request or replay reasoning items; loses client-side reasoning continuity.")
    parser.add_argument("--encrypted-reasoning-fallback", choices=("drop", "fail"), default="drop",
                        help="On invalid_encrypted_content, drop opaque reasoning and retry this turn once (default), or fail.")
    parser.add_argument("--system-prompt-file", help="Optional Responses-native inference instructions (not an SFT prompt).")


def validate_responses_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not (args.model_alias or args.model or args.served_model_name):
        parser.error("Specify --model-alias, --model, or --served-model-name.")
    if args.model_alias and (args.model or args.served_model_name):
        parser.error("Use --model-alias or a direct model name, not both.")
    if args.checkpoint:
        parser.error("Responses API does not load --checkpoint; use --model or --model-alias.")
    for flag in ("top_k", "repetition_penalty", "presence_penalty"):
        if getattr(args, flag) is not None:
            parser.error(f"--{flag.replace('_', '-')} is not a Responses API sampling parameter.")
    if args.max_turns < 1 or args.max_tool_calls < 0 or args.api_max_retries < 0:
        parser.error("--max-turns must be positive; --max-tool-calls and --api-max-retries must be non-negative.")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive (includes reasoning and answer tokens).")
    if args.responses_state_mode == "previous-response-id" and args.responses_store == "false":
        parser.error("previous-response-id requires --responses-store true; use replay with store=false.")


def get_tool_definitions(disabled_tools: tuple[str, ...]) -> list[dict[str, Any]]:
    """Flatten the inference schemas into Responses format without strict coercion."""
    result = []
    for definition in json.loads(tools.get_tools_definition(disabled_tools=disabled_tools)):
        function = definition["function"]
        result.append({"type": "function", **function, "strict": False})
    return result


class ResponsesRunner:
    def __init__(self, *, model: str, base_url: str | None, api_key: str | None,
                 client_type: str = "openai", api_version: str | None = None,
                 timeout: int = 600, max_retries: int = 3,
                 request_options: dict[str, Any] | None = None,
                 state_mode: str = "replay", max_turns: int = 46,
                 max_tool_calls: int = 100, system_prompt: str = SYSTEM_PROMPT,
                 default_headers: dict[str, str] | None = None, model_alias: str | None = None,
                 replay_reasoning: bool = True, encrypted_reasoning_fallback: str = "drop") -> None:
        self.model = model
        self.display_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.client_type = client_type
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_options = deepcopy(request_options or {})
        self.state_mode = state_mode
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.system_prompt = system_prompt
        self.default_headers = default_headers
        self.model_alias = model_alias
        if encrypted_reasoning_fallback not in {"drop", "fail"}:
            raise ValueError("encrypted_reasoning_fallback must be drop or fail")
        self.replay_reasoning = replay_reasoning
        self.encrypted_reasoning_fallback = encrypted_reasoning_fallback
        self.client = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ResponsesRunner":
        alias_config: dict[str, Any] = {}
        if args.model_alias:
            from synthesis.model_worker import LLM_WORKER
            alias_config = LLM_WORKER.get_model(args.model_alias) or {}
            if not alias_config:
                raise ValueError(f"Unknown synthesis model alias: {args.model_alias}")
        model = alias_config.get("served_model") or args.served_model_name or args.model
        client_type = args.client_type or alias_config.get("client_type") or "openai"
        endpoint = (args.azure_endpoint or args.base_url or alias_config.get("azure_endpoint")
                    or os.environ.get("AZURE_OPENAI_ENDPOINT")) if client_type == "azure_openai" else (
                    args.base_url or alias_config.get("base_url") or os.environ.get("AGENT_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        api_version = args.api_version or alias_config.get("api_version") or os.environ.get("OPENAI_API_VERSION")
        if client_type == "azure_openai" and not (endpoint and api_version):
            raise ValueError("Azure Responses requires --azure-endpoint and --api-version (or a configured alias).")
        key = (args.api_key or alias_config.get("api_key") or os.environ.get("AGENT_API_KEY")
               or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        sampling = dict(alias_config.get("sampling_params") or {})
        args.max_tokens = args.max_tokens or next((sampling[k] for k in
                           ("max_output_tokens", "out_seq_length", "max_tokens") if sampling.get(k) is not None), 16384)
        options: dict[str, Any] = {"max_output_tokens": int(args.max_tokens)}
        for name in ("temperature", "top_p"):
            value = getattr(args, name)
            if value is None:
                value = sampling.get(name)
            if value is not None:
                options[name] = value
                setattr(args, name, value)
        reasoning = dict(sampling.get("reasoning") or {})
        effort = args.reasoning_effort or sampling.get("reasoning_effort")
        if effort is not None:
            reasoning["effort"] = effort
        if args.reasoning_summary is not None:
            reasoning["summary"] = args.reasoning_summary
        if reasoning:
            options["reasoning"] = reasoning
        # replay is stateless by default; do not rely on gateway response retention.
        store = args.responses_store
        options["store"] = store == "true" if store is not None else args.responses_state_mode == "previous-response-id"
        if not args.no_encrypted_reasoning:
            options["include"] = ["reasoning.encrypted_content"]
        prompt = Path(args.system_prompt_file).read_text(encoding="utf-8") if args.system_prompt_file else SYSTEM_PROMPT
        if "t2i_search" in (args.ablation or []):
            prompt += "\nExternal image downloading is disabled. read_url may read webpage/PDF text only."
        headers = dict(alias_config.get("default_headers") or {})
        if alias_config.get("generate_tt_logid"):
            from synthesis.model_worker import FIXED_TT_LOGID
            headers.setdefault("X-TT-LOGID", FIXED_TT_LOGID)
        return cls(model=model, base_url=endpoint, api_key=key, client_type=client_type,
                   api_version=api_version, timeout=args.api_timeout, max_retries=args.api_max_retries,
                   request_options=options, state_mode=args.responses_state_mode,
                   max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
                   system_prompt=prompt, default_headers=headers or None, model_alias=args.model_alias,
                   replay_reasoning=not args.no_encrypted_reasoning,
                   encrypted_reasoning_fallback=args.encrypted_reasoning_fallback)

    def load(self) -> None:
        from openai import AzureOpenAI, OpenAI
        common = {"api_key": self.api_key or "EMPTY", "timeout": self.timeout,
                  "max_retries": 0, "default_headers": self.default_headers}
        if self.client_type == "azure_openai":
            self.client = AzureOpenAI(azure_endpoint=self.base_url, api_version=self.api_version, **common)
        else:
            self.client = OpenAI(base_url=self.base_url, **common)

    def configuration(self) -> dict[str, Any]:
        return {"api_mode": "responses", "model_alias": self.model_alias,
                "served_model": self.model, "client_type": self.client_type,
                "base_url": self.base_url, "api_version": self.api_version,
                "state_mode": self.state_mode, "parallel_tool_calls": False,
                "replay_reasoning": self.replay_reasoning,
                "encrypted_reasoning_fallback": self.encrypted_reasoning_fallback,
                "request_options": deepcopy(self.request_options),
                "system_prompt": self.system_prompt, "final_answer_format": "plain_text",
                "transport": "openai_sdk"}

    def create(self, *, input_items: list[dict[str, Any]], definitions: list[dict[str, Any]],
               instructions: str, force_answer: bool, previous_response_id: str | None,
               omit_encrypted_reasoning: bool = False) -> dict[str, Any]:
        if self.client is None:
            self.load()
        kwargs = {"model": self.model, "input": input_items, "instructions": instructions,
                  "tools": definitions, "parallel_tool_calls": False,
                  "tool_choice": "none" if force_answer else "auto", **self.request_options}
        if omit_encrypted_reasoning:
            # Never mutate shared runner options: only the affected case degrades.
            include = [value for value in kwargs.get("include", []) if value != "reasoning.encrypted_content"]
            if include:
                kwargs["include"] = include
            else:
                kwargs.pop("include", None)
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.responses.create(**kwargs)
                return response if isinstance(response, dict) else response.model_dump(exclude_none=True)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                # Retry capacity/server failures, not invalid schemas or parameters.
                if attempt == self.max_retries or (status is not None and status < 500 and status not in (408, 409, 429)):
                    raise
                tracked_sleep(min(2 ** attempt, 8), reason="llm_error", tool="llm",
                              error=exc, url=self.base_url or "", fallback_sleep=time.sleep)
        raise AssertionError("unreachable")


def _input_from_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    chat = messages.to_openai_messages([{"role": "user", "parts": parts}])
    blocks = []
    for message in chat:
        for part in message["content"]:
            if part["type"] == "text":
                blocks.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image_url":
                blocks.append({"type": "input_image", "image_url": part["image_url"]["url"]})
    return {"role": "user", "content": blocks}


def _parse_output(raw: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """Read public assistant text; never conflate reasoning summaries with answers."""
    public_text, final_text, calls, refusals = [], [], [], []
    for item in raw.get("output") or []:
        kind = item.get("type")
        if kind == "function_call":
            calls.append(item)
        elif kind == "message" and item.get("role", "assistant") == "assistant":
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    text = block.get("text", "")
                    public_text.append(text)
                    if item.get("phase") != "commentary":
                        final_text.append(text)
                elif block.get("type") == "refusal":
                    refusals.append(block.get("refusal", ""))
    return "\n".join(public_text).strip(), "\n".join(final_text).strip(), calls, refusals


def _serialise(value: Any, image_urls: dict[str, str]) -> Any:
    # Existing serializer covers parquet bytes; also strip Responses data URLs
    # and opaque reasoning from disk (retain them unmodified in runtime history).
    if isinstance(value, str) and value.startswith("data:image/"):
        return f"<image payload omitted: {len(value)} chars>"
    if isinstance(value, dict):
        value = {key: ("<encrypted reasoning omitted>" if key == "encrypted_content" and val else _serialise(val, image_urls))
                 for key, val in value.items()}
    elif isinstance(value, list):
        value = [_serialise(item, image_urls) for item in value]
    return pipeline._strip_base64_payloads(value, image_urls)


def _previous_response_missing(exc: Exception) -> bool:
    text = str(exc).lower()
    return (getattr(exc, "status_code", None) in (400, 404)
            and ("previous_response" in text or "response with id" in text)
            and any(word in text for word in ("not found", "does not exist", "unavailable", "unknown")))


def _invalid_encrypted_content(exc: Exception) -> bool:
    """Recognize this specific 400, including gateways that wrap the upstream code."""
    if getattr(exc, "status_code", None) != 400:
        return False
    text = " ".join((str(exc), str(getattr(exc, "code", "")),
                     str(getattr(exc, "body", "")))).lower()
    return ("invalid_encrypted_content" in text or (
        ("encrypted content" in text or "encrypted_content" in text)
        and any(term in text for term in (
            "could not be verified", "could not be decrypted", "cannot decrypt", "not issued to this caller",
        ))))


def _replay_without_reasoning(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lossy recovery, not the normal protocol: retain public history and tool pairs.

    Dropping only encrypted_content would leave rs_* server references that may
    resolve to the same broken state. Drop the entire reasoning/compaction item,
    and replay remaining message/call content without server item IDs. call_id,
    phase, tool arguments, observations and images must stay unchanged.
    """
    result = []
    for item in items:
        if item.get("type") in {"reasoning", "compaction"}:
            continue
        copied = deepcopy(item)
        if copied.get("type") in {"message", "function_call", "function_call_output"}:
            copied.pop("id", None)
        result.append(copied)
    return result


def _reasoning_item_count(items: list[dict[str, Any]]) -> int:
    return sum(item.get("type") in {"reasoning", "compaction"} for item in items)


def process_single_case(row: Any, runner: ResponsesRunner, output_dir: str, case_idx: int,
                        image_urls_dict=None, dataset_type: str = "train", visual_lookup=None,
                        inference_cfg=None, timing=None, dynamic_resolution: bool = False,
                        disabled_tools: tuple[str, ...] = ()) -> dict[str, Any]:
    row_dict = pipeline._row_to_dict(row)
    case_id = str(pipeline._first_present(row_dict, ("data_id", "id", "idx", "_id"), f"case_{case_idx}"))
    prompt_list, question = pipeline._prompt_from_row(row_dict)
    filename_prefix = "fvqa_test" if dataset_type == "test" else "fvqa_train"
    image_paths, parts = pipeline._bootstrap_images(row, case_id, case_idx, filename_prefix,
                                                    image_urls_dict, dynamic_resolution=dynamic_resolution)
    parts.append({"text": question})
    history = [_input_from_parts(parts)]
    pending = list(history)
    registry: dict[str, Any] = {}
    definitions = get_tool_definitions(disabled_tools)
    allowed_tools = {definition["name"] for definition in definitions}
    trajectory: dict[str, Any] = {"case_id": case_id, "case_idx": case_idx,
        "category": row_dict.get("category", "unknown"),
        "data_source": row_dict.get("data_source", row_dict.get("source", "unknown")),
        "prompt": prompt_list, "original_data": row_dict, "api_mode": "responses",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turns": [], "inference": runner.configuration(), "disabled_tools": list(disabled_tools),
        "image_download_disabled": "t2i_search" in disabled_tools,
        "max_turns": runner.max_turns, "max_tool_calls": runner.max_tool_calls}
    tool_count = 0
    attempted_count = 0
    seen_call_ids: set[str] = set()
    previous_id = None
    use_previous = runner.state_mode == "previous-response-id"
    reasoning_replay_disabled = not runner.replay_reasoning
    trajectory.update({"fallback_events": [], "reasoning_state_dropped": False,
                       "reasoning_items_dropped": 0,
                       "reasoning_replay_disabled": reasoning_replay_disabled})
    failure_kind = "inference_error"
    turn_num = 0
    final_answer = ""
    try:
        for turn_num in range(runner.max_turns):
            force_answer = turn_num == runner.max_turns - 1 or attempted_count >= runner.max_tool_calls
            instructions = runner.system_prompt
            if force_answer:
                instructions += "\n\n" + FINALIZATION_PROMPT
            if config.TRACE_LOGS:
                logger.info("[responses-trace] LLM_START case_id=%s turn=%d force_answer=%s", case_id, turn_num, force_answer)
            started = time.perf_counter()
            failure_kind = "inference_error"
            try:
                kwargs = {"definitions": definitions, "instructions": instructions, "force_answer": force_answer}
                turn_fallbacks = []
                # At most two recovery transitions: stored ID -> replay, then
                # replay -> replay without reasoning. Neither re-executes tools.
                while True:
                    trajectory["state_mode_effective"] = (
                        "previous-response-id" if use_previous else
                        "replay_without_reasoning" if reasoning_replay_disabled else "replay"
                    )
                    try:
                        raw = runner.create(input_items=pending if use_previous else history,
                                            previous_response_id=previous_id if use_previous else None,
                                            omit_encrypted_reasoning=reasoning_replay_disabled, **kwargs)
                        for event in turn_fallbacks:
                            event["request_succeeded"] = True
                        if turn_fallbacks:
                            logger.warning("[responses-fallback] RECOVERED case_id=%s turn=%d mode=%s",
                                           case_id, turn_num, trajectory["state_fallback"])
                        break
                    except Exception as exc:
                        if previous_id and use_previous and _previous_response_missing(exc):
                            use_previous = False
                            previous_id = None
                            trajectory["state_fallback"] = "full_native_output_replay"
                            event = {"turn": turn_num, "kind": "previous_response_id_unavailable",
                                     "action": "full_native_output_replay", "request_succeeded": False}
                            trajectory["fallback_events"].append(event)
                            turn_fallbacks.append(event)
                            logger.warning("[responses-fallback] case_id=%s turn=%d previous_response_id unavailable; replaying full history",
                                           case_id, turn_num)
                            continue
                        if _invalid_encrypted_content(exc):
                            failure_kind = "invalid_encrypted_content"
                            # A second rejection after sanitation is terminal.
                            # In stored mode the invalid state can be server-side.
                            has_stored_state = bool(use_previous and previous_id)
                            count = _reasoning_item_count(history)
                            can_degrade = (not reasoning_replay_disabled or has_stored_state) and (count or has_stored_state)
                            if runner.encrypted_reasoning_fallback == "drop" and can_degrade:
                                history = _replay_without_reasoning(history)
                                use_previous = False
                                previous_id = None
                                reasoning_replay_disabled = True
                                trajectory["state_fallback"] = "replay_without_reasoning"
                                trajectory["reasoning_state_dropped"] = True
                                trajectory["reasoning_replay_disabled"] = True
                                trajectory["reasoning_items_dropped"] += count
                                event = {"turn": turn_num, "kind": "invalid_encrypted_content",
                                         "action": "replay_without_reasoning", "removed_reasoning_items": count,
                                         "used_stored_state": has_stored_state, "request_succeeded": False}
                                trajectory["fallback_events"].append(event)
                                turn_fallbacks.append(event)
                                logger.warning("[responses-fallback] case_id=%s turn=%d invalid_encrypted_content; "
                                               "dropping %d reasoning items, preserving tool pairs, retrying current turn",
                                               case_id, turn_num, count)
                                continue
                        raise
            finally:
                if timing is not None:
                    timing.record("llm", time.perf_counter() - started)
            text, candidate, calls, refusals = _parse_output(raw)
            record: dict[str, Any] = {"turn": turn_num, "response": raw,
                "response_text": text, "response_id": raw.get("id"),
                "status": raw.get("status"), "force_answer": force_answer,
                "tool_calls": deepcopy(calls)}
            trajectory["turns"].append(record)
            # Preserve native output exactly normally. After explicit lossy
            # recovery, do not reintroduce fresh opaque reasoning on later turns.
            output_items = raw.get("output") or []
            if reasoning_replay_disabled:
                dropped = _reasoning_item_count(output_items)
                trajectory["reasoning_items_dropped"] += dropped
                trajectory["reasoning_state_dropped"] |= bool(dropped)
                history.extend(_replay_without_reasoning(output_items))
            else:
                history.extend(deepcopy(output_items))
            previous_id = raw.get("id") if use_previous else None
            if use_previous and not previous_id:
                use_previous = False
                trajectory["state_fallback"] = "missing_response_id_full_replay"
            trajectory["state_mode_effective"] = (
                "previous-response-id" if use_previous else
                "replay_without_reasoning" if reasoning_replay_disabled else "replay"
            )
            status = raw.get("status")
            if status in {"incomplete", "failed", "cancelled", "in_progress", "queued"} or raw.get("error"):
                failure_kind = "incomplete_response" if status == "incomplete" else "response_error"
                raise RuntimeError(f"Responses status={status}: {raw.get('incomplete_details') or raw.get('error')}")
            if refusals:
                failure_kind = "refusal"
                raise RuntimeError("Model refused: " + "\n".join(refusals))
            if not calls:
                if candidate and not re.search(r"<(?:tool_call|action)>|</?thinking>", candidate):
                    answer_match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", candidate, re.DOTALL)
                    final_answer = answer_match.group(1).strip() if answer_match else candidate
                    if final_answer:
                        break
                if text:  # Commentary is a progress update, not a final answer.
                    reminder = {"role": "user", "content": [{"type": "input_text", "text":
                        "Continue using native function calls if needed, or provide the final answer "
                        "in ordinary text with phase final_answer. Do not output XML tool or thinking tags."}]}
                    history.append(reminder)
                    pending = [reminder]
                    record["tool_output_type"] = "format_or_commentary_continuation"
                    continue
                failure_kind = "empty_response"
                raise RuntimeError("No native function call or final assistant text (reasoning-only output is not an answer).")

            pending = []
            results = []
            for call in calls:
                # call_id is the function's protocol ID; item.id is not a substitute.
                call_id = call.get("call_id")
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    failure_kind = "invalid_tool_call_id"
                    raise RuntimeError(f"Missing or duplicate function call_id: {call_id!r}")
                seen_call_ids.add(call_id)
                attempted_count += 1
                name = call.get("name")
                result: dict[str, Any] = {"call_id": call_id, "name": name, "executed": False}
                new_images: dict[str, Any] = {}
                if force_answer or attempted_count > runner.max_tool_calls:
                    observation = "Tool was not executed. " + FINALIZATION_PROMPT
                    result["type"] = "tool_budget_exhausted"
                else:
                    try:
                        arguments = json.loads(call.get("arguments") or "{}")
                        if not isinstance(arguments, dict) or name not in allowed_tools:
                            raise ValueError("Tool name must be declared and arguments must be a JSON object.")
                    except (TypeError, ValueError) as exc:
                        observation = f"Tool argument error: {exc}. Correct the native call or answer directly."
                        result["type"] = "invalid_tool_arguments"
                    else:
                        tool_count += 1
                        result.update({"executed": True, "arguments": arguments})
                        started = time.perf_counter()
                        try:
                            observation, new_images = tools.execute_tool(
                                json.dumps({"name": name, "parameters": arguments}, ensure_ascii=False),
                                image_paths, case_id, case_idx, turn_num, os.path.join(output_dir, "intermediate"),
                                filename_prefix=filename_prefix, visual_lookup=visual_lookup,
                                url_registry=registry, disabled_tools=disabled_tools)
                        except Exception as exc:
                            observation = f"Tool execution error: {type(exc).__name__}: {exc}"
                            result["type"] = "tool_execution_error"
                        finally:
                            if timing is not None:
                                timing.record(str(name), time.perf_counter() - started)
                result["output"] = str(observation)
                results.append(result)
                pending.append({"type": "function_call_output", "call_id": call_id, "output": str(observation)})
                image_paths.update(new_images)
                for image_id, payload in new_images.items():
                    image_part = pipeline._model_image_part(payload, dynamic_resolution)
                    if image_part is not None:
                        pending.append(_input_from_parts([{"text": f"image_id={image_id}"}, image_part]))
            # Never drop parallel calls: every returned call_id needs an output,
            # even for invalid arguments or deliberately skipped budget calls.
            history.extend(deepcopy(pending))
            record["tool_results"] = results
            if len(calls) == 1:
                result = results[0]
                record["tool_call"] = {"name": result["name"], "arguments": result.get("arguments", {})}
                record["tool_output"] = result["output"]
                record["tool_output_type"] = result.get("type", "tool_result")
        else:
            failure_kind = "max_turns"
            raise RuntimeError(f"Reached {runner.max_turns} Responses turns without a final answer.")
    except Exception as exc:
        trajectory["tool_call_attempt_count"] = attempted_count
        pipeline._save_failure_trajectory(
            _serialise(trajectory, {}), output_dir, image_paths, registry,
            failure_kind=failure_kind, failure_reason=str(exc), failed_turn=turn_num,
            tool_call_count=tool_count, max_turns=runner.max_turns, max_tool_calls=runner.max_tool_calls)
        raise

    trajectory.update({"status": "completed", "tool_call_count": tool_count,
        "tool_call_attempt_count": attempted_count, "final_answer": final_answer,
        # Keep the judge/resume contract while retaining native output separately.
        "final_response_text": f"<answer>{final_answer}</answer>",
        "url_resources": pipeline._serialise_url_resources(registry)})
    image_urls = {key: val for key, val in image_paths.items()
                  if isinstance(val, str) and val.startswith(("http://", "https://"))}
    path = Path(output_dir) / f"{case_id}_trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_serialise(trajectory, image_urls), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)
    logger.info("Responses case=%s completed turns=%d tools=%d -> %s", case_id, len(trajectory["turns"]), tool_count, path)
    return trajectory
