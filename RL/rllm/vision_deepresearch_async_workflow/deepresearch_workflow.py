from io import BytesIO
import asyncio
import json
import os
import re
import traceback
from typing import Any, List, Optional

from PIL import Image

from vision_deepresearch_async_workflow.deepresearch_agent import DeepResearchAgent
from vision_deepresearch_async_workflow.search_redundancy_reward import (
    score_search_redundancy,
)
from vision_deepresearch_async_workflow.partial_credit_reward import (
    PartialCreditConfig,
    partial_credit_metrics,
    score_statement_partial_credit,
)
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationReason, Workflow

import base64


def as_pil_image(image: Any) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, str) and image.startswith("data:image/"):
        try:
            _, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            try:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        data_str = None
        if "data" in image and isinstance(image["data"], str):
            data_str = image["data"]
        elif "path" in image and isinstance(image["path"], str):
            data_str = image["path"]
        elif "url" in image and isinstance(image["url"], str):
            data_str = image["url"]
        if data_str:
            if data_str.startswith("data:image/"):
                try:
                    _, encoded = data_str.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception:  # noqa: BLE001
                    return None
            try:
                return Image.open(data_str).convert("RGB")
            except Exception:  # noqa: BLE001
                return None

    if isinstance(image, str):
        try:
            return Image.open(image).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    return None


def _task_system_prompt(task: dict[str, Any]) -> str | None:
    """Return an optional per-example system prompt from Registry metadata.

    DatasetRegistry puts the complete registered record into the task's
    runtime ``extra_info``.  The converter keeps future metadata in a nested
    ``extra_info`` object, so support both a direct field and the nested form.
    A missing value deliberately falls back to the workflow-level prompt.
    """
    direct_prompt = task.get("system_prompt")
    nested_info = task.get("extra_info")
    nested_prompt = (
        nested_info.get("system_prompt")
        if isinstance(nested_info, dict)
        else None
    )
    prompt = direct_prompt or nested_prompt
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<thinking>.*?</thinking>\s*(<tool_call>.*?</tool_call>|<answer>.*?</answer>)\s*$"
    )
    return re.match(pattern, content, re.DOTALL) is not None


def _format_reward_for_step(step: Step) -> float:
    if step.info.get("step_error"):
        return 0.0
    if _has_tool_error_observation(step.observation):
        return 0.0
    content = step.model_response if isinstance(step.model_response, str) else ""
    return 1.0 if _is_valid_format(content) else 0.0


def _has_tool_error_observation(observation: Any) -> bool:
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
        "[Python Interpreter Error]",
        "Python execution error:",
        "PythonInterpreter tool not available",
        "PythonInterpreter tool is not callable",
        "Tool execution error:",
        "Error executing",
        "Error: Image reference",
        "Error: OpenCV not available",
    )
    return any(marker in observation for marker in error_markers)


def _is_step_error(step: Step) -> bool:
    if step.info.get("step_error"):
        return True
    return _has_tool_error_observation(step.observation)


# ---------------------------------------------------------------------------
# Query-utility reward: LLM judge evaluates search trajectory quality
# ---------------------------------------------------------------------------

_QUERY_JUDGE_MARKER = os.getenv("JUDGE_MODEL_MARKER", os.getenv("JUDGE_MODEL", "gpt-4o-mini"))

# MAX_TURNS remains a diagnostic termination type, but it currently carries no
# reward adjustment. It is retained as a named constant so experiments can
# re-enable a terminal penalty without changing reward composition code.
_MAX_TURNS_TERMINAL_REWARD = 0.0
_ACCURACY_REWARD_WEIGHT = 0.9
_FORMAT_REWARD_WEIGHT = 0.1


def _base_outcome_reward(r_accuracy: float, r_format: float) -> float:
    """Combine final-answer correctness and trajectory-wide format reward."""

    return (
        _ACCURACY_REWARD_WEIGHT * r_accuracy
        + _FORMAT_REWARD_WEIGHT * r_format
    )

_QUERY_UTILITY_PROMPT_TEMPLATE = """\
You are an impartial judge evaluating the quality and utility of an agent's search trajectory.

[Original Question]
{question}

[Ground Truth Answer]
{ground_truth}

[Agent's Final Answer]
{prediction}

[Search Trajectory]
{trajectory_summary}

[Evaluation Criteria]
Evaluate the overall utility of the agent's search queries and retrieved results:

1. **Image search utility**: Did image searches retrieve visual evidence that genuinely supports answering the question? Were the images relevant, or just noise (captchas, unrelated pages)?
2. **Text search utility**: Did text searches find relevant textual information that contributes to the answer? Were queries well-formed and targeted?
3. **Query progression**: Did the queries show logical progression — refining, narrowing, or covering different aspects of the question? Or did they repeat / drift aimlessly?
4. **Complementarity**: Did image and text searches complement each other, providing evidence that one modality alone couldn't supply?
5. **Evidence vs. noise ratio**: What fraction of retrieved results actually contained useful evidence versus irrelevant content?

Score the overall query utility from 0.0 to 1.0:
- 0.0: No useful information retrieved; all searches irrelevant or failed
- 0.3: Mostly noise with occasional marginally relevant results
- 0.5: Mixed — some useful evidence found but significant noise or inefficiency
- 0.7: Good search strategy; majority of results were relevant
- 1.0: Excellent — targeted, efficient queries that retrieved highly relevant evidence

Output format (strictly follow):
score: [a single float between 0.0 and 1.0]
reasoning: [brief explanation in 2-3 sentences]"""


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + " ... " + text[-(limit // 2) :]


def _build_trajectory_summary(steps: list[Step], max_steps: int = 20) -> str:
    """Build a concise summary of search-related steps for the judge."""
    lines: list[str] = []
    search_count = 0

    for idx, step in enumerate(steps):
        action_dict = step.action.action if step.action else {}
        action_type = action_dict.get("type", "")

        if action_type == "tool_call":
            tool_call_raw = action_dict.get("tool_call", "")
            tool_name = "unknown"
            tool_args_str = ""
            try:
                if isinstance(tool_call_raw, str):
                    parsed = json.loads(tool_call_raw)
                else:
                    parsed = tool_call_raw
                tool_name = parsed.get("name", "unknown")
                tool_args_str = json.dumps(
                    parsed.get("arguments", {}), ensure_ascii=False
                )
            except (json.JSONDecodeError, TypeError, AttributeError):
                tool_name = "parse_error"
                tool_args_str = _truncate(str(tool_call_raw), 200)

            obs = step.observation or ""
            if isinstance(obs, str) and "<tool_response>" in obs:
                obs = obs.split("<tool_response>")[-1].split("</tool_response>")[0]

            is_error = step.info.get("step_error", False)
            status = " [ERROR]" if is_error else ""

            lines.append(
                f"Step {idx + 1}: {tool_name}({_truncate(tool_args_str, 200)}){status}\n"
                f"  Result: {_truncate(obs.strip(), 400)}"
            )
            search_count += 1
            if search_count >= max_steps:
                lines.append(f"... ({len(steps) - idx - 1} more steps omitted)")
                break

        elif action_type == "final_answer":
            lines.append(
                f"Step {idx + 1}: final_answer\n"
                f"  Answer: {_truncate(action_dict.get('answer', ''), 300)}"
            )

    if not lines:
        return "(No tool calls in trajectory)"
    return "\n\n".join(lines)


def _parse_query_utility_score(text: str) -> float | None:
    """Extract the score float from judge response."""
    for line in text.split("\n"):
        line_stripped = line.strip().lower()
        if line_stripped.startswith("score:"):
            val_str = line_stripped[len("score:"):].strip()
            try:
                val = float(val_str)
                return max(0.0, min(1.0, val))
            except ValueError:
                continue
    m = re.search(r"score\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return None


async def _judge_query_utility(
    question: str,
    ground_truth: str,
    prediction: str,
    steps: list[Step],
) -> float:
    """Call LLM judge to evaluate search query utility. Returns 0.0-1.0."""

    trajectory_summary = _build_trajectory_summary(steps)
    if trajectory_summary == "(No tool calls in trajectory)":
        return 0.0

    prompt = _QUERY_UTILITY_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth or "(not provided)",
        prediction=prediction or "(no prediction)",
        trajectory_summary=trajectory_summary,
    )

    try:
        from vision_deepresearch_async_workflow.utils.api_gateway_client import (
            api_gateway_chat,
            is_api_gateway_configured,
        )

        if not is_api_gateway_configured():
            return 0.0

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        data = await asyncio.to_thread(
            api_gateway_chat,
            messages=messages,
            model_marker=_QUERY_JUDGE_MARKER,
            max_tokens=500,
            timeout=60,
        )

        choices = data.get("choices") or [{}]
        content = ""
        first_choice = choices[0] or {}
        if isinstance(first_choice, dict):
            msg = first_choice.get("message")
            if isinstance(msg, dict):
                content = msg.get("content", "")

        score = _parse_query_utility_score(content)
        if score is not None:
            return score

        print(f"[QueryUtilityReward] Failed to parse score from: {content[:200]}")
        return 0.0

    except Exception as exc:  # noqa: BLE001
        print(f"[QueryUtilityReward] Judge call failed: {exc}")
        return 0.0


def _get_next_observation(messages: list[dict], current_index: int) -> str:
    if current_index + 1 < len(messages):
        next_msg = messages[current_index + 1]
        if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
            return next_msg["content"]
    return ""


def _is_max_turns_termination(termination: str) -> bool:
    normalized = str(termination or "").strip().lower()
    return normalized in {
        "max_rounds_reached",
        "max_rounds_reached_no_answer",
        "exceed available llm calls",
        "answer not found",
    } or normalized.startswith("answer not found after")


def _map_termination_reason(termination: str) -> TerminationReason:
    # The agent uses both fixed names and dynamic strings such as
    # ``answer not found after 50 rounds`` when it exhausts its call budget.
    # Normalize all of these to the dedicated max-turns reason instead of
    # letting the dynamic forms fall through to UNKNOWN.
    if _is_max_turns_termination(termination):
        return TerminationReason.MAX_TURNS_EXCEEDED

    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "answer_tool_fallback": TerminationReason.ENV_DONE,
        "timeout": TerminationReason.UNKNOWN,
        "token_limit_no_answer": TerminationReason.UNKNOWN,
        "answer_token_limit": TerminationReason.UNKNOWN,
        "prompt_budget_reached": TerminationReason.UNKNOWN,
        "repetition_detected": TerminationReason.UNKNOWN,  # Will be masked
        "tool_errors_too_many": TerminationReason.UNKNOWN,  # Will be masked
        "consecutive_step_errors": TerminationReason.UNKNOWN,  # Will be masked
        "error": TerminationReason.UNKNOWN,  # Will be masked
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


def _build_rollout_markers(
    termination: str,
    steps: list[Step],
    latency: dict,
) -> list[str]:
    """Build diagnostic markers without changing reward/masking semantics."""
    markers: list[str] = []

    # The agent emits either an explicit max-call reason or an
    # ``answer not found`` reason after exhausting its configured call budget.
    if _is_max_turns_termination(termination):
        markers.append("MAX_TURNS")

    raw_termination = str(termination or "").strip().lower()

    tool_error_termination = raw_termination in {
        "tool_errors_too_many",
        "consecutive_step_errors",
        "tool_execution_exception",
        "tool_call_json_parse",
        "tool_call_not_object",
        "tool_call_schema",
        "answer_tool_fallback",
    }
    step_has_error = any(_is_step_error(step) for step in steps)
    latency_tool_failures = 0
    if isinstance(latency, dict):
        tool_summaries = latency.get("tools")
        if isinstance(tool_summaries, dict):
            latency_tool_failures = sum(
                int(summary.get("failures") or 0)
                for summary in tool_summaries.values()
                if isinstance(summary, dict)
            )
    if tool_error_termination or step_has_error or latency_tool_failures > 0:
        markers.append("TOOL_ERROR")

    return markers


def _evaluate_answer(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()


def _should_mask_episode(result: dict, episode: Episode) -> tuple[bool, str]:
    """Determine if the entire episode should be masked based on answer/step error conditions."""
    steps = episode.trajectories[0].steps if episode.trajectories else []
    termination = result.get("termination", "")

    # Mask directly if no final answer is produced.
    if termination != "answer":
        return True, termination or "no_final_answer"

    # Check if there are too many step errors.
    total_steps = len(steps)

    if total_steps > 0:
        step_error_steps = sum(1 for step in steps if _is_step_error(step))
        if step_error_steps / total_steps > 0.5:
            return True, "tool_errors_too_many"

    return False, ""


# ---------------------------------------------------------------------------
# Step-level fatal-after detection (replaces trajectory-level hard mask)
# ---------------------------------------------------------------------------

_CONSECUTIVE_ERROR_THRESHOLD = 3


def _find_fatal_step_index(
    steps: list[Step], termination: str
) -> tuple[int | None, str]:
    """Detect the first fatal step using a consecutive-error counter.

    Returns ``(fatal_step_index, reason)`` where *fatal_step_index* is the
    0-based index of the **first step to mask** (this step and all subsequent
    steps will be masked).  Returns ``(None, "")`` for normal trajectories,
    including a MAX_TURNS terminal failure with otherwise usable steps.

    A single error followed by recovery resets the counter — only cascading
    failures (>= threshold consecutive errors) trigger masking.
    """
    if not steps:
        return 0, "no_steps"

    consecutive_errors = 0

    for idx, step in enumerate(steps):
        if _is_step_error(step):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
            fatal_start = idx - _CONSECUTIVE_ERROR_THRESHOLD + 1
            return fatal_start, "consecutive_errors"

    if termination in {"answer", "answer_tool_fallback"}:
        return None, ""

    # Exhausting the turn budget is an agent-policy failure, not a corrupted
    # rollout.  Keep the complete trajectory learnable so its terminal -0.1
    # reward produces a negative group-relative advantage.  Treating it as
    # fatal would trigger the trainer's one-sided A >= 0 clamp and silently
    # remove precisely the anti-looping signal this reward was added for.
    if _is_max_turns_termination(termination):
        return None, ""

    # Abnormal termination without error cascade — all existing steps are
    # usable but the trajectory is incomplete.  Mark fatal *after* the last
    # step so every step is kept, yet the trajectory is still flagged for
    # advantage clamping.
    return len(steps), termination or "no_final_answer"


def _to_pil_image(img: Any) -> Optional[Image.Image]:
    """Best-effort conversion to PIL.Image for downstream multi-modal pipeline."""
    if isinstance(img, Image.Image):
        return img
    pil = as_pil_image(img)
    if pil is not None:
        return pil
    if isinstance(img, dict) and "bytes" in img:
        try:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    return None


def _search_redundancy_metrics(record: Any) -> dict[str, float]:
    """Convert one private scorer record into safe numeric training metrics.

    The trace itself is consumed and discarded before the Episode reaches the
    trainer. This helper deliberately exports aggregate counts/timings only,
    so trainer and W&B logs cannot expose retrieved pages or user prompts.
    """

    record = record if isinstance(record, dict) else {}

    def nonnegative(name: str) -> float:
        try:
            return max(0.0, float(record.get(name, 0.0)))
        except (TypeError, ValueError):
            return 0.0

    mode = str(record.get("mode") or "").strip().lower()
    # ``off`` can still calculate observation-only records when an embedding
    # URL happens to be supplied, but the scorer is not configured as an
    # enabled experiment mode.  Keep this signal distinct from ``scored``.
    scoring_enabled = mode in {"shadow", "reward"}
    reason = str(record.get("reason") or "")
    raw_penalty = nonnegative("raw_penalty")
    applied_penalty = nonnegative("penalty")
    penalty_cap = nonnegative("trajectory_penalty_cap")
    cap_hit = penalty_cap > 0.0 and raw_penalty >= penalty_cap - 1e-12

    return {
        "search_redundancy/enabled": float(scoring_enabled),
        "search_redundancy/observation_only": float(
            scoring_enabled and mode != "reward"
        ),
        "search_redundancy/reward_enabled": float(mode == "reward"),
        "search_redundancy/mode_shadow": float(mode == "shadow"),
        "search_redundancy/mode_reward": float(mode == "reward"),
        # ``scored`` means that the TEI request returned valid embeddings and
        # the deterministic comparison completed. It is intentionally not the
        # same as "trajectory had >= 2 searches".
        "search_redundancy/scored": float(bool(record.get("scored"))),
        "search_redundancy/embedding_request_failed": float(
            reason == "embedding_request_failed"
        ),
        "search_redundancy/missing_embedding_url": float(
            reason == "missing_embedding_url"
        ),
        "search_redundancy/search_trace_count": nonnegative(
            "search_trace_count"
        ),
        "search_redundancy/eligible_search_count": nonnegative(
            "eligible_search_count"
        ),
        "search_redundancy/embedded_query_count": nonnegative(
            "embedded_query_count"
        ),
        "search_redundancy/pair_count": nonnegative("pair_count"),
        "search_redundancy/hard_constraint_exempt_pair_count": nonnegative(
            "hard_constraint_exempt_pair_count"
        ),
        "search_redundancy/confirmed_event_count": nonnegative(
            "confirmed_event_count"
        ),
        "search_redundancy/raw_penalty": raw_penalty,
        "search_redundancy/applied_penalty": applied_penalty,
        "search_redundancy/raw_penalized_trajectory": float(raw_penalty > 0.0),
        "search_redundancy/penalized_trajectory": float(
            applied_penalty > 0.0
        ),
        "search_redundancy/cap_hit": float(cap_hit),
        "search_redundancy/embedding_request_s": nonnegative(
            "embedding_request_s"
        ),
        "search_redundancy/embedding_queue_wait_s": nonnegative(
            "embedding_queue_wait_s"
        ),
    }


class DeepResearchWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict | None = None,
        system_prompt: str | None = None,
        reward_function: RewardFunction | None = None,
        max_llm_calls: int | None = None,
        max_turns: int | None = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)

        self.tools = tools or {}
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        self.system_prompt = system_prompt
        self.reward_function = reward_function
        configured_max_llm_calls = (
            max_llm_calls if max_llm_calls is not None else max_turns
        )

        self.agent = DeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
            executor=executor,
            max_llm_calls=configured_max_llm_calls,
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")
        task_system_prompt = _task_system_prompt(task)

        try:
            raw_images = None
            if "images" in task:
                raw_images = task.get("images")

            pil_images: List[Image.Image] = []
            if raw_images is not None:
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                for img in raw_images:
                    pil = _to_pil_image(img)
                    if pil is not None:
                        pil_images.append(pil)

            if pil_images:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    images=pil_images,
                    image_path=raw_images[0],
                    system_prompt=task_system_prompt,
                    trajectory_id=uid,
                    **kwargs,
                )
            else:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    system_prompt=task_system_prompt,
                    trajectory_id=uid,
                    **kwargs,
                )

            episode = self._convert_result_to_episode(result, task, uid)

            # The partial-credit audit uses the completed, text-only
            # trajectory and the question-facing merged statements retained
            # by synthesis.  It issues one judge request per trajectory (not
            # one per turn), and fails open so an auxiliary judge outage can
            # never abort rollout.
            first_traj = (
                episode.trajectories[0] if episode.trajectories else None
            )
            all_steps = first_traj.steps if first_traj else []
            try:
                statement_partial_credit = await score_statement_partial_credit(
                    task=task,
                    steps=all_steps,
                )
            except Exception as err:  # noqa: BLE001 - reward shaping is optional.
                print(f"Statement partial-credit scorer failed open: {err}")
                statement_partial_credit = {
                    "mode": PartialCreditConfig.from_env().mode,
                    "reason": "workflow_scorer_failed",
                    "scored": False,
                    "raw_bonus": 0.0,
                    "bonus": 0.0,
                    "events": [],
                }
            statement_partial_metrics = partial_credit_metrics(
                statement_partial_credit
            )
            raw_statement_partial_bonus = float(
                statement_partial_credit.get("raw_bonus", 0.0)
            )
            episode.info["statement_partial_credit"] = {
                # Keep only scalar/structural audit information; never retain
                # the full judge prompt, retrieved webpage text, or image data.
                key: value
                for key, value in statement_partial_credit.items()
                if key not in {"judge_error", "events"}
            }

            # This is intentionally evaluated at the end of *this*
            # trajectory, alongside its terminal reward. The trace contains
            # private t2t URLs and is never shown to the rollout model. The
            # scorer fails open when the auxiliary embedding service is down.
            search_redundancy = await score_search_redundancy(
                result.get("search_trace", [])
            )
            redundancy_penalty = float(search_redundancy.get("penalty", 0.0))
            raw_redundancy_penalty = float(
                search_redundancy.get("raw_penalty", 0.0)
            )
            redundancy_metrics = _search_redundancy_metrics(search_redundancy)
            episode.info["search_redundancy"] = search_redundancy
            episode.info["search_redundancy_penalty"] = redundancy_penalty
            # The scorer has consumed the private query/URL trace. Do not
            # serialize it through Ray, the trainer batch, or trajectory logs.
            episode.info.pop("search_trace", None)

            prediction = result.get("prediction", "")
            if self.reward_function is not None and prediction:
                try:
                    if asyncio.iscoroutinefunction(self.reward_function):
                        reward_out = await self.reward_function(task, prediction)
                    else:
                        loop = asyncio.get_running_loop()
                        reward_out = await loop.run_in_executor(
                            self.executor,
                            lambda: self.reward_function(task, prediction),
                        )
                except Exception as err:  # noqa: BLE001
                    print(f"Reward function failed: {err}")
                else:
                    if reward_out.is_correct is not None:
                        episode.is_correct = bool(reward_out.is_correct)
                    if isinstance(reward_out.metadata, dict):
                        reward_metadata = episode.info.setdefault("reward_metadata", {})
                        for key, value in reward_out.metadata.items():
                            if key not in reward_metadata:
                                reward_metadata[key] = value
                    if getattr(reward_out, "reward", None) is not None:
                        episode.info["reward_function_reward"] = float(
                            reward_out.reward
                        )

            # --- Step-level fatal-after detection ---
            # Replaces the old trajectory-level hard mask.  Instead of
            # discarding entire anomalous episodes, we detect the first
            # "fatal" point (consecutive-error cascade) and only mask
            # tokens *after* that point.  The prefix is kept for learning
            # with a one-sided advantage clamp (see trainer).
            fatal_step_index, fatal_reason = _find_fatal_step_index(
                all_steps, result.get("termination", "")
            )
            is_fatal = fatal_step_index is not None

            num_turns = sum(
                len(t.steps) for t in episode.trajectories
            )

            if is_fatal and fatal_step_index == 0:
                # Nothing salvageable — degenerate to hard mask.
                episode.termination_reason = TerminationReason.UNKNOWN
                episode.info["fatal_step_index"] = 0
                episode.info["is_fatal"] = True
                episode.info["fatal_reason"] = fatal_reason
                for trajectory in episode.trajectories:
                    trajectory.reward = 0.0
                episode.metrics = {
                    "reward/accuracy": 0.0,
                    "reward/query_utility": 0.0,
                    "reward/format": 0.0,
                    "reward/search_redundancy_raw_penalty": raw_redundancy_penalty,
                    "reward/search_redundancy_penalty": redundancy_penalty,
                    "reward/search_redundancy_events": float(
                        search_redundancy.get("confirmed_event_count", 0)
                    ),
                    "reward/statement_partial_credit_raw_bonus": raw_statement_partial_bonus,
                    "reward/statement_partial_credit_applied_bonus": 0.0,
                    **redundancy_metrics,
                    **statement_partial_metrics,
                    "partial_credit/applied_bonus": 0.0,
                    "partial_credit/reward_suppressed_fatal": 1.0,
                    "partial_credit/reward_suppressed_max_turns": 0.0,
                    "reward/total": 0.0,
                    "masked": 1.0,
                    "is_fatal": 1.0,
                    "fatal_step_index": 0,
                    "num_turns": num_turns,
                }
                episode.info["mask_reason"] = fatal_reason
            else:
                # Compute full reward for group advantage statistics.
                # Query utility remains scoped to the learnable prefix below,
                # but format is a trajectory-level hard constraint: one
                # malformed/error step makes the whole format component zero.
                r_accuracy = 1.0 if episode.is_correct else 0.0

                # Keep checking every generated round, including the part
                # after a fatal boundary.  The policy may mask that suffix
                # during the update, but a format error anywhere in the
                # trajectory must still invalidate the format reward.
                format_scores = [_format_reward_for_step(s) for s in all_steps]
                format_error_rounds = [
                    index + 1
                    for index, score in enumerate(format_scores)
                    if score == 0.0
                ]
                if format_scores:
                    r_format = 0.0 if format_error_rounds else 1.0
                else:
                    r_format = 0.0
                episode.info["format_error_rounds"] = format_error_rounds

                # Query-utility/quality reward is intentionally disabled for
                # the current experiment.  Keep the metric field at 0.0 for
                # backward-compatible logging, but do not issue an additional
                # LLM judge request and do not include it in policy reward.
                r_query = 0.0
                # Correctness is the principal objective, with a smaller
                # independent trajectory-format reward.
                base_reward = _base_outcome_reward(r_accuracy, r_format)
                max_turns_terminated = _is_max_turns_termination(
                    result.get("termination", "")
                )
                termination_penalty = (
                    _MAX_TURNS_TERMINAL_REWARD if max_turns_terminated else 0.0
                )
                # Positive process credit remains unavailable for fatal error
                # cascades, but MAX_TURNS itself no longer changes reward.
                statement_partial_bonus = float(
                    statement_partial_credit.get("bonus", 0.0)
                )
                partial_credit_suppressed_fatal = bool(is_fatal)
                partial_credit_suppressed_max_turns = False
                applied_statement_partial_bonus = (
                    0.0
                    if partial_credit_suppressed_fatal
                    else statement_partial_bonus
                )
                total_reward = base_reward
                total_reward -= redundancy_penalty
                total_reward += applied_statement_partial_bonus
                episode.info["reward_base_total"] = base_reward
                episode.info["max_turns_terminated"] = max_turns_terminated

                for trajectory in episode.trajectories:
                    trajectory.reward = total_reward

                if is_fatal:
                    episode.info["fatal_step_index"] = fatal_step_index
                    episode.info["is_fatal"] = True
                    episode.info["fatal_reason"] = fatal_reason

                episode.metrics = {
                    "reward/accuracy": r_accuracy,
                    "reward/query_utility": r_query,
                    "reward/format": r_format,
                    "reward/termination_penalty": termination_penalty,
                    "reward/search_redundancy_raw_penalty": raw_redundancy_penalty,
                    "reward/search_redundancy_penalty": redundancy_penalty,
                    "reward/search_redundancy_events": float(
                        search_redundancy.get("confirmed_event_count", 0)
                    ),
                    "reward/statement_partial_credit_raw_bonus": raw_statement_partial_bonus,
                    "reward/statement_partial_credit_applied_bonus": applied_statement_partial_bonus,
                    **redundancy_metrics,
                    **statement_partial_metrics,
                    "partial_credit/applied_bonus": applied_statement_partial_bonus,
                    "partial_credit/reward_suppressed_fatal": float(
                        partial_credit_suppressed_fatal
                    ),
                    "partial_credit/reward_suppressed_max_turns": float(
                        partial_credit_suppressed_max_turns
                    ),
                    "reward/total": total_reward,
                    "masked": 0.0,
                    "is_fatal": 1.0 if is_fatal else 0.0,
                    "fatal_step_index": (
                        fatal_step_index if fatal_step_index is not None else -1
                    ),
                    "num_turns": num_turns,
                }

            return episode

        except Exception as exc:  # noqa: BLE001
            print(f"❌ DeepResearch workflow failed for task {uid}: {exc}")
            episode = Episode()
            episode.id = uid
            episode.task = task
            episode.termination_reason = TerminationReason.ERROR
            episode.is_correct = False
            episode.trajectories = []
            episode.metrics = {
                "reward/accuracy": 0.0,
                "reward/query_utility": 0.0,
                "reward/format": 0.0,
                "reward/search_redundancy_raw_penalty": 0.0,
                "reward/search_redundancy_penalty": 0.0,
                "reward/search_redundancy_events": 0.0,
                **_search_redundancy_metrics({}),
                **partial_credit_metrics({}),
                "reward/total": 0.0,
                "masked": 1.0,
            }
            episode.info = {
                "error": str(exc),
                "raw_termination": "error",
                "rollout_markers": [],
            }
            return episode

    def _convert_result_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        messages = result.get("messages", [])
        model_outputs = result.get("model_outputs", [])
        last_model_output = result.get("last_model_output")
        prediction = result.get("prediction", "")
        termination = result.get("termination", "unknown")
        rounds = result.get("rounds", 0)
        time_taken = result.get("time_taken", 0.0)
        latency = result.get("latency", {})

        trajectories: list[Trajectory] = []
        steps: list[Step] = []
        assistant_output_index = 0
        assistant_message_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        last_assistant_message_index = (
            assistant_message_indices[-1] if assistant_message_indices else None
        )

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                context = messages[: i + 1]
                assistant_content = msg.get("content", "")
                action = _extract_action_from_response(assistant_content)
                observation = _get_next_observation(messages, i)
                step = Step(
                    chat_completions=context.copy(),
                    model_response=assistant_content,
                    action=action,
                    observation=observation,
                    reward=0.0,
                )
                # Stage 1 stores the exact ModelOutput returned by the
                # multimodal rollout engine for each assistant generation.
                # Legacy and cumulative modes do not populate this list and
                # therefore keep the historical ``None`` behavior.
                candidate_output = (
                    model_outputs[assistant_output_index]
                    if assistant_output_index < len(model_outputs)
                    else None
                )
                if (
                    candidate_output is None
                    and i == last_assistant_message_index
                    and isinstance(last_model_output, ModelOutput)
                ):
                    candidate_output = last_model_output
                step.model_output = (
                    candidate_output
                    if isinstance(candidate_output, ModelOutput)
                    else None
                )
                assistant_output_index += 1
                if msg.get("step_error"):
                    step.info["step_error"] = True
                if _has_tool_error_observation(observation):
                    step.info["step_error"] = True
                steps.append(step)
            i += 1

        trajectory = Trajectory(
            name="deepresearch_agent",
            task=task,
            steps=steps,
            reward=0.0,
            info={},
        )
        trajectories.append(trajectory)

        answer_text = task.get("answer", "")
        is_correct = _evaluate_answer(prediction, answer_text) if answer_text else False

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = _map_termination_reason(termination)
        episode.is_correct = is_correct
        episode.trajectories = trajectories
        episode.metrics = {}
        episode.info = {
            "rounds": rounds,
            "time_taken": time_taken,
            "latency": latency,
            "raw_termination": termination,
            "rollout_markers": _build_rollout_markers(
                termination, steps, latency
            ),
            "prediction": prediction,
            "answer": answer_text,
            "token_usage": result.get("token_usage", {}),
            "search_trace": result.get("search_trace", []),
        }
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        # Keep the Workflow-level lifecycle state in sync with the base class.
        # MultiTurnReactAgent resets its own per-run state through the base
        # implementation's agent reset loop, but the Workflow also needs to
        # retain ``uid`` and ``task`` for timeout/error post-processing.
        super().reset(task=task, uid=uid)

    def _on_timeout(self, *, task: dict, uid: str) -> None:
        """Print one compact snapshot before timeout cleanup starts."""
        if str(os.environ.get("RLLM_TIMEOUT_DEBUG", "1")).strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return

        agent = self.agent
        tool_calls = getattr(agent, "_tool_call_counts", {}) or {}
        tool_failures = getattr(agent, "_tool_latency_failures", {}) or {}
        tool_calls_text = ",".join(
            f"{name}:{count}" for name, count in sorted(tool_calls.items())
        ) or "-"
        tool_failures_text = ",".join(
            f"{name}:{count}" for name, count in sorted(tool_failures.items())
        ) or "-"
        attempted_tools = sum(int(count) for count in tool_calls.values())
        completed_tools = sum(
            len(samples)
            for samples in (getattr(agent, "_tool_latency_samples_s", {}) or {}).values()
        )

        print(
            "[RL_ROLLOUT_TIMEOUT] "
            f"task={uid} timeout_s={self.timeout} "
            f"rounds={getattr(agent, '_current_round', 0)} "
            f"llm_calls_completed={getattr(agent, 'model_call_count', 0)} "
            f"prompt_tokens_sum={getattr(agent, 'total_prompt_tokens', 0)} "
            f"completion_tokens_sum={getattr(agent, 'total_completion_tokens', 0)} "
            f"last_prompt_tokens={getattr(agent, 'last_prompt_tokens', 0)} "
            f"last_completion_tokens={getattr(agent, 'last_completion_tokens', 0)} "
            f"tool_attempts={attempted_tools} tool_completed={completed_tools} "
            f"tool_calls_by_name={tool_calls_text} "
            f"tool_failures_by_name={tool_failures_text}",
            flush=True,
        )

    def is_multithread_safe(self) -> bool:
        return True


__all__ = ["DeepResearchWorkflow"]
