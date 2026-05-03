from io import BytesIO
import asyncio
import json
import os
import re
import traceback
from typing import Any, List, Optional

from PIL import Image

from vision_deepresearch_async_workflow.deepresearch_agent import DeepResearchAgent
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
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


def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<response>" in response and "</response>" in response:
        answer = response.split("<response>")[1].split("</response>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<response>.*?</response>|<answer>.*?</answer>)\s*$"
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
            print("[QueryUtilityReward] API gateway not configured, defaulting to 0.0")
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
            print(f"[QueryUtilityReward] score={score:.2f}")
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


def _map_termination_reason(termination: str) -> TerminationReason:
    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "timeout": TerminationReason.UNKNOWN,
        "max_rounds_reached": TerminationReason.UNKNOWN,
        "token_limit_no_answer": TerminationReason.UNKNOWN,
        "answer_token_limit": TerminationReason.UNKNOWN,
        "exceed available llm calls": TerminationReason.UNKNOWN,
        "prompt_budget_reached": TerminationReason.UNKNOWN,
        "max_rounds_reached_no_answer": TerminationReason.UNKNOWN,
        "repetition_detected": TerminationReason.UNKNOWN,  # Will be masked
        "tool_errors_too_many": TerminationReason.UNKNOWN,  # Will be masked
        "consecutive_step_errors": TerminationReason.UNKNOWN,  # Will be masked
        "error": TerminationReason.UNKNOWN,  # Will be masked
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


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
    steps will be masked).  Returns ``(None, "")`` for normal trajectories
    that terminated with an answer and contain no error cascades.

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

    if termination == "answer":
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


class DeepResearchWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict | None = None,
        system_prompt: str | None = None,
        reward_function: RewardFunction | None = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)

        self.tools = tools or {}
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        self.system_prompt = system_prompt
        self.reward_function = reward_function

        self.agent = DeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")

        print(f"🚀 Starting DeepResearch workflow for task {uid}")
        print(f"   Question: {question}")

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
                    **kwargs,
                )
            else:
                result = await self.agent.run(
                    question=question, answer=answer, **kwargs
                )

            episode = self._convert_result_to_episode(result, task, uid)

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
            first_traj = (
                episode.trajectories[0] if episode.trajectories else None
            )
            all_steps = first_traj.steps if first_traj else []

            fatal_step_index, fatal_reason = _find_fatal_step_index(
                all_steps, result.get("termination", "")
            )
            is_fatal = fatal_step_index is not None

            ACCURACY_WEIGHT = 0.8
            QUERY_WEIGHT = 0.2

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
                    "reward/total": 0.0,
                    "masked": 1.0,
                    "is_fatal": 1.0,
                    "fatal_step_index": 0,
                    "num_turns": num_turns,
                }
                episode.info["mask_reason"] = fatal_reason
            else:
                # Compute full reward for group advantage statistics.
                # For fatal trajectories, r_format and r_query are scored
                # only on the learnable prefix (pre-fatal steps).
                r_accuracy = 1.0 if episode.is_correct else 0.0

                scored_steps = (
                    all_steps[:fatal_step_index]
                    if is_fatal and fatal_step_index < len(all_steps)
                    else all_steps
                )
                if scored_steps:
                    fmt_scores = [_format_reward_for_step(s) for s in scored_steps]
                    r_format = sum(fmt_scores) / len(fmt_scores)
                else:
                    r_format = 0.0

                query_steps = (
                    all_steps[:fatal_step_index]
                    if is_fatal and fatal_step_index < len(all_steps)
                    else all_steps
                )
                try:
                    r_query = await _judge_query_utility(
                        question=question,
                        ground_truth=answer,
                        prediction=prediction,
                        steps=query_steps,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[QueryUtilityReward] Exception: {exc}")
                    r_query = 0.0

                total_reward = r_format * (
                    QUERY_WEIGHT * r_query + ACCURACY_WEIGHT * r_accuracy
                )

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
                    "reward/total": total_reward,
                    "masked": 0.0,
                    "is_fatal": 1.0 if is_fatal else 0.0,
                    "fatal_step_index": (
                        fatal_step_index if fatal_step_index is not None else -1
                    ),
                    "num_turns": num_turns,
                }

            print(f"✅ DeepResearch workflow completed for task {uid}")
            print(f"   Prediction: {result.get('prediction', 'No prediction')}")
            print(f"   True Answer: {answer}")
            print(f"   Metrics: {episode.metrics}")
            if episode.info.get("mask_reason"):
                print(f"   Mask Reason: {episode.info['mask_reason']}")
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
                "reward/total": 0.0,
                "masked": 1.0,
            }
            episode.info = {"error": str(exc)}
            return episode

    def _convert_result_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        messages = result.get("messages", [])
        prediction = result.get("prediction", "")
        termination = result.get("termination", "unknown")
        rounds = result.get("rounds", 0)
        time_taken = result.get("time_taken", 0.0)

        trajectories: list[Trajectory] = []
        steps: list[Step] = []

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
                step.model_output = None
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
            "prediction": prediction,
            "answer": answer_text,
            "token_usage": result.get("token_usage", {}),
        }
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        # MultiTurnReactAgent handles per-run state; nothing to reset here.
        return

    def is_multithread_safe(self) -> bool:
        return True


__all__ = ["DeepResearchWorkflow"]
