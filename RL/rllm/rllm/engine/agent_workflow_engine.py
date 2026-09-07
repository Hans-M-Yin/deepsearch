import asyncio
import logging
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from rllm.agents.agent import Episode
from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.utils.multimodal_debug import (
    abort_on_missing_payload,
    count_token,
    event,
    sequence_mode,
)
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)

# Print a lightweight terminal progress summary after every N completed
# rollouts.  This is intentionally local to rollout collection and does not
# affect scheduling, retry, or training behavior.
ROLLOUT_PROGRESS_INTERVAL = 50


def _apply_fatal_step_mask(
    response_mask: torch.Tensor, fatal_step_index: int
) -> torch.Tensor:
    """Zero out tokens for assistant steps >= *fatal_step_index* in a
    cumulative ``response_mask``.

    In cumulative mode the mask alternates between blocks of **1** (assistant
    tokens) and **0** (user / observation tokens).  Each contiguous 1-block
    corresponds to one agent step.  This helper walks through the mask,
    counts 1-blocks, and zeros everything from the *fatal_step_index*-th
    block onward.
    """
    mask = response_mask.clone()
    step = 0
    prev_val = 0
    for t in range(len(mask)):
        val = mask[t].item()
        if val == 1 and prev_val == 0:
            if step >= fatal_step_index:
                mask[t:] = 0
                return mask
        if val == 0 and prev_val == 1:
            step += 1
        prev_val = val
    return mask


def _aggregate_rollout_latency(episodes: list[Episode]) -> dict[str, float]:
    """Aggregate per-trajectory latency samples for trainer/WandB logging.

    The agent records individual LLM/tool call samples in ``Episode.info``.
    Aggregate them here, before the large multimodal payload is converted to
    a DataProto, so latency telemetry does not travel through the training
    tensors or get mixed with reward metrics.
    """

    trajectory_times: list[float] = []
    llm_call_samples: list[float] = []
    tool_call_samples: list[float] = []
    llm_total_times: list[float] = []
    tool_total_times: list[float] = []
    llm_call_counts: list[float] = []
    tool_call_counts: list[float] = []
    tool_samples_by_name: dict[str, list[float]] = defaultdict(list)
    tool_call_counts_by_name: dict[str, list[float]] = defaultdict(list)
    tool_failures_by_name: dict[str, int] = defaultdict(int)
    wikimedia_download_calls = 0
    wikimedia_download_success = 0
    wikimedia_download_cache_success = 0
    wikimedia_download_failure = 0
    wikimedia_download_times: list[float] = []
    llm_failures = 0
    tool_failures = 0
    valid_episodes = 0

    for episode in episodes:
        if episode is None or not isinstance(episode.info, dict):
            continue
        latency = episode.info.get("latency")
        if not isinstance(latency, dict):
            continue

        valid_episodes += 1
        trajectory_time = latency.get("trajectory_time_s")
        if trajectory_time is not None:
            trajectory_times.append(max(0.0, float(trajectory_time)))

        llm = latency.get("llm")
        if isinstance(llm, dict):
            samples = [
                max(0.0, float(value))
                for value in (llm.get("samples_s") or [])
            ]
            llm_call_samples.extend(samples)
            llm_total_times.append(max(0.0, float(llm.get("total_s") or 0.0)))
            llm_call_counts.append(float(llm.get("calls") or 0))
            llm_failures += int(llm.get("failures") or 0)

        tool = latency.get("tool")
        if isinstance(tool, dict):
            tool_total_times.append(max(0.0, float(tool.get("total_s") or 0.0)))
            tool_call_counts.append(float(tool.get("calls") or 0))
            tool_failures += int(tool.get("failures") or 0)

        wikimedia_download = latency.get("wikimedia_download")
        if isinstance(wikimedia_download, dict):
            wikimedia_download_calls += int(wikimedia_download.get("calls") or 0)
            wikimedia_download_success += int(
                wikimedia_download.get("download_success") or 0
            )
            wikimedia_download_cache_success += int(
                wikimedia_download.get("cache_success") or 0
            )
            wikimedia_download_failure += int(wikimedia_download.get("failure") or 0)
            calls = int(wikimedia_download.get("calls") or 0)
            total_s = max(0.0, float(wikimedia_download.get("total_s") or 0.0))
            if calls > 0:
                wikimedia_download_times.extend([total_s / calls] * calls)

        tools = latency.get("tools")
        if not isinstance(tools, dict):
            continue
        for tool_name, tool_summary in tools.items():
            if not isinstance(tool_summary, dict):
                continue
            name = str(tool_name)
            samples = [
                max(0.0, float(value))
                for value in (tool_summary.get("samples_s") or [])
            ]
            tool_samples_by_name[name].extend(samples)
            tool_call_samples.extend(samples)
            tool_call_counts_by_name[name].append(
                float(tool_summary.get("calls") or 0)
            )
            failures = int(tool_summary.get("failures") or 0)
            tool_failures_by_name[name] += failures

    metrics: dict[str, float] = {
        "rollout/latency_episodes": float(valid_episodes),
        "rollout/llm_calls_total": float(len(llm_call_samples)),
        "rollout/llm_failures_total": float(llm_failures),
        "rollout/tool_calls_total": float(len(tool_call_samples)),
        "rollout/tool_failures_total": float(tool_failures),
    }

    def add_distribution(prefix: str, values: list[float]) -> None:
        if not values:
            return
        array = np.asarray(values, dtype=np.float64)
        metrics[f"rollout/{prefix}_mean_s"] = float(np.mean(array))
        metrics[f"rollout/{prefix}_p95_s"] = float(np.percentile(array, 95))
        metrics[f"rollout/{prefix}_max_s"] = float(np.max(array))

    add_distribution("trajectory_time", trajectory_times)
    add_distribution("llm_call_time", llm_call_samples)
    add_distribution("tool_call_time", tool_call_samples)
    add_distribution("llm_total_time", llm_total_times)
    add_distribution("tool_total_time", tool_total_times)
    add_distribution("wikimedia_download_time", wikimedia_download_times)

    if llm_call_counts:
        metrics["rollout/llm_calls_mean"] = float(np.mean(llm_call_counts))
    if tool_call_counts:
        metrics["rollout/tool_calls_mean"] = float(np.mean(tool_call_counts))
    metrics["rollout/wikimedia_download_calls_total"] = float(
        wikimedia_download_calls
    )
    metrics["rollout/wikimedia_download_success_total"] = float(
        wikimedia_download_success
    )
    metrics["rollout/wikimedia_download_cache_success_total"] = float(
        wikimedia_download_cache_success
    )
    metrics["rollout/wikimedia_download_failure_total"] = float(
        wikimedia_download_failure
    )

    for tool_name, samples in sorted(tool_samples_by_name.items()):
        prefix = f"tool/{tool_name}/call_time"
        add_distribution(prefix, samples)
        counts = tool_call_counts_by_name[tool_name]
        if counts:
            metrics[f"rollout/tool/{tool_name}/calls_mean"] = float(np.mean(counts))
        metrics[f"rollout/tool/{tool_name}/calls_total"] = float(sum(counts))
        metrics[f"rollout/tool/{tool_name}/failures_total"] = float(
            tool_failures_by_name[tool_name]
        )

    return metrics


def _expected_image_token_count(
    multimodal_inputs: dict, merge_size: int = 1
) -> int | None:
    """Return image-pad count represented by grid metadata.

    Qwen-VL's ``image_grid_thw`` is expressed before spatial merge on some
    processor versions and after merge on others.  The processor's own
    ``merge_size`` is therefore applied here, matching the feature-token
    calculation used by the Qwen processor smoke test.
    """
    if not isinstance(multimodal_inputs, dict):
        return None
    image_grid_thw = multimodal_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return 0
    try:
        grid = torch.as_tensor(image_grid_thw)
        if grid.numel() == 0:
            return 0
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
        if grid.shape[-1] < 3:
            return None
        merge_size = max(int(merge_size), 1)
        spatial = (grid[..., 1] // merge_size) * (grid[..., 2] // merge_size)
        return int((grid[..., 0] * spatial).sum().item())
    except Exception:  # noqa: BLE001
        return None


def _validate_multimodal_alignment(
    *,
    input_ids: torch.Tensor,
    multimodal_inputs: dict,
    image_token_id: int | None,
    merge_size: int,
    row_idx: int,
) -> None:
    """Fail fast if expanded image ids and visual features disagree.

    The legacy path intentionally does not call this check.  In either new
    mode, silently replacing visual embeddings with text embeddings is worse
    than stopping with an actionable error.
    """
    if image_token_id is None:
        return
    observed = count_token(input_ids, image_token_id)
    expected = _expected_image_token_count(multimodal_inputs, merge_size)
    if observed is None or expected is None:
        return
    has_pixel_values = (
        isinstance(multimodal_inputs, dict)
        and multimodal_inputs.get("pixel_values") is not None
    )
    if expected > 0 and not has_pixel_values:
        raise RuntimeError(
            "[RLLM_MM_SEQUENCE] visual features are missing after sequence "
            f"construction at row={row_idx}: observed_image_tokens={observed}, "
            f"expected_image_tokens={expected}, keys={sorted(multimodal_inputs)}"
        )
    if observed != expected:
        raise RuntimeError(
            "[RLLM_MM_SEQUENCE] image token/payload mismatch after sequence "
            f"construction at row={row_idx}: observed_image_tokens={observed}, "
            f"expected_image_tokens={expected}, keys={sorted(multimodal_inputs)}"
        )


class AgentWorkflowEngine:
    def __init__(
        self,
        workflow_cls: type[Workflow],
        workflow_args: dict,
        rollout_engine: RolloutEngine,
        config=None,
        n_parallel_tasks: int = 128,
        n_parallel_tools: int | None = None,
        retry_limit: int = 3,
        raise_on_error: bool = True,
        episode_logger=None,
        **kwargs,
    ):
        """Initialize the AgentWorkflowEngine.

        Args:
            workflow_cls: The workflow class to instantiate for each task.
            workflow_args: Arguments to pass to workflow instances.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            n_parallel_tools: Number of tool worker threads; defaults to n_parallel_tasks * 8.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            episode_logger: Optional logger for saving episode data to files.
            **kwargs: Additional keyword arguments.
        """
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args or {}

        self.rollout_engine = rollout_engine
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.n_parallel_tools = (
            n_parallel_tools
            if n_parallel_tools is not None
            else self.n_parallel_tasks * 8
        )
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tools)
        self.workflow_queue = None

        # Episode logging support
        self.episode_logger = episode_logger
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"  # "train" or "val"

        # Stores the raw Episode list from the most recent execute_tasks call
        self._last_episodes: list[Episode] = []

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        """Set current training step for episode logging.

        Args:
            step: Current training step number
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(
                rollout_engine=self.rollout_engine,
                executor=self.executor,
                **self.workflow_args,
            )
            assert (
                workflow.is_multithread_safe()
            ), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(
        self, task: dict, task_id: str, rollout_idx: int, **kwargs
    ) -> tuple[str, int, Episode]:
        """Process a single task rollout with retry logic based on termination reasons.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            **kwargs: Additional arguments passed to the workflow.

        Returns:
            tuple[str, int, Episode]: Task ID, rollout index, and completed episode.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        workflow = await self.workflow_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                episode = await workflow.run_with_termination_handling(
                    task=task, uid=uid, **kwargs
                )

                # Temporarily suppress one terminal line per rollout.  The
                # every-50-rollout progress summary retains aggregate reward,
                # termination, and timing information without producing
                # high-frequency stdout traffic.

                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, episode

                error_tb = episode.info.get("error", {}).get("traceback")
                if error_tb:
                    print(error_tb)

                if retry_attempt < self.retry_limit:
                    print(
                        f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying..."
                    )
                    continue

            if not self.raise_on_error:
                print(
                    f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts."
                )
            else:
                raise Exception(
                    f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts."
                )

            return task_id, rollout_idx, episode

        finally:
            await self.workflow_queue.put(workflow)

    async def execute_tasks(
        self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs
    ) -> list[Episode]:
        """Run asynchronous workflow execution with retry logic for multiple tasks.

        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers. If None, UUIDs are generated.
            **kwargs: Additional arguments passed to individual task processing.

        Returns:
            list[Episode]: List of completed episodes from all tasks.
        """
        if self.workflow_queue is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        # Optional local callback used by the RL trainer to emit a compact
        # live W&B record while one long logical training step is still
        # collecting rollouts.  It is deliberately removed here so it is not
        # passed into individual workflow/tool calls.
        progress_callback = kwargs.pop("progress_callback", None)

        # ``tool_cache`` is an optional RL-only epoch cache.  Resolve one
        # sample-scoped cache for each task before launching its sibling
        # rollouts; inference callers simply do not pass this argument.
        tool_cache = kwargs.pop("tool_cache", None)

        task_states = defaultdict(
            lambda: {
                "idx": None,
                "task": None,
                "episodes": [],
                "completed": 0,
                "total_rollouts": 0,
                "is_complete": False,
            }
        )

        futures = []
        idx_counter = 0
        for task, task_id in zip(tasks, task_ids, strict=True):
            state = task_states[task_id]
            if state["idx"] is None:  # First time seeing this task_id
                state["idx"] = idx_counter
                state["task"] = task
                idx_counter += 1
            rollout_idx = state["total_rollouts"]
            rollout_kwargs = dict(kwargs)
            if tool_cache is not None:
                if hasattr(tool_cache, "for_sample"):
                    rollout_kwargs["tool_cache"] = tool_cache.for_sample(
                        task, task_id
                    )
                else:
                    rollout_kwargs["tool_cache"] = tool_cache
            futures.append(
                self.process_task_with_retry(
                    task, task_id, rollout_idx, **rollout_kwargs
                )
            )
            state["total_rollouts"] += 1

        completed_count = 0
        correct_count = 0
        trajectory_time_sum = 0.0
        trajectory_time_count = 0
        termination_counts: dict[str, int] = defaultdict(int)
        raw_termination_counts: dict[str, int] = defaultdict(int)
        marker_counts: dict[str, int] = defaultdict(int)
        llm_total_s = 0.0
        llm_call_count = 0
        tool_total_s = 0.0
        tool_call_count = 0
        tool_timing: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_s": 0.0, "calls": 0.0}
        )
        tool_dispatch_timing: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_s": 0.0, "calls": 0.0}
        )
        read_url_stage_timing: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_s": 0.0, "calls": 0.0}
        )
        wikimedia_download_stats: dict[str, float] = defaultdict(float)

        # Windowed counters are reset after every progress report. The window
        # is over completed rollouts (not individual LLM/tool calls), which
        # makes it useful for spotting latency drift and long-tail batches.
        recent_trajectory_time_sum = 0.0
        recent_trajectory_time_count = 0
        recent_llm_total_s = 0.0
        recent_llm_call_count = 0
        recent_tool_total_s = 0.0
        recent_tool_call_count = 0
        recent_tool_timing: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_s": 0.0, "calls": 0.0}
        )
        recent_tool_dispatch_timing: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_s": 0.0, "calls": 0.0}
        )
        recent_read_url_stage_timing: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_s": 0.0, "calls": 0.0}
        )
        recent_wikimedia_download_stats: dict[str, float] = defaultdict(float)

        def add_timing(
            target: dict[str, dict[str, float]],
            name: str,
            total_s: float,
            calls: int,
        ) -> None:
            if calls <= 0:
                return
            target[str(name)]["total_s"] += max(0.0, float(total_s))
            target[str(name)]["calls"] += int(calls)

        def format_named_timing(
            timing: dict[str, dict[str, float]],
        ) -> str:
            return ";".join(
                f"{name}(avg_s={values['total_s'] / values['calls']:.2f},"
                f"total_s={values['total_s']:.1f},calls={int(values['calls'])})"
                for name, values in sorted(timing.items())
                if values["calls"] > 0
            ) or "-"

        def format_read_url_stage_timing(
            timing: dict[str, dict[str, float]],
        ) -> str:
            stage_names = (
                "enhanced_reader",
                "http_download",
                "firecrawl_scrape",
                "wikimedia_download",
            )
            return ";".join(
                f"{name}(avg_s={(timing[name]['total_s'] / timing[name]['calls']) if timing[name]['calls'] else 0.0:.2f},"
                f"total_s={timing[name]['total_s']:.1f},calls={int(timing[name]['calls'])})"
                for name in stage_names
            )

        def format_wikimedia_download_stats(stats: dict[str, float]) -> str:
            calls = int(stats.get("calls", 0.0))
            total_s = float(stats.get("total_s", 0.0))
            avg_s = total_s / calls if calls else 0.0
            return (
                f"download_success={int(stats.get('download_success', 0.0))},"
                f"cache_success={int(stats.get('cache_success', 0.0))},"
                f"failure={int(stats.get('failure', 0.0))},"
                f"avg_s={avg_s:.2f},total_s={total_s:.1f},calls={calls}"
            )
        progress_started_at = time.monotonic()

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, episode = await future

                state = task_states[task_id]
                state["episodes"].append(episode)
                state["completed"] += 1
                completed_count += 1
                correct_count += int(bool(getattr(episode, "is_correct", False)))

                termination = getattr(episode, "termination_reason", None)
                termination_name = getattr(termination, "value", None) or str(
                    termination or "unknown"
                )
                termination_counts[termination_name] += 1

                episode_info = getattr(episode, "info", None)
                if not isinstance(episode_info, dict):
                    episode_info = {}

                raw_termination = episode_info.get("raw_termination")
                if raw_termination:
                    raw_termination_counts[str(raw_termination)] += 1

                markers = episode_info.get("rollout_markers", [])
                if isinstance(markers, (list, tuple, set)):
                    for marker in markers:
                        marker_counts[str(marker)] += 1

                latency = episode_info.get("latency")
                if isinstance(latency, dict):
                    llm_summary = latency.get("llm")
                    if isinstance(llm_summary, dict):
                        llm_samples = [
                            max(0.0, float(value))
                            for value in (llm_summary.get("samples_s") or [])
                        ]
                        llm_total_s += float(
                            llm_summary.get("total_s") or sum(llm_samples)
                        )
                        llm_calls = int(
                            llm_summary.get("calls") or len(llm_samples)
                        )
                        llm_call_count += llm_calls
                        recent_llm_total_s += float(
                            llm_summary.get("total_s") or sum(llm_samples)
                        )
                        recent_llm_call_count += llm_calls

                    tool_summary = latency.get("tool")
                    if isinstance(tool_summary, dict):
                        tool_samples = [
                            max(0.0, float(value))
                            for value in (tool_summary.get("samples_s") or [])
                        ]
                        tool_total = float(
                            tool_summary.get("total_s") or sum(tool_samples)
                        )
                        tool_calls = int(
                            tool_summary.get("calls") or len(tool_samples)
                        )
                        tool_total_s += tool_total
                        tool_call_count += tool_calls
                        recent_tool_total_s += tool_total
                        recent_tool_call_count += tool_calls

                    tool_summaries = latency.get("tools")
                    if isinstance(tool_summaries, dict):
                        for tool_name, summary in tool_summaries.items():
                            if not isinstance(summary, dict):
                                continue
                            samples = [
                                max(0.0, float(value))
                                for value in (summary.get("samples_s") or [])
                            ]
                            tool_total = float(
                                summary.get("total_s") or sum(samples)
                            )
                            tool_calls = int(
                                summary.get("calls") or len(samples)
                            )
                            add_timing(tool_timing, str(tool_name), tool_total, tool_calls)
                            add_timing(
                                recent_tool_timing,
                                str(tool_name),
                                tool_total,
                                tool_calls,
                            )

                    tool_dispatch_summaries = latency.get("tool_dispatch")
                    if isinstance(tool_dispatch_summaries, dict):
                        for metric_name, summary in tool_dispatch_summaries.items():
                            if not isinstance(summary, dict):
                                continue
                            dispatch_total = float(summary.get("total_s") or 0.0)
                            dispatch_calls = int(summary.get("calls") or 0)
                            add_timing(
                                tool_dispatch_timing,
                                str(metric_name),
                                dispatch_total,
                                dispatch_calls,
                            )
                            add_timing(
                                recent_tool_dispatch_timing,
                                str(metric_name),
                                dispatch_total,
                                dispatch_calls,
                            )

                    wikimedia_download = latency.get("wikimedia_download")
                    if isinstance(wikimedia_download, dict):
                        for key in (
                            "calls",
                            "download_success",
                            "cache_success",
                            "failure",
                            "total_s",
                        ):
                            try:
                                value = max(
                                    0.0,
                                    float(wikimedia_download.get(key) or 0.0),
                                )
                            except (TypeError, ValueError):
                                continue
                            wikimedia_download_stats[key] += value
                            recent_wikimedia_download_stats[key] += value

                    read_url_stages = latency.get("read_url_stages")
                    if isinstance(read_url_stages, dict):
                        for stage_name, summary in read_url_stages.items():
                            if not isinstance(summary, dict):
                                continue
                            stage_total = float(summary.get("total_s") or 0.0)
                            stage_calls = int(summary.get("calls") or 0)
                            add_timing(
                                read_url_stage_timing,
                                str(stage_name),
                                stage_total,
                                stage_calls,
                            )
                            add_timing(
                                recent_read_url_stage_timing,
                                str(stage_name),
                                stage_total,
                                stage_calls,
                            )

                trajectory_time = (
                    latency.get("trajectory_time_s")
                    if isinstance(latency, dict)
                    else None
                )
                if trajectory_time is not None:
                    try:
                        trajectory_time_sum += max(0.0, float(trajectory_time))
                        trajectory_time_count += 1
                        recent_trajectory_time_sum += max(0.0, float(trajectory_time))
                        recent_trajectory_time_count += 1
                    except (TypeError, ValueError):
                        pass

                pbar.update(1)

                if completed_count % ROLLOUT_PROGRESS_INTERVAL == 0:
                    elapsed = max(time.monotonic() - progress_started_at, 1e-6)
                    rate = completed_count / elapsed
                    termination_summary = ",".join(
                        f"{name}:{count}"
                        for name, count in sorted(termination_counts.items())
                    )
                    raw_termination_summary = ",".join(
                        f"{name}:{count}"
                        for name, count in sorted(raw_termination_counts.items())
                    ) or "-"
                    marker_summary = ",".join(
                        f"{name}:{count}"
                        for name, count in sorted(marker_counts.items())
                    ) or "-"
                    tool_timing_summary = format_named_timing(tool_timing)
                    recent_tool_timing_summary = format_named_timing(recent_tool_timing)
                    tool_dispatch_summary = format_named_timing(tool_dispatch_timing)
                    recent_tool_dispatch_summary = format_named_timing(
                        recent_tool_dispatch_timing
                    )
                    avg_trajectory_time = (
                        trajectory_time_sum / trajectory_time_count
                        if trajectory_time_count
                        else 0.0
                    )
                    recent_avg_trajectory_time = (
                        recent_trajectory_time_sum / recent_trajectory_time_count
                        if recent_trajectory_time_count
                        else 0.0
                    )
                    llm_avg_s = llm_total_s / llm_call_count if llm_call_count else 0.0
                    tool_avg_s = tool_total_s / tool_call_count if tool_call_count else 0.0
                    recent_llm_avg_s = (
                        recent_llm_total_s / recent_llm_call_count
                        if recent_llm_call_count
                        else 0.0
                    )
                    recent_tool_avg_s = (
                        recent_tool_total_s / recent_tool_call_count
                        if recent_tool_call_count
                        else 0.0
                    )
                    tqdm.write(
                        "[RL_ROLLOUT_PROGRESS] "
                        f"completed={completed_count}/{len(tasks)} "
                        f"correct={correct_count} "
                        f"elapsed_s={elapsed:.1f} "
                        f"rate={rate:.2f}/s "
                        f"avg_trajectory_s={avg_trajectory_time:.1f} "
                        f"recent50_avg_trajectory_s={recent_avg_trajectory_time:.1f} "
                        f"termination={termination_summary} "
                        f"raw_termination={raw_termination_summary} "
                        f"markers={marker_summary} "
                        f"llm_request(avg_s={llm_avg_s:.2f},recent50_avg_s={recent_llm_avg_s:.2f},"
                        f"total_s={llm_total_s:.1f},calls={llm_call_count};"
                        f"recent50_total_s={recent_llm_total_s:.1f},recent50_calls={recent_llm_call_count}) "
                        f"tools(avg_s={tool_avg_s:.2f},recent50_avg_s={recent_tool_avg_s:.2f},"
                        f"total_s={tool_total_s:.1f},calls={tool_call_count};"
                        f"by_name={tool_timing_summary};recent50_by_name={recent_tool_timing_summary}) "
                        f"tool_dispatch(overall={tool_dispatch_summary};"
                        f"recent50={recent_tool_dispatch_summary}) "
                        f"read_url_stages(overall={format_read_url_stage_timing(read_url_stage_timing)};"
                        f"recent50={format_read_url_stage_timing(recent_read_url_stage_timing)}) "
                        f"wikimedia_download(overall={format_wikimedia_download_stats(wikimedia_download_stats)};"
                        f"recent50={format_wikimedia_download_stats(recent_wikimedia_download_stats)})"
                    )
                    sys.stdout.flush()

                    if progress_callback is not None:
                        live_metrics = {
                            "live/rollouts_completed": completed_count,
                            "live/rollouts_total": len(tasks),
                            "live/rollouts_correct": correct_count,
                            "live/elapsed_s": elapsed,
                            "live/rate_per_s": rate,
                            "live/trajectory_avg_s": avg_trajectory_time,
                            "live/trajectory_recent50_avg_s": recent_avg_trajectory_time,
                            "live/llm_avg_s": llm_avg_s,
                            "live/llm_recent50_avg_s": recent_llm_avg_s,
                            "live/llm_total_s": llm_total_s,
                            "live/llm_calls": llm_call_count,
                            "live/tool_avg_s": tool_avg_s,
                            "live/tool_recent50_avg_s": recent_tool_avg_s,
                            "live/tool_total_s": tool_total_s,
                            "live/tool_calls": tool_call_count,
                            "live/wikimedia_download/download_success": wikimedia_download_stats.get(
                                "download_success", 0.0
                            ),
                            "live/wikimedia_download/cache_success": wikimedia_download_stats.get(
                                "cache_success", 0.0
                            ),
                            "live/wikimedia_download/failure": wikimedia_download_stats.get(
                                "failure", 0.0
                            ),
                            "live/wikimedia_download/calls": wikimedia_download_stats.get(
                                "calls", 0.0
                            ),
                            "live/wikimedia_download/total_s": wikimedia_download_stats.get(
                                "total_s", 0.0
                            ),
                            "live/wikimedia_download/recent50_download_success": recent_wikimedia_download_stats.get(
                                "download_success", 0.0
                            ),
                            "live/wikimedia_download/recent50_cache_success": recent_wikimedia_download_stats.get(
                                "cache_success", 0.0
                            ),
                            "live/wikimedia_download/recent50_failure": recent_wikimedia_download_stats.get(
                                "failure", 0.0
                            ),
                            "live/wikimedia_download/recent50_calls": recent_wikimedia_download_stats.get(
                                "calls", 0.0
                            ),
                            "live/wikimedia_download/recent50_total_s": recent_wikimedia_download_stats.get(
                                "total_s", 0.0
                            ),
                        }
                        for stage_name, stage_values in read_url_stage_timing.items():
                            stage_prefix = f"live/read_url/{stage_name}"
                            stage_calls = int(stage_values["calls"])
                            live_metrics[f"{stage_prefix}/total_s"] = stage_values[
                                "total_s"
                            ]
                            live_metrics[f"{stage_prefix}/calls"] = stage_calls
                            live_metrics[f"{stage_prefix}/avg_s"] = (
                                stage_values["total_s"] / stage_calls
                                if stage_calls
                                else 0.0
                            )
                        try:
                            progress_callback(live_metrics, self.current_step)
                        except Exception as exc:
                            # Monitoring must never turn into a rollout error.
                            logger.warning(
                                "Live rollout logging failed at completed=%s: %s",
                                completed_count,
                                exc,
                            )

                    # Start the next completed-rollout window.
                    recent_trajectory_time_sum = 0.0
                    recent_trajectory_time_count = 0
                    recent_llm_total_s = 0.0
                    recent_llm_call_count = 0
                    recent_tool_total_s = 0.0
                    recent_tool_call_count = 0
                    recent_tool_timing.clear()
                    recent_tool_dispatch_timing.clear()
                    recent_read_url_stage_timing.clear()
                    recent_wikimedia_download_stats.clear()

        results = []
        sorted_tasks = sorted(
            task_states.keys(), key=lambda task_id: task_states[task_id]["idx"]
        )
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])

        # Keep a reference so the trainer can inspect raw episodes
        self._last_episodes = results

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(
                    f"Logging {len(results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}"
                )
                self.episode_logger.log_episodes_batch(
                    results, self.current_step, self.current_mode, self.current_epoch
                )
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")
                import traceback

                traceback.print_exc()

        return results

    async def execute_tasks_verl(self, batch: "DataProto", **kwargs) -> "DataProto":
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
        await self.rollout_engine.wake_up()

        is_validation = batch.meta_info.get("validate", False)
        if is_validation:
            self.rollout_engine.validate = True
            self.current_mode = "val"
        else:
            self.current_mode = "train"
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        results = await self.execute_tasks(
            tasks, task_ids, **kwargs
        )  # list of Episodes
        self.rollout_engine.validate = False

        await self.rollout_engine.sleep()

        self.current_mode = "train"
        transformed = self.transform_results_for_verl(results, task_ids)
        transformed.meta_info["rollout_latency"] = _aggregate_rollout_latency(results)
        return transformed

    def transform_results_for_verl(
        self, episodes: list[Episode], task_ids: np.ndarray
    ) -> "DataProto":
        """Transform episode results into Verl-compatible DataProto format.

        Args:
            episodes: List of completed episodes from workflow execution.
            task_ids: Array of task identifiers corresponding to episodes.

        Returns:
            DataProto: Formatted data ready for Verl training pipeline.
        """
        # Local import to keep verl optional
        from verl import DataProto
        from verl.utils.torch_functional import pad_sequence_to_length

        prompts = []
        responses = []
        traj_rewards = []
        step_rewards = []
        episode_ids = []
        trajectory_ids = []
        step_ids = []
        step_nums = []
        repeat_counts = []
        is_last_step = []
        is_correct = []
        traj_mask = []
        termination_reasons = []
        rollout_markers = []
        metrics = []
        multi_modal_inputs_list = []
        is_fatal_flags = []
        fatal_step_indices = []

        multimodal_mode = sequence_mode()
        configured_stepwise = bool(self.config.rllm.stepwise_advantage.enable)
        if multimodal_mode == "stepwise":
            stepwise_enabled = True
        elif multimodal_mode == "cumulative":
            stepwise_enabled = False
        else:
            stepwise_enabled = configured_stepwise

        if multimodal_mode != "legacy":
            logger.info(
                "RLLM multimodal sequence mode=%s (configured stepwise=%s, effective stepwise=%s)",
                multimodal_mode,
                configured_stepwise,
                stepwise_enabled,
            )

        for i, episode in enumerate(episodes):
            total_steps = 0

            if episode is None:
                print(f"Episode {i} is None (failed task), dropping it from the batch")
                repeat_counts.append(0)
                continue

            if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
                # termination hits before an agent finishes it's first step
                # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
                # we delete the episode from the batch by setting repeat_counts to 0
                print(
                    f"Episode {episode.id} has no valid trajectories, dropping it from the batch"
                )
                repeat_counts.append(0)
                continue

            ep_info = episode.info or {}
            ep_is_fatal = ep_info.get("is_fatal", False)
            ep_fatal_step_idx = ep_info.get("fatal_step_index")

            for trajectory in episode.trajectories:
                name = trajectory.name
                trajectory_id = f"{task_ids[i]}_{name}"  # unique trajectory identifier e.g., 1234567890_solver

                if len(trajectory.steps) == 0:
                    logger.info(f"Trajectory {trajectory_id} has no steps, skipping")
                    continue

                if not stepwise_enabled:
                    if len(trajectory.steps) > 1:
                        if not trajectory.is_cumulative():
                            logger.warning(
                                f"Warning: Multi-step trajectory {trajectory_id} is not cumulative, but stepwise mode is not enabled. There could be a token mismatch during trajectory generation."
                            )

                        chat_completions = trajectory.steps[-1].chat_completions
                        if multimodal_mode == "cumulative":
                            (
                                prompt,
                                response,
                                mask,
                                multimodal_inputs,
                            ) = self.rollout_engine.chat_parser.tokenize_and_mask_cumulative_multimodal(
                                chat_completions,
                                model_output=trajectory.steps[-1].model_output,
                            )
                        else:
                            prompt, response, mask = (
                                self.rollout_engine.chat_parser.tokenize_and_mask_cumulative(
                                    chat_completions
                                )
                            )
                            multimodal_inputs = {}
                        if ep_is_fatal and ep_fatal_step_idx is not None:
                            mask = _apply_fatal_step_mask(mask, ep_fatal_step_idx)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append(multimodal_inputs)

                    elif multimodal_mode == "cumulative":
                        (
                            prompt,
                            response,
                            mask,
                            multimodal_inputs,
                        ) = self.rollout_engine.chat_parser.tokenize_and_mask_cumulative_multimodal(
                            trajectory.steps[0].chat_completions,
                            model_output=trajectory.steps[0].model_output,
                        )
                        if ep_is_fatal and ep_fatal_step_idx == 0:
                            mask = torch.zeros_like(mask)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append(multimodal_inputs)

                    elif isinstance(trajectory.steps[0].model_output, ModelOutput):
                        step = trajectory.steps[0]

                        prompt_ids = torch.tensor(
                            step.model_output.prompt_ids, dtype=torch.long
                        )
                        prompts.append(prompt_ids)

                        response_ids = torch.tensor(
                            step.model_output.completion_ids, dtype=torch.long
                        )
                        responses.append(response_ids)

                        mask = torch.ones_like(response_ids, dtype=torch.long)
                        if ep_is_fatal and ep_fatal_step_idx == 0:
                            mask = torch.zeros_like(response_ids, dtype=torch.long)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append(
                            step.model_output.multi_modal_inputs or {}
                        )

                    else:
                        chat_completions = trajectory.steps[0].chat_completions
                        prompt, response, mask = (
                            self.rollout_engine.chat_parser.tokenize_and_mask(
                                chat_completions
                            )
                        )
                        if ep_is_fatal and ep_fatal_step_idx == 0:
                            mask = torch.zeros_like(mask)
                        prompts.append(prompt)
                        responses.append(response)
                        traj_mask.append(mask)
                        multi_modal_inputs_list.append({})  # empty dict

                    step_rewards.append(trajectory.reward)
                    step_ids.append(trajectory_id)
                    n_steps = 1

                else:
                    for step_idx, step in enumerate(trajectory.steps):
                        _mask_this_step = (
                            ep_is_fatal
                            and ep_fatal_step_idx is not None
                            and step_idx >= ep_fatal_step_idx
                        )

                        if isinstance(step.model_output, ModelOutput):
                            prompt_ids = torch.tensor(
                                step.model_output.prompt_ids, dtype=torch.long
                            )
                            prompts.append(prompt_ids)

                            response_ids = torch.tensor(
                                step.model_output.completion_ids, dtype=torch.long
                            )
                            responses.append(response_ids)

                            mask = torch.ones_like(response_ids, dtype=torch.long)
                            if _mask_this_step:
                                mask = torch.zeros_like(response_ids, dtype=torch.long)
                            traj_mask.append(mask)
                            multi_modal_inputs_list.append(
                                step.model_output.multi_modal_inputs or {}
                            )

                        else:
                            chat_completions = step.chat_completions
                            prompt, response, mask = (
                                self.rollout_engine.chat_parser.tokenize_and_mask(
                                    chat_completions
                                )
                            )
                            if _mask_this_step:
                                mask = torch.zeros_like(mask)
                            prompts.append(prompt)
                            responses.append(response)
                            traj_mask.append(mask)
                            multi_modal_inputs_list.append({})  # empty dict

                        step_rewards.append(step.reward)
                        step_ids.append(
                            f"{trajectory_id}_step{step_idx}"
                        )  # unique step identifier e.g., 1234567890_solver_step0

                    n_steps = len(trajectory.steps)

                trajectory_ids.extend([trajectory_id] * n_steps)
                step_nums.extend([n_steps] * n_steps)
                traj_rewards.extend([trajectory.reward] * n_steps)
                is_last_step.extend([False] * n_steps)
                is_last_step[-1] = True
                is_fatal_flags.extend([ep_is_fatal] * n_steps)
                fatal_step_indices.extend(
                    [ep_fatal_step_idx if ep_fatal_step_idx is not None else -1]
                    * n_steps
                )
                total_steps += n_steps

            episode_ids.extend([episode.id] * total_steps)
            is_correct.extend([episode.is_correct] * total_steps)
            termination_reasons.extend(
                [
                    (
                        episode.termination_reason
                        if episode.termination_reason is not None
                        else TerminationReason.UNKNOWN
                    )
                ]
                * total_steps
            )
            metrics.extend([episode.metrics] * total_steps)
            episode_markers = ep_info.get("rollout_markers", [])
            if isinstance(episode_markers, (list, tuple, set)):
                episode_markers = ",".join(
                    sorted({str(marker) for marker in episode_markers})
                )
            else:
                episode_markers = str(episode_markers or "")
            rollout_markers.extend([episode_markers] * total_steps)
            repeat_counts.append(total_steps)

        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in prompts],
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        ).flip(dims=[1])
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = pad_sequence_to_length(
            prompts_batch,
            max_prompt_length,
            self.rollout_engine.tokenizer.pad_token_id,
            left_pad=True,
        )
        prompts_batch = prompts_batch[:, -max_prompt_length:]  # truncate if necessary

        response_batch = torch.nn.utils.rnn.pad_sequence(
            responses,
            batch_first=True,
            padding_value=self.rollout_engine.tokenizer.pad_token_id,
        )
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(
            response_batch,
            max_response_length,
            self.rollout_engine.tokenizer.pad_token_id,
            left_pad=False,
        )
        response_batch = response_batch[
            :, :max_response_length
        ]  # truncate if necessary

        input_ids = torch.concat([prompts_batch, response_batch], dim=1)

        # This is the boundary between workflow rollout and policy/ref-policy
        # computation.  Log only metadata: never serialize image tensors.
        image_token_id = getattr(self.rollout_engine.processor, "image_token_id", None)
        if image_token_id is None and self.rollout_engine.processor is not None:
            image_token_id = self.rollout_engine.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        merge_size = int(
            getattr(
                getattr(self.rollout_engine.processor, "image_processor", None),
                "merge_size",
                1,
            )
        )
        for row_idx, mm_inputs in enumerate(multi_modal_inputs_list):
            mm_keys = sorted(mm_inputs.keys()) if isinstance(mm_inputs, dict) else []
            has_visual_payload = isinstance(mm_inputs, dict) and (
                mm_inputs.get("pixel_values") is not None
                or mm_inputs.get("image_grid_thw") is not None
            )
            event(
                "trajectory_to_batch",
                row_idx=row_idx,
                input_ids_shape=getattr(input_ids, "shape", None),
                image_token_count=count_token(input_ids[row_idx], image_token_id),
                multimodal_keys=mm_keys,
                has_visual_payload=has_visual_payload,
                pixel_values_shape=getattr(
                    mm_inputs.get("pixel_values") if isinstance(mm_inputs, dict) else None,
                    "shape",
                    None,
                ),
                image_grid_thw_shape=getattr(
                    mm_inputs.get("image_grid_thw") if isinstance(mm_inputs, dict) else None,
                    "shape",
                    None,
                ),
                expected_image_token_count=_expected_image_token_count(
                    mm_inputs, merge_size
                ),
                sequence_mode=multimodal_mode,
                stepwise_enabled=stepwise_enabled,
            )
            image_token_count = count_token(input_ids[row_idx], image_token_id)
            if (
                abort_on_missing_payload()
                and image_token_count
                and not has_visual_payload
            ):
                raise RuntimeError(
                    "[RLLM_MM_DEBUG] multimodal payload is missing at "
                    f"trajectory_to_batch row={row_idx}: "
                    f"image_token_count={image_token_count}, keys={mm_keys}"
                )

            if multimodal_mode in {"stepwise", "cumulative"}:
                _validate_multimodal_alignment(
                    input_ids=input_ids[row_idx],
                    multimodal_inputs=mm_inputs,
                    image_token_id=image_token_id,
                    merge_size=merge_size,
                    row_idx=row_idx,
                )

        prompt_lengths = torch.as_tensor([len(t) for t in prompts]).clamp_(
            min=0, max=max_prompt_length
        )
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in responses]).clamp_(
            min=0, max=max_response_length
        )
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        if (
            hasattr(self.rollout_engine, "processor")
            and self.rollout_engine.processor is not None
        ):
            position_ids = self._handle_multimodal_position_ids(
                processor=self.rollout_engine.processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs_list,
            )
        else:
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        traj_mask = torch.nn.utils.rnn.pad_sequence(
            traj_mask, batch_first=True, padding_value=0
        )
        traj_mask = pad_sequence_to_length(
            traj_mask, max_response_length, 0, left_pad=False
        )
        traj_mask = traj_mask[:, :max_response_length]  # truncate if necessary

        # Place all rewards to last response token of the last_step response
        traj_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        step_rewards_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, (traj_reward, step_reward) in enumerate(
            zip(traj_rewards, step_rewards, strict=False)
        ):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= traj_rewards_batch.shape[1]:
                traj_rewards_batch[i, resp_len - 1] = traj_reward
                step_rewards_batch[i, resp_len - 1] = step_reward

        # compact filtering
        cf = self.config.rllm.compact_filtering
        is_valid = [True] * len(episode_ids)
        if cf.enable:
            for i in range(len(episode_ids)):
                termination_reason = termination_reasons[i]
                if (
                    (
                        cf.mask_max_prompt_length_exceeded
                        and termination_reason
                        == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
                    )
                    or (
                        cf.mask_max_response_length_exceeded
                        and termination_reason
                        == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
                    )
                    or (
                        cf.mask_env_done
                        and termination_reason == TerminationReason.ENV_DONE
                    )
                    or (
                        cf.mask_max_turns_exceeded
                        and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED
                    )
                    or (
                        cf.mask_timeout
                        and termination_reason == TerminationReason.TIMEOUT
                    )
                    or (
                        cf.mask_unknown
                        and termination_reason == TerminationReason.UNKNOWN
                    )
                    or (cf.mask_error and termination_reason == TerminationReason.ERROR)
                ):
                    is_valid[i] = (
                        False  # set flag to filter out the episode later (after advantages are computed)
                    )

        # Override: keep fatal trajectories that have a recoverable prefix.
        # Their response_mask already zeros post-fatal tokens; the trainer
        # will apply advantage clamping.
        for i in range(len(is_valid)):
            if (
                not is_valid[i]
                and i < len(is_fatal_flags)
                and is_fatal_flags[i]
                and fatal_step_indices[i] > 0
            ):
                is_valid[i] = True

        non_tensors = {
            "episode_ids": np.array(episode_ids),  # unique identifier for each rollout
            "trajectory_ids": np.array(
                trajectory_ids
            ),  # unique identifier for each trajectory (shares prefix with task_id) and shared across rollouts
            "step_ids": np.array(
                step_ids
            ),  # unique identifier for each step (shares prefix with task_id) and shared across rollouts
            "batch_ids": np.array(
                [str(uuid.uuid4())] * len(episode_ids)
            ),  # unique identifier for each batch
            "step_nums": np.array(step_nums),
            "is_correct": np.array(is_correct),
            "termination_reasons": np.array([x.value for x in termination_reasons]),
            "rollout_markers": np.array(rollout_markers, dtype=object),
            "metrics": np.array(metrics),
            "is_valid": np.array(is_valid),
            "is_last_step": np.array(is_last_step),
            "is_pad_step": np.array([False] * len(episode_ids)),
            "is_fatal": np.array(is_fatal_flags) if is_fatal_flags else np.array([False] * len(episode_ids)),
        }

        if any(mm_inputs is not None for mm_inputs in multi_modal_inputs_list):
            non_tensors["multi_modal_inputs"] = np.array(
                multi_modal_inputs_list, dtype=object
            )

        # The DataProto now owns the processor payload.  Do not keep another
        # reference to large prompt/image tensors through _last_episodes or
        # trajectory dumps after conversion.
        if multimodal_mode in {"stepwise", "cumulative"}:
            for episode in episodes:
                if episode is None:
                    continue
                for trajectory in episode.trajectories:
                    for step in trajectory.steps:
                        step.model_output = None

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_batch,
                "responses": response_batch,
                "response_mask": traj_mask,
                "traj_rewards": traj_rewards_batch,
                "step_rewards": step_rewards_batch,
            },
            non_tensors=non_tensors,
            meta_info={
                "repeat_counts": repeat_counts,
            },
        )

    def _handle_multimodal_position_ids(
        self,
        processor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: list[dict],
    ) -> torch.Tensor:
        """Handle multimodal position ids calculation. Borrowed from verl.utils.dataset.rl_dataset.py"""
        batch_size = input_ids.shape[0]
        position_ids_list = []

        if (
            processor is not None
            and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
        ):
            # qwen-vl mrope
            if "Qwen3VLProcessor" in processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            for i in range(batch_size):
                model_inputs = (
                    multi_modal_inputs[i] if i < len(multi_modal_inputs) else {}
                )
                vision_position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids[i],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[i],
                )  # (3, seq_length)
                valid_mask = attention_mask[i].bool()
                text_position_ids = torch.ones((1, len(input_ids[i])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids_list.append(
                    torch.cat((text_position_ids, vision_position_ids), dim=0)
                )  # (4, seq_length)

        else:
            # Fallback: should not reach here if called correctly
            raise ValueError(
                f"Unsupported processor type: {processor.__class__.__name__ if processor else None}"
            )

        # Stack all position_ids to form batch: (batch_size, 4, seq_length)
        position_ids = torch.stack(position_ids_list, dim=0)
        return position_ids

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
