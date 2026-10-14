#!/usr/bin/env python3
"""Unified entrypoint for OpenSearch-VL inference.

Usage examples
--------------

Run the dense 8B model on the FVQA training split using two GPUs::

    python run_infer.py --model 8b --gpus 0,1 \
        --data-path /data/fvqa_train.parquet \
        --output-dir ./outputs/fvqa_train_8b

Use Claude Opus 4.5 (no local GPUs needed; requires CLAUDE_API_* env vars)::

    python run_infer.py --model claude \
        --data-path /data/fvqa_test.parquet \
        --output-dir ./outputs/fvqa_test_claude

Switch to the MoE 30B-A3B variant::

    python run_infer.py --model 30b-a3b --gpus 0,1,2,3 \
        --checkpoint /models/Qwen3-VL-30B-A3B-Instruct \
        --data-path /data/fvqa_train.parquet \
        --output-dir ./outputs/fvqa_train_30b_a3b

The model registry is defined in ``opensearch_infer/config.py``. All
default values pull from environment variables so the codebase is safe
to open-source.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback
from typing import Any, Optional

from tqdm.auto import tqdm

from opensearch_infer import config
from synthesis.retry_monitor import RetryMonitor, case_context, install_monitor


_REDACTED_VALUE = "<redacted>"
_SENSITIVE_ARGUMENT_TOKENS = ("api-key", "token", "secret", "password")
_REPRO_ENVIRONMENT_NAMES = (
    "AGENT_BASE_URL",
    "OPENAI_BASE_URL",
    "AGENT_API_KEY",
    "OPENAI_API_KEY",
    "QWEN3VL_8B_PATH",
    "QWEN3VL_32B_PATH",
    "QWEN3VL_30B_A3B_PATH",
    "AGENT_MAX_TURNS",
    "AGENT_MAX_TOOL_CALLS",
    "AGENT_TRACE_LOGS",
    "CUDA_VISIBLE_DEVICES",
)


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    if normalized.lstrip("-") in {"max-tokens", "max-output-tokens", "max-completion-tokens"}:
        return False
    return any(token in normalized for token in _SENSITIVE_ARGUMENT_TOKENS)


def _redact_argv(argv: list[str]) -> list[str]:
    """Return CLI arguments with values for secret-like flags redacted."""

    redacted: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next:
            redacted.append(_REDACTED_VALUE)
            redact_next = False
            continue

        option, separator, _ = token.partition("=")
        if option.startswith("--") and _is_sensitive_name(option):
            redacted.append(f"{option}={_REDACTED_VALUE}" if separator else option)
            redact_next = not separator
        else:
            redacted.append(token)
    return redacted


def _redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _REDACTED_VALUE if _is_sensitive_name(key) and value else value
        for key, value in values.items()
    }


def _file_metadata(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    metadata: dict[str, Any] = {
        "requested_path": path_value,
        "resolved_path": str(path),
    }
    try:
        stat = path.stat()
    except OSError:
        metadata["exists"] = False
        return metadata

    metadata.update(
        {
            "exists": True,
            "size_bytes": stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    )
    return metadata


def _git_output(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _runner_metadata(runner: Any) -> dict[str, Any]:
    """Capture reproducibility-relevant runner fields without credentials."""

    metadata = {
        "class": type(runner).__name__,
        "display_name": getattr(runner, "display_name", None),
    }
    for field in ("model", "checkpoint", "base_url"):
        value = getattr(runner, field, None)
        if value is not None:
            metadata[field] = str(value)
    return metadata


def _run_configuration_snapshot(
    args: argparse.Namespace,
    raw_argv: list[str],
    runner: Any,
    *,
    entrypoint: Optional[str] = None,
    no_tools: bool = False,
) -> dict[str, Any]:
    redacted_argv = _redact_argv(raw_argv)
    entrypoint_path = str(Path(entrypoint or __file__).resolve())
    replay_command = [sys.executable, entrypoint_path, *redacted_argv]
    environment = {
        name: (_REDACTED_VALUE if _is_sensitive_name(name) else os.environ[name])
        for name in _REPRO_ENVIRONMENT_NAMES
        if name in os.environ
    }
    snapshot = {
        "schema_version": 1,
        "ablation_policy_version": 2 if "t2i_search" in (getattr(args, "ablation", None) or []) else 1,
        "mode": "no_tools" if no_tools else "tools",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "entrypoint": entrypoint_path,
            "argv": redacted_argv,
            "replay_command": shlex.join(replay_command),
        },
        "arguments": _redact_mapping(vars(args)),
        "runner": _runner_metadata(runner),
        "inference": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
        },
        "agent_limits": {
            "max_turns": 1 if no_tools else config.MAX_TURNS,
            "max_tool_calls": 0 if no_tools else config.MAX_TOOL_CALLS,
            "trace_logs": config.TRACE_LOGS,
        },
        "dataset": _file_metadata(args.data_path),
        "environment": environment,
        "code": {
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_dirty": bool(_git_output("status", "--porcelain")),
            "python_version": platform.python_version(),
        },
    }
    if getattr(args, "responses_api", False):
        snapshot["api_mode"] = "responses"
        snapshot["inference"] = runner.configuration()
        snapshot["agent_limits"].update({
            "max_turns": args.max_turns,
            "max_tool_calls": args.max_tool_calls,
        })
    return snapshot


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _persist_run_configuration(
    args: argparse.Namespace,
    raw_argv: list[str],
    runner: Any,
    logger: logging.Logger,
    *,
    entrypoint: Optional[str] = None,
    no_tools: bool = False,
) -> None:
    """Save an immutable initial config plus one record for every invocation."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _run_configuration_snapshot(
        args, raw_argv, runner, entrypoint=entrypoint, no_tools=no_tools
    )
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        _write_json_atomically(config_path, snapshot)
        logger.info("Saved reproducibility config to %s", config_path)
    else:
        logger.info("Keeping existing initial reproducibility config at %s", config_path)

    invocations_path = output_dir / "run_invocations.jsonl"
    with invocations_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _build_arg_parser(*, responses_api: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenSearch-VL Visual Investigation Agent. Supported tools: "
            "web_search, text_search, image_search, crop, layout_parsing, "
            "perspective_correct, super_resolution, sharpen."
        )
    )
    parser.add_argument(
        "--ablation", choices=["t2i_search", "i2i_search"], nargs="+",
        action="extend", default=None,
        help="Disable one or more tools in both the system prompt and execution.",
    )
    parser.add_argument(
        "--model",
        choices=None if responses_api else sorted(config.MODEL_REGISTRY.keys()),
        default=None if responses_api else "8b",
        help="API model name." if responses_api else "Model variant to run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=argparse.SUPPRESS if responses_api else (
            "Override the checkpoint path or HuggingFace id. Defaults to "
            "the per-model env variable, then the registry default."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help=argparse.SUPPRESS if responses_api else (
            "Comma-separated CUDA device ids for Qwen3-VL. Single id uses "
            "single-GPU placement; multiple ids enable model parallelism "
            'via device_map="auto".'
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help=argparse.SUPPRESS if responses_api else "Floating point dtype for Qwen3-VL weights.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="api" if responses_api else "local",
        choices=["api"] if responses_api else ["local", "api"],
        help="Responses API endpoint (no local weights)." if responses_api else (
            "Inference backend. 'local' loads HF weights in-process; 'api' calls "
            "an OpenAI-compatible endpoint such as vLLM serve."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for --backend api, e.g. http://localhost:8001/v1.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key; otherwise use the model alias or AGENT_API_KEY / OPENAI_API_KEY." if responses_api else
             "API key for --backend api. Defaults to AGENT_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="API model name; alternative to --model." if responses_api else (
            "Model name sent to the API server. Defaults to --checkpoint, then "
            "the registry display name."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Parquet file with FVQA-style rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where per-case trajectories and artifacts are written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Tag stored in trajectories; controls intermediate filename prefix.",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First row index (inclusive)."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last row index (exclusive). Defaults to len(df).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows starting from --start.",
    )
    parser.add_argument(
        "--dynamic-resolution",
        action="store_true",
        help=(
            "Resize every image before it is sent to the model so its longest "
            "side is at most 1600 pixels."
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Restrict to rows whose 'category' column equals this value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None if responses_api else 16384,
        help="Per-response max_output_tokens, including reasoning tokens; CLI > alias > 16384." if responses_api else
             "Per-turn generation cap.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of benchmark examples to run concurrently. Each trajectory "
            "remains turn-by-turn serial."
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds for --backend api.",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="HTTP retry count for --backend api.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None if responses_api else 0.7,
        help="Optional temperature; omitted unless configured. Not all reasoning models support it." if responses_api else
             "Sampling temperature (Qwen3-VL VL default: 0.7; 0 disables sampling).",
    )
    parser.add_argument("--top-p", type=float, default=None if responses_api else 0.8)
    parser.add_argument("--top-k", type=int, default=None if responses_api else 20,
                        help=argparse.SUPPRESS if responses_api else None)
    parser.add_argument("--repetition-penalty", type=float, default=None if responses_api else 1.0,
                        help=argparse.SUPPRESS if responses_api else None)
    parser.add_argument("--presence-penalty", type=float, default=None if responses_api else 1.5,
                        help=argparse.SUPPRESS if responses_api else None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logger level.",
    )
    parser.add_argument(
        "--retry-monitor-interval",
        type=float,
        default=10.0,
        help=(
            "Seconds between retry-status reports. Set to 0 to disable the "
            "process-local retry monitor."
        ),
    )
    parser.add_argument(
        "--retry-error-log",
        type=str,
        default=None,
        help=(
            "Path for the detailed per-retry error log. Defaults to "
            "<output-dir>/retry_errors.txt."
        ),
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
        level=getattr(logging, level),
    )


def _row_to_case_id(row: Any, idx: int) -> str:
    if hasattr(row, "to_dict"):
        try:
            row_dict = row.to_dict()
        except Exception:
            row_dict = {}
    elif isinstance(row, dict):
        row_dict = row
    else:
        row_dict = {}

    if isinstance(row_dict, dict):
        for key in ("data_id", "id", "idx", "_id"):
            value = row_dict.get(key)
            if value is not None:
                return str(value)
    return f"case_{idx}"


def _completed_case_ids(output_dir: str) -> set[str]:
    """Skip successful trajectories and explicitly archived terminal failures.

    Root-level max-turn failure files are deliberately not evaluation inputs.
    Failures still in ``failures/`` remain retryable.
    """

    completed: set[str] = set()
    if not os.path.isdir(output_dir):
        return completed
    suffix = "_trajectory.json"
    for name in os.listdir(output_dir):
        if name.endswith("_failure.json"):
            try:
                with open(os.path.join(output_dir, name), encoding="utf-8") as handle:
                    archived = json.load(handle)
            except (OSError, ValueError, UnicodeError):
                continue
            if (
                isinstance(archived, dict)
                and archived.get("status") == "failed"
                and archived.get("failure_kind") == "max_turns"
            ):
                completed.add(name[: -len("_failure.json")])
            continue
        if name.endswith(suffix):
            try:
                with open(os.path.join(output_dir, name), encoding="utf-8") as handle:
                    trajectory = json.load(handle)
            except (OSError, ValueError, UnicodeError):
                continue
            if not isinstance(trajectory, dict):
                continue
            if trajectory.get("status") not in (None, "completed"):
                continue
            turns = trajectory.get("turns") or []
            if not isinstance(turns, list) or any(
                isinstance(turn, dict)
                and (
                    turn.get("error")
                    or (
                        isinstance(turn.get("response"), dict)
                        and turn["response"].get("inference_error")
                    )
                )
                for turn in turns
            ):
                continue
            response_text = trajectory.get("final_response_text", "")
            if (
                isinstance(response_text, str)
                and response_text.strip()
                and (
                    trajectory.get("mode") == "no_tools"
                    or ("<answer>" in response_text and "</answer>" in response_text)
                )
            ):
                completed.add(name[: -len(suffix)])
    return completed


def main(
    argv: Optional[list[str]] = None,
    *,
    no_tools: bool = False,
    entrypoint: Optional[str] = None,
    responses_api: bool = False,
) -> int:
    """Shared benchmark execution for the agent and single-turn baseline."""

    parser = _build_arg_parser(responses_api=responses_api)
    if responses_api:
        from opensearch_infer.responses import add_responses_arguments
        add_responses_arguments(parser)
    if no_tools:
        parser.description = "OpenSearch-VL single-turn inference without tool use."
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if responses_api:
        from opensearch_infer.responses import validate_responses_arguments
        validate_responses_arguments(parser, args)
    if args.ablation:
        args.ablation = sorted(set(args.ablation))
    if no_tools and args.ablation:
        parser.error("--ablation applies only to tool-enabled inference")
    previous_config = Path(args.output_dir) / "run_config.json"
    if previous_config.is_file():
        with previous_config.open(encoding="utf-8") as handle:
            previous_snapshot = json.load(handle)
            previous_ablation = previous_snapshot.get("arguments", {}).get("ablation")
        # Older run configs stored a single tool as a string. Ordering and
        # duplicates do not change an ablation's meaning.
        if isinstance(previous_ablation, str):
            previous_ablation = [previous_ablation]
        if sorted(set(previous_ablation or [])) != (args.ablation or []):
            parser.error("Ablation differs from the existing output directory; use a new --output-dir")
        if "t2i_search" in (args.ablation or []) and previous_snapshot.get("ablation_policy_version", 1) != 2:
            parser.error("Existing t2i ablation allowed image downloads; use a new --output-dir for the text-only read_url policy")
        if responses_api != (previous_snapshot.get("api_mode") == "responses"):
            parser.error("API mode differs from the existing output directory; use a new --output-dir")

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer_no_tools" if no_tools else "run_infer")

    from opensearch_infer import pipeline
    from opensearch_infer.runners import InferenceConfig, build_runner

    if responses_api:
        from opensearch_infer import responses as case_pipeline
    elif no_tools:
        from opensearch_infer import no_tools_pipeline as case_pipeline
    else:
        case_pipeline = pipeline

    if responses_api:
        from opensearch_infer.responses import ResponsesRunner
        try:
            runner = ResponsesRunner.from_args(args)
        except (ValueError, KeyError) as exc:
            parser.error(str(exc))
    else:
        runner = build_runner(
            model_name=args.model,
            checkpoint=args.checkpoint,
            gpus=args.gpus,
            dtype=args.dtype,
            backend=args.backend,
            base_url=args.base_url,
            api_key=args.api_key,
            served_model_name=args.served_model_name,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
        )
    logger.info(
        "Model: %s (backend=%s, mode=%s)",
        runner.display_name, args.backend, "no_tools" if no_tools else "tools",
    )

    try:
        _persist_run_configuration(
            args, raw_argv, runner, logger, entrypoint=entrypoint, no_tools=no_tools
        )
    except OSError as exc:
        logger.warning("Could not save reproducibility config: %s", exc)

    try:
        runner.load()
    except Exception as exc:
        logger.error("Failed to initialize runner: %s", exc, exc_info=True)
        return 1

    if not os.path.exists(args.data_path):
        logger.error("Data file not found: %s", args.data_path)
        return 1

    import pandas as pd

    df = pd.read_parquet(args.data_path)
    logger.info("Loaded %d rows from %s", len(df), args.data_path)

    if args.category:
        df = df[df["category"] == args.category]
        logger.info("Filtered to category=%s: %d rows", args.category, len(df))

    start = max(0, int(args.start))
    if args.end is not None:
        end = min(int(args.end), len(df))
    elif args.limit is not None:
        end = min(start + int(args.limit), len(df))
    else:
        end = len(df)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Processing rows [%d, %d) -> %s", start, end, args.output_dir)

    inference_cfg = InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    workers = max(1, int(args.parallel_workers))
    if args.backend == "local" and workers > 1:
        logger.warning(
            "--parallel-workers=%d with --backend local shares one in-process model; "
            "API/vLLM backend is recommended for parallel evaluation.",
            workers,
        )

    def _run_one(idx: int) -> tuple[int, bool, Optional[str]]:
        row = df.iloc[idx]
        case_id = _row_to_case_id(row, idx)
        timing = pipeline.InferenceTiming() if logger.isEnabledFor(logging.INFO) else None
        status = "failed"
        with case_context(case_id, idx):
            try:
                case_pipeline.process_single_case(
                    row=row,
                    runner=runner,
                    output_dir=args.output_dir,
                    case_idx=idx,
                    dataset_type=args.dataset,
                    inference_cfg=inference_cfg,
                    timing=timing,
                    dynamic_resolution=args.dynamic_resolution,
                    **({"disabled_tools": tuple(args.ablation)} if args.ablation else {}),
                )
                status = "completed"
                return idx, True, None
            except Exception as exc:
                logger.error("Case %d failed: %s", idx, exc)
                traceback.print_exc()
                return idx, False, str(exc)
            finally:
                if timing is not None:
                    timing.emit(case_id=case_id, case_idx=idx, status=status)

    success, failure, skipped = 0, 0, 0
    completed_case_ids = _completed_case_ids(args.output_dir)
    all_indices = list(range(start, end))
    indices = []
    for idx in all_indices:
        case_id = _row_to_case_id(df.iloc[idx], idx)
        if case_id in completed_case_ids:
            skipped += 1
            continue
        indices.append(idx)

    if skipped:
        logger.info(
            "Resume detected: skipped %d completed or archived max-turns case(s) already present in %s",
            skipped,
            args.output_dir,
        )

    retry_monitor = None
    progress_description = "No-tools inference" if no_tools else "Inference"
    if args.retry_monitor_interval > 0:
        retry_error_log = args.retry_error_log or os.path.join(
            args.output_dir,
            "retry_errors.txt",
        )
        retry_monitor = RetryMonitor(
            interval_s=args.retry_monitor_interval,
            error_log_path=retry_error_log,
        )
        install_monitor(retry_monitor)
        retry_monitor.start()
    try:
        if workers == 1:
            for idx in tqdm(indices, desc=progress_description, unit="case"):
                _, ok, _ = _run_one(idx)
                success += int(ok)
                failure += int(not ok)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_one, idx): idx for idx in indices}
                with tqdm(total=len(futures), desc=progress_description, unit="case") as progress:
                    for future in concurrent.futures.as_completed(futures):
                        idx, ok, error = future.result()
                        success += int(ok)
                        failure += int(not ok)
                        progress.update(1)
                        if ok:
                            logger.info(
                                "Case %d completed (%d/%d)",
                                idx,
                                success + failure,
                                len(indices),
                            )
                        else:
                            logger.error(
                                "Case %d failed (%d/%d): %s",
                                idx,
                                success + failure,
                                len(indices),
                                error,
                            )

        logger.info(
            "Done. success=%d failure=%d skipped=%d output=%s",
            success,
            failure,
            skipped,
            args.output_dir,
        )
        return 0
    finally:
        if retry_monitor is not None:
            retry_monitor.stop()
            install_monitor(None)


if __name__ == "__main__":
    sys.exit(main())
