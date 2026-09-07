"""Small, training-free W&B diagnostics used by the RL/SFT probe scripts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


def add_common_args(parser: argparse.ArgumentParser, default_project: str) -> None:
    parser.add_argument("--project", default=default_project)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--post-finish-seconds", type=float, default=5.0)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument(
        "--wandb-dir",
        default=None,
        help="Optional local W&B directory. If omitted, WANDB_DIR is preserved.",
    )


def configure_environment(args: argparse.Namespace) -> None:
    if args.mode is not None:
        os.environ["WANDB_MODE"] = args.mode
    else:
        os.environ.setdefault("WANDB_MODE", "online")
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = str(Path(args.wandb_dir).expanduser())
    # Make the command-line project deterministic even when a training env was
    # sourced before launching the probe.
    os.environ["WANDB_PROJECT"] = args.project


def _safe_setting_values(settings: Any) -> dict[str, Any]:
    values = {}
    for attr in (
        "mode",
        "_offline",
        "offline",
        "_sync",
        "sync",
        "x_disable_stats",
        "_disable_stats",
        "disable_stats",
        "x_stats_sampling_interval",
        "root_dir",
        "wandb_dir",
        "dir",
        "files_dir",
        "sync_dir",
    ):
        try:
            value = getattr(settings, attr, None)
        except Exception as exc:
            value = f"<error:{type(exc).__name__}>"
        if value is not None:
            values[attr] = value
    return values


def _run_root(run: Any) -> tuple[str | None, Path | None]:
    run_dir_value = getattr(run, "dir", None)
    if not run_dir_value:
        return None, None
    run_dir = Path(str(run_dir_value))
    return str(run_dir), run_dir.parent if run_dir.name == "files" else run_dir


def _system_metric_snapshot(run: Any) -> tuple[int | None, list[str], dict[str, Any]]:
    try:
        collected = getattr(run, "_system_metrics", {}) or {}
        keys = sorted(str(key) for key in collected)
        latest = {}
        for key, records in collected.items():
            if not records:
                continue
            try:
                latest[str(key)] = records[-1][1]
            except Exception:
                latest[str(key)] = "<unreadable>"
        interesting = {
            key: latest[key]
            for key in sorted(latest)
            if any(token in key.lower() for token in ("cpu", "gpu", "memory", "ram", "power"))
        }
        return len(collected), keys[:40], dict(list(interesting.items())[:20])
    except Exception as exc:
        return None, [f"<query_error:{type(exc).__name__}:{exc}>"], {}


def _sender_tail(root_dir: Path | None) -> tuple[str | None, str | None]:
    if root_dir is None:
        return None, None
    internal_log = root_dir / "logs" / "debug-internal.log"
    if not internal_log.exists():
        return None, None
    try:
        with internal_log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 16384), os.SEEK_SET)
            tail = stream.read().decode("utf-8", errors="replace")
        interesting_tokens = (
            "send",
            "upload",
            "sync",
            "request failed",
            "run started",
            "starting system monitor",
            "started gpu",
        )
        lines = [
            line.strip().replace("\n", " ")[:500]
            for line in tail.splitlines()
            if any(token in line.lower() for token in interesting_tokens)
        ]
        send_stats_count = sum("send: stats" in line.lower() for line in lines)
        return send_stats_count, " || ".join(lines[-3:]) or None
    except Exception as exc:
        return None, f"<log_error:{type(exc).__name__}:{exc}>"


def print_snapshot(run: Any, event: str, step: int | None = None) -> None:
    """Print one compact JSON diagnostic line for local collection/sending."""
    run_dir_value, root_dir = _run_root(run)
    metric_count, metric_keys, latest_values = _system_metric_snapshot(run)

    event_file_state = []
    summary_state = None
    if root_dir is not None and root_dir.exists():
        for path in sorted(root_dir.glob("*.wandb")):
            try:
                event_file_state.append({"name": path.name, "bytes": path.stat().st_size})
            except OSError:
                event_file_state.append({"name": path.name, "bytes": None})
        summary_path = root_dir / "files" / "wandb-summary.json"
        if summary_path.exists():
            try:
                summary_state = summary_path.stat().st_size
            except OSError:
                summary_state = None

    stats_tail_count, sender_tail = _sender_tail(root_dir)
    settings = getattr(run, "settings", None)
    settings_mode = getattr(settings, "mode", None)
    payload = {
        "event": event,
        "step": step,
        "pid": os.getpid(),
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "project": getattr(run, "project", None),
        "url": getattr(run, "url", None),
        # Prefer Settings.mode.  Accessing run.mode emits a deprecation warning
        # in newer W&B clients and may report the internal state "run" rather
        # than the configured mode "online".
        "mode": settings_mode,
        "env": {
            key: os.environ.get(key)
            for key in ("WANDB_MODE", "WANDB_DIR", "WANDB_BASE_URL", "WANDB_PROJECT")
            if os.environ.get(key) is not None
        },
        "settings": _safe_setting_values(getattr(run, "settings", None)),
        "run_dir": run_dir_value,
        "root_dir": str(root_dir) if root_dir is not None else None,
        "event_files": event_file_state,
        "summary_bytes": summary_state,
        "system_metric_count": metric_count,
        "system_metric_keys": metric_keys,
        "latest_system_values": latest_values,
        "tail_send_stats_count": stats_tail_count,
        "sender_tail": sender_tail,
    }
    print("[WANDB_PROBE] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def run_probe_loop(
    run: Any,
    log_fn: Callable[[dict[str, Any], int], None],
    duration_seconds: float,
    interval_seconds: float,
    post_finish_seconds: float,
    finish_fn: Callable[[], None],
) -> None:
    interval_seconds = max(0.1, interval_seconds)
    duration_seconds = max(0.0, duration_seconds)
    print_snapshot(run, event="init")

    started = time.monotonic()
    step = 0
    while time.monotonic() - started < duration_seconds:
        time.sleep(min(interval_seconds, max(0.0, duration_seconds - (time.monotonic() - started))))
        step += 1
        elapsed = time.monotonic() - started
        log_fn(
            {
                "probe/heartbeat": 1,
                "probe/elapsed_seconds": elapsed,
                "probe/python_pid": os.getpid(),
            },
            step,
        )
        print_snapshot(run, event="log", step=step)

    print_snapshot(run, event="pre_finish", step=step)
    finish_fn()
    if post_finish_seconds > 0:
        time.sleep(post_finish_seconds)
    print_snapshot(run, event="post_finish", step=step)
