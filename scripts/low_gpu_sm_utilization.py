#!/usr/bin/env python3
"""Add a small, adaptive, low-priority CUDA workload to local GPUs.

This is a best-effort utilization filler, not an isolation mechanism.  It
tries to keep each selected GPU near ``--target-utilization`` while pausing
when the observed utilization reaches ``--max-utilization`` or free memory is
too low.  The workload uses a small FP16 matrix and a low-priority CUDA stream
so higher-priority work has a better chance to run first.

The default settings target roughly 30% utilization and stop adding work at
40%.  GPU utilization is sampled by ``nvidia-smi`` and can fluctuate; no
program can guarantee zero impact on other workloads sharing the same GPU.

Example:
    python scripts/low_gpu_sm_utilization.py
    python scripts/low_gpu_sm_utilization.py --gpus 4,5 --target-utilization 40
"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass


DEFAULT_NVIDIA_SMI = "nvidia-smi"
DEFAULT_TARGET_UTILIZATION = 30.0
DEFAULT_MAX_UTILIZATION = 40.0
DEFAULT_MIN_FREE_MEMORY_MIB = 2048.0
DEFAULT_CONTROL_INTERVAL_SECONDS = 2.0
DEFAULT_MATRIX_SIZE = 2048
DEFAULT_BUSY_WINDOW_SECONDS = 0.05
DEFAULT_CYCLE_SECONDS = 0.20


@dataclass(frozen=True)
class GpuStats:
    index: int
    utilization_percent: float
    free_memory_mib: float


def resolve_executable(command: str) -> str:
    if "/" in command:
        return command
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"executable not found: {command}")
    return resolved


def parse_gpu_stats(output: str) -> list[GpuStats]:
    stats: list[GpuStats] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError(f"malformed nvidia-smi output line {line_number}: {raw_line!r}")
        try:
            index = int(fields[0])
            utilization = float(fields[1].rstrip("% "))
            free_memory_mib = float(fields[2].rstrip("MiB "))
        except ValueError as exc:
            raise ValueError(f"invalid nvidia-smi output line {line_number}: {raw_line!r}") from exc
        if not 0.0 <= utilization <= 100.0 or free_memory_mib < 0.0:
            raise ValueError(f"out-of-range nvidia-smi output line {line_number}: {raw_line!r}")
        stats.append(GpuStats(index, utilization, free_memory_mib))
    if not stats:
        raise ValueError("nvidia-smi returned no GPU rows")
    return stats


def query_gpu_stats(nvidia_smi: str, timeout_seconds: float) -> list[GpuStats]:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return parse_gpu_stats(result.stdout)


def parse_gpu_selection(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid GPU index: {token!r}") from exc
        if index < 0:
            raise ValueError(f"GPU index must be non-negative: {index}")
        selected.add(index)
    if not selected:
        raise ValueError("--gpus did not contain any GPU indices")
    return selected


def low_priority_stream(torch: object, device_index: int) -> object:
    cuda = torch.cuda  # type: ignore[attr-defined]
    try:
        least_priority, _greatest_priority = cuda.Stream.priority_range()
        return cuda.Stream(device=device_index, priority=least_priority)
    except (AttributeError, RuntimeError):
        # Older PyTorch builds may not expose stream priorities.  Falling back
        # to the default stream keeps the script usable, but is less gentle.
        print(
            f"[gpu-fill] warning: low-priority CUDA stream unavailable on GPU {device_index}; "
            "using the default stream",
            flush=True,
        )
        return cuda.Stream(device=device_index)


def gpu_worker(
    torch: object,
    device_index: int,
    run_event: threading.Event,
    stop_event: threading.Event,
    failure_queue: queue.Queue[tuple[int, str]],
    matrix_size: int,
    busy_window_seconds: float,
    cycle_seconds: float,
) -> None:
    try:
        cuda = torch.cuda  # type: ignore[attr-defined]
        cuda.set_device(device_index)
        device = f"cuda:{device_index}"
        stream = low_priority_stream(torch, device_index)

        # About 24 MiB for three FP16 matrices at the default size.  Keeping
        # this allocation small reduces pressure on other jobs' allocators.
        a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
        b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
        cuda.synchronize()

        while not stop_event.is_set():
            if not run_event.wait(timeout=0.25):
                continue
            cycle_start = time.monotonic()
            busy_deadline = cycle_start + busy_window_seconds
            while (
                not stop_event.is_set()
                and run_event.is_set()
                and time.monotonic() < busy_deadline
            ):
                with cuda.stream(stream):
                    _result = torch.mm(a, b)
                stream.synchronize()

            remaining = cycle_seconds - (time.monotonic() - cycle_start)
            if remaining > 0:
                stop_event.wait(remaining)
    except BaseException as exc:  # report worker failures to the controller
        failure_queue.put((device_index, repr(exc)))
        stop_event.set()


def state_for_gpu(
    stats: GpuStats,
    target_utilization: float,
    max_utilization: float,
    min_free_memory_mib: float,
) -> tuple[bool, str]:
    if stats.free_memory_mib <= min_free_memory_mib:
        return False, f"free={stats.free_memory_mib:.0f}MiB<=min={min_free_memory_mib:g}MiB"
    if stats.utilization_percent >= max_utilization:
        return False, f"util={stats.utilization_percent:.0f}%>=max={max_utilization:g}%"
    if stats.utilization_percent >= target_utilization:
        return False, f"util={stats.utilization_percent:.0f}%>=target={target_utilization:g}%"
    return True, f"util={stats.utilization_percent:.0f}% free={stats.free_memory_mib:.0f}MiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=None, help="Comma-separated local GPU indices; default: all visible GPUs.")
    parser.add_argument("--nvidia-smi", default=DEFAULT_NVIDIA_SMI, help="nvidia-smi executable.")
    parser.add_argument("--target-utilization", type=float, default=DEFAULT_TARGET_UTILIZATION)
    parser.add_argument("--max-utilization", type=float, default=DEFAULT_MAX_UTILIZATION)
    parser.add_argument("--min-free-memory-mib", type=float, default=DEFAULT_MIN_FREE_MEMORY_MIB)
    parser.add_argument("--control-interval-seconds", type=float, default=DEFAULT_CONTROL_INTERVAL_SECONDS)
    parser.add_argument("--nvidia-smi-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--matrix-size", type=int, default=DEFAULT_MATRIX_SIZE)
    parser.add_argument("--busy-window-seconds", type=float, default=DEFAULT_BUSY_WINDOW_SECONDS)
    parser.add_argument("--cycle-seconds", type=float, default=DEFAULT_CYCLE_SECONDS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.target_utilization < 100.0:
        raise SystemExit("--target-utilization must be between 0 and 100")
    if not args.target_utilization < args.max_utilization <= 100.0:
        raise SystemExit("require target-utilization < max-utilization <= 100")
    if args.min_free_memory_mib < 0:
        raise SystemExit("--min-free-memory-mib cannot be negative")
    if args.control_interval_seconds <= 0 or args.nvidia_smi_timeout_seconds <= 0:
        raise SystemExit("control and nvidia-smi timeouts must be positive")
    if args.matrix_size <= 0 or args.busy_window_seconds <= 0 or args.cycle_seconds <= 0:
        raise SystemExit("matrix size and workload windows must be positive")
    if args.busy_window_seconds >= args.cycle_seconds:
        raise SystemExit("--busy-window-seconds must be smaller than --cycle-seconds")

    try:
        selected_gpus = parse_gpu_selection(args.gpus)
        nvidia_smi = resolve_executable(args.nvidia_smi)
        import torch
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment")

    torch_gpu_count = torch.cuda.device_count()
    try:
        initial_stats = query_gpu_stats(nvidia_smi, args.nvidia_smi_timeout_seconds)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SystemExit(f"unable to query local GPUs with nvidia-smi: {exc}") from exc

    visible_indices = {item.index for item in initial_stats}
    if selected_gpus is None:
        selected_gpus = visible_indices
    unknown = selected_gpus - visible_indices
    if unknown:
        raise SystemExit(f"requested GPU(s) not returned by nvidia-smi: {sorted(unknown)}")
    if max(selected_gpus) >= torch_gpu_count:
        raise SystemExit(
            "GPU indices are incompatible with the current CUDA_VISIBLE_DEVICES "
            f"mapping: nvidia-smi={sorted(visible_indices)}, torch_device_count={torch_gpu_count}"
        )

    run_events = {index: threading.Event() for index in selected_gpus}
    stop_event = threading.Event()
    failures: queue.Queue[tuple[int, str]] = queue.Queue()
    threads = [
        threading.Thread(
            target=gpu_worker,
            args=(
                torch,
                index,
                run_events[index],
                stop_event,
                failures,
                args.matrix_size,
                args.busy_window_seconds,
                args.cycle_seconds,
            ),
            name=f"gpu-fill-{index}",
            daemon=True,
        )
        for index in sorted(selected_gpus)
    ]
    for thread in threads:
        thread.start()

    print(
        f"[gpu-fill] GPUs={sorted(selected_gpus)} target={args.target_utilization:g}% "
        f"max={args.max_utilization:g}% min_free={args.min_free_memory_mib:g}MiB "
        f"matrix={args.matrix_size} low_priority_stream=best_effort",
        flush=True,
    )
    last_states: dict[int, str] = {}
    last_report = 0.0

    try:
        while not stop_event.is_set():
            try:
                failure_gpu, failure = failures.get_nowait()
            except queue.Empty:
                failure_gpu = None
                failure = ""
            if failure_gpu is not None:
                print(f"[gpu-fill] worker gpu={failure_gpu} failed: {failure}", flush=True)
                return 1

            try:
                stats_by_index = {
                    item.index: item
                    for item in query_gpu_stats(nvidia_smi, args.nvidia_smi_timeout_seconds)
                }
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                for event in run_events.values():
                    event.clear()
                print(f"[gpu-fill] nvidia-smi query failed; pausing all GPUs: {exc}", flush=True)
                stop_event.wait(args.control_interval_seconds)
                continue

            now = time.monotonic()
            states: dict[int, str] = {}
            for index, event in run_events.items():
                stats = stats_by_index.get(index)
                if stats is None:
                    event.clear()
                    states[index] = "pause: GPU missing from nvidia-smi"
                    continue
                should_run, reason = state_for_gpu(
                    stats,
                    args.target_utilization,
                    args.max_utilization,
                    args.min_free_memory_mib,
                )
                if should_run:
                    event.set()
                    states[index] = f"run: {reason}"
                else:
                    event.clear()
                    states[index] = f"pause: {reason}"

            if states != last_states or now - last_report >= 30.0:
                print(
                    "[gpu-fill] " + "; ".join(
                        f"gpu={index} {states[index]}" for index in sorted(states)
                    ),
                    flush=True,
                )
                last_states = states
                last_report = now
            stop_event.wait(args.control_interval_seconds)
    except KeyboardInterrupt:
        print("\n[gpu-fill] stopping", flush=True)
    finally:
        stop_event.set()
        for event in run_events.values():
            event.clear()
        for thread in threads:
            thread.join(timeout=10)
        for index in sorted(selected_gpus):
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
