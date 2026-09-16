#!/usr/bin/env python3
"""Supervise ``holder.py`` on locally idle GPUs.

For each local GPU, this script waits until the sampled GPU utilization is at
or below a threshold (0% by default), with sufficient free memory, for a
continuous interval (30 minutes by default). Existing CUDA processes do not
prevent a launch; their PIDs are recorded as a baseline for that launch. It
then starts the target with only the eligible GPUs in ``CUDA_VISIBLE_DEVICES``.

The script remains running after launch. If a *new* CUDA process not belonging
to the target's process group appears on any GPU assigned to the target, the
script terminates only that target process group and starts the idle timer
again. It never terminates the foreign process.

``nvidia-smi`` utilization is sampled, so this supervisor cannot reserve a
GPU against a scheduler race. Use scheduler reservations or MIG as well when
strict isolation is required.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO


DEFAULT_NVIDIA_SMI = "nvidia-smi"
DEFAULT_TARGET_SCRIPT = Path(
    "/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/verl/holder.py"
)
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_REQUIRED_IDLE_SECONDS = 30 * 60
DEFAULT_MIN_FREE_MEMORY_MIB = 2048.0
DEFAULT_UTILIZATION_THRESHOLD_PERCENT = 0.0
DEFAULT_STOP_GRACE_SECONDS = 30.0
DEFAULT_LOCK_FILE = Path("/tmp/monitor_gpu_power_zero.lock")


@dataclass(frozen=True)
class GpuReading:
    index: int
    uuid: str
    utilization_percent: float
    free_memory_mib: float


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int


@dataclass
class ManagedTarget:
    process: subprocess.Popen[object]
    process_group_id: int
    gpu_indices: tuple[int, ...]
    baseline_foreign_pids_by_gpu: dict[int, frozenset[int]]


def _parse_gpu_output(output: str) -> list[GpuReading]:
    readings: list[GpuReading] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(
                f"malformed nvidia-smi output on line {line_number}: {raw_line!r}"
            )
        try:
            index = int(fields[0])
            utilization = float(fields[2].rstrip("% "))
            free_memory_mib = float(fields[3].rstrip("MiB "))
        except ValueError as exc:
            raise ValueError(
                f"invalid nvidia-smi GPU reading on line {line_number}: {raw_line!r}"
            ) from exc
        uuid = fields[1]
        if not uuid:
            raise ValueError(f"missing GPU UUID on line {line_number}")
        if index in seen_indices or uuid in seen_uuids:
            raise ValueError(f"duplicate GPU identity on line {line_number}")
        if utilization < 0 or free_memory_mib < 0:
            raise ValueError(f"negative GPU metric on line {line_number}")
        seen_indices.add(index)
        seen_uuids.add(uuid)
        readings.append(GpuReading(index, uuid, utilization, free_memory_mib))

    if not readings:
        raise ValueError("nvidia-smi returned no local GPU readings")
    return readings


def _parse_compute_process_output(output: str) -> list[ComputeProcess]:
    processes: list[ComputeProcess] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line in {
            "No running processes found",
            "No running compute processes found",
        }:
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            raise ValueError(
                "malformed nvidia-smi compute-process output on "
                f"line {line_number}: {raw_line!r}"
            )
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise ValueError(
                "invalid nvidia-smi compute-process output on "
                f"line {line_number}: {raw_line!r}"
            ) from exc
        if not fields[0] or pid <= 0:
            raise ValueError(f"invalid compute process on line {line_number}")
        processes.append(ComputeProcess(gpu_uuid=fields[0], pid=pid))
    return processes


def _run_nvidia_smi(nvidia_smi: str, query: str, timeout_seconds: float) -> str:
    completed = subprocess.run(
        [
            nvidia_smi,
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.stdout


def _read_local_gpu_state(
    nvidia_smi: str,
    timeout_seconds: float,
) -> tuple[list[GpuReading], list[ComputeProcess]]:
    gpu_output = _run_nvidia_smi(
        nvidia_smi,
        "gpu=index,uuid,utilization.gpu,memory.free",
        timeout_seconds,
    )
    process_output = _run_nvidia_smi(
        nvidia_smi,
        "compute-apps=gpu_uuid,pid",
        timeout_seconds,
    )
    return _parse_gpu_output(gpu_output), _parse_compute_process_output(process_output)


def _resolve_executable(command: str) -> str:
    if "/" in command:
        return command
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found: {command}")
    return resolved


def _launch_target(
    target_script: Path,
    python_command: str,
    gpu_indices: list[int],
    baseline_foreign_pids_by_gpu: dict[int, set[int]],
) -> ManagedTarget:
    python_executable = _resolve_executable(python_command)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
    process = subprocess.Popen(
        [python_executable, str(target_script)],
        env=environment,
        start_new_session=True,
    )
    return ManagedTarget(
        process=process,
        process_group_id=process.pid,
        gpu_indices=tuple(gpu_indices),
        baseline_foreign_pids_by_gpu={
            index: frozenset(baseline_foreign_pids_by_gpu.get(index, set()))
            for index in gpu_indices
        },
    )


def _pid_in_process_group(pid: int, process_group_id: int) -> bool:
    try:
        return os.getpgid(pid) == process_group_id
    except ProcessLookupError:
        return False


def _foreign_processes_by_gpu(
    readings: list[GpuReading],
    processes: list[ComputeProcess],
    target_process_group_id: int | None,
) -> dict[int, set[int]]:
    index_by_uuid = {reading.uuid: reading.index for reading in readings}
    foreign: dict[int, set[int]] = {}
    for process in processes:
        try:
            index = index_by_uuid[process.gpu_uuid]
        except KeyError as exc:
            raise ValueError(
                f"compute process PID {process.pid} belongs to unknown GPU UUID "
                f"{process.gpu_uuid!r}"
            ) from exc
        if (
            target_process_group_id is not None
            and _pid_in_process_group(process.pid, target_process_group_id)
        ):
            continue
        foreign.setdefault(index, set()).add(process.pid)
    return foreign


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_target(target: ManagedTarget, grace_seconds: float) -> None:
    if not _process_group_exists(target.process_group_id):
        target.process.poll()
        return

    try:
        os.killpg(target.process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        target.process.poll()
        return

    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(target.process_group_id) and time.monotonic() < deadline:
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    if _process_group_exists(target.process_group_id):
        try:
            os.killpg(target.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        target.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _reset_idle_timers(idle_since: dict[int, float], indices: tuple[int, ...]) -> None:
    for index in indices:
        idle_since.pop(index, None)


def _format_processes(processes_by_gpu: dict[int, set[int]]) -> str:
    return ", ".join(
        f"gpu={index}:pid={','.join(str(pid) for pid in sorted(pids))}"
        for index, pids in sorted(processes_by_gpu.items())
    )


def _acquire_lock(lock_file: Path) -> IO[str]:
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_file.open("a+", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot open lock file {lock_file}: {exc}") from exc
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SystemExit(
            f"Another monitor already holds lock file: {lock_file}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvidia-smi",
        default=DEFAULT_NVIDIA_SMI,
        help="Local nvidia-smi executable (default: nvidia-smi).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL_SECONDS:g}).",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=DEFAULT_REQUIRED_IDLE_SECONDS,
        help=(
            "Per-GPU continuous low-utilization duration in seconds "
            f"(default: {DEFAULT_REQUIRED_IDLE_SECONDS})."
        ),
    )
    parser.add_argument(
        "--utilization-threshold-percent",
        type=float,
        default=DEFAULT_UTILIZATION_THRESHOLD_PERCENT,
        help=(
            "Treat a GPU as idle when utilization.gpu is less than or equal to "
            f"this percentage (default: {DEFAULT_UTILIZATION_THRESHOLD_PERCENT:g})."
        ),
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=float,
        default=DEFAULT_MIN_FREE_MEMORY_MIB,
        help=(
            "Strict minimum free memory per eligible GPU in MiB "
            f"(default: {DEFAULT_MIN_FREE_MEMORY_MIB:g})."
        ),
    )
    parser.add_argument(
        "--nvidia-smi-timeout-seconds",
        type=float,
        default=60,
        help="Timeout for each local nvidia-smi call (default: 60).",
    )
    parser.add_argument(
        "--stop-grace-seconds",
        type=float,
        default=DEFAULT_STOP_GRACE_SECONDS,
        help=(
            "Seconds to wait after SIGTERM before SIGKILLing the holder "
            f"process group (default: {DEFAULT_STOP_GRACE_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--target-script",
        type=Path,
        default=DEFAULT_TARGET_SCRIPT,
        help=f"Python script to launch (default: {DEFAULT_TARGET_SCRIPT})",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used to launch the target (default: python).",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_FILE,
        help=f"Local single-instance lock file (default: {DEFAULT_LOCK_FILE}).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.idle_seconds <= 0:
        raise SystemExit("--idle-seconds must be positive")
    if args.utilization_threshold_percent < 0:
        raise SystemExit("--utilization-threshold-percent cannot be negative")
    if args.min_free_memory_mib < 0:
        raise SystemExit("--min-free-memory-mib cannot be negative")
    if args.nvidia_smi_timeout_seconds <= 0:
        raise SystemExit("--nvidia-smi-timeout-seconds must be positive")
    if args.stop_grace_seconds < 0:
        raise SystemExit("--stop-grace-seconds cannot be negative")
    if not args.target_script.is_file():
        raise SystemExit(f"Target script does not exist: {args.target_script}")


def _install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    try:
        nvidia_smi = _resolve_executable(args.nvidia_smi)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    _install_signal_handlers()
    lock_handle = _acquire_lock(args.lock_file)
    idle_since: dict[int, float] = {}
    target: ManagedTarget | None = None
    next_check_at = time.monotonic()
    print(
        "Monitoring local nvidia-smi every "
        f"{args.interval_seconds:g}s; each GPU must stay at <= "
        f"{args.utilization_threshold_percent:g}% for "
        f"{args.idle_seconds / 3600:g}h, have > {args.min_free_memory_mib:g} MiB "
        "free before holder is launched; existing CUDA PIDs are allowed and "
        "recorded as the launch baseline; "
        f"target={args.target_script}",
        flush=True,
    )

    try:
        while True:
            checked_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            try:
                readings, processes = _read_local_gpu_state(
                    nvidia_smi,
                    args.nvidia_smi_timeout_seconds,
                )
                foreign_processes = _foreign_processes_by_gpu(
                    readings,
                    processes,
                    target.process_group_id if target is not None else None,
                )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                idle_since.clear()
                print(
                    f"[{checked_at}] local GPU state unavailable: {exc}; "
                    "idle timers reset",
                    flush=True,
                )
                if target is not None:
                    print(
                        f"[{checked_at}] stopping holder PID {target.process.pid} "
                        "because GPU state can no longer be monitored",
                        flush=True,
                    )
                    _stop_target(target, args.stop_grace_seconds)
                    target = None
            else:
                now = time.monotonic()
                current_indices = {reading.index for reading in readings}
                for index in list(idle_since):
                    if index not in current_indices:
                        del idle_since[index]

                if target is not None and target.process.poll() is not None:
                    print(
                        f"[{checked_at}] holder PID {target.process.pid} exited "
                        f"with status {target.process.returncode}; restarting its "
                        "idle timer",
                        flush=True,
                    )
                    _reset_idle_timers(idle_since, target.gpu_indices)
                    target = None

                if target is not None:
                    conflicts = {
                        index: pids - target.baseline_foreign_pids_by_gpu.get(
                            index, frozenset()
                        )
                        for index, pids in foreign_processes.items()
                        if index in target.gpu_indices
                    }
                    conflicts = {
                        index: pids for index, pids in conflicts.items() if pids
                    }
                    if conflicts:
                        print(
                            f"[{checked_at}] new foreign CUDA process detected on "
                            f"holder GPU(s): {_format_processes(conflicts)}; "
                            f"stopping holder PID {target.process.pid}",
                            flush=True,
                        )
                        _stop_target(target, args.stop_grace_seconds)
                        _reset_idle_timers(idle_since, target.gpu_indices)
                        target = None
                    else:
                        print(
                            f"[{checked_at}] holder PID {target.process.pid} running "
                            "without new foreign CUDA processes on its assigned GPU(s)",
                            flush=True,
                        )

                if target is None:
                    for reading in readings:
                        is_idle = (
                            reading.utilization_percent
                            <= args.utilization_threshold_percent
                        )
                        if is_idle:
                            idle_since.setdefault(reading.index, now)
                        else:
                            idle_since.pop(reading.index, None)

                    eligible = [
                        reading
                        for reading in readings
                        if reading.utilization_percent
                        <= args.utilization_threshold_percent
                        and now - idle_since[reading.index] >= args.idle_seconds
                        and reading.free_memory_mib > args.min_free_memory_mib
                    ]
                    if eligible:
                        gpu_indices = [reading.index for reading in eligible]
                        memory_summary = ", ".join(
                            f"gpu={reading.index}:{reading.free_memory_mib:.0f}MiB"
                            for reading in eligible
                        )
                        print(
                            f"[{checked_at}] launching holder on local GPU(s): "
                            f"{memory_summary}; CUDA_VISIBLE_DEVICES="
                            f"{','.join(map(str, gpu_indices))}",
                            flush=True,
                        )
                        try:
                            target = _launch_target(
                                args.target_script,
                                args.python,
                                gpu_indices,
                                foreign_processes,
                            )
                        except (OSError, subprocess.SubprocessError) as exc:
                            print(
                                f"[{checked_at}] failed to launch holder: {exc}",
                                flush=True,
                            )
                            _reset_idle_timers(idle_since, tuple(gpu_indices))
                        else:
                            print(
                                f"[{checked_at}] holder launched with PID "
                                f"{target.process.pid}",
                                flush=True,
                            )
                    else:
                        summaries: list[str] = []
                        for reading in readings:
                            existing_process_note = ""
                            if reading.index in foreign_processes:
                                foreign_pids = ",".join(
                                    str(pid)
                                    for pid in sorted(foreign_processes[reading.index])
                                )
                                existing_process_note = (
                                    f" existing_foreign_pid={foreign_pids}"
                                )
                            if (
                                reading.utilization_percent
                                > args.utilization_threshold_percent
                            ):
                                summaries.append(
                                    f"gpu={reading.index} busy="
                                    f"{reading.utilization_percent:.0f}%"
                                    f"{existing_process_note}"
                                )
                                continue
                            idle_for = now - idle_since[reading.index]
                            if reading.free_memory_mib <= args.min_free_memory_mib:
                                summaries.append(
                                    f"gpu={reading.index} free="
                                    f"{reading.free_memory_mib:.0f}MiB <= "
                                    f"{args.min_free_memory_mib:g}MiB"
                                    f"{existing_process_note}"
                                )
                            else:
                                remaining = max(0.0, args.idle_seconds - idle_for)
                                summaries.append(
                                    f"gpu={reading.index} idle_for="
                                    f"{idle_for / 3600:.2f}h remaining="
                                    f"{remaining / 3600:.2f}h free="
                                    f"{reading.free_memory_mib:.0f}MiB"
                                    f"{existing_process_note}"
                                )
                        print(
                            f"[{checked_at}] no eligible local GPU; "
                            + "; ".join(summaries),
                            flush=True,
                        )

            next_check_at += args.interval_seconds
            sleep_seconds = next_check_at - time.monotonic()
            if sleep_seconds <= 0:
                next_check_at = time.monotonic()
                continue
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("\nMonitor interrupted.", flush=True)
        return 130
    finally:
        if target is not None:
            print(
                f"Stopping holder PID {target.process.pid} before monitor exits.",
                flush=True,
            )
            _stop_target(target, args.stop_grace_seconds)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
