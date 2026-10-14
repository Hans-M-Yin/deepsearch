#!/usr/bin/env python3
"""HTTP start/stop controller for one holder.py process group on local GPUs.

POST /start: start holder on all configured GPUs (idempotent).
POST /stop: stop this controller's holder and its children (idempotent).
GET /status: inspect process state. See monitor_v2.md for Python examples.
Uses only the Python standard library; holder's interpreter needs PyTorch.
"""
from __future__ import annotations

import argparse
import fcntl
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

LOG = logging.getLogger("monitor-v2")
DEFAULT_HOLDER = "/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/verl/holder.py"


def resolve_gpus(value: str | None, nvidia_smi: str) -> list[str]:
    if value is None:
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        result = subprocess.run([nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=10)
        value = ",".join(result.stdout.splitlines())
    gpus = [v.strip() for v in value.split(",") if v.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or any(
        not re.fullmatch(r"\d+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9/.-]+", v) for v in gpus
    ):
        raise ValueError("GPU list must contain distinct indices/UUIDs, e.g. --gpus 0,1,2,3")
    return gpus


def live_group_members(pgid: int) -> list[int]:
    """Linux process groups, ignoring exited zombies which cannot be signalled."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] not in ("Z", "X") and int(fields[2]) == pgid:
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue  # process exited while reading /proc
    return members


class HolderManager:
    def __init__(self, script: Path, python: str, gpus: list[str], log_file: Path, grace: float):
        self.script, self.python, self.gpus = script, python, gpus
        self.log_file, self.grace = log_file, grace
        self.process = None
        self.lock = threading.RLock()
        self.closed = False
        self.last_exit_code = None

    def status(self):
        with self.lock:
            code = self.process.poll() if self.process is not None else None
            if code is not None:
                self.last_exit_code = code
            return {
                "running": self.process is not None and code is None,
                "pid": self.process.pid if self.process is not None else None,
                "gpus": self.gpus,
                "last_exit_code": self.last_exit_code,
                "holder_log": str(self.log_file),
            }

    def start(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("monitor is shutting down")
            if self.process is not None and self.process.poll() is None:
                return {"action": "already_running", **self.status()}
            # A crashed parent may leave children behind; clean them before relaunch.
            self.stop()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
            with self.log_file.open("ab", buffering=0) as output:
                self.process = subprocess.Popen(
                    [self.python, "-u", str(self.script)], cwd=str(self.script.parent),
                    env=environment, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
            LOG.info("started holder pid=%s GPUs=%s", self.process.pid, ",".join(self.gpus))
            return {"action": "started", **self.status()}

    def stop(self):
        with self.lock:
            if self.process is None:
                return {"action": "already_stopped", **self.status()}
            process = self.process
            pgid = process.pid  # start_new_session gives holder its own process group
            process.poll()
            if live_group_members(pgid):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + self.grace
                while time.monotonic() < deadline:
                    process.poll()  # reap parent; zombies are not live group members
                    if not live_group_members(pgid):
                        break
                    time.sleep(0.05)
                if live_group_members(pgid):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while live_group_members(pgid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if live_group_members(pgid):
                raise RuntimeError(f"holder group {pgid} has not exited; retry /stop")
            self.last_exit_code = process.returncode
            self.process = None
            LOG.info("stopped holder process group=%s", pgid)
            return {"action": "stopped", **self.status()}

    def close(self):
        with self.lock:
            self.closed = True
            self.stop()


def make_handler(manager: HolderManager, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                self.reply(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path in ("/status", "/health"):
                self.reply(200, manager.status())
            else:
                self.reply(404, {"error": "use GET /status, POST /start or POST /stop"})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path not in ("/start", "/stop"):
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 4096 or self.headers.get("Transfer-Encoding"):
                    self.reply(400, {"error": "send an empty request or a small body with Content-Length"})
                    return
                if length:
                    self.rfile.read(length)
                result = manager.start() if self.path == "/start" else manager.stop()
                self.reply(200, result)
            except Exception as exc:
                LOG.exception("%s failed", self.path)
                self.reply(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            LOG.info("client=%s %s", self.client_address[0], fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--gpus", help="GPU indices/UUIDs, e.g. 0,1,2,3; default: CUDA_VISIBLE_DEVICES or all local GPUs")
    parser.add_argument("--target-script", type=Path, default=Path(DEFAULT_HOLDER))
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for holder; must have PyTorch/CUDA")
    parser.add_argument("--holder-log", type=Path, default=Path("/tmp/monitor_v2_holder.log"))
    parser.add_argument("--stop-grace-seconds", type=float, default=5)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--token-env", default="MONITOR_V2_TOKEN", help="Optional Bearer token environment variable")
    parser.add_argument("--lock-file", type=Path, help="Default: /tmp/monitor_v2_PORT.lock")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.stop_grace_seconds) or args.stop_grace_seconds < 0:
        parser.error("--stop-grace-seconds must be finite and nonnegative")
    script = args.target_script.resolve()
    if not script.is_file():
        parser.error(f"holder script does not exist: {script}")
    python = shutil.which(args.python)
    if python is None:
        parser.error(f"Python executable not found: {args.python}")
    try:
        gpus = resolve_gpus(args.gpus, args.nvidia_smi)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    args.holder_log.parent.mkdir(parents=True, exist_ok=True)
    lock_file = args.lock_file or Path(f"/tmp/monitor_v2_{args.port}.lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with lock_file.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another monitor_v2 already holds this lock file")
        manager = HolderManager(script, python, gpus, args.holder_log.resolve(), args.stop_grace_seconds)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(manager, os.environ.get(args.token_env)))
        def stop_monitor(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_monitor)
        LOG.info("listening on %s:%s GPUs=%s; holder waits for POST /start", args.host, args.port, ",".join(gpus))
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            # Block future starts before closing listeners and managed processes.
            manager.close()
            server.server_close()


if __name__ == "__main__":
    main()
