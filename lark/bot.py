"""Minimal single-user Feishu bot for executing commands on a Linux host.

The process uses Feishu's long-connection mode to receive messages and calls
the Feishu HTTP API to send periodic progress, completion, or failure replies.

This first version executes commands on the host where this process runs. If
your cluster uses Slurm, Kubernetes, or another scheduler, replace
``run_shell_command`` with the corresponding execution adapter.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import signal
import sqlite3
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import lark_oapi as lark
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - environment files are optional
    load_dotenv = None


LOGGER = logging.getLogger("lark_command_bot")
PROGRESS_INTERVAL_SECONDS = 60
PROGRESS_LINE_COUNT = 30


def _load_environment() -> None:
    if load_dotenv is not None:
        # Keep deployment secrets in the project-local file requested by the
        # deployment instructions. Existing process environment variables
        # still take precedence by default.
        load_dotenv(Path(__file__).with_name(".lark_env"))


def _csv_env(name: str) -> frozenset[str]:
    return frozenset(
        item.strip()
        for item in os.getenv(name, "").split(",")
        if item.strip()
    )


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)


@dataclass(frozen=True)
class Config:
    app_id: str
    app_secret: str
    allowed_open_ids: frozenset[str]
    allowed_tenant_keys: frozenset[str]
    command_cwd: str
    command_timeout_seconds: int
    max_output_chars: int
    state_db: str
    max_workers: int
    log_level: str
    api_base_url: str

    @classmethod
    def from_environment(cls) -> "Config":
        app_id = os.getenv("LARK_APP_ID", "").strip()
        app_secret = os.getenv("LARK_APP_SECRET", "").strip()
        if not app_id or not app_secret:
            raise RuntimeError("LARK_APP_ID and LARK_APP_SECRET are required")

        default_state_db = Path(__file__).with_name("lark_state.sqlite3")
        command_cwd = os.path.abspath(
            os.path.expanduser(
                os.getenv("LARK_COMMAND_CWD", str(Path(__file__).parent))
            )
        )
        if not os.path.isdir(command_cwd):
            raise RuntimeError(f"LARK_COMMAND_CWD is not a directory: {command_cwd}")

        return cls(
            app_id=app_id,
            app_secret=app_secret,
            allowed_open_ids=_csv_env("LARK_ALLOWED_OPEN_IDS"),
            allowed_tenant_keys=_csv_env("LARK_ALLOWED_TENANT_KEYS"),
            command_cwd=command_cwd,
            command_timeout_seconds=_int_env(
                "LARK_COMMAND_TIMEOUT_SECONDS", default=300, minimum=1, maximum=3600
            ),
            max_output_chars=_int_env(
                "LARK_MAX_OUTPUT_CHARS", default=12000, minimum=1000, maximum=100000
            ),
            state_db=os.path.abspath(
                os.path.expanduser(os.getenv("LARK_STATE_DB", str(default_state_db)))
            ),
            max_workers=_int_env("LARK_MAX_WORKERS", default=1, minimum=1, maximum=8),
            log_level=os.getenv("LARK_LOG_LEVEL", "INFO").upper(),
            api_base_url=os.getenv(
                "LARK_API_BASE_URL", "https://open.feishu.cn/open-apis"
            ).rstrip("/"),
        )


class MessageState:
    """Persistent message-id deduplication without an external database."""

    def __init__(self, database_path: str) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(path)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    first_seen_at INTEGER NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=30)

    def claim(self, message_id: str) -> bool:
        """Return True only for the first observation of a message."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO processed_messages(message_id, first_seen_at) "
                "VALUES (?, ?)",
                (message_id, int(time.time())),
            )
            return cursor.rowcount == 1


class LarkApi:
    """Small HTTP client for tenant token acquisition and message replies."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self._token: str | None = None
        self._token_expire_at = 0.0
        self._token_lock = threading.Lock()

    def _get_tenant_access_token(self, force_refresh: bool = False) -> str:
        with self._token_lock:
            if (
                not force_refresh
                and self._token
                and time.time() < self._token_expire_at
            ):
                return self._token

            response = self.session.post(
                f"{self.config.api_base_url}/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.config.app_id,
                    "app_secret": self.config.app_secret,
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(
                    f"failed to obtain tenant_access_token: {payload}"
                )

            self._token = payload["tenant_access_token"]
            expire_seconds = int(payload.get("expire", 7200))
            self._token_expire_at = time.time() + max(expire_seconds - 60, 60)
            return self._token

    def reply_text(self, message_id: str, text: str) -> None:
        url_message_id = quote(message_id, safe="")
        body = {
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }

        for attempt in range(2):
            token = self._get_tenant_access_token(force_refresh=attempt == 1)
            response = self.session.post(
                f"{self.config.api_base_url}/im/v1/messages/{url_message_id}/reply",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") == 0:
                return

            # Retry once if the cached access token has become invalid.
            if str(payload.get("code")) != "99991663" or attempt == 1:
                raise RuntimeError(f"failed to reply to Feishu message: {payload}")

        raise RuntimeError("failed to reply to Feishu message")


@dataclass
class ExecutionResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def _append_tail(buffer: bytearray, chunk: bytes, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    buffer.extend(chunk)
    if len(buffer) > max_bytes:
        del buffer[: len(buffer) - max_bytes]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_shell_command(
    command: str,
    config: Config,
    on_progress: Callable[[str, int], None] | None = None,
) -> ExecutionResult:
    """Run a command locally and optionally report live output periodically."""
    start_time = time.monotonic()
    deadline = start_time + config.command_timeout_seconds
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=config.command_cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    max_bytes = config.max_output_chars * 4
    stdout_tail = bytearray()
    stderr_tail = bytearray()
    line_buffers = {"stdout": bytearray(), "stderr": bytearray()}
    latest_lines: deque[str] = deque(maxlen=PROGRESS_LINE_COUNT)
    next_progress_at = start_time + PROGRESS_INTERVAL_SECONDS
    timed_out = False

    def consume_lines(stream_name: str, chunk: bytes) -> None:
        buffer = line_buffers[stream_name]
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            del buffer[: len(buffer) - max_bytes]

        while b"\n" in buffer:
            raw_line, _, remaining = buffer.partition(b"\n")
            line_buffers[stream_name] = bytearray(remaining)
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
            latest_lines.append(f"[{stream_name}] {line}")
            buffer = line_buffers[stream_name]

    def progress_snapshot() -> str:
        lines = list(latest_lines)
        for stream_name, buffer in line_buffers.items():
            if buffer:
                partial = buffer.decode("utf-8", errors="replace")
                lines.append(f"[{stream_name}] {partial}")
        if not lines:
            return "暂无输出。"
        return "\n".join(lines[-PROGRESS_LINE_COUNT:])

    def maybe_report_progress() -> None:
        nonlocal next_progress_at
        if on_progress is None or process.poll() is not None:
            return
        now = time.monotonic()
        if now < next_progress_at:
            return
        elapsed_seconds = int(now - start_time)
        on_progress(progress_snapshot(), elapsed_seconds)
        next_progress_at = time.monotonic() + PROGRESS_INTERVAL_SECONDS

    with selectors.DefaultSelector() as selector:
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        while selector.get_map():
            if process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _terminate_process_group(process)
                    break
                wait_timeout = min(remaining, 0.5)
            else:
                # The process has exited; finish draining both pipes.
                wait_timeout = 0.5

            for key, _ in selector.select(wait_timeout):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    _append_tail(stdout_tail, chunk, max_bytes)
                else:
                    _append_tail(stderr_tail, chunk, max_bytes)
                consume_lines(key.data, chunk)

            maybe_report_progress()

    if process.poll() is None:
        process.wait()

    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()

    return ExecutionResult(
        returncode=process.returncode,
        stdout=bytes(stdout_tail).decode("utf-8", errors="replace"),
        stderr=bytes(stderr_tail).decode("utf-8", errors="replace"),
        timed_out=timed_out,
    )


def _tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "...[内容过长，已截取末尾]...\n" + text[-max_chars:]


def _event_as_dict(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    serialized = lark.JSON.marshal(data)
    if isinstance(serialized, bytes):
        serialized = serialized.decode("utf-8")
    return json.loads(serialized)


def _message_text(content: Any) -> str:
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if not isinstance(content, str):
        return ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict):
        return str(parsed.get("text", ""))
    return ""


def _parse_command(raw_command: str) -> tuple[str, bool]:
    """Strip the optional SLICENCE prefix and return its progress mode."""
    prefix = "SLICENCE"
    if raw_command == prefix:
        return "", True

    remainder = raw_command[len(prefix) :]
    if raw_command.startswith(prefix) and remainder and (
        remainder[0].isspace() or remainder[0] == ":"
    ):
        return remainder.lstrip(" \t\r\n:"), True
    return raw_command, False


class CommandBot:
    def __init__(
        self, config: Config, state: MessageState, lark_api: LarkApi
    ) -> None:
        self.config = config
        self.state = state
        self.lark_api = lark_api
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

    def _is_allowed(self, open_id: str, tenant_key: str) -> bool:
        if open_id not in self.config.allowed_open_ids:
            return False
        if (
            self.config.allowed_tenant_keys
            and tenant_key not in self.config.allowed_tenant_keys
        ):
            return False
        return True

    def _send_reply(self, message_id: str, text: str) -> None:
        try:
            self.lark_api.reply_text(message_id, text)
        except Exception:
            LOGGER.exception("failed to send Feishu reply: message_id=%s", message_id)

    def _reply_progress(
        self, message_id: str, output: str, elapsed_seconds: int
    ) -> None:
        progress_text = (
            f"运行中（已运行 {elapsed_seconds} 秒）\n"
            f"最近 {PROGRESS_LINE_COUNT} 行输出：\n"
            f"{output}"
        )
        self._send_reply(
            message_id,
            _tail_text(progress_text, self.config.max_output_chars),
        )

    def _execute_and_report(
        self, message_id: str, command: str, suppress_progress: bool
    ) -> None:
        try:
            result = run_shell_command(
                command,
                self.config,
                on_progress=(
                    None
                    if suppress_progress
                    else lambda output, elapsed: self._reply_progress(
                        message_id, output, elapsed
                    )
                ),
            )
            if not result.timed_out and result.returncode == 0:
                LOGGER.info("command succeeded: message_id=%s", message_id)
                self._send_reply(message_id, "Done!")
                return

            if result.timed_out:
                headline = (
                    "命令执行超时："
                    f"超过 {self.config.command_timeout_seconds} 秒"
                )
            else:
                headline = f"命令执行失败，退出码：{result.returncode}"

            output_parts = [headline, f"$ {command}"]
            if result.stderr.strip():
                output_parts.append(f"stderr:\n{result.stderr.strip()}")
            if result.stdout.strip():
                output_parts.append(f"stdout:\n{result.stdout.strip()}")
            if len(output_parts) == 2:
                output_parts.append("命令没有输出。")

            self._send_reply(
                message_id,
                _tail_text("\n\n".join(output_parts), self.config.max_output_chars),
            )
        except Exception as exc:
            LOGGER.exception("command execution failed unexpectedly: message_id=%s", message_id)
            self._send_reply(message_id, f"机器人执行命令时发生异常：{exc}")

    def handle_message(self, data: Any) -> None:
        """Receive one SDK event and return quickly; execution is asynchronous."""
        try:
            payload = _event_as_dict(data)
            event = payload.get("event", {})
            sender = event.get("sender", {})
            sender_id = sender.get("sender_id", {})
            message = event.get("message", {})
            header = payload.get("header", {})

            message_id = str(message.get("message_id", ""))
            open_id = str(sender_id.get("open_id", ""))
            tenant_key = str(
                sender.get("tenant_key") or header.get("tenant_key") or ""
            )
            sender_type = sender.get("sender_type")

            if not message_id or sender_type != "user":
                return
            if message.get("chat_type") != "p2p":
                return
            if not self._is_allowed(open_id, tenant_key):
                LOGGER.warning(
                    "rejected message: tenant_key=%s open_id=%s message_id=%s",
                    tenant_key,
                    open_id,
                    message_id,
                )
                return
            if not self.state.claim(message_id):
                LOGGER.info("ignored duplicate message: message_id=%s", message_id)
                return
            if message.get("message_type") != "text":
                LOGGER.info("ignored non-text message: message_id=%s", message_id)
                return

            command, suppress_progress = _parse_command(
                _message_text(message.get("content")).strip()
            )
            if not command:
                self.executor.submit(
                    self._send_reply, message_id, "没有收到可执行的命令。"
                )
                return

            LOGGER.info(
                "accepted command: tenant_key=%s open_id=%s message_id=%s "
                "suppress_progress=%s",
                tenant_key,
                open_id,
                message_id,
                suppress_progress,
            )
            self.executor.submit(
                self._execute_and_report,
                message_id,
                command,
                suppress_progress,
            )
        except Exception:
            # Never let malformed input make Feishu retry the event indefinitely.
            LOGGER.exception("failed to process Feishu message event")


def main() -> None:
    _load_environment()
    config = Config.from_environment()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not config.allowed_open_ids:
        LOGGER.warning(
            "LARK_ALLOWED_OPEN_IDS is empty; all users will be rejected until it is set"
        )

    state = MessageState(config.state_db)
    lark_api = LarkApi(config)
    command_bot = CommandBot(config, state, lark_api)

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(command_bot.handle_message)
        .build()
    )
    sdk_log_level = getattr(lark.LogLevel, config.log_level, lark.LogLevel.DEBUG)
    client = lark.ws.Client(
        config.app_id,
        config.app_secret,
        event_handler=event_handler,
        log_level=sdk_log_level,
    )

    LOGGER.info("starting Feishu long-connection bot")
    client.start()


if __name__ == "__main__":
    main()
