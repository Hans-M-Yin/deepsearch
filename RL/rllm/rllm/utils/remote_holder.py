"""Drive monitor_v2 at synchronous RL rollout boundaries (driver only)."""
from contextlib import contextmanager
import json
import logging
import math
import os
import time
import urllib.request
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class RemoteHolder:
    def __init__(self, url="", token="", timeout=30.0, attempts=3):
        self.url = url.strip().rstrip("/")
        self.token = token
        self.timeout = timeout
        self.attempts = attempts
        if self.url:
            parts = urlsplit(self.url)
            if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("RLLM_HOLDER_MONITOR_URL must be an HTTP base URL without credentials/query/fragment")
        if not math.isfinite(timeout) or timeout <= 0 or attempts < 1:
            raise ValueError("holder timeout and attempts must be positive")
        # Monitor is an internal control endpoint; do not send commands/tokens to HTTP proxies.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_environment(cls):
        return cls(
            url=os.environ.get("RLLM_HOLDER_MONITOR_URL", ""),
            token=os.environ.get("RLLM_HOLDER_MONITOR_TOKEN", ""),
            timeout=float(os.environ.get("RLLM_HOLDER_MONITOR_TIMEOUT", "30")),
        )

    def command(self, action, required=False):
        if action not in {"start", "stop"}:
            raise ValueError("holder action must be start or stop")
        if not self.url:
            return False
        headers = {}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        last_error = None
        for attempt in range(self.attempts):
            try:
                request = urllib.request.Request(self.url + "/" + action, data=b"", headers=headers, method="POST")
                with self.opener.open(request, timeout=self.timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"monitor HTTP status {response.status}")
                    result = json.loads(response.read(65536))
                if not isinstance(result, dict) or result.get("running") is not (action == "start"):
                    raise RuntimeError("monitor response did not confirm requested holder state")
                logger.info("[RL_HOLDER] %s confirmed at %s", action, self.url)
                return True
            except Exception as exc:
                last_error = exc
                logger.warning("[RL_HOLDER] %s failed (%d/%d): %s", action, attempt + 1, self.attempts, type(exc).__name__)
                if attempt + 1 < self.attempts:
                    time.sleep(1)
        if required:
            raise RuntimeError("Remote holder stop was not confirmed; refusing to begin rollout") from last_error
        logger.warning("[RL_HOLDER] %s was not confirmed; continuing without guaranteed holder state", action)
        return False

    @contextmanager
    def rollout(self):
        self.command("stop", required=True)
        # Only a successful rollout starts holder; failures may leave judge work in flight.
        yield
        self.command("start")

    def close(self):
        self.command("stop")
