#!/usr/bin/env python3
"""Launch independent SGLang workers on a Ray cluster and front them with SMG.

The process intentionally does not use SGLang's native multi-node TP launch.
For a 2-node x 8-GPU cluster, four independent TP4 workers avoid making a
single TP group cross the node boundary and leave routing/failure handling to
the Model Gateway.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import ray


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def find_free_tcp_port() -> int:
    """Ask the kernel for a free local TCP port for the Router metrics endpoint."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def csv_values(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def quote_host(host: str) -> str:
    """Return an IPv4/IPv6 host suitable for an HTTP URL."""
    host = host.strip().strip("[]")
    return f"[{host}]" if ":" in host else host


def http_url(host: str, port: int, path: str = "") -> str:
    return f"http://{quote_host(host)}:{port}{path}"


def wait_http(urls: list[str], timeout_s: float, interval_s: float = 2.0) -> None:
    """Wait until one of the candidate health URLs returns HTTP 200."""
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=min(10.0, max(1.0, deadline - time.monotonic()))) as response:
                    if response.status == 200:
                        return
                    last_error = RuntimeError(f"HTTP {response.status} from {url}")
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for health endpoint {urls}; last_error={last_error!r}")


def wait_process_http(
    process: subprocess.Popen[Any], urls: list[str], timeout_s: float, interval_s: float = 2.0
) -> None:
    """Wait for health while failing quickly if the child exits during startup."""
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"SGLang worker exited before becoming healthy: returncode={returncode}"
            )
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=min(10.0, max(1.0, deadline - time.monotonic()))) as response:
                    if response.status == 200:
                        return
                    last_error = RuntimeError(f"HTTP {response.status} from {url}")
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for health endpoint {urls}; last_error={last_error!r}")


def command_help(python: str, module: str) -> str:
    result = subprocess.run(
        [python, "-m", module, "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Unable to inspect {module}; exit_code={result.returncode}\n{result.stdout}"
        )
    return result.stdout


def first_supported(help_text: str, *options: str) -> str | None:
    for option in options:
        if option in help_text:
            return option
    return None


@dataclass(frozen=True)
class Config:
    python: str
    model_path: str
    served_model_name: str
    router_host: str
    router_port: int
    worker_port_base: int
    tensor_parallel_size: int
    data_parallel_size: int
    node_count: int
    replicas_per_node: int
    max_model_len: int
    mem_fraction_static: float
    max_running_requests: int
    chunked_prefill_size: int
    attention_backend: str
    mm_attention_backend: str
    router_policy: str
    router_log_level: str
    router_request_timeout_s: int
    router_prometheus_port: int
    startup_timeout_s: int
    restart_delay_s: float
    ray_address: str
    ray_namespace: str | None
    node_ips: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "Config":
        tp = env_int("TENSOR_PARALLEL_SIZE", 4)
        dp = env_int("DATA_PARALLEL_SIZE", 4)
        node_count = env_int("RAY_NODE_COUNT", 2)
        replicas_per_node = env_int("REPLICAS_PER_NODE", max(1, dp // node_count))
        if dp != node_count * replicas_per_node:
            raise ValueError(
                "DATA_PARALLEL_SIZE must equal RAY_NODE_COUNT * REPLICAS_PER_NODE; "
                f"got {dp} != {node_count} * {replicas_per_node}"
            )
        node_ips = tuple(csv_values("RAY_NODE_IPS"))
        if node_ips and len(node_ips) != node_count:
            raise ValueError(
                f"RAY_NODE_IPS must contain exactly {node_count} comma-separated addresses; "
                f"got {len(node_ips)}"
            )
        mem_fraction_static = env_float(
            "MEM_FRACTION_STATIC", 0.90 if tp >= 4 else 0.88
        )
        if mem_fraction_static >= 1.0:
            raise ValueError("MEM_FRACTION_STATIC must be smaller than 1.0")

        configured_prometheus_port = os.environ.get("ROUTER_PROMETHEUS_PORT", "").strip()
        if configured_prometheus_port:
            router_prometheus_port = env_int(
                "ROUTER_PROMETHEUS_PORT", int(configured_prometheus_port)
            )
        else:
            # Avoid the common SGLang default (29000) and any locally reserved
            # port range, while still allowing an explicit port through
            # ROUTER_PROMETHEUS_PORT.
            router_prometheus_port = find_free_tcp_port()

        return cls(
            python=os.environ.get("SGLANG_PYTHON", sys.executable),
            model_path=os.environ.get(
                "MODEL_PATH",
                "/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-vl-32b",
            ),
            served_model_name=os.environ.get("SERVED_MODEL_NAME", "Qwen3-VL-32B"),
            router_host=os.environ.get("SGLANG_HOST", "0.0.0.0"),
            router_port=env_int("SGLANG_PORT", 6658),
            worker_port_base=env_int("WORKER_PORT_BASE", 7001),
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            node_count=node_count,
            replicas_per_node=replicas_per_node,
            max_model_len=env_int("MAX_MODEL_LEN", 48000),
            # 0.90 leaves a small safety margin while making more KV cache
            # available than the previous 0.88 default.
            mem_fraction_static=mem_fraction_static,
            # This is per worker. Four workers provide roughly 4x this
            # request capacity at the router, subject to token/KV pressure.
            max_running_requests=env_int(
                "MAX_RUNNING_REQUESTS", 64 if tp >= 4 else 48
            ),
            # Smaller chunks improve prefill/decode overlap for long contexts.
            chunked_prefill_size=env_int("CHUNKED_PREFILL_SIZE", 8192),
            attention_backend=os.environ.get("ATTENTION_BACKEND", "fa3"),
            mm_attention_backend=os.environ.get("MM_ATTENTION_BACKEND", "fa3"),
            router_policy=os.environ.get("ROUTER_POLICY", "cache_aware"),
            router_log_level=os.environ.get("ROUTER_LOG_LEVEL", "warn"),
            router_request_timeout_s=env_int("ROUTER_REQUEST_TIMEOUT_S", 1800),
            router_prometheus_port=router_prometheus_port,
            startup_timeout_s=env_int("STARTUP_TIMEOUT_S", 1800),
            restart_delay_s=env_float("RESTART_DELAY_S", 2.0),
            ray_address=os.environ.get("RAY_ADDRESS", "auto"),
            ray_namespace=os.environ.get("RAY_NAMESPACE") or None,
            node_ips=node_ips,
        )


def worker_command(config: Config, help_text: str, host: str, port: int) -> list[str]:
    command = [config.python, "-m", "sglang.launch_server"]

    model_flag = first_supported(help_text, "--model-path", "--model")
    if model_flag is None:
        raise RuntimeError("sglang.launch_server exposes neither --model-path nor --model")
    command += [model_flag, config.model_path]

    tp_flag = first_supported(help_text, "--tp-size", "--tp")
    if tp_flag is None:
        raise RuntimeError("sglang.launch_server exposes neither --tp-size nor --tp")
    command += [tp_flag, str(config.tensor_parallel_size)]

    required_options: list[tuple[tuple[str, ...], str]] = [
        (("--served-model-name",), config.served_model_name),
        (("--context-length",), str(config.max_model_len)),
        (("--host",), "0.0.0.0"),
        (("--port",), str(port)),
        (("--mem-fraction-static",), str(config.mem_fraction_static)),
        (("--max-running-requests",), str(config.max_running_requests)),
        (("--chunked-prefill-size",), str(config.chunked_prefill_size)),
        (("--attention-backend",), config.attention_backend),
        (("--mm-attention-backend",), config.mm_attention_backend),
    ]
    for options, value in required_options:
        flag = first_supported(help_text, *options)
        if flag is None:
            print(f"[sglang-ray] warning: skipping unsupported option {options[0]}", flush=True)
        else:
            command += [flag, value]

    for flag in ("--enable-multimodal", "--trust-remote-code"):
        if flag in help_text:
            command.append(flag)
        else:
            print(f"[sglang-ray] warning: skipping unsupported option {flag}", flush=True)

    if "--log-level" in help_text:
        command += ["--log-level", os.environ.get("SGLANG_WORKER_LOG_LEVEL", "error")]

    # Mixed chunking can improve prefill/decode overlap for long-context,
    # high-concurrency workloads.  It is opt-out and version-gated so older
    # SGLang releases keep working unchanged.
    if os.environ.get("ENABLE_MIXED_CHUNK", "1").lower() in {"1", "true", "yes"}:
        if "--enable-mixed-chunk" in help_text:
            command.append("--enable-mixed-chunk")
    return command


def router_module_and_help(config: Config) -> tuple[str, str]:
    requested = os.environ.get("ROUTER_MODULE", "").strip()
    candidates = [requested] if requested else [
        "sglang_router.launch_router",
        "sglang_router.launch_server",
    ]
    for module in candidates:
        if not module:
            continue
        try:
            if importlib.util.find_spec(module) is None:
                continue
            return module, command_help(config.python, module)
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            if requested:
                raise
            print(f"[sglang-ray] router module {module} unavailable: {exc}", flush=True)
    raise RuntimeError(
        "No SGLang Router launcher found. Install sglang-router or set ROUTER_MODULE."
    )


def router_command(config: Config, module: str, help_text: str, worker_urls: list[str]) -> list[str]:
    required = first_supported(help_text, "--worker-urls")
    if required is None:
        raise RuntimeError(
            f"{module} does not expose --worker-urls; separate-worker mode is unavailable."
        )

    command = [config.python, "-m", module, "--host", config.router_host, "--port", str(config.router_port)]
    command += [required, *worker_urls]

    policy_flag = first_supported(help_text, "--policy", "--router-policy")
    if policy_flag is not None:
        command += [policy_flag, config.router_policy]

    log_flag = first_supported(help_text, "--log-level", "--router-log-level")
    if log_flag is not None:
        command += [log_flag, config.router_log_level]

    timeout_flag = first_supported(
        help_text,
        "--request-timeout-secs",
        "--router-request-timeout-secs",
    )
    if timeout_flag is not None:
        command += [timeout_flag, str(config.router_request_timeout_s)]

    prometheus_flag = first_supported(
        help_text,
        "--router-prometheus-port",
        "--prometheus-port",
    )
    if prometheus_flag is not None:
        command += [prometheus_flag, str(config.router_prometheus_port)]
    else:
        print(
            "[sglang-ray] warning: Router has no recognized Prometheus port option; "
            "it may fall back to its default port",
            flush=True,
        )
    return command


@ray.remote(num_cpus=1)
class SGLangWorkerProcess:
    def __init__(self, command: list[str], port: int, startup_timeout_s: int):
        self.command = command
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[Any] | None = None

    def start(self) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            return self.info()

        print(
            f"[sglang-ray] starting worker port={self.port} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"command={' '.join(self.command)}",
            flush=True,
        )
        self.process = subprocess.Popen(
            self.command,
            env=os.environ.copy(),
            start_new_session=True,
        )
        try:
            wait_process_http(
                self.process,
                [
                    f"http://127.0.0.1:{self.port}/health",
                    f"http://127.0.0.1:{self.port}/health_generate",
                ],
                timeout_s=self.startup_timeout_s,
            )
        except BaseException:
            self.stop()
            raise
        return self.info()

    def info(self) -> dict[str, Any]:
        return {
            "pid": self.process.pid if self.process is not None else None,
            "returncode": self.process.poll() if self.process is not None else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.process = None


def select_nodes(config: Config) -> list[dict[str, Any]]:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    requested_ips = set(config.node_ips)
    if requested_ips:
        nodes = [node for node in alive_nodes if node.get("NodeManagerAddress") in requested_ips]
        missing = requested_ips - {node.get("NodeManagerAddress") for node in nodes}
        if missing:
            raise RuntimeError(f"RAY_NODE_IPS not found among alive Ray nodes: {sorted(missing)}")
    else:
        nodes = alive_nodes

    nodes = [node for node in nodes if float(node.get("Resources", {}).get("GPU", 0)) >= config.replicas_per_node * config.tensor_parallel_size]
    nodes.sort(key=lambda node: str(node.get("NodeManagerAddress", "")))
    if len(nodes) < config.node_count:
        summary = [
            (node.get("NodeManagerAddress"), node.get("Resources", {}).get("GPU", 0))
            for node in alive_nodes
        ]
        raise RuntimeError(
            f"Need {config.node_count} Ray nodes with at least "
            f"{config.replicas_per_node * config.tensor_parallel_size} GPUs each; "
            f"available={summary}"
        )
    return nodes[: config.node_count]


class Fleet:
    def __init__(self, config: Config):
        self.config = config
        self.actors: list[ray.actor.ActorHandle] = []
        self.router: subprocess.Popen[Any] | None = None

    def launch(self) -> None:
        nodes = select_nodes(self.config)
        sglang_help = command_help(self.config.python, "sglang.launch_server")
        router_module, router_help = router_module_and_help(self.config)
        worker_urls: list[str] = []

        print(
            f"[sglang-ray] topology={self.config.data_parallel_size}xTP{self.config.tensor_parallel_size} "
            f"nodes={[(n.get('NodeManagerAddress'), n.get('NodeID')) for n in nodes]} "
            f"router_module={router_module}",
            flush=True,
        )

        for node_index, node in enumerate(nodes):
            node_id = node["NodeID"]
            node_ip = str(node["NodeManagerAddress"])
            for local_replica in range(self.config.replicas_per_node):
                replica_index = node_index * self.config.replicas_per_node + local_replica
                port = self.config.worker_port_base + local_replica
                command = worker_command(self.config, sglang_help, node_ip, port)
                actor = SGLangWorkerProcess.options(
                    num_gpus=self.config.tensor_parallel_size,
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                    name=f"qwen3_vl_32b_tp{self.config.tensor_parallel_size}_replica_{replica_index}",
                ).remote(command, port, self.config.startup_timeout_s)
                self.actors.append(actor)
                worker_urls.append(http_url(node_ip, port))

        print(f"[sglang-ray] starting {len(self.actors)} workers: {worker_urls}", flush=True)
        results = ray.get([actor.start.remote() for actor in self.actors])
        print(f"[sglang-ray] workers_ready={results}", flush=True)

        command = router_command(self.config, router_module, router_help, worker_urls)
        print(f"[sglang-ray] starting router command={' '.join(command)}", flush=True)
        self.router = subprocess.Popen(command, env=os.environ.copy(), start_new_session=True)
        wait_http(
            [
                "http://127.0.0.1:%d/health" % self.config.router_port,
                "http://127.0.0.1:%d/health_generate" % self.config.router_port,
            ],
            timeout_s=self.config.startup_timeout_s,
        )
        print(
            f"[sglang-ray] ready endpoint=http://{quote_host(self.config.router_host)}:{self.config.router_port} "
            f"workers={len(worker_urls)}",
            flush=True,
        )

    def monitor(self) -> None:
        while not stopping:
            if self.router is None or self.router.poll() is not None:
                code = None if self.router is None else self.router.returncode
                raise RuntimeError(f"SGLang Router exited unexpectedly: returncode={code}")
            statuses = ray.get([actor.alive.remote() for actor in self.actors])
            if not all(statuses):
                raise RuntimeError(f"SGLang worker exited unexpectedly: alive={statuses}")
            time.sleep(5)

    def stop(self) -> None:
        if self.router is not None and self.router.poll() is None:
            try:
                os.killpg(self.router.pid, signal.SIGTERM)
                self.router.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.router.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.router = None
        for actor in self.actors:
            try:
                ray.get(actor.stop.remote(), timeout=45)
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                print(f"[sglang-ray] worker cleanup failed: {exc}", flush=True)
        self.actors = []


stopping = False


def handle_signal(signum: int, _frame: Any) -> None:
    global stopping
    stopping = True
    print(f"[sglang-ray] received signal={signum}; stopping", flush=True)


def main() -> int:
    global stopping
    config = Config.from_environment()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(
        f"[sglang-ray] connecting ray address={config.ray_address} namespace={config.ray_namespace!r} "
        f"model={config.model_path} tp={config.tensor_parallel_size} dp={config.data_parallel_size} "
        f"max_model_len={config.max_model_len} mem_fraction_static={config.mem_fraction_static} "
        f"max_running_requests={config.max_running_requests} "
        f"chunked_prefill_size={config.chunked_prefill_size} "
        f"router_prometheus_port={config.router_prometheus_port}",
        flush=True,
    )
    ray.init(address=config.ray_address, namespace=config.ray_namespace, ignore_reinit_error=True)

    try:
        while not stopping:
            fleet = Fleet(config)
            try:
                fleet.launch()
                fleet.monitor()
            except KeyboardInterrupt:
                stopping = True
            except Exception as exc:  # noqa: BLE001 - restart loop is intentional
                print(f"[sglang-ray] fleet failed: {exc}", flush=True)
                traceback.print_exc()
            finally:
                fleet.stop()

            if not stopping:
                print(
                    f"[sglang-ray] restarting fleet in {config.restart_delay_s}s",
                    flush=True,
                )
                time.sleep(config.restart_delay_s)
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
