# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import os
import threading
from collections.abc import Mapping
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any


def _wandb_debug_enabled() -> bool:
    return os.environ.get("RLLM_WANDB_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _wandb_immediate_enabled() -> bool:
    """Enable the opt-in live W&B path used by long-running RL rollouts."""
    return os.environ.get("RLLM_WANDB_IMMEDIATE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sanitize_wandb_config(config: Any, max_key_length: int = 64) -> tuple[Any, list[str]]:
    """Remove config leaves whose flattened key is too long for the tracking API.

    The internal tracking service used by this cluster validates flattened
    W&B config keys and rejects keys longer than 64 characters.  Hydra itself
    still receives the original config; this function only prepares the copy
    passed to ``wandb.init``.
    """
    dropped_keys: list[str] = []

    def visit(value: Any, prefix: str) -> Any:
        if isinstance(value, Mapping):
            sanitized = {}
            for raw_key, child in value.items():
                key = str(raw_key)
                full_key = f"{prefix}.{key}" if prefix else key
                if len(full_key) > max_key_length:
                    dropped_keys.append(full_key)
                    continue
                sanitized[key] = visit(child, full_key)
            return sanitized
        return value

    return visit(config, ""), dropped_keys


def _debug_wandb_system_metrics(run, event: str, step: int | None = None) -> None:
    """Print a low-overhead diagnostic for W&B's automatic system monitor.

    W&B collects system metrics in its internal process rather than through the
    trainer's ``wandb.log`` call.  ``run._system_metrics`` asks that process for
    the metrics it has collected so far.  This verifies local collection and
    the online/offline mode; the W&B service uploads those records
    asynchronously after they are handed to the internal process.
    """
    if not _wandb_debug_enabled():
        return

    try:
        settings = getattr(run, "settings", None)
        mode = getattr(run, "mode", None) or getattr(settings, "mode", None)
        disable_stats = None
        for attr in ("x_disable_stats", "_disable_stats", "disable_stats"):
            value = getattr(settings, attr, None) if settings is not None else None
            if value is not None:
                disable_stats = value
                break

        collected = {}
        try:
            collected = getattr(run, "_system_metrics", {}) or {}
        except Exception as exc:
            print(
                f"[RLLM_WANDB_DEBUG] event={event} step={step} "
                f"system_metrics_query_error={type(exc).__name__}:{exc}",
                flush=True,
            )

        latest = {}
        for key, records in collected.items():
            if records:
                latest[key] = records[-1][1]
        interesting = {
            key: latest[key]
            for key in sorted(latest)
            if any(token in key.lower() for token in ("cpu", "gpu", "memory", "ram", "power"))
        }
        if len(interesting) > 20:
            interesting = dict(list(interesting.items())[:20])

        print(
            f"[RLLM_WANDB_DEBUG] event={event} step={step} "
            f"run_id={getattr(run, 'id', None)} mode={mode} "
            f"disable_stats={disable_stats} "
            f"local_run_dir={getattr(run, 'dir', None)} "
            f"collected_metric_count={len(collected)} "
            f"collected_metric_keys={','.join(sorted(collected)[:40]) or '-'} "
            f"latest_system_values={interesting or '-'}",
            flush=True,
        )
    except Exception as exc:
        # Diagnostics must never affect training or W&B logging.
        print(
            f"[RLLM_WANDB_DEBUG] event={event} diagnostic_error="
            f"{type(exc).__name__}:{exc}",
            flush=True,
        )


def _debug_wandb_local_state(run, event: str, step: int | None = None) -> None:
    """Report the local W&B event file and sender state.

    This is intentionally diagnostic-only.  In particular, the existence of a
    ``.wandb`` file does not prove that the file has reached the tracking
    service, so this reports both the local file state and the latest sender
    messages when they are available.
    """
    if not _wandb_debug_enabled():
        return

    try:
        settings = getattr(run, "settings", None)
        setting_values = {}
        for attr in (
            "mode",
            "_offline",
            "offline",
            "_sync",
            "sync",
            "x_disable_stats",
            "_disable_stats",
            "disable_stats",
            "dir",
            "files_dir",
            "sync_dir",
        ):
            value = getattr(settings, attr, None) if settings is not None else None
            if value is not None:
                setting_values[attr] = value

        run_dir_value = getattr(run, "dir", None)
        run_dir = Path(str(run_dir_value)) if run_dir_value else None
        if run_dir is not None and run_dir.name == "files":
            root_dir = run_dir.parent
        else:
            root_dir = run_dir

        event_files = []
        internal_log = None
        internal_tail = []
        if root_dir is not None and root_dir.exists():
            event_files = sorted(root_dir.glob("*.wandb"))
            internal_log = root_dir / "logs" / "debug-internal.log"
            if internal_log.exists():
                with internal_log.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 8192), os.SEEK_SET)
                    tail = stream.read().decode("utf-8", errors="replace")
                interesting = (
                    "send",
                    "upload",
                    "sync",
                    "Request failed",
                    "run started",
                    "Starting system monitor",
                    "Started gpu",
                )
                internal_tail = [
                    line.strip().replace("\n", " ")[:500]
                    for line in tail.splitlines()
                    if any(token.lower() in line.lower() for token in interesting)
                ][-3:]

        event_file_state = ",".join(
            f"{path.name}:{path.stat().st_size}B" for path in event_files
        ) or "-"
        summary_path = root_dir / "files" / "wandb-summary.json" if root_dir else None
        summary_state = "missing"
        if summary_path is not None and summary_path.exists():
            summary_state = f"{summary_path.stat().st_size}B"

        env_state = {
            key: os.environ.get(key)
            for key in ("WANDB_MODE", "WANDB_DIR", "WANDB_BASE_URL")
            if os.environ.get(key) is not None
        }
        tail_state = " || ".join(internal_tail) or "-"
        print(
            f"[RLLM_WANDB_DEBUG] event={event} step={step} "
            f"run_id={getattr(run, 'id', None)} "
            f"run_dir={run_dir_value} root_dir={root_dir} "
            f"env={env_state or '-'} settings={setting_values or '-'} "
            f"event_files={event_file_state} summary={summary_state} "
            f"internal_tail={tail_state}",
            flush=True,
        )
    except Exception as exc:
        # Diagnostics must never affect training or W&B logging.
        print(
            f"[RLLM_WANDB_DEBUG] event={event} local_state_error="
            f"{type(exc).__name__}:{exc}",
            flush=True,
        )


class _WandbImmediateLogger:
    """Small adapter that gives the RL trainer a live W&B history stream.

    W&B's SDK still uploads asynchronously, but every call is handed directly
    to the returned ``Run`` object and gets a monotonically increasing history
    step.  This is important for live rollout progress: multiple records may
    be emitted while one logical RL training step is running.

    The caller's training step is retained in ``training/global_step``.  The
    W&B internal ``_step`` is a logging-record counter in this opt-in mode so
    live progress records cannot collide with the final batch record.
    """

    def __init__(self, run):
        self.run = run
        self._next_record_step = 0
        self._lock = threading.Lock()

    def _log(self, data, training_step):
        payload = dict(data)
        payload.setdefault("training/global_step", training_step)
        with self._lock:
            record_step = self._next_record_step
            self._next_record_step += 1
            # ``Run.log`` is the public per-run primitive used by the SFT
            # integration.  commit=True makes each live record a history row.
            self.run.log(payload, step=record_step, commit=True)

    def log(self, data, step):
        self._log(data, step)

    def log_live(self, data, step):
        self._log(data, step)

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        self._finished = False
        self._wandb_immediate = _wandb_immediate_enabled()
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            if self._wandb_immediate:
                # W&B service is the recommended mode for distributed/Ray
                # training.  Keep this best-effort for older internal clients.
                require_service = getattr(wandb, "require", None)
                if require_service is not None:
                    try:
                        require_service("service")
                    except Exception as exc:
                        print(
                            "[RLLM_WANDB] service mode unavailable; continuing "
                            f"with the installed SDK ({type(exc).__name__}: {exc})",
                            flush=True,
                        )

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            entity = os.environ.get("WANDB_ENTITY", None)
            wandb_config, dropped_config_keys = _sanitize_wandb_config(config)
            if dropped_config_keys:
                examples = ";".join(dropped_config_keys[:5])
                print(
                    "[RLLM_WANDB_CONFIG] filtered "
                    f"{len(dropped_config_keys)} overlong config keys "
                    f"(max_length=64), examples={examples}",
                    flush=True,
                )
            run = wandb.init(
                project=project_name,
                name=experiment_name,
                entity=entity,
                config=wandb_config,
                settings=settings,
            )
            self.logger["wandb"] = (
                _WandbImmediateLogger(run) if self._wandb_immediate else wandb
            )
            if self._wandb_immediate:
                print(
                    "[RLLM_WANDB] immediate_mode=1 "
                    f"run_id={getattr(run, 'id', None)} "
                    f"run_url={getattr(run, 'url', None)}",
                    flush=True,
                )
            _debug_wandb_system_metrics(run, event="init", step=None)
            _debug_wandb_local_state(run, event="init", step=None)

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)
        if "wandb" in self.logger and _wandb_debug_enabled():
            try:
                debug_every = max(1, int(os.environ.get("RLLM_WANDB_DEBUG_EVERY", "10")))
                if int(step) % debug_every == 0:
                    _debug_wandb_system_metrics(self.logger["wandb"].run, event="log", step=int(step))
                    _debug_wandb_local_state(self.logger["wandb"].run, event="log", step=int(step))
            except Exception as exc:
                print(
                    f"[RLLM_WANDB_DEBUG] event=log step={step} "
                    f"diagnostic_error={type(exc).__name__}:{exc}",
                    flush=True,
                )

    @property
    def live_enabled(self) -> bool:
        """Whether the trainer should emit rollout-progress history records."""
        return self._wandb_immediate and "wandb" in self.logger

    def log_live(self, data, step):
        """Log a small in-flight progress record to W&B only.

        This intentionally does not send partial metrics to other backends and
        is a no-op unless ``RLLM_WANDB_IMMEDIATE=1`` is enabled.
        """
        if not self.live_enabled:
            return
        logger_instance = self.logger["wandb"]
        log_live = getattr(logger_instance, "log_live", None)
        if log_live is None:
            return
        try:
            log_live(data=data, step=step)
        except Exception as exc:
            # Live observability must never abort an RL rollout.
            print(
                "[RLLM_WANDB] live log failed "
                f"step={step} error={type(exc).__name__}:{exc}",
                flush=True,
            )

    def finish(self, exit_code=0):
        """Explicitly flush and close all initialized tracking backends."""
        if getattr(self, "_finished", False):
            return
        self._finished = True

        for backend, logger_instance in list(getattr(self, "logger", {}).items()):
            if backend == "wandb":
                run = getattr(logger_instance, "run", None)
                if run is None:
                    run = getattr(logger_instance, "run", None)
                if run is not None:
                    _debug_wandb_system_metrics(run, event="finish", step=None)
                    _debug_wandb_local_state(run, event="finish", step=None)

                finisher = getattr(logger_instance, "finish", None)
                if finisher is not None:
                    try:
                        finisher(exit_code=exit_code)
                    except TypeError:
                        finisher()
            elif backend == "swanlab":
                logger_instance.finish()
            elif backend == "vemlp_wandb":
                logger_instance.finish(exit_code=exit_code)
            elif backend in {"tensorboard", "clearml", "trackio", "file"}:
                logger_instance.finish()

        self.logger.clear()

    def __del__(self):
        try:
            self.finish()
        except Exception:
            # Destructors run during interpreter/Ray teardown; never surface a
            # tracking cleanup error as a training failure.
            pass


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "w")

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(json.dumps(data) + "\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.

    def log(self, data, step):
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)
            if sanitized != key:
                self.logger.warning(
                    "[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized
                )
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log.  In immediate mode use the same Run-owned
        # adapter as scalar metrics so validation tables share the monotonic
        # live history counter instead of colliding with training steps.
        wandb_logger = self.logger.get("wandb")
        if isinstance(wandb_logger, _WandbImmediateLogger):
            wandb_logger.log({"val/generations": new_table}, step=step)
        else:
            wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
