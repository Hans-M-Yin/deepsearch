"""Test RL control ordering without loading Torch/Ray or contacting GPU hosts."""
import ast
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
RLLM = ROOT / "RL/rllm/rllm"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load_module("remote_holder_under_test", RLLM / "utils/remote_holder.py")


class RemoteHolderTests(unittest.TestCase):
    def holder(self):
        holder = remote.RemoteHolder("http://monitor:18090", token="secret", attempts=2)
        holder.opener = MagicMock()
        return holder

    def response(self, running):
        response = MagicMock()
        response.__enter__.return_value = SimpleNamespace(status=200, read=lambda _: json.dumps({"running": running}).encode())
        return response

    def test_http_paths_token_and_confirmation(self):
        holder = self.holder()
        holder.opener.open.side_effect = [self.response(False), self.response(True)]
        with holder.rollout():
            pass
        calls = holder.opener.open.call_args_list
        self.assertEqual([call.args[0].full_url for call in calls], ["http://monitor:18090/stop", "http://monitor:18090/start"])
        self.assertEqual(calls[0].args[0].get_header("Authorization"), "Bearer secret")

    @patch.object(remote.time, "sleep")
    def test_stop_failure_blocks_rollout(self, sleep):
        holder = self.holder()
        holder.opener.open.side_effect = OSError("unavailable")
        entered = False
        with self.assertRaisesRegex(RuntimeError, "refusing to begin rollout"):
            with holder.rollout():
                entered = True
        self.assertFalse(entered)
        self.assertEqual(holder.opener.open.call_count, 2)

    @patch.object(remote.time, "sleep")
    def test_unconfirmed_stop_does_not_count_as_success(self, sleep):
        holder = self.holder()
        holder.opener.open.return_value = self.response(True)
        with self.assertRaises(RuntimeError):
            holder.command("stop", required=True)

    @patch.object(remote.time, "sleep")
    def test_start_failure_does_not_discard_completed_rollout(self, sleep):
        holder = self.holder()
        holder.opener.open.side_effect = [self.response(False), OSError(), OSError()]
        with holder.rollout():
            result = "finished"
        self.assertEqual(result, "finished")

    def test_rollout_exception_does_not_start_holder(self):
        holder = self.holder()
        holder.opener.open.return_value = self.response(False)
        with self.assertRaisesRegex(ValueError, "rollout failed"):
            with holder.rollout():
                raise ValueError("rollout failed")
        self.assertEqual(holder.opener.open.call_count, 1)

    def test_disabled_monitor_has_no_http_calls(self):
        holder = remote.RemoteHolder()
        holder.opener = MagicMock()
        with holder.rollout():
            pass
        holder.close()
        holder.opener.open.assert_not_called()


class TrainerBoundaryTests(unittest.TestCase):
    def method(self, name, namespace):
        tree = ast.parse((RLLM / "trainer/verl/agent_workflow_trainer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentWorkflowPPOTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer_hook", "exec"), namespace)
        return namespace[name]

    def test_actual_generation_method_order_for_train_and_validation(self):
        for batch in ("train", "validation"):
            events = []
            holder = remote.RemoteHolder("http://monitor")
            holder.command = lambda action, **kwargs: events.append(action)
            engine = SimpleNamespace(execute_tasks_verl=lambda *a, **kw: events.append("execute") or "coroutine")
            def submit(coro, loop):
                return SimpleNamespace(result=lambda: events.append("reward_and_postprocessing_done") or "result")
            method = self.method("generate_trajectories", {
                "marked_timer": lambda *a, **kw: nullcontext(),
                "asyncio": SimpleNamespace(run_coroutine_threadsafe=submit),
            })
            trainer = SimpleNamespace(_holder_monitor=holder, agent_execution_engine=engine, _loop=None)
            self.assertEqual(method(trainer, batch), "result")
            self.assertEqual(events, ["stop", "execute", "reward_and_postprocessing_done", "start"])

    def test_actual_shutdown_method_stops_holder(self):
        holder = MagicMock()
        trainer = SimpleNamespace(_holder_monitor=holder)
        self.method("shutdown", {})(trainer)
        holder.close.assert_called_once_with()

    def test_ray_forwards_monitor_configuration(self):
        env_module = load_module("holder_runtime_env", RLLM / "trainer/verl/ray_runtime_env.py")
        env = {"RLLM_HOLDER_MONITOR_URL": "http://monitor:18090",
               "RLLM_HOLDER_MONITOR_TOKEN": "secret", "RLLM_HOLDER_MONITOR_TIMEOUT": "30"}
        with patch.dict(os.environ, env, clear=True):
            forwarded = env_module.get_ppo_ray_runtime_env()["env_vars"]
        for key, value in env.items():
            self.assertEqual(forwarded[key], value)


if __name__ == "__main__":
    unittest.main()
