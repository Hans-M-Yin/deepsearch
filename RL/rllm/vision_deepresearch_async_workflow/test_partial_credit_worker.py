"""Offline worker-routing checks; never send provider requests."""
import sys
import asyncio
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vision_deepresearch_async_workflow import partial_credit_reward as pc


class PartialCreditWorkerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Exercise request construction without starting executor threads.
        async def inline_call(fn, request, concurrency):
            return fn(request)
        self.real_run_judge_request = pc._run_judge_request
        patcher = patch.object(pc, "_run_judge_request", new=inline_call)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_module(self, registered=True):
        worker = Mock()
        worker.get_model.return_value = {"served_model": "deployment"} if registered else None
        worker.generate.return_value = SimpleNamespace(content='{"verdicts": []}')
        return SimpleNamespace(
            LLM_WORKER=worker, ModelRequest=SimpleNamespace, ModelMessage=SimpleNamespace
        )

    async def test_routes_alias_and_preserves_prompt(self):
        module = self.fake_module()
        config = pc.PartialCreditConfig(judge_model="gpt55_internal_azure", include_judge_reasons=True)
        with patch.dict(sys.modules, {"synthesis.model_worker": module}):
            content, elapsed, queued = await pc._request_judge("trace", config)
        request = module.LLM_WORKER.generate.call_args.args[0]
        self.assertEqual(request.model, config.judge_model)
        self.assertEqual(request.metadata["llm_retry_count"], 1)
        self.assertFalse(hasattr(request, "temperature"))
        self.assertFalse(hasattr(request, "max_tokens"))
        self.assertEqual(request.messages[0].content, pc._JUDGE_SYSTEM_PROMPT + pc._JUDGE_REASON_INSTRUCTION)
        self.assertEqual(request.messages[1].content, "trace")
        self.assertEqual(content, '{"verdicts": []}')
        self.assertGreaterEqual(elapsed, 0)
        self.assertGreaterEqual(queued, 0)

    async def test_unknown_alias_never_dispatches(self):
        module = self.fake_module(registered=False)
        with patch.dict(sys.modules, {"synthesis.model_worker": module}):
            with self.assertRaises(pc.PartialCreditJudgeConfigError):
                await pc._request_judge("trace", pc.PartialCreditConfig(judge_model="gpt88_internal_azure"))
        module.LLM_WORKER.generate.assert_not_called()

    async def test_provider_error_propagates_to_scorer(self):
        module = self.fake_module()
        module.LLM_WORKER.generate.side_effect = TimeoutError("provider timeout")
        with patch.dict(sys.modules, {"synthesis.model_worker": module}):
            with self.assertRaises(TimeoutError):
                await pc._request_judge("trace", pc.PartialCreditConfig(judge_model="registered"))

    async def test_config_preserves_rollout_sized_concurrency(self):
        with patch.dict(os.environ, {"PARTIAL_CREDIT_JUDGE_CONCURRENCY": "768"}):
            self.assertEqual(pc.PartialCreditConfig.from_env().judge_concurrency, 768)

    async def test_dedicated_pool_exceeds_default_executor_capacity(self):
        # All 40 fake requests must start before any returns. This would stall
        # on asyncio's default pool (at most 32 threads). No provider calls.
        release = threading.Event()
        lock = threading.Lock()
        started = 0

        def generate(request):
            nonlocal started
            with lock:
                started += 1
            if not release.wait(timeout=10):
                raise TimeoutError("test requests not released")
            return SimpleNamespace(content="ok")

        module = self.fake_module()
        module.LLM_WORKER.generate.side_effect = generate
        config = pc.PartialCreditConfig(judge_model="registered", judge_concurrency=40)
        with patch.dict(sys.modules, {"synthesis.model_worker": module}), patch.object(
            pc, "_run_judge_request", new=self.real_run_judge_request
        ):
            jobs = [asyncio.create_task(pc._request_judge("trace", config)) for _ in range(40)]
            try:
                async def wait_for_all():
                    while True:
                        with lock:
                            if started == 40:
                                return
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_for_all(), timeout=5)
            finally:
                release.set()
                results = await asyncio.gather(*jobs)
                executor = pc._JUDGE_EXECUTORS.pop(asyncio.get_running_loop())[1]
                executor.shutdown(wait=True)
        self.assertEqual(len(results), 40)
        self.assertTrue(all(result[0] == "ok" for result in results))


if __name__ == "__main__":
    unittest.main()
