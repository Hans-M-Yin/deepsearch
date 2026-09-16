import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.utils.multimodal_debug import count_token, event
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        # Qwen3-VL image fetching/resizing and patch construction are
        # synchronous CPU work.  Running them on the workflow asyncio loop
        # blocks tool-result callbacks and other trajectories.  Keep a
        # dedicated pool separate from the tool executor so tool bursts
        # cannot starve multimodal preprocessing.
        workflow_config = config.get("rllm", {}).get("workflow", {})
        processor_workers = int(workflow_config.get("n_processor_workers", 128))
        if processor_workers < 1:
            raise ValueError("rllm.workflow.n_processor_workers must be at least 1")
        self.processor_executor = ThreadPoolExecutor(
            max_workers=processor_workers,
            thread_name_prefix="qwen-vl-input-preprocess",
        )
        self.server_manager = AsyncLLMServerManager(
            config,
            server_handles=rollout_manager.server_handles,
            request_submit_executor=self.processor_executor,
        )

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    def _prepare_request_inputs(
        self,
        messages: list[dict],
        tools: list,
        accumulate_reasoning: bool,
    ):
        """Prepare all synchronous model inputs outside the asyncio loop."""
        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )
        request_prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        has_images = any(
            msg.get("images", None) is not None and msg["role"] == "user"
            for msg in messages
        )
        if has_images and self.processor is not None:
            image_data = self.chat_parser.process_image_data(messages)
            model_inputs = self.processor(
                text=[prompt],
                images=image_data,
                return_tensors="pt",
            )
        else:
            image_data = None
            model_inputs = None

        return prompt, request_prompt_ids, image_data, model_inputs

    def shutdown(self):
        """Release the dedicated CPU preprocessing workers."""
        if self.processor_executor is not None:
            self.processor_executor.shutdown(wait=True, cancel_futures=True)
            self.processor_executor = None

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        # ``max_tokens`` is the per-request/per-turn generation limit supplied
        # by the agent.  Keep the trajectory-level ``max_response_length``
        # separate: the latter may be very large (e.g. 56000) while a single
        # ReAct turn is intentionally capped at a much smaller value (currently
        # 4096 in DeepResearchAgent).
        max_tokens = sampling_params.pop(
            "max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length)
        )
        max_tokens = min(int(max_tokens), int(self.max_response_length))
        sampling_params["max_new_tokens"] = max_tokens

        # Prompt rendering, tokenization, image loading/resizing, and Qwen-VL
        # patch construction are all synchronous CPU work. Submit the whole
        # preparation pipeline as one task so the event loop only waits for a
        # Future and remains available for tool callbacks and other samples.
        prompt, request_prompt_ids, image_data, model_inputs = await asyncio.get_running_loop().run_in_executor(
            self.processor_executor,
            partial(
                self._prepare_request_inputs,
                messages,
                tools,
                accumulate_reasoning,
            ),
        )

        if model_inputs is not None:
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is None:
                image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            event(
                "rollout_processor",
                request_id=application_id,
                image_count=len(image_data),
                raw_prompt_length=len(request_prompt_ids),
                expanded_prompt_length=int(model_inputs["input_ids"].shape[-1]),
                raw_image_token_count=count_token(request_prompt_ids, image_token_id),
                expanded_image_token_count=count_token(model_inputs["input_ids"][0], image_token_id),
                image_grid_thw_shape=getattr(model_inputs.get("image_grid_thw"), "shape", None),
                pixel_values_shape=getattr(model_inputs.get("pixel_values"), "shape", None),
                processor_keys=sorted(model_inputs.keys()),
            )
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            prompt_ids = request_prompt_ids
            multi_modal_inputs = None

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        try:
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=application_id,
                prompt_ids=request_prompt_ids,
                image_data=image_data,
                sampling_params=sampling_params,
            )  # type: ignore
        except asyncio.CancelledError:
            # workflow.wait_for() cancels this coroutine on rollout timeout.
            # Cancel the corresponding backend request before the next
            # release/wake cycle; otherwise the backend can retain an
            # in-flight request after the workflow coroutine is cancelled.
            if self.config.actor_rollout_ref.rollout.name in ("sglang", "vllm"):
                try:
                    await self.server_manager.abort_request(request_id=application_id)
                except Exception:
                    logger.exception("Failed to abort cancelled rollout request %s", application_id)
            raise
        completion_ids: list[int] = token_output.token_ids

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        event(
            "rollout_complete",
            request_id=application_id,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
            image_count=len(image_data or []),
        )
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=[],
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        if self.config.actor_rollout_ref.rollout.name in ("sglang", "vllm"):
            await self.server_manager.abort_request(abort_all=True)
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])
