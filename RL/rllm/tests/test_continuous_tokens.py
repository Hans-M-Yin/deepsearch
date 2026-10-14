import ast
import asyncio
from functools import partial
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("continuous_tokens", ROOT / "rllm/utils/continuous_tokens.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return ord(token)


class QwenChatTemplateParser:
    image_token = "@"
    eot_token = "!\n"

    def parse(self, messages, **kwargs):
        return "".join(m["role"] + ":" + m["content"] + self.eot_token for m in messages) + "assistant:"

    def process_image_data(self, messages):
        return [im for m in messages for im in m.get("images", [])]


class Processor:
    image_processor = SimpleNamespace(merge_size=2)
    image_token_id = ord("@")

    def __init__(self):
        self.calls = []

    def __call__(self, text, images, **kwargs):
        self.calls.append(list(images))
        # Image sizes 2 and 4 -> 1 and 4 expanded tokens, respectively.
        sizes = iter(images)
        ids = []
        for c in text[0]:
            ids.extend([ord(c)] * ((next(sizes) // 2) ** 2 if c == "@" else 1))
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids)),
                    pixel_values=torch.cat([torch.full((s*s, 3), s) for s in images]),
                    image_grid_thw=torch.tensor([[1, s, s] for s in images]))


class ContinuousTest(unittest.TestCase):
    def test_batch_widths_preserve_long_image_prompt(self):
        prompts = [torch.zeros(6000), torch.zeros(1000)]
        responses = [torch.zeros(10), torch.zeros(500)]
        self.assertEqual(ct.batch_widths(prompts, responses, 60096, 56000), (6000, 500))
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            ct.batch_widths(prompts, responses, 6000, 56000)
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            ct.batch_widths(prompts, responses, 60096, 100)

    def setUp(self):
        self.parser, self.tok, self.proc = QwenChatTemplateParser(), Tokenizer(), Processor()

    def prepare(self, messages, state):
        return ct.prepare(self.parser, self.tok, self.proc, messages, [], state)

    def test_multiturn_images_noncanonical_ids_and_training_mask(self):
        messages = [dict(role="user", content="image @", images=[2])]
        first = self.prepare(messages, {})
        original = [100001, 100002, ord("!")]  # not reproducible from message text
        state = ct.commit(first, original, self.tok, self.parser)
        old_expanded = list(state["expanded"])
        messages += [dict(role="assistant", content="reencoded text is NOT authoritative"),
                     dict(role="user", content="two images @@", images=[4, 2])]
        second = self.prepare(messages, state)
        self.assertEqual(second["expanded"][:len(old_expanded)], old_expanded)
        self.assertEqual(second["expanded"][len(old_expanded)], ord("\n"))
        self.assertEqual(second["mask"][len(old_expanded)], 0)
        self.assertEqual(self.proc.calls, [[2], [4, 2]])  # never process old images again
        self.assertEqual(second["images"], [2, 4, 2])
        self.assertEqual(second["features"]["image_grid_thw"].tolist(), [[1,2,2], [1,4,4], [1,2,2]])
        prompt = list(second["expanded"])
        final_ids = [100003, ord("!")]
        final = ct.commit(second, final_ids, self.tok, self.parser)
        self.assertEqual([x for x, m in zip(final["expanded"], final["mask"]) if m], original + final_ids)
        # Execute the production parser method without importing GPU/Ray deps.
        tree = ast.parse((ROOT / "rllm/parser/chat_template_parser.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChatTemplateParser")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_tokenize_cumulative_multimodal_from_model_output")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "parser", "exec"), scope)
        output = SimpleNamespace(prompt_ids=prompt, completion_ids=final_ids,
                                 cumulative_loss_mask=final["mask"],
                                 cumulative_initial_prompt_length=final["initial_length"],
                                 multi_modal_inputs=final["features"])
        p, r, mask, mm = scope[fn.name](None, [], output)
        self.assertEqual(torch.cat([p, r]).tolist(), final["expanded"])
        self.assertEqual(r[mask.bool()].tolist(), original + final_ids)
        self.assertEqual(mm["pixel_values"].shape[0], 24)
        alignment_spec = importlib.util.spec_from_file_location("alignment", ROOT / "rllm/utils/rollout_consistency.py")
        alignment = importlib.util.module_from_spec(alignment_spec)
        alignment_spec.loader.exec_module(alignment)
        values, valid = alignment.align_rollout_logprobs(r, mask, [
            {"token_ids": original, "log_probs": [-.1, -.2, -.3]},
            {"token_ids": final_ids, "log_probs": [-.4, -.5]},
        ])
        self.assertTrue(valid[mask.bool()].all())
        self.assertEqual(int(valid.sum()), 5)
        shaping_spec = importlib.util.spec_from_file_location("shaping", ROOT / "rllm/utils/partial_credit_shaping.py")
        shaping = importlib.util.module_from_spec(shaping_spec)
        shaping_spec.loader.exec_module(shaping)
        turn_mask, aligned = shaping.build_credit_turn_token_mask(mask, [2], 2)
        self.assertTrue(aligned)
        self.assertEqual(r[turn_mask].tolist(), final_ids)

    def test_failed_preparation_does_not_change_state(self):
        messages = [dict(role="user", content="q")]
        state = ct.commit(self.prepare(messages, {}), [99], self.tok, self.parser)
        before = list(state["expanded"])
        new = messages + [dict(role="assistant", content="c"), dict(role="user", content="next")]
        candidate = self.prepare(new, state)
        self.assertEqual(state["expanded"], before)
        self.assertGreater(len(candidate["expanded"]), len(before))
        self.assertEqual(candidate["expanded"][len(before):len(before)+2], [ord("!"), ord("\n")])
        self.assertTrue(all(m == 0 for m in candidate["mask"][len(before):]))

    def test_history_edit_rejected(self):
        state = ct.commit(self.prepare([dict(role="user", content="q")], {}), [99], self.tok, self.parser)
        with self.assertRaisesRegex(ValueError, "history was edited"):
            self.prepare([dict(role="user", content="changed")], state)

    def test_missing_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "without image payload"):
            self.prepare([dict(role="user", content="@")], {})

    def test_unknown_processor_fields_rejected(self):
        original = self.proc
        def processor(**kwargs):
            output = original(**kwargs)
            output["mm_token_type_ids"] = torch.ones(1)
            return output
        self.proc = processor
        with self.assertRaisesRegex(ValueError, "processor fields"):
            self.prepare([dict(role="user", content="@", images=[2])], {})

    def test_generated_image_token_rejected(self):
        state = self.prepare([dict(role="user", content="q")], {})
        with self.assertRaisesRegex(ValueError, "Generated image"):
            ct.commit(state, [ord("@")], self.tok, self.parser)

    def test_no_double_boundary(self):
        messages = [dict(role="user", content="q")]
        state = ct.commit(self.prepare(messages, {}), [99, ord("!"), ord("\n")], self.tok, self.parser)
        size = len(state["raw"])
        result = self.prepare(messages + [dict(role="assistant", content="c"), dict(role="user", content="n")], state)
        self.assertEqual(result["raw"][size], ord("u"))

    def test_real_engine_method_two_calls_and_failed_backend(self):
        # Run the production async orchestration with a fake transport, avoiding
        # importing Ray/vLLM on this CPU-only test host.
        tree = ast.parse((ROOT / "rllm/engine/rollout/verl_engine.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VerlEngine")
        fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_model_response")
        import uuid
        scope = dict(asyncio=asyncio, partial=partial, uuid=uuid, ModelOutput=SimpleNamespace,
                     TokenOutput=SimpleNamespace, event=lambda *a, **kw: None,
                     count_token=lambda ids, token: 0)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "engine", "exec"), scope)
        class Rollout(dict):
            name = "vllm"
        parser, tok = self.parser, self.tok
        parser.parse_completion = lambda ids: dict(content="text", reasoning=None, tool_calls=None)
        tok.decode = lambda ids, **kw: "display only"
        requests = []
        async def generate(**kw):
            requests.append(kw)
            sizes = iter(kw["image_data"] or [])
            expanded = []
            for token in kw["prompt_ids"]:
                expanded.extend([token] * ((next(sizes)//2)**2 if token == ord("@") else 1))
            return SimpleNamespace(token_ids=[100001, ord("!")], log_probs=[-.1, -.2], prompt_ids=expanded)
        engine = SimpleNamespace(continuous_tito=True, validate=False, accumulate_reasoning=False,
                                 train_sampling_params={}, val_sampling_params={},
                                 max_response_length=100, max_prompt_length=1000,
                                 processor_executor=None, chat_parser=parser, tokenizer=tok,
                                 processor=self.proc, server_manager=SimpleNamespace(generate=generate),
                                 config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=Rollout(max_model_len=2000)),
                                                        data=SimpleNamespace(max_prompt_length=10, max_response_length=1000)))
        async def run():
            state = {}
            messages = [dict(role="user", content="@", images=[4])]
            first = await scope[fn.name](engine, messages, continuous_token_state=state)
            old = list(state["expanded"])
            messages += [dict(role="assistant", content="text"), dict(role="user", content="@", images=[2])]
            second = await scope[fn.name](engine, messages, continuous_token_state=state)
            self.assertEqual(second.prompt_ids.tolist()[:len(old)], old)
            self.assertEqual(sum(second.cumulative_loss_mask), 4)
            self.assertEqual(state["turn"], 2)
            self.assertEqual(sum(first.cumulative_loss_mask), 2)  # old output immutable
            self.assertEqual(requests[1]["prompt_ids"].count(ord("@")), 2)
            before = list(state["expanded"])
            async def wrong(**kwargs):
                return SimpleNamespace(token_ids=[99], prompt_ids=[123], log_probs=[-.1])
            engine.server_manager.generate = wrong
            messages += [dict(role="assistant", content="text"), dict(role="user", content="next")]
            with self.assertRaisesRegex(ValueError, "processed prompt differs"):
                await scope[fn.name](engine, messages, continuous_token_state=state)
            self.assertEqual(state["expanded"], before)
        with patch.dict("sys.modules", {"rllm.utils.continuous_tokens": ct}):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
