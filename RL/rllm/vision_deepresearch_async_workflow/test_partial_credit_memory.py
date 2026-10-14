"""Offline regressions for pending evidence, direct progress and image credit."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vision_deepresearch_async_workflow import partial_credit_reward as pc


def step(text="Search for the target", observation="No useful information was returned.", final=False):
    return SimpleNamespace(model_response=text, observation=observation,
                           action=SimpleNamespace(action={"type": "final_answer" if final else "tool_call", "tool_call": "search"}))


def task(image=False):
    return {"question": "Find the pictured object. " + ("<image>" if image else ""),
            "extra_info": {"partial_credit": {"statements": [
                {"statement_id": "s", "statement": "The scene identifies Alpha as Beta.",
                 "source_object": "" if image else "Alpha", "target_object": "Beta"}]}}}


def verdict(turn=1, adoption=2, pending=False, quote="Alpha is connected to Beta."):
    return {"statement_id": "s", "verdict": "evidence_only" if pending else "established_and_used",
            "evidence_source": "tool", "evidence_turns": [turn],
            "evidence_quotes": [{"turn": turn, "quote": quote}], "credit_turn": turn,
            "credit_turns": [turn], "adoption_turn": None if pending else adoption,
            "reason": "Tool supports relation; action uses the target."}


class PartialCreditMemoryTest(unittest.IsolatedAsyncioTestCase):
    def test_quote_representation_normalization(self):
        cases = [
            ('He called it "the main keep".', json.dumps({"content": 'He called it "the main keep".'})),
            ("formerly known as the Joseph Henry Memorial", json.dumps({"content": "formerly known as the **Joseph Henry Memorial**"})),
            ("Café has a main keep.", json.dumps({"content": "Cafe\u0301 has a\n main keep."})),
            ('He said "castle" & left.', '{"content": "He said “castle” &amp; left."}'),
            ('"title": "Nagoya Castle"', '{"title": "Nagoya Castle"}'),
            ("Alpha links to Beta.", '<tool_response>{"content": "Alpha links to Beta."}</tool_response>'),
            ('In Middlebury, Vermont, she married John Willard.', json.dumps({'content': 'In [Middlebury](https://example.org/place/Middlebury), Vermont, she married John Willard.'})),
            ('Alpha is connected to Beta.', json.dumps({'content': '[Alpha](https://example.org/Alpha_(name)) is connected to **Beta**.'})),
        ]
        for quote, observation in cases:
            with self.subTest(quote=quote):
                self.assertTrue(pc._evidence_quote_exists(quote, observation))

    def test_quote_does_not_allow_semantic_substitution_or_stitching(self):
        cases = [
            ("Elisabeth of Austria was Casimir's mother.", "Elizabeth of Poland was Louis's mother."),
            ("Alpha was built in 1815.", "Alpha was built in 1816."),
            ("Alpha is Beta.", "Alpha is not Beta."),
            ("Alpha links to Beta.", json.dumps({"title": "Alpha links", "content": "to Beta."})),
            ("Alpha links to Beta.", json.dumps(["Alpha links", "to Beta."])),
            ("Alpha ... Beta", "Alpha is connected to Beta"),
            ("", "Alpha"),
            ('Alpha is Beta.', '[Alpha](https://example.org) is not Beta.'),
            ('Alpha was born in Berlin.', '[Beta](https://example.org/Alpha) was born in Berlin.'),
        ]
        for quote, observation in cases:
            with self.subTest(quote=quote):
                self.assertFalse(pc._evidence_quote_exists(quote, observation))

    def test_grounded_quotes_uses_decoded_text_and_keeps_turn_checks(self):
        item = verdict(quote='Alpha is connected to "Beta".')
        records = {1: {"is_tool_turn": True, "observation": json.dumps({"content": 'Alpha is connected to "Beta".'})}}
        self.assertIsNotNone(pc._grounded_quotes(item, records, 12))
        self.assertIsNone(pc._grounded_quotes(item, {2: records[1]}, 12))
        records[1]["is_tool_turn"] = False
        self.assertIsNone(pc._grounded_quotes(item, records, 12))

    def test_long_quote_is_checked_in_full_not_truncated(self):
        quote = 'Alpha is connected to Beta. ' * 100
        records = {1: {'is_tool_turn': True, 'observation': json.dumps({'content': quote})}}
        self.assertIsNotNone(pc._grounded_quotes(verdict(quote=quote), records, 12))
        self.assertIsNone(pc._grounded_quotes(verdict(quote=quote + 'Invented ending.'), records, 12))
        oversized = 'A' * 16001
        records[1]['observation'] = oversized
        self.assertIsNone(pc._grounded_quotes(verdict(quote=oversized), records, 12))

    def test_adoption_prompt_allows_explicit_resolution_not_unrelated_action(self):
        prompt = pc._JUDGE_SYSTEM_PROMPT
        self.assertIn('reasoning BEFORE a tool call', prompt)
        self.assertIn('on that basis', prompt)
        self.assertIn('taking an unrelated action', prompt)
        self.assertIn('Final-answer-only use remains excluded', prompt)

    async def test_pending_evidence_crosses_slice_and_long_gap(self):
        steps = [step() for _ in range(17)]
        steps[7] = step(observation="Alpha is connected to Beta.")
        prompts = []
        async def request(prompt, config):
            prompts.append(prompt)
            return json.dumps({"statements": [verdict(8, 16, pending=len(prompts) == 1)]}), .01, 0.
        with patch.object(pc, "_request_judge", new=request):
            result = await pc.score_statement_partial_credit(task=task(), steps=steps,
                config=pc.PartialCreditConfig(mode="shadow", include_judge_reasons=True))
        self.assertEqual(len(prompts), 2)
        self.assertIn('"quote": "Alpha is connected to Beta."', prompts[1])
        self.assertEqual(result["chunk_results"][0]["events"], [])
        self.assertEqual(result["events"][0]["credit_turns"], [8])
        self.assertEqual(result["events"][0]["adoption_turn"], 16)
        self.assertEqual(result["bonus"], 0)
        self.assertNotIn("_evidence_memory", result["chunk_results"][0])

    async def test_fabricated_pending_quote_cannot_cross_slice(self):
        calls = []
        async def request(prompt, config):
            calls.append(prompt)
            return json.dumps({"statements": [verdict(8, 16, pending=len(calls) == 1)]}), .01, 0.
        with patch.object(pc, "_request_judge", new=request):
            result = await pc.score_statement_partial_credit(task=task(), steps=[step() for _ in range(17)],
                config=pc.PartialCreditConfig(mode="shadow", include_judge_reasons=True))
        self.assertEqual(result["verified_statement_count"], 0)
        self.assertEqual(result["chunk_results"][0]["evidence_memory_packet_count"], 0)

    async def test_evidence_only_never_earns_reward(self):
        async def request(prompt, config):
            return json.dumps({"statements": [verdict(pending=True)]}), .01, 0.
        with patch.object(pc, "_request_judge", new=request):
            result = await pc.score_statement_partial_credit(task=task(), steps=[step(observation="Alpha is connected to Beta.")],
                config=pc.PartialCreditConfig(mode="reward"))
        self.assertEqual(result["raw_bonus"], 0)
        self.assertEqual(result["events"], [])

    def test_original_image_requires_corroboration_and_credits_first_action(self):
        steps = [step("The input photo shows Beta; search Beta now."),
                 step(observation="The image match identifies this pictured building as Beta.")]
        _, records, _ = pc.build_text_only_judge_trace(steps, question="Identify <image>")
        item = verdict(turn=2)
        item.update(evidence_source="input_image", credit_turn=1, credit_turns=[1], adoption_turn=1,
                    visual_claim_quote="The input photo shows Beta",
                    evidence_quotes=[{"turn": 2, "quote": steps[1].observation}])
        statements, _, _ = pc.resolve_partial_credit_statements(task(image=True))
        events, rejected = pc.validate_judge_verdicts({"statements": [item]}, statements=statements,
            turn_records=records, config=pc.PartialCreditConfig())
        self.assertEqual(rejected, 0)
        self.assertEqual(events[0]["credit_turns"], [1])
        for update in ({"evidence_quotes": []}, {"visual_claim_quote": "Invented visual identification"}):
            events, rejected = pc.validate_judge_verdicts({"statements": [{**item, **update}]}, statements=statements,
                turn_records=records, config=pc.PartialCreditConfig())
            self.assertEqual(events, [])
            self.assertEqual(rejected, 1)
        records[1]["original_image_available"] = False
        events, _ = pc.validate_judge_verdicts({"statements": [item]}, statements=statements,
            turn_records=records, config=pc.PartialCreditConfig())
        self.assertEqual(events, [])

    async def test_old_adoption_in_memory_is_not_recounted(self):
        async def request(prompt, config):
            return json.dumps({"statements": [verdict()]}), .01, 0.
        steps = [step(observation="Alpha is connected to Beta.")] + [step() for _ in range(16)]
        with patch.object(pc, "_request_judge", new=request):
            result = await pc.score_statement_partial_credit(task=task(), steps=steps,
                config=pc.PartialCreditConfig(mode="shadow", include_judge_reasons=True))
        self.assertEqual(result["verified_statement_count"], 1)
        self.assertEqual(result["chunk_results"][1]["events"], [])

    async def test_input_image_credit_can_wait_for_later_slice_corroboration(self):
        steps = [step("The input photo shows Beta; search Beta now.")] + [step() for _ in range(15)]
        steps[15] = step(observation="The image match identifies this pictured building as Beta.")
        prompts = []
        async def request(prompt, config):
            prompts.append(prompt)
            if len(prompts) == 1:
                item = {"statement_id": "s", "verdict": "unverifiable", "evidence_turns": [],
                        "credit_turns": [], "credit_turn": None, "adoption_turn": None}
            else:
                item = verdict(turn=16, quote=steps[15].observation)
                item.update(evidence_source="input_image", credit_turn=1, credit_turns=[1],
                            adoption_turn=1, visual_claim_quote="The input photo shows Beta")
            return json.dumps({"statements": [item]}), .01, 0.
        with patch.object(pc, "_request_judge", new=request):
            result = await pc.score_statement_partial_credit(task=task(image=True), steps=steps,
                config=pc.PartialCreditConfig(mode="shadow"))
        self.assertIn("The input photo shows Beta", prompts[1])
        self.assertEqual(result["events"][0]["credit_turns"], [1])
        self.assertEqual(result["events"][0]["evidence_turns"], [16])

    async def test_later_chunk_cannot_invent_missing_prior_observation(self):
        async def request(prompt, config):
            return json.dumps({"statements": [verdict(1, 16)]}), .01, 0.
        with patch.object(pc, "_request_judge", new=request):
            chunk = await pc._score_trace_chunk(question="Find object", statements=pc.resolve_partial_credit_statements(task())[0],
                steps=[step()], turn_offset=15, chunk_index=1, config=pc.PartialCreditConfig())
        self.assertEqual(chunk["events"], [])
        self.assertEqual(chunk["judge_rejected_verdict_count"], 1)

    async def test_final_answer_is_not_cross_chunk_adoption(self):
        steps = [step(observation="Alpha is connected to Beta.")] + [step() for _ in range(14)] + [step(final=True)]
        calls = []
        async def request(prompt, config):
            calls.append(prompt)
            return json.dumps({"statements": [verdict(1, 16, pending=len(calls) == 1)]}), .01, 0.
        with patch.object(pc, "_request_judge", new=request):
            result = await pc.score_statement_partial_credit(task=task(), steps=steps,
                config=pc.PartialCreditConfig(mode="shadow"))
        self.assertEqual(result["verified_statement_count"], 0)

    def test_optional_core_criteria_preserved(self):
        data = task()
        data["extra_info"]["partial_credit"]["statements"][0].update(
            required_relation="Identify the foreground vessel in this event.",
            identity_constraints="Same vessel and same depicted event.",
            background_details="Department affiliation is background.")
        statements, _, _ = pc.resolve_partial_credit_statements(data)
        self.assertEqual(statements[0]["required_relation"], "Identify the foreground vessel in this event.")
        self.assertIn("Mastico", pc._JUDGE_SYSTEM_PROMPT)
        self.assertIn("different painting, vessel, or event", pc._JUDGE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
