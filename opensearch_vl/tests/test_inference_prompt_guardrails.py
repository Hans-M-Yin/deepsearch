"""Regression checks for targeted inference prompt guardrails."""

from opensearch_vl.opensearch_infer.prompts import build_system_prompt


def test_prompt_contains_the_targeted_stop_and_retry_rules() -> None:
    prompt = build_system_prompt("[]")

    assert "do not keep searching merely for extra confirmation" in prompt
    assert "Retry the same resource at most once" in prompt
    assert "at most 45 tool calls" in prompt
