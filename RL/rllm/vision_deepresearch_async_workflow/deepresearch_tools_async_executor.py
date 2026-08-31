"""SFT tool registry used by the RL DeepResearch workflow.

The actual execution path intentionally lives in ``synthesis.sft.api_tools``:
the RL side only exposes the SFT schemas to the agent and marks the tools for
the dispatcher in ``deepresearch_agent.py``.  This keeps the SFT implementation
unchanged while preserving its resource registry, image handling, timeout and
retry behavior.
"""

from __future__ import annotations

import copy
from typing import Any

from synthesis.sft import tools as sft_tools
from synthesis.sft.qwen3_vl_template import format_sft_qwen_tool_prompt


class SFTToolAdapter:
    """Schema-only adapter for one tool from ``synthesis.sft.tools``.

    Calls are dispatched by :class:`DeepResearchAgent` through
    ``synthesis.sft.api_tools.execute_tool_call``.  Keeping this object small
    is deliberate: a second implementation of the SFT tool backends here
    would silently diverge in retries, cache handling, or image provenance.
    """

    uses_sft_dispatcher = True

    def __init__(self, definition: dict[str, Any]) -> None:
        function = definition["function"]
        self._json = copy.deepcopy(definition)
        self.name = str(function["name"])
        self.description = str(function.get("description") or "")
        self.executor = None

    @property
    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self._json)

    def set_executor(self, executor: Any) -> None:
        """Retain the workflow executor for interface compatibility."""

        self.executor = executor

    def __repr__(self) -> str:
        return f"SFTToolAdapter(name={self.name!r})"


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return the exact OpenAI schemas declared by the SFT tool module."""

    return copy.deepcopy(sft_tools.get_tool_definitions())


def get_all_tools() -> dict[str, SFTToolAdapter]:
    """Return the SFT tool set used by RL.

    ``finish`` is intentionally not included: it is represented by the RL
    agent's ``<answer>...</answer>`` termination format, not a backend tool.
    """

    return {
        str(definition["function"]["name"]): SFTToolAdapter(definition)
        for definition in get_tool_definitions()
    }


def get_tool(name: str) -> SFTToolAdapter | None:
    return get_all_tools().get(name)


def build_sft_react_system_prompt() -> str:
    """Build an RL-compatible prompt from the SFT schemas.

    The exported SFT training data uses ``<thinking>``/``<tool_call>`` and
    ``<answer>`` blocks.  RL uses the same model-facing protocol while keeping
    its own internal trajectory representation.
    """

    definitions = get_tool_definitions()
    qwen_tool_prompt = format_sft_qwen_tool_prompt(definitions)
    return """You are a visual research agent. Answer the question with evidence and do not guess.

Before every action, write a concise analysis inside <thinking>...</thinking>.
Use exactly one tool call per turn and wait for its observation:
<tool_call>{{"name":"tool_name","arguments":{{...}}}}</tool_call>

Tool rules:
1. t2t_search searches text/web pages. Use its returned source_page_id with read_url.
2. t2i_search searches images. Use its returned image_id or source_page_id with read_url before treating an image as evidence.
3. i2i_search performs reverse-image search on the current image; region is an optional normalized [x1,y1,x2,y2] box on a 0-1000 scale.
4. read_url reads a prior search resource or a directly available URL. Include a focused goal describing the evidence to extract.
5. Use resource IDs returned by search tools instead of inventing or exposing backend URLs.

When enough evidence has been collected, finish with:
<answer>final answer</answer>

Do not call tools not listed above. Do not emit more than one tool call in a turn.
""" + "\n\n" + qwen_tool_prompt


__all__ = [
    "SFTToolAdapter",
    "get_tool_definitions",
    "get_all_tools",
    "get_tool",
    "build_sft_react_system_prompt",
]
