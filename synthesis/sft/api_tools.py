"""Runtime context, tool dispatcher, and OpenAI-based tool-calling agent."""

from __future__ import annotations

import argparse
import base64
# #### START Response 0720 ####
from copy import deepcopy
# #### END Response 0720 ####
import contextvars
import io
import json
import logging
import os
import re
import threading
import time
import uuid
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageOps

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.sft"

from synthesis.model_worker import LLM_WORKER
from synthesis.model_worker import ModelMessage
from synthesis.model_worker import ModelRequest
# #### START Response 0720 ####
from synthesis.model_worker import ResponsesModelRequest
# #### END Response 0720 ####
from . import tools
import sys

logger = logging.getLogger(__name__)


# Temporary generation diagnostics.  A process-wide batch is intentional:
# ``debug_vqa_batch`` uses a ThreadPoolExecutor, so this reports the aggregate
# over the next ten completed trajectories across all workers in the process.
_SFT_TIMING_ENABLED_ENV = "SFT_TIMING_DEBUG"
_SFT_TIMING_BATCH_SIZE_ENV = "SFT_TIMING_BATCH_SIZE"


def _sft_timing_enabled() -> bool:
    return str(os.environ.get(_SFT_TIMING_ENABLED_ENV, "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _sft_timing_batch_size() -> int:
    try:
        return max(1, int(os.environ.get(_SFT_TIMING_BATCH_SIZE_ENV, "10")))
    except (TypeError, ValueError):
        return 10


@dataclass(slots=True)
class _TrajectoryTiming:
    events: dict[str, tuple[float, int]] = field(default_factory=dict)

    def add(self, name: str, elapsed_s: float) -> None:
        total_s, calls = self.events.get(name, (0.0, 0))
        self.events[name] = (total_s + max(0.0, float(elapsed_s)), calls + 1)


_ACTIVE_TRAJECTORY_TIMING: contextvars.ContextVar[_TrajectoryTiming | None] = contextvars.ContextVar(
    "sft_active_trajectory_timing",
    default=None,
)
_TIMING_BATCH_LOCK = threading.Lock()
_TIMING_BATCH_SAMPLE_COUNT = 0
_TIMING_BATCH_NUMBER = 0
_TIMING_BATCH_EVENTS: dict[str, tuple[float, int]] = {}


def _record_active_timing(name: str, elapsed_s: float) -> None:
    if not _sft_timing_enabled():
        return
    active = _ACTIVE_TRAJECTORY_TIMING.get()
    if active is not None:
        active.add(name, elapsed_s)


def _timed_call(name: str, callback: Callable[[], Any]) -> Any:
    """Run a module call and record wall time, including internal sleeps/retries."""

    if not _sft_timing_enabled() or _ACTIVE_TRAJECTORY_TIMING.get() is None:
        return callback()
    started_at = time.perf_counter()
    try:
        return callback()
    finally:
        _record_active_timing(name, time.perf_counter() - started_at)


def _finish_trajectory_timing(timing: _TrajectoryTiming, elapsed_s: float) -> None:
    """Add one completed trajectory and print every configured batch."""

    global _TIMING_BATCH_SAMPLE_COUNT, _TIMING_BATCH_NUMBER, _TIMING_BATCH_EVENTS
    if not _sft_timing_enabled():
        return
    timing.add("trajectory", elapsed_s)
    batch_size = _sft_timing_batch_size()
    with _TIMING_BATCH_LOCK:
        _TIMING_BATCH_SAMPLE_COUNT += 1
        for name, (total_s, calls) in timing.events.items():
            previous_total, previous_calls = _TIMING_BATCH_EVENTS.get(name, (0.0, 0))
            _TIMING_BATCH_EVENTS[name] = (previous_total + total_s, previous_calls + calls)
        if _TIMING_BATCH_SAMPLE_COUNT < batch_size:
            return
        _TIMING_BATCH_NUMBER += 1
        batch_number = _TIMING_BATCH_NUMBER
        sample_count = _TIMING_BATCH_SAMPLE_COUNT
        snapshot = dict(_TIMING_BATCH_EVENTS)
        _TIMING_BATCH_SAMPLE_COUNT = 0
        _TIMING_BATCH_EVENTS = {}

        print(
            f"[sft-timing] batch={batch_number} samples={sample_count}",
            file=sys.stderr,
            flush=True,
        )
        for name in sorted(snapshot):
            total_s, calls = snapshot[name]
            average_s = total_s / calls if calls else 0.0
            print(
                f"[sft-timing]   module={name} calls={calls} "
                f"total_s={total_s:.3f} avg_per_call_s={average_s:.3f}",
                file=sys.stderr,
                flush=True,
            )

# COS uploads are external network operations.  Keep the initial attempt plus
# a bounded retry budget so a transient upload failure does not discard an
# otherwise valid crop, while permanent uploader/configuration errors still
# fall back to the local cache promptly.
_COS_UPLOAD_RETRIES = 5
_COS_UPLOAD_RETRY_DELAY_S = 5


DEFAULT_SYSTEM_PROMPT = """
You are writing a full solution for a multi-hop knowledge question. Specifically, based on the question provided to you, you need to produce a complete solution process that includes scientifically rigorous, logically sound reasoning steps. This solution process should contain analysis and reasoning about the question, tool calls, analysis and reflection on tool results, replanning of the solution steps, multiple search attempts, and a final accurate solution.

Requirements:
1. You may think freely during your internal reasoning phase, but the statements ultimately included in the written solution process must also follow rigorous logic, ensuring that the solution remains sound and error-free even if one reads only the written solution process and ignores your private thinking.
2. In the solution you write, the following logic should be explicitly visible: after each tool call and its returned result, you must carefully analyze the new clues in detail, review the existing clues and the question, determine and plan the next step in detail, and then call a new tool as needed with an explanation.
3. In the written solution, every factual claim must be grounded in the question, the image, or tool-returned evidence. Do not introduce any fact, entity, date, name, or relationship unless it is directly supported by the available evidence. If something is not yet supported, explicitly treat it as uncertain and continue searching instead of guessing.
4. Once you believe the evidence is sufficient and there are no remaining unclear or uncertain points, provide the final answer and end the solution.
5. In your solution, DO NOT use tools to directly search for pages related to Wikipedia or Wiki Commons, in order to avoid shortcuts. However, you can read related Wikipedia or Wiki Commons pages which are the results of the search tools. When a search yields no useful results, try switching the form of the content you search for.
6. When writing the solution, make full use of the tool results. For example, a searched URL may seem irrelevant to the clues, but you should still analyze whether the webpage may contain the clues needed to solve the problem based on any available snippets, and then use read_url to read it further.
7. The intermediate factual statements related to the question, as well as the final answer, will be provided to you, but you must never reveal any of that content in your response. You may only use it to verify whether your current solution process is correct. If the results of your search and analysis contradict the provided factual statements, that means there is a flaw in your current search or reasoning. You should promptly reflect that reconsideration and rejection process in your solution, then continue analyzing and searching for new clues until your answer is correct.  Do not search directly for statements provided to you out of nowhere, and do not follow a pattern of first searching for an unsupported claim and only then trying to verify it.
8. Note: you should not mention the above requirements in your solution.

Next, I will provide some excerpted examples, and you can learn from them how to write a high-quality answer process.

** Example 1: Avoid using internal knowledge or other fabricated evidence.

Bad writing:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository.
<action>
{
  "tool_name": "t2t_search",
  "arguments": {
    "query": "Alamy sources content from Wikimedia Commons"
  },
}
</action>

Good writing: 

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository.  I should first search which repository Alamy sources content from.
<action>
{
"tool_name": "t2t_search",
"arguments": {
"query": "The large repository Alamy sources content from"
}
}
</action>

Discuss: In the bad version, the query target 'Wiki Commons' comes from nowhere, which is a typical knowledge leakage of the writer model. In good version, the answer is more logically rigorous, the reasoning is more careful, and there are no clues appearing from nowhere.

** Example 2: Effective Tool Use/Coordination

Original question: In 2022, Messi, as captain, lifted the World Cup trophy on the podium, surrounded by his teammates. How many goals did the player standing immediately to Messi’s left score in the 2016–17 season?

Good writing:
To determine the goal total of the player standing to Messi’s left on the World Cup final podium, we can search for a photo of the *full Argentina team on the podium at the 2022 World Cup final award ceremony*. Then I can inspect the player standing to Messi’s left in the image and identify who he is. Once I know his identity, we can search his career history and find how many goals he scored in the 2016–17 season. Next, I will call t2t_search to search for that image.
<action>
{
"tool_name": "t2t_search",
"arguments": {
"query": "2022 World Cup final Argentina team award ceremony photo"
}
}
</action>
(...tool result omitted...)
The tool returned several compact search-result records. The titles of the first, second, and fourth results mention “Argentina winning the World Cup,” so the corresponding images are likely candidates. Search metadata is only a clue, not visual proof. I will inspect the first candidate image using its returned image_id.
<action>
{
"tool_name": "read_url",
"arguments": {
 "resource_id": "image_id from the selected search result",
 "goal": "Download the candidate image and check whether it is the target trophy-presentation photo."
}
}
</action>
(... tool result omitted ...)
The image has been successfully downloaded, and it is indeed a photo of the full Argentina team during the trophy presentation. In the image, Messi is wearing a black bisht and lifting the World Cup trophy. The player on his left is wearing the number 21 shirt. From this image alone, I cannot be fully certain who he is. My next step should be to use i2i_search, crop out this player, and see whether I can find similar images to determine his identity. In addition, since the image shows that he is wearing number 21, if i2i_search does not produce a satisfactory result, I can also directly search who wore number 21 for Argentina in the 2022 World Cup squad.
<action>
{
"tool_name": "i2i_search",
"arguments": {
"region": [420,340,520,370]
}
}
</action>
(... tool result omitted ...)
Several of the search result titles mention Dybala, which suggests that this player is very likely Dybala. However, I still cannot be certain, because it is possible that the retrieved images are not actually of the same person as the cropped player. Therefore, I should download another image and compare it with the person in the original image to see whether they are indeed the same individual. So I will use read_url to download a new image. If it does turn out to be the same person, then I will only need to search Dybala’s historical goal records, and by reading the relevant sources I will be able to determine his goal total for the 2016–17 season.
(... omitted below ...)

Discuss: In the example above, the solution first analyzes the question and then describes the target image in a detailed and precise way, rather than simply searching for something like “Argentina championship celebration photo,” which could return many different images that fit that description. It then opens a returned image resource by image_id and confirms that it is indeed the target photo. After that, it uses i2i_search on that photo for identification, and read_url with another result ID to further verify an unfamiliar person. The whole process is natural and rigorous, and it does not reveal any internal knowledge.

"""

MANUAL_REACT_PROTOCOL = """
When writing the solution, follow these tool rules:

1. t2t_search returns compact records with title, snippet, and source_page_id. Use read_url with source_page_id to inspect a selected webpage.
2. t2i_search returns compact records with image_id and source_page_id. Use read_url with image_id to inspect an image, or source_page_id to read its source page.
3. i2i_search performs reverse-image search. It returns compact records with image_id and source_page_id instead of raw URLs. Titles, source labels, snippets, and URL-derived keywords expose useful source metadata and can help choose which ID to read; verify the selected resource before treating it as evidence.
4. Search-result images are not visible until read_url succeeds with their image_id. The runtime may privately use a thumbnail fallback if the primary image cannot be downloaded.
5. For read_url, use resource_id from a prior search result whenever available. Include a focused arguments.goal explaining the evidence to extract. The legacy url argument is only for direct links already available in the conversation.

You must answer exactly one step at a time. Then end your response with exactly one action block in the following format:

<action>
{
  "tool_name": "tool_name",
  "arguments": {
    "parm1": "your param here"
  }
}
</action>

Rules:
- Output exactly one <action>...</action> block in each round.
- The content inside <action> must be valid JSON.
- The JSON must contain exactly these top-level keys: tool_name, arguments.
- For most tools, put only the real execution arguments inside arguments.
- For read_url, use resource_id from a search result whenever available and include arguments.goal.
"""


# #### START Response 0720 ####
RESPONSES_SYSTEM_PROMPT_V2 = """
You are writing a complete solution for a multi-hop knowledge question. Specifically, based on the question provided to you, you need to produce a full problem-solving process that includes scientifically rigorous and logically coherent reasoning steps. This solution process should include analysis and reasoning about the question, native tool calls, analysis and reflection on tool results, replanning of the solution steps, multiple search attempts when necessary, and a final accurate answer.

Requirements:
1. Ensure that the solution process is strictly logical. You need to break the question down step by step and solve each part in sequence, so that a reader can understand your reasoning clearly just by reading the written solution process.
2. In the solution you write, the following logic must be explicitly visible: before each native function call, first write a detailed and substantive progress update. This update should analyze what the latest tool observation has actually established, what uncertainties still remain, what tool should be called next to resolve the problem, and why that function and its arguments are the right choice.
3. In the solution you write, every factual statement must be grounded in evidence from the question, the image, or tool-returned results. Do not introduce any fact, entity, date, name, or relationship unless it is directly supported by the available evidence. As for non-evidential content, such as intermediate answers and the final answer provided to you separately, those must never appear or even be mentioned anywhere in your response process.
4. For widely known common knowledge or very easily identifiable visual content—such as recognizing a flag in an image—you may skip search and reason directly. However, to ensure accuracy and avoid hallucinations, this should be limited to knowledge that is truly universally familiar or to visual entities that are extremely clear and easy to identify. To be safe, you should make the logic explicit: first make a tentative guess based on common knowledge, and then search for clues to verify it. Note that this “guess first, verify later” approach does not apply to any intermediate statements or final answers.
5. Pay attention to the pace of reasoning: once you believe the evidence is sufficient and no unclear or uncertain points remain, you should provide the final answer directly and must not call any more tools. During the response process, when the evidence in the main reasoning chain is sufficient, do not keep searching repeatedly just to eliminate minor local ambiguities; the core objective is to solve the final question, not to remove every small ambiguity in the process.
6. While writing the solution, you must not search specific websites such as Wikipedia or Wiki Commons directly. However, if such sources appear in search results, you may freely read them.
7. Please analyze the task carefully and choose the most appropriate tool, rather than relying on text search by default. This is because some detailed, visual, or scene-related clues may exist only in relevant images. For any sub-question, you may either search for textual clues directly, or use t2i_search to find relevant images and then inspect those images for clues. Therefore, you need to judge whether the evidence needed is more likely to be recorded in webpage text or easier to obtain from related images, and then choose the tool most likely to yield useful evidence. If the intermediate factual statements provided to you mention images, that is a hint that this clue is more likely to be found in relevant images, so you should use t2i_search.
8. Intermediate factual statements related to the question, as well as the final answer, will be provided to you. Before you obtain search evidence, you must not mention or reveal any of that content, or even the name of any entity contained in it. These provided items are only for checking whether your current step-by-step solution process is correct. If your search and analysis results conflict with those provided factual statements, that means there is a flaw in your current search results or reasoning. You should then promptly reflect that reconsideration, self-correction, and re-examination process in the written solution, and continue analyzing and searching for new clues until the answer is correct. If you do not find any problem in your reasoning, then you should trust your reasoning and continue moving forward. Do not directly search for the content of the provided statements without cause. Also, do not follow a pattern where you first search for a claim derived from an intermediate statement that is not yet supported by evidence, and only afterward try to verify it. Do not use the following logic in your response: improperly anchoring the reasoning process to an intermediate statement—for example, seeing the name of an entity mentioned in an intermediate statement in the search results and then directly assuming that entity is correct without any substantive evidence or analysis.
9. The separately provided intermediate statements and final answer are not evidence and must remain completely isolated from the reasoning process unless independently established through tool-grounded evidence. They must not influence query formulation, candidate selection, source preference, conflict resolution, confidence calibration, or final answer wording. After repeated low-information searches, never narrow a query with unverified candidate details. Replan from the last evidence-supported unresolved variable using a broader, open-ended query. A query string must contain ONLY tokens that already appear in the question, the image, or in evidence you have already obtained from a prior tool result in this trajectory. It is strictly forbidden to place the wording of the intermediate/final answer — in whole or in part, verbatim or lightly paraphrased — into any query, especially into an exact-phrase ("...") search, when that wording has not yet appeared in any inspected tool result.
10. Close the chain, then stop. Treat each hop as closed the moment one inspected source establishes it. Do not open additional "more authoritative" sources to re-confirm a hop that is already grounded (this is redundant even when harmless). Escalate to a second source ONLY when the first is truncated/blocked/ambiguous, or when the question's wording demands a stricter form of the fact (e.g. it asks "founded" but you only have "co-editor") — and say explicitly which of these triggers the extra step.
11. The answer is provided to verify whether your reasoning arrives at the correct endpoint — it is not a target you must reproduce word for word. As long as your response completely and correctly addresses what the question itself asks for, the task is considered complete; you do not need to match the answer verbatim in wording, dates, numbers, or other specific expressions (but you must ensure your answer is factually correct). Do not treat details that appear only in the answer — and that are not surfaced by the question or by the tool results you have already obtained (e.g., a specific year, a particular phrasing) — as search targets to be repeatedly verified. The criterion for whether a detail is worth continuing to investigate is "whether it is naturally raised by the question or by the evidence already obtained," not "whether it is mentioned in the answer."
Note: the solution must not mention any of the above writing requirements. In every round of solution writing, you must check one by one that the above requirements are satisfied.

Tool-use tips:
1. t2t_search returns compact records with title, snippet, and source_page_id. Use read_url with source_page_id to inspect a promising page before treating page content as verified evidence.
2. t2i_search returns compact image-search records with image_id and source_page_id. The images are not visible yet. Use read_url with image_id to inspect an image or source_page_id to inspect its page before making visual claims.
3. i2i_search returns compact reverse-image-search records with image_id and source_page_id. Matches may be noisy. Use titles, sources, and URL-derived keyword hints to select the most appropriate image_id or source_page_id for the next read_url call; verify the selected resource before making factual claims.
4. After i2i_search or t2i_search, do not claim that you have seen a returned image unless a successful read_url call with its image_id has downloaded/read that image. Search metadata and URL-derived keyword hints help select which resource ID to read, but a successful read_url inspection is still required before making visual claims.
5. Use search tools **flexibly**. If you still cannot find a specific detail after multiple search attempts, try searching indirectly for related pages that may contain the information. For example, if repeated searches for a certain Olympic delegation with 108 athletes yield no results, you can instead search for statistics on delegation sizes by country, or for participation statistics from that edition of the Olympics.
6. For i2i_search, region coordinates are x-first normalized coordinates on a 0-1000 scale in the order [x1, y1, x2, y2]. x increases left-to-right and y increases top-to-bottom. Use [0, 0, 1000, 1000] for the full image.
7. For read_url, use a source_page_id or image_id returned by search whenever available. A source_page_id reads the webpage; an image_id downloads the image. Legacy raw URLs are supported only for direct links already available in the conversation. The tool cannot see your prior reasoning history, so clearly state what evidence you need in goal.

Next, I will provide some excerpted examples, and you can learn from them how to write a high-quality answer process. The examples use native function calls, so the written text shows the public progress update before the call and then names the intended native function call.

** Example 1: Avoid using internal knowledge or other fabricated evidence.

Bad writing:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository. I will call t2t_search with query: "Alamy sources content from Wikimedia Commons".

Good writing:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository. I should first search which repository Alamy sources content from, without assuming the repository name in advance. I will call t2t_search with query: "The large repository Alamy sources content from".

Discuss: In the bad version, the query target 'Wiki Commons' comes from nowhere, which is a typical knowledge leakage of the writer model. In good version, the answer is more logically rigorous, the reasoning is more careful, and there are no clues appearing from nowhere.

** Example 2: Effective Tool Use/Coordination

Original question: In 2022, Messi, as captain, lifted the World Cup trophy on the podium, surrounded by his teammates. How many goals did the player standing immediately to Messi's left score in the 2016-17 season?

Good writing:
To determine the goal total of the player standing to Messi's left on the World Cup final podium, we can search for a photo of the full Argentina team on the podium at the 2022 World Cup final award ceremony. Then I can inspect the player standing to Messi's left in the image and identify who he is. Once I know his identity, we can search his career history and find how many goals he scored in the 2016-17 season. Next, I will call t2i_search with query: "2022 World Cup final Argentina team award ceremony photo".
(...tool result omitted...)
The tool returned several compact image-search records. Among them, the titles of the first, second, and fourth results mention Argentina winning the World Cup, so they are plausible candidates. However, search metadata is not visual evidence. Next, I will inspect the first candidate image using its image_id to check whether it is the target image and whether it can provide clues for the question.
(... tool result omitted ...)
The image has been successfully downloaded, and it is indeed a photo of the full Argentina team during the trophy presentation. In the image, Messi is wearing a black bisht and lifting the World Cup trophy. The player on his left is wearing the number 21 shirt. From this image alone, I cannot be fully certain who he is. My next step should be to use i2i_search, crop out this player, and see whether I can find similar images to determine his identity. In addition, since the image shows that he is wearing number 21, if i2i_search does not produce a satisfactory result, I can also directly search who wore number 21 for Argentina in the 2022 World Cup squad. I will call i2i_search with region [420, 340, 520, 370].
(... tool result omitted ...)
Several of the search result titles mention Dybala, which suggests that this player is very likely Dybala. However, I still cannot be certain, because it is possible that the retrieved images are not actually of the same person as the cropped player. Therefore, I should download another image and compare it with the person in the original image to see whether they are indeed the same individual. So I will use read_url to download a new image. If it does turn out to be the same person, then I will only need to search Dybala's historical goal records, and by reading the relevant sources I will be able to determine his goal total for the 2016-17 season.
(... omitted below ...)

Discuss: In the example above, the solution first analyzes the question and then describes the target image in a detailed and precise way, rather than simply searching for something like "Argentina championship celebration photo," which could return many different images that fit that description. It then opens a returned image resource by image_id and confirms that it is indeed the target photo. After that, it uses i2i_search on that photo for identification, and read_url with another result ID to further verify an unfamiliar person. The whole process is natural and rigorous, and it does not reveal any internal knowledge.
""".strip()

# Public compatibility metadata for callers/tests that need to discover the
# Responses prompt's image-resource guidance without altering the prompt text.
# The actual instructions remain embedded verbatim in the prompt constants
# below; this value is intentionally not interpolated into them.
RESPONSES_TOOL_USE_TIPS = """
t2i_search returns compact image-search records with image_id and source_page_id.
i2i_search returns compact reverse-image-search records with image_id and source_page_id.
Use read_url with image_id before making visual claims.
""".strip()

# #### START Response 0720 ####
RESPONSES_SYSTEM_PROMPT = """
You are writing a complete solution for a multi-hop knowledge question. Specifically, based on the question provided to you, you need to produce a full problem-solving process that includes scientifically rigorous and logically coherent reasoning steps. This solution process should include analysis and reasoning about the question, native tool calls, analysis and reflection on tool results, replanning of the solution steps, multiple search attempts when necessary, and a final accurate answer.

Requirements:
1. Ensure that the solution process is strictly logical. You need to break the question down step by step and solve each part in sequence, so that a reader can understand your reasoning clearly just by reading the written solution process.
2. In the solution you write, the following logic must be explicitly visible: before each native function call, first write a detailed and substantive progress update. This update should analyze what the latest tool observation has actually established, what uncertainties still remain, what tool should be called next to resolve the problem, and why that function and its arguments are the right choice.
3. In the solution you write, every factual statement must be grounded in evidence from the question, the image, or tool-returned results. Do not introduce any fact, entity, date, name, or relationship unless it is directly supported by the available evidence. As for non-evidential content, such as intermediate answers and the final answer provided to you separately, those must never appear or even be mentioned anywhere in your response process.
4. For widely known common knowledge or very easily identifiable visual content—such as recognizing a flag in an image—you may skip search and reason directly. However, to ensure accuracy and avoid hallucinations, this should be limited to knowledge that is truly universally familiar or to visual entities that are extremely clear and easy to identify. To be safe, you should make the logic explicit: first make a tentative guess based on common knowledge, and then search for clues to verify it. Note that this “guess first, verify later” approach does not apply to any intermediate statements or final answers.
5. Pay attention to the pace of reasoning: once you believe the evidence is sufficient and no unclear or uncertain points remain, you should provide the final answer directly and must not call any more tools. During the response process, when the evidence in the main reasoning chain is sufficient, do not keep searching repeatedly just to eliminate minor local ambiguities; the core objective is to solve the final question, not to remove every small ambiguity in the process.
6. While writing the solution, you must not search specific websites such as Wikipedia or Wiki Commons directly. However, if such sources appear in search results, you may freely read them.
7. Analyze the task carefully and choose the most appropriate tool. For any sub-question, you may either search for textual clues directly, or use t2i_search to find relevant images and then inspect those images for clues. Therefore, you should judge whether the evidence needed is more likely to appear in webpage text or to be found more easily through related images, and then choose the tool that is most likely to produce useful evidence. For example, when a question refers to a specific scene, group photo, or event, it is usually better to use t2i_search first to look for relevant images and then examine them for clues. As another example, if you need to determine which province lies directly to the west of Guizhou Province in China, you could use t2i_search to look for a map of China’s administrative divisions. Note that the images returned by the search may not be the exact ones you need, so you may need to look for other images or switch to a different line of reasoning.
8. Intermediate factual statements related to the question, as well as the final answer, will be provided to you. Before you obtain search evidence, you must not mention or reveal any of that content, or even the name of any entity contained in it. These provided items are only for checking whether your current step-by-step solution process is correct. If your search and analysis results conflict with those provided factual statements, that means there is a flaw in your current search results or reasoning. You should then promptly reflect that reconsideration, self-correction, and re-examination process in the written solution, and continue analyzing and searching for new clues until the answer is correct. If you do not find any problem in your reasoning, then you should trust your reasoning and continue moving forward. Do not directly search for the content of the provided statements without cause. Also, do not follow a pattern where you first search for a claim derived from an intermediate statement that is not yet supported by evidence, and only afterward try to verify it. Do not use the following logic in your response: improperly anchoring the reasoning process to an intermediate statement—for example, seeing the name of an entity mentioned in an intermediate statement in the search results and then directly assuming that entity is correct without any substantive evidence or analysis.
9. The separately provided intermediate statements and final answer are not evidence and must remain completely isolated from the reasoning process unless independently established through tool-grounded evidence. They must not influence query formulation, candidate selection, source preference, conflict resolution, confidence calibration, or final answer wording. After repeated low-information searches, never narrow a query with unverified candidate details. Replan from the last evidence-supported unresolved variable using a broader, open-ended query. A query string must contain ONLY tokens that already appear in the question, the image, or in evidence you have already obtained from a prior tool result in this trajectory. It is strictly forbidden to place the wording of the intermediate/final answer — in whole or in part, verbatim or lightly paraphrased — into any query, especially into an exact-phrase ("...") search, when that wording has not yet appeared in any inspected tool result.
10. Close the chain, then stop. Treat each hop as closed the moment one inspected source establishes it. Do not open additional "more authoritative" sources to re-confirm a hop that is already grounded (this is redundant even when harmless). Escalate to a second source ONLY when the first is truncated/blocked/ambiguous, or when the question's wording demands a stricter form of the fact (e.g. it asks "founded" but you only have "co-editor") — and say explicitly which of these triggers the extra step.
11. The answer is provided to verify whether your reasoning arrives at the correct endpoint — it is not a target you must reproduce word for word. As long as your response completely and correctly addresses what the question itself asks for, the task is considered complete; you do not need to match the answer verbatim in wording, dates, numbers, or other specific expressions (but you must ensure your answer is factually correct). Do not treat details that appear only in the answer — and that are not surfaced by the question or by the tool results you have already obtained (e.g., a specific year, a particular phrasing) — as search targets to be repeatedly verified. The criterion for whether a detail is worth continuing to investigate is "whether it is naturally raised by the question or by the evidence already obtained," not "whether it is mentioned in the answer."
Note: the solution must not mention any of the above writing requirements. In every round of solution writing, you must check one by one that the above requirements are satisfied.

Tool-use tips:
1. t2t_search returns compact records with title, snippet, and source_page_id. Use read_url with source_page_id to inspect a promising page before treating page content as verified evidence.
2. t2i_search returns compact image-search records with image_id and source_page_id. The images are not visible yet. Use read_url with image_id to inspect an image or source_page_id to inspect its page before making visual claims.
3. i2i_search returns compact reverse-image-search records with image_id and source_page_id. Matches may be noisy. Use titles, sources, and URL-derived keyword hints to select the most appropriate image_id or source_page_id for the next read_url call; verify the selected resource before making factual claims.
4. After i2i_search or t2i_search, do not claim that you have seen a returned image unless a successful read_url call with its image_id has downloaded/read that image. Search metadata and URL-derived keyword hints help select which resource ID to read, but a successful read_url inspection is still required before making visual claims.
5. Use search tools **flexibly**. If you still cannot find a specific detail after multiple search attempts, try searching indirectly for related pages that may contain the information. For example, if repeated searches for a certain Olympic delegation with 108 athletes yield no results, you can instead search for statistics on delegation sizes by country, or for participation statistics from that edition of the Olympics.
6. For i2i_search, region coordinates are x-first normalized coordinates on a 0-1000 scale in the order [x1, y1, x2, y2]. x increases left-to-right and y increases top-to-bottom. Use [0, 0, 1000, 1000] for the full image.
7. For read_url, use a source_page_id or image_id returned by search whenever available. A source_page_id reads the webpage; an image_id downloads the image. Legacy raw URLs are supported only for direct links already available in the conversation. The tool cannot see your prior reasoning history, so clearly state what evidence you need in goal.

Next, I will provide some excerpted examples, and you can learn from them how to write a high-quality answer process. The examples use native function calls, so the written text shows the public progress update before the call and then names the intended native function call.

** Example 1: Avoid using internal knowledge or other fabricated evidence.

Bad writing:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository. I will call t2t_search with query: "Alamy sources content from Wikimedia Commons".

Good writing:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository. I should first search which repository Alamy sources content from, without assuming the repository name in advance. I will call t2t_search with query: "The large repository Alamy sources content from".

Discuss: In the bad version, the query target 'Wiki Commons' comes from nowhere, which is a typical knowledge leakage of the writer model. In good version, the answer is more logically rigorous, the reasoning is more careful, and there are no clues appearing from nowhere.

** Example 2: Effective Tool Use/Coordination

Original question: In 2022, Messi, as captain, lifted the World Cup trophy on the podium, surrounded by his teammates. How many goals did the player standing immediately to Messi's left score in the 2016-17 season?

Good writing:
To determine the goal total of the player standing to Messi's left on the World Cup final podium, we can search for a photo of the full Argentina team on the podium at the 2022 World Cup final award ceremony. Then I can inspect the player standing to Messi's left in the image and identify who he is. Once I know his identity, we can search his career history and find how many goals he scored in the 2016-17 season. Next, I will call t2i_search with query: "2022 World Cup final Argentina team award ceremony photo".
(...tool result omitted...)
The tool returned several compact image-search records. Among them, the titles of the first, second, and fourth results mention Argentina winning the World Cup, so they are plausible candidates. However, search metadata is not visual evidence. Next, I will inspect the first candidate image using its image_id to check whether it is the target image and whether it can provide clues for the question.
(... tool result omitted ...)
The image has been successfully downloaded, and it is indeed a photo of the full Argentina team during the trophy presentation. In the image, Messi is wearing a black bisht and lifting the World Cup trophy. The player on his left is wearing the number 21 shirt. From this image alone, I cannot be fully certain who he is. My next step should be to use i2i_search, crop out this player, and see whether I can find similar images to determine his identity. In addition, since the image shows that he is wearing number 21, if i2i_search does not produce a satisfactory result, I can also directly search who wore number 21 for Argentina in the 2022 World Cup squad. I will call i2i_search with region [420, 340, 520, 370].
(... tool result omitted ...)
Several of the search result titles mention Dybala, which suggests that this player is very likely Dybala. However, I still cannot be certain, because it is possible that the retrieved images are not actually of the same person as the cropped player. Therefore, I should download another image and compare it with the person in the original image to see whether they are indeed the same individual. So I will use read_url to download a new image. If it does turn out to be the same person, then I will only need to search Dybala's historical goal records, and by reading the relevant sources I will be able to determine his goal total for the 2016-17 season.
(... omitted below ...)

Discuss: In the example above, the solution first analyzes the question and then describes the target image in a detailed and precise way, rather than simply searching for something like "Argentina championship celebration photo," which could return many different images that fit that description. It then opens a returned image resource by image_id and confirms that it is indeed the target photo. After that, it uses i2i_search on that photo for identification, and read_url with another result ID to further verify an unfamiliar person. The whole process is natural and rigorous, and it does not reveal any internal knowledge.
""".strip()


def _build_responses_instructions(base_system_prompt: str) -> str:
    # Use the Responses-specific system prompt only for the default SFT prompt.
    # Its complete tool-use guidance is embedded exactly once in that prompt.
    if base_system_prompt.strip() == DEFAULT_SYSTEM_PROMPT.strip():
        return RESPONSES_SYSTEM_PROMPT
    return base_system_prompt.strip()


_MANUAL_REACT_ACTIONS = {"t2t_search", "t2i_search", "i2i_search", "read_url", "finish"}
_MANUAL_REACT_ACTION_RE = re.compile(r"<action>\s*(?P<json>\{.*?\})\s*</action>", re.DOTALL | re.IGNORECASE)
_I2I_WRAPPER_DEFAULT_MODEL_ALIAS = "multimodal_process"
_I2I_WRAPPER_MAX_TOKENS = 2048
_SFT_FIXED_REQUEST_ID = "3200636808"


def _sft_worker_metadata(
    trace_label: str,
    *,
    extra_body: dict[str, Any] | None = None,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "trace_label": trace_label,
        "session_id": _SFT_FIXED_REQUEST_ID,
        "prompt_cache_key": _SFT_FIXED_REQUEST_ID,
        "user_id": _SFT_FIXED_REQUEST_ID,
        "x_tt_logid": _SFT_FIXED_REQUEST_ID,
    }
    if stop:
        metadata["stop"] = stop
    if extra_body:
        metadata["extra_body"] = extra_body
    return metadata


# #### START Response 0720 ####
def _sft_request_extra_headers() -> dict[str, str]:
    """Return per-request cache/session headers for direct OpenAI SDK calls."""

    return {
        "extra": json.dumps(
            {
                "session_id": _SFT_FIXED_REQUEST_ID,
                "prompt_cache_key": _SFT_FIXED_REQUEST_ID,
                "user_id": _SFT_FIXED_REQUEST_ID,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "X-TT-LOGID": _SFT_FIXED_REQUEST_ID,
    }
# #### END Response 0720 ####

PROMPT_I2I_REWRITE_ASSISTANT = """
I will give you an image and a passage containing analysis and tool-call process text for a certain question. This passage is missing context. Your goal is to determine, based only on this single passage, which object in the image the passage is focusing on. Then, summarize that object as a noun phrase (possibly with a descriptive referring expression). Finally, polish the parts of the passage that are related to tool calling so that the logic becomes tighter and more coherent.

Rules:
1. Only polish the text related to tool calling detailedly, such as the purpose of calling the tool, the motivation, what is intended to be searched, and so on. Do not modify other content or the overall logic.
2. When polishing, besides making the logic more rigorous and detailed, you may also appropriately add text describing that the next step is to locate the target object of interest. But make sure the logic remains rigorous. Do not use structured polishment such as "Goal: ...".
3. Any text that is kept unchanged must remain exactly the same as the original in content, format, and even punctuation.
4. Output in the following format:
...Think process first...
<object>The entity in the image that this passage is trying to find</object>
<refined>The polished text</refined>

Example:
Input: "To answer this question, I need to follow a multi-step process. First, I need to identify the celestial body shown in the image to determine the orbiter that discovered its prominent equatorial ridge. Once the orbiter is identified, I can find its launch vehicle program. Then, I will research the three consecutive launch failures of that program between August 1998 and April 1999 and find the distinct root cause for each.

My first step is to use the provided image to identify the celestial body.

<action>
{
  "tool_name": "i2i_search",
  "arguments": {...ignored...}
}
</action>"
Your output:
...The detailed thinking process is ignored in this example, but you should think step by step in your response...
<object>Celestial body</object>
<refined>To answer this question, I need to follow a multi-step process. First, I need to identify the celestial body shown in the image to determine the orbiter that discovered its prominent equatorial ridge. Once the orbiter is identified, I can find its launch vehicle program. Then, I will research the three consecutive launch failures of that program between August 1998 and April 1999 and find the distinct root cause for each. So based on the above plan, for the input image, I first need to locate the position of this celestial body within the image, crop out the relevant local region, and pass its position to the i2i_search tool so that I can search for this celestial body. Ideally, by using similar images and their descriptions, I can determine exactly which celestial body it is.

<action>
{
  "tool_name": "i2i_search",
  "arguments": {...ignored...}
}
</action></refined>
"""

# #### START Response 0720 ####
PROMPT_I2I_GROUND_OBJECT = """
You need to localize a target object in an image for reverse image search.
Please return one bounding box for the target object using normalized coordinates on a 0-1000 scale:
Format: [x1, y1, x2, y2]
If most of the image consists of the target object, or if the target is not a clearly defined entity (for example, it is a description of a scene), please return the full image:
[0, 0, 1000, 1000]
Please return strict JSON:
{
"label": "...",
"bbox": [x1, y1, x2, y2]
}
"""
# #### END Response 0720 ####
_NORMALIZED_COORD_SCALE = 1000.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _prepare_region_for_crop(
    region: object,
    image_size: tuple[int, int],
    *,
    coordinate_order: str | None = None,
) -> object:
    """Convert model-style normalized coordinates into absolute crop coordinates."""

    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return region

    try:
        coords = [float(value) for value in region]
    except (TypeError, ValueError):
        return region

    # #### START Response 0720 ####
    effective_order = (coordinate_order or "").strip().lower()
    if not effective_order:
        effective_order = "yxyx" if _env_flag("REVERSE_IMAGE_CROP_COORDS") else "xyxy"
    if effective_order == "yxyx":
        coords = [coords[1], coords[0], coords[3], coords[2]]
    # #### END Response 0720 ####

    image_width, image_height = image_size
    x1 = int(round(coords[0] / _NORMALIZED_COORD_SCALE * image_width))
    y1 = int(round(coords[1] / _NORMALIZED_COORD_SCALE * image_height))
    x2 = int(round(coords[2] / _NORMALIZED_COORD_SCALE * image_width))
    y2 = int(round(coords[3] / _NORMALIZED_COORD_SCALE * image_height))

    x1 = min(max(x1, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    x2 = min(max(x2, 0), image_width)
    y2 = min(max(y2, 0), image_height)
    return [x1, y1, x2, y2]


def _truncate_tool_calls(tool_calls: list[Any], *, source: str) -> list[Any]:
    if len(tool_calls) > 1:
        logger.warning(
            "Expected at most one tool call per turn; keeping only the first from %s and dropping %d extra call(s).",
            source,
            len(tool_calls) - 1,
        )
        return tool_calls[:1]
    return tool_calls


@dataclass(slots=True)
class ToolRuntimeContext:
    """Per-session runtime state used by the tool dispatcher."""

    working_dir: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    image_registry: dict[str, Any] = field(default_factory=dict)
    url_registry: dict[str, tools.UrlResource] = field(default_factory=dict)
    resource_registry: dict[str, tools.UrlResource] = field(default_factory=dict)
    filename_prefix: str = "sft"
    case_id: str = "sft_session"
    visual_lookup: Callable[..., object] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _image_counter: int = 0

    def __post_init__(self) -> None:
        self.working_dir = os.path.abspath(self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        if self.image_registry:
            for key in self.image_registry:
                if key.startswith("img_"):
                    try:
                        self._image_counter = max(self._image_counter, int(key.split("_", 1)[1]))
                    except (TypeError, ValueError):
                        continue

    @property
    def intermediate_dir(self) -> str:
        path = os.path.join(self.working_dir, "artifacts")
        os.makedirs(path, exist_ok=True)
        return path

    def next_image_id(self) -> str:
        self._image_counter += 1
        return f"img_{self._image_counter}"

    def register_image(self, payload: Any) -> str:
        image_id = self.next_image_id()
        self.image_registry[image_id] = payload
        return image_id

    def latest_image_reference(self) -> str | None:
        if not self.image_registry:
            return None
        return next(reversed(self.image_registry))

    def image_summary(self) -> str:
        if not self.image_registry:
            return ""
        lines = ["Available image refs:"]
        for image_id, payload in self.image_registry.items():
            lines.append(f"- {image_id}: {str(payload)[:120]}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_resource_url(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            parsed = urllib.parse.urlsplit(raw)
        except Exception:
            return raw
        if parsed.scheme not in {"http", "https"}:
            return raw
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
        netloc = hostname
        if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        return urllib.parse.urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))

    def register_url_resource(self, resource: tools.UrlResource) -> None:
        if resource.resource_id:
            existing = self.resource_registry.get(resource.resource_id)
            if existing is not None and existing.primary_url != resource.primary_url:
                raise ValueError(f"Resource ID collision: {resource.resource_id}")
            self.resource_registry[resource.resource_id] = resource
        for candidate in (
            resource.primary_url,
            resource.image_url,
            resource.thumbnail_url,
            resource.source_page_url,
        ):
            key = self._normalize_resource_url(candidate)
            if key:
                self.url_registry[key] = resource

    def resolve_url_resource(self, url: str) -> tools.UrlResource | None:
        return self.url_registry.get(self._normalize_resource_url(url))

    def resolve_resource_id(self, resource_id: str) -> tools.UrlResource | None:
        return self.resource_registry.get(str(resource_id or "").strip())

    def register_search_output(self, tool_name: str, output: dict[str, Any]) -> None:
        _compact, resources = tools.postprocess_search_output(
            tool_name=tool_name,
            output=output,
        )
        for resource in resources:
            self.register_url_resource(resource)

    def postprocess_search_output(self, tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
        compact, resources = tools.postprocess_search_output(
            tool_name=tool_name,
            output=output,
        )
        for resource in resources:
            self.register_url_resource(resource)
        return compact


@dataclass(slots=True)
class ToolExecutionResult:
    """Structured result for one tool invocation."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    output_text: str
    new_images: dict[str, Any] = field(default_factory=dict)


_DEFAULT_FAILURE_JSONL_PATH = Path("synthesis/ignore/failure.jsonl")
_FAILURE_JSONL_ENV = "SFT_FAILURE_JSONL"
_TOOL_FAILURE_RECORDS: list[dict[str, Any]] = []
_TOOL_FAILURE_LOCK = threading.Lock()


def get_tool_failure_records() -> list[dict[str, Any]]:
    """Return a snapshot of tool failures recorded in this process."""
    with _TOOL_FAILURE_LOCK:
        return deepcopy(_TOOL_FAILURE_RECORDS)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _failure_jsonl_path() -> Path:
    configured = str(os.environ.get(_FAILURE_JSONL_ENV) or "").strip()
    return Path(configured) if configured else _DEFAULT_FAILURE_JSONL_PATH


def _record_tool_failure(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    error_output: Any,
) -> None:
    """Keep an in-memory failure list and append one durable JSONL record."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "arguments": _json_safe(arguments),
        "error_output": _json_safe(error_output),
    }
    with _TOOL_FAILURE_LOCK:
        _TOOL_FAILURE_RECORDS.append(record)
        try:
            path = _failure_jsonl_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, default=str)
                handle.write("\n")
                handle.flush()
        except Exception:
            # Failure logging must not turn the original tool failure into a
            # separate dispatcher failure.
            logger.exception("Unable to persist tool failure to %s", _failure_jsonl_path())


def _read_url_image_attachment(
    result: ToolExecutionResult,
    context: ToolRuntimeContext,
) -> dict[str, Any] | None:
    """Build a multimodal follow-up message for an image downloaded by read_url."""
    if result.name != "read_url" or not result.new_images:
        return None
    image_id, _local_path = next(iter(result.new_images.items()))
    try:
        model_url = _image_source_to_model_url(image_id, context)
    except Exception as exc:
        logger.warning("Unable to attach read_url image %s to the next model turn: %s", image_id, exc)
        return None
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"image_id={image_id}",
            },
            {"type": "image_url", "image_url": {"url": model_url, "detail": "auto"}},
        ],
    }


@dataclass(slots=True)
class OpenAIToolAgentConfig:
    """Configuration for OpenAI-compatible chat-completions tool calling."""

    model: str
    api_key: str | None = None
    client_type: str = "azure_openai"
    azure_endpoint: str | None = None
    base_url: str | None = None
    api_version: str = "2024-03-01-preview"
    api_mode: str = "manual_react"
    max_tokens: int | None = 1024
    temperature: float | None = None
    timeout_s: float = 120.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    default_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    max_turns: int = 8
    print_rounds: bool = True
    # #### START Response 0720 ####
    responses_reasoning_effort: str | None = None
    responses_reasoning_summary: str | None = "auto"
    responses_reasoning_mode: str | None = None
    responses_reasoning_context: str | None = "all_turns"
    responses_parallel_tool_calls: bool = False
    responses_store: bool | None = None
    responses_prompt_public_reasoning: bool = True
    responses_i2i_wrapper_enabled: bool = False
    # #### END Response 0720 ####


@dataclass(slots=True)
class AgentRunResult:
    """Final answer plus the intermediate tool trace."""

    final_text: str
    messages: list[dict[str, Any]]
    tool_results: list[ToolExecutionResult]
    raw_responses: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ManualReActStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    goal: str
    raw_text: str


@dataclass(slots=True)
class I2IRepairResult:
    assistant_text: str
    display_arguments: dict[str, Any]
    execution_arguments: dict[str, Any]
    target_object: str
    used_full_image: bool = False


def _guess_mime_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _normalize_region_bbox(region: object) -> tuple[tuple[int, int, int, int] | None, str | None]:
    if region in (None, ""):
        return None, None
    if isinstance(region, dict):
        if not all(key in region for key in ("x", "y", "width", "height")):
            return None, "Region dict must contain x, y, width, and height."
        try:
            x = int(region["x"])
            y = int(region["y"])
            width = int(region["width"])
            height = int(region["height"])
        except (TypeError, ValueError):
            return None, "Region dict values must be numeric."
        if width <= 0 or height <= 0:
            return None, "Region width and height must be positive."
        return (x, y, width, height), None
    if isinstance(region, (list, tuple)):
        if len(region) != 4:
            return None, "Region list must contain exactly 4 numbers."
        try:
            x1 = int(region[0])
            y1 = int(region[1])
            x2 = int(region[2])
            y2 = int(region[3])
        except (TypeError, ValueError):
            return None, "Region list values must be numeric."
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None, "Region list must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1."
        return (x1, y1, width, height), None
    return None, "Region must be a 4-number list or a dict with x/y/width/height."


def _resolve_registered_model_alias(alias_or_model: str | None) -> dict[str, Any] | None:
    if not alias_or_model:
        return None
    try:
        return LLM_WORKER.get_model(alias_or_model)
    except Exception:
        return None


def _i2i_wrapper_model_alias() -> str | None:
    alias = (
        os.environ.get("SFT_I2I_WRAPPER_MODEL")
        or tools.get_sft_qwen_model_alias()
        or _I2I_WRAPPER_DEFAULT_MODEL_ALIAS
    )
    return alias if _resolve_registered_model_alias(alias) is not None else None


def _i2i_wrapper_max_tokens() -> int:
    raw_value = os.environ.get("SFT_I2I_WRAPPER_MAX_TOKENS")
    if raw_value is None or str(raw_value).strip() == "":
        return _I2I_WRAPPER_MAX_TOKENS
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return _I2I_WRAPPER_MAX_TOKENS


def _worker_generate_json_message(
    *,
    model_alias: str,
    system_prompt: str,
    user_content: Any,
    max_tokens: int,
    trace_label: str,
) -> dict[str, Any]:
    response = _timed_call(
        "llm:auxiliary_json",
        lambda: LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=system_prompt),
                    ModelMessage(role="user", content=user_content),
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                metadata=_sft_worker_metadata(trace_label),
            )
        ),
    )
    parsed = _extract_json_object(response.content or "")
    if parsed is None:
        raise ValueError(f"Model response is not valid JSON: {response.content[:500]}")
    return parsed


def _worker_generate_text_message(
    *,
    model_alias: str,
    system_prompt: str,
    user_content: Any,
    max_tokens: int,
    trace_label: str,
) -> str:
    response = _timed_call(
        "llm:auxiliary_text",
        lambda: LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=system_prompt),
                    ModelMessage(role="user", content=user_content),
                ],
                max_tokens=max_tokens,
                metadata=_sft_worker_metadata(trace_label),
            )
        ),
    )
    return response.content or ""


def _extract_xml_tag_content(text: str, tag_name: str) -> str:
    pattern = rf"<{re.escape(tag_name)}>\s*(.*?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _strip_action_blocks(text: str) -> str:
    if not text:
        return ""
    cleaned = _MANUAL_REACT_ACTION_RE.sub("", text)
    return cleaned.strip()


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and str(part.get("type") or "") in {"text", "input_text"}:
                    text = str(part.get("text", "")).strip()
                    if text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
    return ""


def _parse_manual_react_goal(text: str) -> str:
    matches = list(_MANUAL_REACT_ACTION_RE.finditer(text.strip()))
    if not matches:
        return ""
    action_payload = _extract_json_object(matches[-1].group("json"))
    if not isinstance(action_payload, dict):
        return ""
    arguments = action_payload.get("arguments")
    if isinstance(arguments, dict):
        nested_goal = str(arguments.get("goal") or "").strip()
        if nested_goal:
            return nested_goal
    return str(action_payload.get("goal") or "").strip()


def _render_manual_react_text(
    *,
    thought: str,
    action: str,
    arguments: dict[str, Any],
    goal: str,
) -> str:
    normalized_arguments = dict(arguments)
    if action == "read_url" and goal and "goal" not in normalized_arguments:
        normalized_arguments["goal"] = goal
    payload = {
        "tool_name": action,
        "arguments": normalized_arguments,
    }
    action_text = json.dumps(payload, ensure_ascii=False, indent=2)
    cleaned_thought = thought.strip()
    if cleaned_thought:
        return f"{cleaned_thought}\n\n<action>\n{action_text}\n</action>"
    return f"<action>\n{action_text}\n</action>"


def _clip_box_coord(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return min(max(numeric, 0), int(_NORMALIZED_COORD_SCALE))


def _full_image_box_xyxy() -> list[int]:
    max_coord = int(_NORMALIZED_COORD_SCALE)
    return [0, 0, max_coord, max_coord]


def _normalize_xyxy_box_1000(raw_box: Any) -> tuple[list[int], bool]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return _full_image_box_xyxy(), True
    x1, y1, x2, y2 = [_clip_box_coord(value) for value in raw_box]
    if x2 <= x1 or y2 <= y1:
        return _full_image_box_xyxy(), True
    return [x1, y1, x2, y2], False


def _xyxy_to_yxyx(box_xyxy: list[int]) -> list[int]:
    return [int(box_xyxy[1]), int(box_xyxy[0]), int(box_xyxy[3]), int(box_xyxy[2])]


def _maybe_repair_i2i_tool_call(
    *,
    assistant_text: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
    context: ToolRuntimeContext,
    question_text: str,
) -> I2IRepairResult | None:
    if tool_name != "i2i_search":
        return None

    model_alias = _i2i_wrapper_model_alias()
    if not model_alias:
        return None

    image_source = tool_arguments.get("image") or tool_arguments.get("url") or context.latest_image_reference() or ""
    if not image_source:
        return None

    try:
        image = _load_pil_image(image_source, context)
        image_width, image_height = image.size
    except Exception as exc:
        logger.warning("Failed to load image for i2i wrapper repair: %s", exc)
        return None

    try:
        image_url = _image_source_to_model_url(image_source, context)
    except Exception as exc:
        logger.warning("Failed to materialize image input for i2i wrapper repair: %s", exc)
        return None

    wrapper_max_tokens = _i2i_wrapper_max_tokens()
    question = question_text.strip()
    original_text = assistant_text.strip()
    original_region = tool_arguments.get("region")

    try:
        rewrite_payload = {
            "question": question,
            "assistant_text": original_text,
            "current_tool_name": tool_name,
            "current_tool_arguments": tool_arguments,
            "current_region": original_region,
            "image_size": {"width": image_width, "height": image_height},
            "required_sentence_style": (
                "Make it explicit what object in the image needs to be identified first and why reverse image search helps."
            ),
        }
        rewrite_text = _worker_generate_text_message(
            model_alias=model_alias,
            system_prompt=PROMPT_I2I_REWRITE_ASSISTANT,
            user_content=[
                {"type": "text", "text": json.dumps(rewrite_payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            max_tokens=wrapper_max_tokens,
            trace_label=f"i2i_wrapper_rewrite:{context.case_id}",
        )
        target_object = _extract_xml_tag_content(rewrite_text, "object")
        revised_assistant_text = _strip_action_blocks(
            _extract_xml_tag_content(rewrite_text, "refined")
        ) or original_text
        if not target_object:
            target_object = "the relevant object in the image"
    except Exception as exc:
        logger.warning("i2i wrapper rewrite failed: %s", exc)
        return None

    try:
        grounding_payload = {
            "question": question,
            "assistant_text": revised_assistant_text,
            "target_object": target_object,
            "current_region": original_region,
            "image_size": {"width": image_width, "height": image_height},
            "coordinate_format": {
                "required_output": "[x1, y1, x2, y2]",
                "normalized_scale": [0, int(_NORMALIZED_COORD_SCALE)],
            },
        }
        grounding_result = _worker_generate_json_message(
            model_alias=model_alias,
            system_prompt=PROMPT_I2I_GROUND_OBJECT,
            user_content=[
                {"type": "text", "text": json.dumps(grounding_payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            max_tokens=wrapper_max_tokens,
            trace_label=f"i2i_wrapper_ground:{context.case_id}",
        )
    except Exception as exc:
        logger.warning("i2i wrapper grounding failed: %s", exc)
        grounding_result = {"bbox": _full_image_box_xyxy(), "used_full_image": True}

    bbox_xyxy, invalid_box = _normalize_xyxy_box_1000(
        grounding_result.get("bbox")
        or grounding_result.get("bbox_xyxy")
        or grounding_result.get("region")
    )
    used_full_image = bool(grounding_result.get("used_full_image")) or invalid_box

    display_arguments = dict(tool_arguments)
    display_arguments["region"] = _xyxy_to_yxyx(bbox_xyxy)

    execution_arguments = dict(tool_arguments)
    execution_arguments["region"] = (
        list(display_arguments["region"])
        if _env_flag("REVERSE_IMAGE_CROP_COORDS")
        else list(bbox_xyxy)
    )

    return I2IRepairResult(
        assistant_text=revised_assistant_text,
        display_arguments=display_arguments,
        execution_arguments=execution_arguments,
        target_object=target_object,
        used_full_image=used_full_image,
    )


def _decode_data_url(data_url: str) -> bytes:
    payload = data_url.split("base64,", 1)[-1]
    return base64.b64decode(payload)


def _encode_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"


def _ensure_png_compatible_image(image: Image.Image) -> Image.Image:
    """Convert PIL modes unsupported by PNG to RGB/RGBA before serialization."""
    if image.mode in {"RGB", "RGBA", "L", "LA"}:
        return image
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB")


def _normalize_image_data_url(data: bytes) -> str:
    """Validate arbitrary image bytes and serialize one API-supported PNG.

    Search and reader backends can return AVIF/SVG/TIFF files, HTML error
    pages with an image-like extension, or content whose extension disagrees
    with its bytes.  Azure validates the actual bytes behind a data URL, so a
    guessed MIME type is insufficient.  Decoding and re-encoding here makes
    every accepted local/data/remote image a valid supported input.
    """

    try:
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            image = ImageOps.exif_transpose(decoded)
            image = _ensure_png_compatible_image(image)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
    except Exception as exc:
        raise ValueError("Image input is not a decodable raster image.") from exc
    return _encode_data_url(output.getvalue(), "image/png")


def _resolve_image_payload(source: Any, context: ToolRuntimeContext) -> Any:
    if isinstance(source, str) and source in context.image_registry:
        return context.image_registry[source]
    return source


def _image_source_to_model_url(source: Any, context: ToolRuntimeContext) -> str:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, str):
        if payload.startswith("data:image"):
            return _normalize_image_data_url(_decode_data_url(payload))
        if payload.startswith(("http://", "https://")):
            response = requests.get(payload, timeout=60)
            response.raise_for_status()
            return _normalize_image_data_url(response.content)
        if os.path.exists(payload):
            with open(payload, "rb") as handle:
                return _normalize_image_data_url(handle.read())
    if isinstance(payload, bytes):
        return _normalize_image_data_url(payload)
    if isinstance(payload, Image.Image):
        buffer = io.BytesIO()
        _ensure_png_compatible_image(payload).save(buffer, format="PNG")
        return _normalize_image_data_url(buffer.getvalue())
    raise ValueError(f"Unsupported image source for model input: {type(payload)!r}")


def _ensure_inline_image_registered(source: Any, context: ToolRuntimeContext) -> None:
    if not isinstance(source, str) or not source.startswith("data:image"):
        return
    if source in context.image_registry:
        return
    for payload in context.image_registry.values():
        if payload == source:
            return
    context.register_image(source)


def _normalize_content_part(part: Any, context: ToolRuntimeContext) -> dict[str, Any]:
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    part_type = str(part.get("type") or "").strip()
    if part_type in {"text", "input_text"}:
        return {"type": "text", "text": str(part.get("text", ""))}

    if part_type == "image_url":
        image_url = part.get("image_url")
        detail = part.get("detail")
        if isinstance(image_url, dict):
            source = image_url.get("url", "")
        else:
            source = image_url
        _ensure_inline_image_registered(source, context)
        normalized = {"type": "image_url", "image_url": {"url": _image_source_to_model_url(source, context)}}
        if detail:
            normalized["image_url"]["detail"] = detail
        return normalized

    if part_type in {"image", "input_image", "image_path", "image_ref"}:
        source = (
            part.get("image")
            or part.get("path")
            or part.get("url")
            or part.get("image_url")
            or part.get("ref")
        )
        _ensure_inline_image_registered(source, context)
        normalized = {"type": "image_url", "image_url": {"url": _image_source_to_model_url(source, context)}}
        if part.get("detail"):
            normalized["image_url"]["detail"] = part["detail"]
        return normalized

    return dict(part)


def _tool_reference_text() -> str:
    lines = ["Available tools and their full definitions:"]
    for item in tools.get_tool_definitions():
        function = item["function"]
        lines.append(f"- {function['name']}:")
        lines.append(json.dumps(function, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append('- finish: End the trajectory. Full definition:')
    lines.append(
        json.dumps(
            {
                "name": "finish",
                "description": "End the trajectory and provide the final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer text.",
                        }
                    },
                    "required": ["answer"],
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return "\n".join(lines)




def _build_manual_react_system_prompt(
    *,
    base_system_prompt: str,
    messages: list[dict[str, Any]],
    context: ToolRuntimeContext,
) -> str:
    # Dynamic state guidance is disabled for now so the prompt stays stable across turns.
    parts = [base_system_prompt.strip(), _tool_reference_text(), MANUAL_REACT_PROTOCOL.strip()]
    return "\n\n".join(part for part in parts if part).strip()


def _build_manual_react_request_messages(
    conversation_messages: list[dict[str, Any]],
    context: ToolRuntimeContext,
    base_system_prompt: str,
) -> list[dict[str, Any]]:
    system_seed = base_system_prompt
    for message in conversation_messages:
        if message.get("role") == "system" and message.get("content"):
            system_seed = str(message.get("content"))
            break
    full_system_prompt = _build_manual_react_system_prompt(
        base_system_prompt=system_seed,
        messages=conversation_messages,
        context=context,
    )
    request_messages: list[dict[str, Any]] = []
    system_replaced = False
    for message in conversation_messages:
        role = message.get("role")
        if role == "tool":
            tool_name = str(message.get("name") or "tool").strip()
            observation_text = str(message.get("content") or "")
            copied = {
                "role": "user",
                "content": f"Observation from {tool_name}:\n{observation_text}",
            }
        else:
            copied = dict(message)
        if copied.get("role") == "system" and not system_replaced:
            copied["content"] = full_system_prompt
            system_replaced = True
        request_messages.append(copied)
    if not system_replaced:
        request_messages.insert(0, {"role": "system", "content": full_system_prompt})
    return request_messages


def _apply_system_prompt_to_messages(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    updated_messages: list[dict[str, Any]] = []
    system_replaced = False
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "system" and not system_replaced:
            copied["content"] = system_prompt
            system_replaced = True
        updated_messages.append(copied)
    if not system_replaced:
        updated_messages.insert(0, {"role": "system", "content": system_prompt})
    return updated_messages


def _strip_code_fence(text: str) -> str:
    candidate = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return candidate


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _complete_trailing_action_block(text: str, finish_reason: str | None) -> str:
    """Close a trailing <action> block if generation stopped at the stop sequence."""

    stripped = text.rstrip()
    lower = stripped.lower()
    last_open = lower.rfind("<action>")
    last_close = lower.rfind("</action>")
    if last_open == -1 or last_close > last_open:
        return stripped
    if finish_reason != "stop":
        return stripped
    if stripped.endswith("}"):
        return f"{stripped}\n</action>"
    return f"{stripped}</action>"


def _parse_manual_react_step(text: str) -> ManualReActStep | None:
    stripped = text.strip()
    matches = list(_MANUAL_REACT_ACTION_RE.finditer(stripped))
    if not matches:
        return None
    match = matches[-1]
    thought = stripped[: match.start()].strip()
    action_payload = _extract_json_object(match.group("json"))
    if not isinstance(action_payload, dict):
        return None
    action = str(action_payload.get("tool_name") or "").strip()
    arguments = action_payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = action_payload.get("params")
    nested_goal = ""
    if isinstance(arguments, dict):
        nested_goal = str(arguments.get("goal") or "").strip()
    goal = nested_goal or str(action_payload.get("goal") or "").strip()
    if action not in _MANUAL_REACT_ACTIONS or not isinstance(arguments, dict):
        return None
    normalized_text = stripped[: match.end()].strip()
    return ManualReActStep(
        thought=thought,
        action=action,
        action_input=arguments,
        goal=goal,
        raw_text=normalized_text,
    )


def _normalize_message(message: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [_normalize_content_part(part, context) for part in content]
    elif isinstance(content, dict):
        normalized["content"] = [_normalize_content_part(content, context)]
    elif content is None:
        normalized["content"] = ""
    else:
        normalized["content"] = content
    return normalized


def _build_initial_messages(
    *,
    prompt: str | None,
    messages: list[dict[str, Any]] | None,
    context: ToolRuntimeContext,
    system_prompt: str | None,
    default_system_prompt: str,
) -> list[dict[str, Any]]:
    if prompt is None and messages is None:
        raise ValueError("Either prompt or messages must be provided.")
    if prompt is not None and messages is not None:
        raise ValueError("Provide either prompt or messages, not both.")

    effective_system_prompt = system_prompt or default_system_prompt

    if messages is None:
        initial_messages: list[dict[str, Any]] = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": prompt or ""},
        ]
        return initial_messages

    normalized_messages = [_normalize_message(message, context) for message in messages]
    has_system = any(message.get("role") == "system" for message in normalized_messages)
    if not has_system and effective_system_prompt:
        normalized_messages.insert(0, {"role": "system", "content": effective_system_prompt})
    return normalized_messages


def _load_pil_image(source: Any, context: ToolRuntimeContext) -> Image.Image:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, Image.Image):
        return payload.copy()
    if isinstance(payload, bytes):
        return Image.open(io.BytesIO(payload))
    if isinstance(payload, str):
        if payload.startswith("data:image"):
            return Image.open(io.BytesIO(_decode_data_url(payload)))
        if payload.startswith(("http://", "https://")):
            response = requests.get(payload, timeout=60)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        if os.path.exists(payload):
            return Image.open(payload)
    raise ValueError(f"Unsupported image source: {type(payload)!r}")


def _persist_pil_image(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> tuple[str, str]:
    image_id = context.next_image_id()
    filename = f"{context.filename_prefix}_{context.session_id}_{tool_name}_{image_id}.png"
    save_path = os.path.join(context.intermediate_dir, filename)
    _ensure_png_compatible_image(image).save(save_path, format="PNG")
    context.image_registry[image_id] = save_path
    return image_id, save_path


def _persist_pil_image_to_cache(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> tuple[str, str]:
    image_id = context.next_image_id()
    cache_dir = Path(__file__).resolve().parents[1] / ".image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{context.filename_prefix}_{context.session_id}_{tool_name}_{image_id}.png"
    save_path = str(cache_dir / filename)
    _ensure_png_compatible_image(image).save(save_path, format="PNG")
    context.image_registry[image_id] = save_path
    return image_id, save_path


def _try_upload_pil_image(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> str | None:
    image = _ensure_png_compatible_image(image)
    try:
        from opensearch_vl.opensearch_infer import cos_upload
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("COS uploader import failed: %s", exc)
        return None

    for attempt in range(1, _COS_UPLOAD_RETRIES + 2):
        try:
            uploaded_url = cos_upload.upload_pil_image(
                image,
                context.filename_prefix,
                0,
                0,
                tool_name,
            )
            if uploaded_url:
                return uploaded_url
            error = "uploader returned an empty URL"
        except Exception as exc:  # pragma: no cover - optional dependency
            error = str(exc)
        if attempt >= _COS_UPLOAD_RETRIES + 1:
            logger.warning(
                "COS upload failed for %s after %d attempt(s): %s",
                tool_name,
                attempt,
                error,
            )
            break
        tools._tool_retry_debug(
            "cos_upload",
            attempt=attempt,
            max_attempts=_COS_UPLOAD_RETRIES + 1,
            sleep_seconds=_COS_UPLOAD_RETRY_DELAY_S * attempt,
            error=error,
            tool_name=tool_name,
        )
        logger.warning(
            "COS upload failed for %s (attempt %d/%d); retrying in %ds: %s",
            tool_name,
            attempt,
            _COS_UPLOAD_RETRIES + 1,
            _COS_UPLOAD_RETRY_DELAY_S * attempt,
            error,
        )
        time.sleep(_COS_UPLOAD_RETRY_DELAY_S * attempt)
    return None


def _materialize_remote_image_url(source: Any, context: ToolRuntimeContext, tool_name: str) -> tuple[str | None, str | None]:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, str) and payload.startswith(("http://", "https://")):
        return payload, None

    try:
        image = _load_pil_image(payload, context)
    except Exception as exc:
        return None, f"Failed to load image for {tool_name}: {exc}"

    uploaded_url = _try_upload_pil_image(image, context, tool_name)
    if uploaded_url:
        return uploaded_url, None
    return None, (
        f"{tool_name} requires a publicly reachable image URL. "
        "Uploading the local image failed; configure the optional COS uploader first."
    )


def _assistant_message_for_followup(message: Any) -> dict[str, Any]:
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    assistant_message = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if getattr(message, "content", None):
        assistant_message["content"] = message.content
    return assistant_message


def _assistant_message_for_followup_from_dict(
    *,
    content: str,
    tool_calls: list[dict[str, Any]],
    # #### START Response 0720 ####
    phase: str | None = None,
    # #### END Response 0720 ####
) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    # #### START Response 0720 ####
    if phase:
        assistant_message["phase"] = phase
    # #### END Response 0720 ####
    if content:
        assistant_message["content"] = content
    return assistant_message


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_message_content(content: Any, *, include_images: bool = True) -> str:
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        formatted_parts: list[str] = []
        for index, part in enumerate(content, start=1):
            if not isinstance(part, dict):
                formatted_parts.append(f"[part {index}] {part}")
                continue
            part_type = str(part.get("type") or "")
            if part_type in {"text", "input_text"}:
                formatted_parts.append(str(part.get("text", "")))
            elif part_type == "image_url":
                if not include_images:
                    continue
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url
                formatted_parts.append(f"[image_url] {url}")
            elif part_type in {"image", "input_image", "image_path", "image_ref"}:
                if not include_images:
                    continue
                source = (
                    part.get("image")
                    or part.get("path")
                    or part.get("url")
                    or part.get("image_url")
                    or part.get("ref")
                    or ""
                )
                formatted_parts.append(f"[{part_type}] {source}")
            else:
                formatted_parts.append(_json_text(part))
        return "\n".join(item for item in formatted_parts if item)
    return _json_text(content) if isinstance(content, (dict, tuple)) else str(content)


def _print_conversation_trace(messages: list[dict[str, Any]]) -> None:
    print("\n=== Conversation Trace ===")
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        print(f"\n[{index}] {role}")
        content_text = _format_message_content(message.get("content"))
        if content_text:
            print(content_text)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            print("tool_calls:")
            print(_json_text(tool_calls))
        if role == "tool":
            tool_name = message.get("name")
            tool_call_id = message.get("tool_call_id")
            if tool_name:
                print(f"name: {tool_name}")
            if tool_call_id:
                print(f"tool_call_id: {tool_call_id}")


def _print_round_output(turn_index: int, assistant_message: Any) -> None:
    print(f"\n=== Model Round {turn_index + 1} ===")
    content = getattr(assistant_message, "content", None)
    if content:
        print(content)
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    if tool_calls:
        print("tool_calls:")
        for tool_call in tool_calls:
            print(
                _json_text(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            )


def _print_round_output_from_responses(
    turn_index: int,
    *,
    content: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    print(f"\n=== Model Round {turn_index + 1} ===")
    if content:
        print(content)
    if tool_calls:
        print("tool_calls:")
        for tool_call in tool_calls:
            print(_json_text(tool_call))


def _print_manual_react_round_io(
    turn_index: int,
    *,
    request_messages: list[dict[str, Any]],
    assistant_text: str,
) -> None:
    print(f"\n=== Manual ReAct Round {turn_index + 1} Input ===")
    for index, message in enumerate(request_messages, start=1):
        role = str(message.get("role") or "")
        if role == "system":
            continue
        print(f"\n[{index}] {role}")
        content_text = _format_message_content(message.get("content"), include_images=False)
        if content_text:
            print(content_text)
    print(f"\n=== Manual ReAct Round {turn_index + 1} Output ===")
    if assistant_text:
        print(assistant_text)


def _message_content_to_responses_content(
    content: Any,
    *,
    role: str = "user",
) -> list[dict[str, Any]]:
    """Convert local message content to role-valid Responses API content.

    The Responses API accepts ``input_text`` for user/developer input, but
    assistant history must be replayed as ``output_text``. This matters when a
    backend cannot continue with ``previous_response_id`` and the agent falls
    back to full conversation replay.
    """
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if isinstance(content, list):
        normalized_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                normalized_parts.append({"type": text_type, "text": str(part)})
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text", "output_text"}:
                normalized_parts.append({"type": text_type, "text": str(part.get("text", ""))})
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                    detail = image_url.get("detail")
                else:
                    url = image_url
                    detail = part.get("detail")
                item = {"type": "input_image", "image_url": url}
                if detail:
                    item["detail"] = detail
                normalized_parts.append(item)
            elif part_type == "input_image":
                normalized_parts.append(dict(part))
            else:
                normalized_parts.append(dict(part))
        return normalized_parts
    if content is None:
        return [{"type": text_type, "text": ""}]
    return [{"type": text_type, "text": str(content)}]


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "tool":
            continue
        items.append(
            # #### START Response 0720 ####
            {
                "role": role,
                "content": _message_content_to_responses_content(
                    message.get("content"), role=role
                ),
            }
            | ({"phase": message.get("phase")} if role == "assistant" and message.get("phase") else {})
            # #### END Response 0720 ####
        )
    return items


def _conversation_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the local conversation history to Responses API input items."""

    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role in {"system", "user", "assistant"}:
            content = message.get("content")
            has_textual_content = bool(content not in (None, "", []))
            if has_textual_content:
                    # #### START Response 0720 ####
                    input_item = {
                            "role": role,
                            "content": _message_content_to_responses_content(content, role=role),
                        }
                    if role == "assistant" and message.get("phase"):
                        input_item["phase"] = message.get("phase")
                    items.append(input_item)
                    # #### END Response 0720 ####
            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                }
            )
    return items


def _extract_responses_content_and_tool_calls(raw_response: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    output_items = raw_response.get("output") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for index, item in enumerate(output_items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content_item in item.get("content") or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if text:
                        text_parts.append(str(text))
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )

    return "\n".join(part for part in text_parts if part).strip(), tool_calls


# #### START Response 0720 ####
def _responses_reasoning_summaries(raw_response: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    for item in raw_response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for summary_item in item.get("summary") or []:
            if not isinstance(summary_item, dict):
                continue
            text = summary_item.get("text") or summary_item.get("summary_text")
            if text:
                summaries.append(str(text).strip())
    return [item for item in summaries if item]


def _responses_turn_trace(
    *,
    turn_index: int,
    raw_response: dict[str, Any],
    assistant_content: str,
    assistant_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    output_items = raw_response.get("output") or []
    return {
        "turn_index": turn_index + 1,
        "response_id": raw_response.get("id"),
        "status": raw_response.get("status"),
        "incomplete_details": raw_response.get("incomplete_details"),
        "assistant_phase": next(
            (
                item.get("phase")
                for item in output_items
                if isinstance(item, dict) and item.get("type") == "message" and item.get("phase")
            ),
            None,
        ),
        "visible_text": assistant_content,
        "visible_text_chars": len(assistant_content),
        "visible_rationale_present": bool(assistant_content.strip()) and bool(assistant_tool_calls),
        "reasoning_summaries": _responses_reasoning_summaries(raw_response),
        "function_calls": deepcopy(assistant_tool_calls),
        "output_item_types": [
            str(item.get("type") or "") for item in output_items if isinstance(item, dict)
        ],
    }


def _responses_rationale_summary(turn_traces: list[dict[str, Any]]) -> dict[str, Any]:
    tool_turns = [item for item in turn_traces if item.get("function_calls")]
    with_visible = [item for item in tool_turns if item.get("visible_rationale_present")]
    total_chars = sum(int(item.get("visible_text_chars") or 0) for item in with_visible)
    return {
        "tool_call_turn_count": len(tool_turns),
        "tool_call_turns_with_visible_rationale": len(with_visible),
        "tool_call_turns_without_visible_rationale": len(tool_turns) - len(with_visible),
        "visible_rationale_coverage": (
            len(with_visible) / len(tool_turns) if tool_turns else None
        ),
        "average_visible_rationale_chars": (
            total_chars / len(with_visible) if with_visible else 0.0
        ),
        "reasoning_summary_turn_count": sum(
            1 for item in turn_traces if item.get("reasoning_summaries")
        ),
    }


def _responses_reasoning_payload(config: OpenAIToolAgentConfig) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if config.responses_reasoning_effort:
        payload["effort"] = config.responses_reasoning_effort
    if config.responses_reasoning_summary:
        payload["summary"] = config.responses_reasoning_summary
    if config.responses_reasoning_mode:
        payload["mode"] = config.responses_reasoning_mode
    if config.responses_reasoning_context:
        payload["context"] = config.responses_reasoning_context
    return payload or None
# #### END Response 0720 ####


def _is_previous_response_not_found_error(exc: Exception) -> bool:
    message = str(exc)
    return "previous_response_not_found" in message or "Previous response with id" in message


def _read_url_page_id(
    *,
    resource_id: str,
    resource: tools.UrlResource | None,
    url: str,
    context: ToolRuntimeContext,
) -> str:
    """Return a stable opaque page ID, registering direct legacy URLs privately."""

    page_id = resource_id or (resource.resource_id if resource is not None else "")
    if not page_id:
        page_id = tools._resource_id("page", url)
        context.register_url_resource(
            tools.UrlResource(primary_url=url, resource_id=page_id, kind="page")
        )
    return page_id


def _public_read_url_text_observation(
    *,
    output: dict[str, Any],
    page_id: str,
    goal: str,
) -> dict[str, str]:
    """Expose text-page reads without URL/backend metadata to the model.

    The underlying reader result retains provenance for runtime diagnostics, but
    tool observations must not reveal raw URLs or Firecrawl metadata.  Keep the
    public schema identical for Enhanced Reader, Firecrawl, PDFs, and failures.
    """

    if output.get("ok"):
        content = str(output.get("content") or "")
        title = str(output.get("title") or "")
    else:
        # Reader errors commonly interpolate the requested URL (and sometimes
        # redirect URLs).  Do not forward any such diagnostic text to the model.
        content = "Unable to read the requested page."
        title = ""
    return {
        "page_id": page_id,
        "title": title,
        "goal": goal,
        "content": content,
    }


def _execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    context: ToolRuntimeContext,
    question_text: str = "",
    assistant_text: str = "",
    tool_goal: str = "",
) -> ToolExecutionResult:
    """Execute one tool call against the runtime context."""

    params = tools.normalize_tool_arguments(name, arguments)

    if name == "t2t_search":
        query = params.get("query") or params.get("q") or ""
        if not query:
            output = {"ok": False, "error": "query is required for t2t_search"}
        else:
            output = tools.t2t_search(
                query=query,
                lang=params.get("lang") or params.get("hl") or "en",
                top_k=int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
            )
        if isinstance(output, dict) and output.get("ok"):
            output = context.postprocess_search_output("t2t_search", output)
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    if name == "t2i_search":
        query = params.get("query") or params.get("q") or ""
        if not query:
            output = {"ok": False, "error": "query is required for t2i_search"}
        else:
            output = tools.t2i_search(
                query=query,
                lang=params.get("lang") or params.get("hl") or "en",
                top_k=int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
            )
        if isinstance(output, dict) and output.get("ok"):
            output = context.postprocess_search_output("t2i_search", output)
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    if name == "read_url":
        resource_id = str(params.get("resource_id") or "").strip()
        direct_url = str(params.get("url") or params.get("URL") or "").strip()
        resource = context.resolve_resource_id(resource_id) if resource_id else None
        effective_goal = str(params.get("goal") or tool_goal or "").strip()
        if resource_id and resource is None and not direct_url:
            failure_output = {
                "ok": False,
                "error": f"resource_id not found: {resource_id}",
            }
            _record_tool_failure(
                tool_name=name,
                arguments=params,
                error_output=failure_output,
            )
            output = _public_read_url_text_observation(
                output=failure_output,
                page_id=resource_id,
                goal=effective_goal,
            )
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        url = resource.primary_url if resource is not None else direct_url
        if not url:
            failure_output = {"ok": False, "error": "URL is required."}
            _record_tool_failure(
                tool_name=name,
                arguments=params,
                error_output=failure_output,
            )
            output = _public_read_url_text_observation(
                output=failure_output,
                page_id=resource_id,
                goal=effective_goal,
            )
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        print(
            "[read_url debug] "
            f"resource_id={resource_id or '-'} "
            f"url={url}",
            file=sys.stderr,
            flush=True,
        )
        output = tools.read_url(
            url=url,
            goal=effective_goal,
            assistant_output=assistant_text,
            resource=resource or context.resolve_url_resource(url),
        )
        if isinstance(output, dict) and output.get("ok") is False:
            _record_tool_failure(
                tool_name=name,
                arguments=params,
                error_output=output,
            )
        new_images: dict[str, Any] = {}
        if output.get("ok") and output.get("local_path"):
            image_id = context.register_image(output["local_path"])
            new_images[image_id] = output["local_path"]
            output = {"image_id": image_id}
        else:
            page_id = _read_url_page_id(
                resource_id=resource_id,
                resource=resource,
                url=url,
                context=context,
            )
            output = _public_read_url_text_observation(
                output=output,
                page_id=page_id,
                goal=effective_goal,
            )
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)

    if name == "i2i_search":
        image_source = params.get("image") or params.get("url") or context.latest_image_reference() or ""
        if not image_source:
            output = {
                "ok": False,
                "error": "No image is available in the current context for i2i_search.",
            }
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

        region = params.get("region")
        new_images: dict[str, Any] = {}
        if region not in (None, ""):
            image = _load_pil_image(image_source, context)
            # #### START Response 0720 ####
            region = _prepare_region_for_crop(
                region,
                image.size,
                coordinate_order=str((context.metadata or {}).get("bbox_coordinate_order") or ""),
            )
            # #### END Response 0720 ####
            bbox, err = _normalize_region_bbox(region)
            if err:
                output = {"ok": False, "error": err}
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
            assert bbox is not None
            x, y, width, height = bbox
            cropped = image.crop((x, y, x + width, y + height))
            # The COS uploader derives its object key from ``tool_name``.  Keep
            # each crop immutable so a previous Lens request never observes a
            # later crop written to the same public URL.
            uploaded_url = _try_upload_pil_image(
                cropped,
                context,
                f"i2i_region_{uuid.uuid4().hex}",
            )
            if not uploaded_url:
                cropped_id, cropped_path = _persist_pil_image_to_cache(cropped, context, "i2i_region")
                new_images[cropped_id] = cropped_path
                output = {
                    "ok": False,
                    "error": (
                        "Cropped region was created, but reverse image search needs a public URL. "
                        "Uploading the cropped image failed, so it was saved to the local image cache instead."
                    ),
                    "cropped_image_id": cropped_id,
                    "cropped_image_path": cropped_path,
                }
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)
            context.register_image(uploaded_url)
            output = tools.i2i_search(
                image_url=uploaded_url,
                visual_lookup=context.visual_lookup,
                top_k=int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
            )
            output = dict(output)
            output["cropped_image_url"] = uploaded_url
            if output.get("ok"):
                output = context.postprocess_search_output("i2i_search", output)
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)

        remote_url, err = _materialize_remote_image_url(image_source, context, "i2i_search")
        if err:
            output = {"ok": False, "error": err}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        output = tools.i2i_search(
            image_url=remote_url or "",
            visual_lookup=context.visual_lookup,
            top_k=int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
        )
        if isinstance(output, dict) and output.get("ok"):
            output = context.postprocess_search_output("i2i_search", output)
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    output = {"ok": False, "error": f"Unknown tool: {name}"}
    return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))


def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    context: ToolRuntimeContext,
    question_text: str = "",
    assistant_text: str = "",
    tool_goal: str = "",
) -> ToolExecutionResult:
    """Execute one tool and record failures before returning or re-raising."""
    try:
        result = _timed_call(
            f"tool:{name}",
            lambda: _execute_tool_call(
                name=name,
                arguments=arguments,
                context=context,
                question_text=question_text,
                assistant_text=assistant_text,
                tool_goal=tool_goal,
            ),
        )
    except Exception as exc:
        _record_tool_failure(
            tool_name=name,
            arguments=arguments,
            error_output={
                "ok": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            },
        )
        raise

    if isinstance(result.output, dict) and result.output.get("ok") is False:
        _record_tool_failure(
            tool_name=name,
            arguments=result.arguments,
            error_output=result.output,
        )
    return result


class OpenAIToolAgent:
    """OpenAI/AzureOpenAI-based multi-turn chat-completions agent with tool calling."""

    def __init__(self, config: OpenAIToolAgentConfig) -> None:
        self.config = config
        self._worker_model_alias = None
        self.client = None
        if config.api_mode == "manual_react":
            resolved_alias = _resolve_registered_model_alias(config.model)
            if resolved_alias is None:
                raise ValueError(
                    "manual_react requires `config.model` to be a registered LLM_WORKER model alias. "
                    f"Got: {config.model!r}"
                )
            self._worker_model_alias = config.model
            return

        # #### START Response 0720 ####
        if config.api_mode == "responses":
            # Responses API calls are routed through LLM_WORKER.responses_generate
            # so model aliases, cache/session headers, retries, and endpoint
            # configuration stay centralized in synthesis/model_worker.py.
            return
        # #### END Response 0720 ####

        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenAIToolAgent requires the `openai` package."
            ) from exc

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        client_type = str(config.client_type or "azure_openai").strip().lower()
        if client_type == "openai":
            base_url = (
                config.base_url
                or os.environ.get("SFT_OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=config.timeout_s,
                default_headers=config.default_headers,
            )
        else:
            azure_endpoint = (
                config.azure_endpoint
                or config.base_url
                or os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
                or os.environ.get("SFT_OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=config.api_version,
                timeout=config.timeout_s,
                default_headers=config.default_headers,
            )

    def _call_worker_chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        trace_label: str,
        stop: list[str] | None = None,
    ) -> tuple[ModelRequest, Any]:
        if self._worker_model_alias is None:
            raise RuntimeError("LLM_WORKER chat-completions call requires a registered model alias.")
        request = ModelRequest(
            model=self._worker_model_alias,
            messages=[
                ModelMessage(
                    role=str(message.get("role") or "user"),
                    content=message.get("content"),
                )
                for message in messages
                if isinstance(message, dict)
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            metadata=_sft_worker_metadata(
                trace_label,
                stop=stop,
                extra_body=self.config.extra_body,
            ),
        )
        response = _timed_call(
            "llm:chat_worker",
            lambda: LLM_WORKER.generate(request),
        )
        return request, response

    def run(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        timing = _TrajectoryTiming()
        timing_token = _ACTIVE_TRAJECTORY_TIMING.set(timing)
        started_at = time.perf_counter()
        try:
            if self.config.api_mode == "manual_react":
                return self._run_manual_react(
                    prompt=prompt,
                    messages=messages,
                    context=context,
                    system_prompt=system_prompt,
                )
            if self.config.api_mode == "responses":
                return self._run_responses(
                    prompt=prompt,
                    messages=messages,
                    context=context,
                    system_prompt=system_prompt,
                )
            return self._run_chat_completions(
                prompt=prompt,
                messages=messages,
                context=context,
                system_prompt=system_prompt,
            )
        finally:
            elapsed_s = time.perf_counter() - started_at
            _ACTIVE_TRAJECTORY_TIMING.reset(timing_token)
            _finish_trajectory_timing(timing, elapsed_s)

    def _run_manual_react(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        context = context or ToolRuntimeContext(working_dir=os.getcwd())
        conversation_messages = _build_initial_messages(
            prompt=prompt,
            messages=messages,
            context=context,
            system_prompt=system_prompt,
            default_system_prompt=self.config.system_prompt,
        )
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []
        final_text = ""
        generation_status = "unknown"
        stop_reason = "unknown"
        print(
            f"[manual_react backend] llm_worker alias={self._worker_model_alias}",
            file=sys.stderr,
            flush=True,
        )

        for turn_index in range(self.config.max_turns):
            request_messages = _build_manual_react_request_messages(
                conversation_messages,
                context,
                system_prompt or self.config.system_prompt,
            )
            _, worker_response = self._call_worker_chat_completions(
                messages=request_messages,
                trace_label=f"manual_react_turn_{turn_index + 1}",
                stop=["</action>"],
            )
            raw_response = worker_response.raw_response or {"content": worker_response.content}
            raw_responses.append(raw_response)
            finish_reason = worker_response.metadata.get("finish_reason")
            assistant_text = _complete_trailing_action_block(
                worker_response.content or "",
                finish_reason,
            )
            if self.config.print_rounds:
                _print_manual_react_round_io(
                    turn_index,
                    request_messages=request_messages,
                    assistant_text=assistant_text,
                )

            step = _parse_manual_react_step(assistant_text)
            if step is None:
                conversation_messages.append({"role": "assistant", "content": assistant_text})
                logger.warning("Failed to parse manual ReAct step; treating the latest assistant text as final output.")
                final_text = assistant_text
                generation_status = "parse_error_finalized"
                stop_reason = "manual_react_parse_error"
                break
            repaired_step_text = step.raw_text
            execution_action_input = step.action_input
            if step.action == "i2i_search":
                repaired = _maybe_repair_i2i_tool_call(
                    assistant_text=step.thought or step.raw_text,
                    tool_name=step.action,
                    tool_arguments=step.action_input,
                    context=context,
                    question_text=_latest_user_text(conversation_messages),
                )
                if repaired is not None:
                    repaired_step_text = _render_manual_react_text(
                        thought=repaired.assistant_text,
                        action=step.action,
                        arguments=repaired.display_arguments,
                        goal=_parse_manual_react_goal(step.raw_text),
                    )
                    execution_action_input = repaired.execution_arguments
            conversation_messages.append({"role": "assistant", "content": repaired_step_text})
            if step.action == "finish":
                final_text = str(step.action_input.get("answer") or step.raw_text).strip()
                generation_status = "finished"
                stop_reason = "finish_action"
                break

            result = execute_tool_call(
                step.action,
                execution_action_input,
                context,
                question_text=_latest_user_text(conversation_messages),
                assistant_text=_strip_action_blocks(repaired_step_text),
                tool_goal=step.goal,
            )
            tool_results.append(result)
            conversation_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"manual_{turn_index + 1}_{step.action}",
                    "name": step.action,
                    "content": result.output_text,
                    "type": "manual_react_tool",
                }
            )
            if attachment := _read_url_image_attachment(result, context):
                conversation_messages.append(attachment)
        else:
            final_text = "Max ReAct turns reached before the model produced a final answer."
            generation_status = "max_turns_reached"
            stop_reason = "max_react_turns"

        if request_messages:
            effective_system_prompt = str(request_messages[0].get("content") or "")
            conversation_messages = _apply_system_prompt_to_messages(
                conversation_messages,
                effective_system_prompt,
            )

        return AgentRunResult(
            final_text=final_text,
            messages=conversation_messages,
            tool_results=tool_results,
            raw_responses=raw_responses,
            metadata={
                "api_mode": "manual_react",
                "generation_status": generation_status,
                "generation_complete": generation_status == "finished",
                "stop_reason": stop_reason,
                "max_turns": self.config.max_turns,
                "turn_count": len(raw_responses),
                "tool_call_count": len(tool_results),
            },
        )

    def _run_chat_completions(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        context = context or ToolRuntimeContext(working_dir=os.getcwd())
        conversation_messages = _build_initial_messages(
            prompt=prompt,
            messages=messages,
            context=context,
            system_prompt=system_prompt,
            default_system_prompt=self.config.system_prompt,
        )
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []
        final_text = ""
        generation_status = "unknown"
        stop_reason = "unknown"

        for turn_index in range(self.config.max_turns):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": conversation_messages,
                "tools": tools.get_tool_definitions(),
                "max_tokens": self.config.max_tokens,
                "stream": False,
                # #### START Response 0720 ####
                "extra_headers": _sft_request_extra_headers(),
                # #### END Response 0720 ####
            }
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature
            if self.config.extra_body:
                kwargs["extra_body"] = self.config.extra_body
            completion = _timed_call(
                "llm:chat_completions",
                lambda: self.client.chat.completions.create(**kwargs),
            )
            raw_responses.append(
                completion.model_dump() if hasattr(completion, "model_dump") else {"repr": repr(completion)}
            )
            choice = completion.choices[0]
            assistant_message = choice.message
            if self.config.print_rounds:
                _print_round_output(turn_index, assistant_message)
            tool_calls = _truncate_tool_calls(list(assistant_message.tool_calls or []), source="chat_completions")

            if not tool_calls:
                final_text = assistant_message.content or ""
                conversation_messages.append({"role": "assistant", "content": final_text})
                generation_status = "finished"
                stop_reason = "no_tool_calls"
                break
            assistant_content = assistant_message.content or ""
            followup_tool_calls: list[dict[str, Any]] = []
            execution_payloads: list[tuple[str, dict[str, Any], str]] = []
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                parsed_args = json.loads(tool_call.function.arguments or "{}")
                tool_call_id = tool_call.id
                display_args = parsed_args
                execution_args = parsed_args
                if tool_name == "i2i_search":
                    repaired = _maybe_repair_i2i_tool_call(
                        assistant_text=assistant_content,
                        tool_name=tool_name,
                        tool_arguments=parsed_args,
                        context=context,
                        question_text=_latest_user_text(conversation_messages),
                    )
                    if repaired is not None:
                        assistant_content = repaired.assistant_text
                        display_args = repaired.display_arguments
                        execution_args = repaired.execution_arguments
                followup_tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(display_args, ensure_ascii=False),
                        },
                    }
                )
                execution_payloads.append((tool_name, execution_args, tool_call_id))
            conversation_messages.append(
                _assistant_message_for_followup_from_dict(
                    content=assistant_content,
                    tool_calls=followup_tool_calls,
                )
            )
            for tool_name, execution_args, tool_call_id in execution_payloads:
                result = execute_tool_call(
                    tool_name,
                    execution_args,
                    context,
                    question_text=_latest_user_text(conversation_messages),
                    assistant_text=assistant_content,
                )
                tool_results.append(result)
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result.output_text,
                        "type": "function",
                    }
                )
                if attachment := _read_url_image_attachment(result, context):
                    conversation_messages.append(attachment)
        else:
            final_text = "Max tool-calling turns reached before the model produced a final answer."
            generation_status = "max_turns_reached"
            stop_reason = "max_tool_calling_turns"

        return AgentRunResult(
            final_text=final_text,
            messages=conversation_messages,
            tool_results=tool_results,
            raw_responses=raw_responses,
            metadata={
                "api_mode": "chat_completions",
                "generation_status": generation_status,
                "generation_complete": generation_status == "finished",
                "stop_reason": stop_reason,
                "max_turns": self.config.max_turns,
                "turn_count": len(raw_responses),
                "tool_call_count": len(tool_results),
            },
        )

    def _run_responses(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        context = context or ToolRuntimeContext(working_dir=os.getcwd())
        conversation_messages = _build_initial_messages(
            prompt=prompt,
            messages=messages,
            context=context,
            system_prompt=system_prompt,
            default_system_prompt=self.config.system_prompt,
        )
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []
        final_text = ""
        generation_status = "unknown"
        stop_reason = "unknown"
        # #### START Response 0720 ####
        responses_turn_traces: list[dict[str, Any]] = []
        base_system_prompt = system_prompt or self.config.system_prompt
        uses_default_system_prompt = base_system_prompt.strip() == DEFAULT_SYSTEM_PROMPT.strip()
        public_reasoning_enabled = bool(self.config.responses_prompt_public_reasoning)
        uses_responses_system_prompt = public_reasoning_enabled and uses_default_system_prompt
        print(
            "[responses-prompt-debug] "
            "api_mode=responses "
            f"responses_prompt_public_reasoning={public_reasoning_enabled} "
            f"uses_default_system_prompt={uses_default_system_prompt} "
            f"uses_responses_system_prompt={uses_responses_system_prompt} "
            f"instructions_source={('responses_system_prompt' if public_reasoning_enabled and uses_default_system_prompt else 'base_system_prompt_only')}",
            file=sys.stderr,
            flush=True,
        )
        responses_instructions = (
            _build_responses_instructions(base_system_prompt)
            if public_reasoning_enabled
            else base_system_prompt
        )
        # Responses receives system behavior through instructions. Remove the
        # system message from input to avoid duplicating the base prompt.
        response_input_messages = [
            message for message in conversation_messages if message.get("role") != "system"
        ]
        # #### END Response 0720 ####
        current_input = _messages_to_responses_input(response_input_messages)
        previous_response_id: str | None = None
        use_previous_response_id = True

        for turn_index in range(self.config.max_turns):
            reasoning_payload = _responses_reasoning_payload(self.config)
            previous_id_for_request = previous_response_id if use_previous_response_id else None

            try:
                worker_response = _timed_call(
                    "llm:responses",
                    lambda: LLM_WORKER.responses_generate(
                        ResponsesModelRequest(
                            model=self.config.model,
                            input=current_input,
                            tools=tools.get_responses_tool_definitions(),
                            instructions=responses_instructions,
                            previous_response_id=previous_id_for_request,
                            max_output_tokens=self.config.max_tokens,
                            reasoning=reasoning_payload,
                            parallel_tool_calls=self.config.responses_parallel_tool_calls,
                            store=self.config.responses_store,
                            temperature=self.config.temperature,
                            metadata=_sft_worker_metadata(
                                f"responses_turn_{turn_index + 1}",
                                extra_body=self.config.extra_body,
                            ),
                        )
                    ),
                )
                raw_response = worker_response.raw_response
            except Exception as exc:
                if previous_response_id and use_previous_response_id and _is_previous_response_not_found_error(exc):
                    logger.warning(
                        "Responses API previous_response_id is unavailable on this backend; "
                        "falling back to full-context replay."
                    )
                    use_previous_response_id = False
                    current_input = _conversation_messages_to_responses_input(conversation_messages)
                    worker_response = _timed_call(
                        "llm:responses",
                        lambda: LLM_WORKER.responses_generate(
                            ResponsesModelRequest(
                                model=self.config.model,
                                input=current_input,
                                tools=tools.get_responses_tool_definitions(),
                                instructions=responses_instructions,
                                max_output_tokens=self.config.max_tokens,
                                reasoning=reasoning_payload,
                                parallel_tool_calls=self.config.responses_parallel_tool_calls,
                                store=self.config.responses_store,
                                temperature=self.config.temperature,
                                metadata=_sft_worker_metadata(
                                    f"responses_turn_{turn_index + 1}:full_replay",
                                    extra_body=self.config.extra_body,
                                ),
                            )
                        ),
                    )
                    raw_response = worker_response.raw_response
                else:
                    raise
            raw_responses.append(raw_response)
            if use_previous_response_id:
                previous_response_id = raw_response.get("id")
            else:
                previous_response_id = None

            assistant_content, assistant_tool_calls = _extract_responses_content_and_tool_calls(raw_response)
            assistant_tool_calls = _truncate_tool_calls(assistant_tool_calls, source="responses")
            # #### START Response 0720 ####
            responses_turn_traces.append(
                _responses_turn_trace(
                    turn_index=turn_index,
                    raw_response=raw_response,
                    assistant_content=assistant_content,
                    assistant_tool_calls=assistant_tool_calls,
                )
            )
            # #### END Response 0720 ####
            if self.config.print_rounds:
                _print_round_output_from_responses(
                    turn_index,
                    content=assistant_content,
                    tool_calls=assistant_tool_calls,
                )

            if not assistant_tool_calls:
                final_text = assistant_content
                # #### START Response 0720 ####
                conversation_messages.append({"role": "assistant", "content": final_text, "phase": "final_answer"})
                # #### END Response 0720 ####
                generation_status = "finished"
                stop_reason = "no_tool_calls"
                break

            followup_tool_calls: list[dict[str, Any]] = []
            execution_payloads: list[tuple[str, dict[str, Any], str]] = []
            current_input = []
            for tool_call in assistant_tool_calls:
                parsed_args = json.loads(tool_call["function"]["arguments"] or "{}")
                display_args = parsed_args
                execution_args = parsed_args
                if (
                    tool_call["function"]["name"] == "i2i_search"
                    and self.config.responses_i2i_wrapper_enabled
                ):
                    # #### START Response 0720 ####
                    repaired = _maybe_repair_i2i_tool_call(
                        assistant_text=assistant_content,
                        tool_name=tool_call["function"]["name"],
                        tool_arguments=parsed_args,
                        context=context,
                        question_text=_latest_user_text(conversation_messages),
                    )
                    if repaired is not None:
                        assistant_content = repaired.assistant_text
                        display_args = repaired.display_arguments
                        execution_args = repaired.execution_arguments
                    # #### END Response 0720 ####
                followup_tool_calls.append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(display_args, ensure_ascii=False),
                        },
                    }
                )
                execution_payloads.append((tool_call["function"]["name"], execution_args, tool_call["id"]))
            conversation_messages.append(
                _assistant_message_for_followup_from_dict(
                    content=assistant_content,
                    tool_calls=followup_tool_calls,
                    # #### START Response 0720 ####
                    phase="commentary",
                    # #### END Response 0720 ####
                )
            )
            for tool_name, execution_args, tool_call_id in execution_payloads:
                result = execute_tool_call(
                    tool_name,
                    execution_args,
                    context,
                    question_text=_latest_user_text(conversation_messages),
                    assistant_text=assistant_content,
                )
                tool_results.append(result)
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result.output_text,
                        "type": "function",
                    }
                )
                current_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": result.output_text,
                    }
                )
                if attachment := _read_url_image_attachment(result, context):
                    conversation_messages.append(attachment)
                    current_input.extend(_conversation_messages_to_responses_input([attachment]))
            if not use_previous_response_id:
                current_input = _conversation_messages_to_responses_input(conversation_messages)
        else:
            final_text = "Max tool-calling turns reached before the model produced a final answer."
            generation_status = "max_turns_reached"
            stop_reason = "max_tool_calling_turns"

        return AgentRunResult(
            final_text=final_text,
            messages=conversation_messages,
            tool_results=tool_results,
            raw_responses=raw_responses,
            metadata={
                "api_mode": "responses",
                "generation_status": generation_status,
                "generation_complete": generation_status == "finished",
                "stop_reason": stop_reason,
                "max_turns": self.config.max_turns,
                "turn_count": len(raw_responses),
                "tool_call_count": len(tool_results),
                # #### START Response 0720 ####
                "responses_public_reasoning_prompted": self.config.responses_prompt_public_reasoning,
                "responses_i2i_wrapper_enabled": self.config.responses_i2i_wrapper_enabled,
                "responses_turn_traces": responses_turn_traces,
                "responses_rationale_summary": _responses_rationale_summary(responses_turn_traces),
                "responses_raw_response_count": len(raw_responses),
                # #### END Response 0720 ####
            },
        )


def _parse_json_flag(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _parse_messages_json(value: str | None) -> list[dict[str, Any]] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of messages.")
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a JSON object.")
    return parsed


def _build_context_from_args(args: argparse.Namespace) -> ToolRuntimeContext:
    context = ToolRuntimeContext(
        working_dir=args.workdir,
        filename_prefix=args.filename_prefix,
        case_id=args.case_id,
    )
    for path in args.image or []:
        image_id = context.register_image(os.path.abspath(path))
        logger.info("Registered local image %s -> %s", image_id, path)
    for url in args.image_url or []:
        image_id = context.register_image(url)
        logger.info("Registered remote image %s -> %s", image_id, url)
    return context


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-based synthesis SFT tool-calling framework.")
    parser.add_argument("--prompt", help="Simple user prompt sent to the model.")
    parser.add_argument("--messages-json", help="Full messages list as a JSON array.")
    parser.add_argument("--messages-file", help="Path to a JSON file containing a full messages list.")
    # #### START Response 0720 ####
    parser.add_argument(
        "--model-alias",
        "--model",
        dest="model_alias",
        default=os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "",
        help="Registered synthesis/models.json alias for the answer model.",
    )
    # #### END Response 0720 ####
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    # #### START Response 0720 ####
    parser.add_argument(
        "--client-type",
        choices=("azure_openai", "openai"),
        default=os.environ.get("SFT_OPENAI_CLIENT_TYPE") or "azure_openai",
        help="OpenAI SDK client type. Use openai for standard/base_url Responses endpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SFT_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        help="Optional base URL for --client-type openai.",
    )
    # #### END Response 0720 ####
    parser.add_argument(
        "--api-mode",
        choices=("manual_react", "chat_completions", "responses"),
        default=os.environ.get("SFT_OPENAI_API_MODE") or "manual_react",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=(
            os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
    )
    parser.add_argument(
        "--api-version",
        default=os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SFT_OPENAI_MAX_TOKENS", "1024")))
    parser.add_argument(
        "--temperature",
        type=float,
        default=(float(os.environ["SFT_OPENAI_TEMPERATURE"]) if os.environ.get("SFT_OPENAI_TEMPERATURE") else None),
    )
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("SFT_OPENAI_MAX_TURNS", "8")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("SFT_OPENAI_TIMEOUT_S", "120")))
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--headers-json", default=os.environ.get("SFT_OPENAI_HEADERS_JSON"))
    parser.add_argument("--extra-body-json", default=os.environ.get("SFT_OPENAI_EXTRA_BODY_JSON"))
    # #### START Response 0720 ####
    parser.add_argument("--responses-reasoning-effort", default=os.environ.get("SFT_RESPONSES_REASONING_EFFORT"))
    parser.add_argument("--responses-reasoning-summary", default=os.environ.get("SFT_RESPONSES_REASONING_SUMMARY", "auto"))
    parser.add_argument("--responses-reasoning-mode", default=os.environ.get("SFT_RESPONSES_REASONING_MODE"))
    parser.add_argument("--responses-reasoning-context", default=os.environ.get("SFT_RESPONSES_REASONING_CONTEXT", "all_turns"))
    parser.add_argument("--responses-store", choices=("true", "false"), default=os.environ.get("SFT_RESPONSES_STORE"))
    parser.add_argument("--no-responses-public-reasoning", action="store_true", help="Do not append the Responses public-reasoning prompt.")
    parser.add_argument("--responses-parallel-tool-calls", action="store_true", help="Allow parallel Responses tool calls. Defaults to false.")
    parser.add_argument("--responses-i2i-wrapper", action="store_true", help="Enable the legacy i2i wrapper rewrite in Responses mode.")
    # #### END Response 0720 ####
    parser.add_argument("--workdir", default=os.path.join(os.getcwd(), "synthesis_sft_runs"))
    parser.add_argument("--filename-prefix", default="sft")
    parser.add_argument("--case-id", default="sft_session")
    parser.add_argument("--image", action="append", help="Preload a local image path as img_n.")
    parser.add_argument("--image-url", action="append", help="Preload a remote image URL as img_n.")
    parser.add_argument("--gpt54", action="store_true", help="Use the GPT-5.4 manual-ReAct branch from .sft_env.")
    parser.add_argument(
        "--gemini35-flash",
        action="store_true",
        help="Use the Gemini 3.5 Flash manual-ReAct branch from .sft_env.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.model_alias:
        parser.error("--model-alias is required (or set SFT_OPENAI_MODEL / OPENAI_MODEL).")
    if not any([args.prompt, args.messages_json, args.messages_file]):
        parser.error("One of --prompt, --messages-json, or --messages-file is required.")
    if sum(1 for item in [args.prompt, args.messages_json, args.messages_file] if item) > 1:
        parser.error("Use only one of --prompt, --messages-json, or --messages-file.")
    if args.gpt54 and args.gemini35_flash:
        parser.error("Use only one model shortcut: --gpt54 or --gemini35-flash.")

    if args.gpt54:
        args.api_mode = "manual_react"
        # #### START Response 0720 ####
        args.client_type = os.environ.get("SFT_GPT54_CLIENT_TYPE") or args.client_type
        args.base_url = os.environ.get("SFT_GPT54_BASE_URL") or args.base_url
        # #### END Response 0720 ####
        args.model_alias = os.environ.get("SFT_GPT54_MODEL") or "gpt-5.4-2026-03-05"
        args.api_key = os.environ.get("SFT_GPT54_API_KEY") or args.api_key
        args.azure_endpoint = os.environ.get("SFT_GPT54_AZURE_ENDPOINT") or args.azure_endpoint
        args.api_version = os.environ.get("SFT_GPT54_API_VERSION") or args.api_version
    elif args.gemini35_flash:
        args.api_mode = "manual_react"
        # #### START Response 0720 ####
        args.client_type = os.environ.get("SFT_GEMINI35_FLASH_CLIENT_TYPE") or args.client_type
        args.base_url = os.environ.get("SFT_GEMINI35_FLASH_BASE_URL") or args.base_url
        # #### END Response 0720 ####
        args.model_alias = os.environ.get("SFT_GEMINI35_FLASH_MODEL") or "gemini-3.5-flash"
        args.api_key = os.environ.get("SFT_GEMINI35_FLASH_API_KEY") or args.api_key
        args.azure_endpoint = os.environ.get("SFT_GEMINI35_FLASH_AZURE_ENDPOINT") or args.azure_endpoint
        args.api_version = os.environ.get("SFT_GEMINI35_FLASH_API_VERSION") or args.api_version

    config = OpenAIToolAgentConfig(
        model=args.model_alias,
        api_key=args.api_key,
        # #### START Response 0720 ####
        client_type=args.client_type,
        base_url=args.base_url,
        # #### END Response 0720 ####
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        api_mode=args.api_mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        system_prompt=args.system_prompt,
        default_headers=_parse_json_flag(args.headers_json),
        extra_body=_parse_json_flag(args.extra_body_json),
        max_turns=args.max_turns,
        # #### START Response 0720 ####
        responses_reasoning_effort=args.responses_reasoning_effort,
        responses_reasoning_summary=args.responses_reasoning_summary,
        responses_reasoning_mode=args.responses_reasoning_mode,
        responses_reasoning_context=args.responses_reasoning_context,
        responses_store=(None if args.responses_store is None else args.responses_store == "true"),
        responses_prompt_public_reasoning=not args.no_responses_public_reasoning,
        responses_parallel_tool_calls=args.responses_parallel_tool_calls,
        responses_i2i_wrapper_enabled=args.responses_i2i_wrapper,
        # #### END Response 0720 ####
    )
    context = _build_context_from_args(args)
    agent = OpenAIToolAgent(config)
    input_messages = _parse_messages_json(args.messages_json)
    if args.messages_file:
        input_messages = _parse_messages_json(Path(args.messages_file).read_text(encoding="utf-8"))
    result = agent.run(prompt=args.prompt, messages=input_messages, context=context)

    _print_conversation_trace(result.messages)

    print("=== Final Answer ===")
    print(result.final_text)
    if result.tool_results:
        print("\n=== Tool Trace ===")
        for idx, item in enumerate(result.tool_results, start=1):
            print(f"[{idx}] {item.name}")
            print("arguments:")
            print(_json_text(item.arguments))
            print("result:")
            print(item.output_text)
            if item.new_images:
                print("new_images:")
                print(_json_text(item.new_images))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
