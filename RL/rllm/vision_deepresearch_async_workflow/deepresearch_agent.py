from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from typing import Any, List, Optional

from PIL import Image
import re
from collections import Counter
import json5
from synthesis.sft import api_tools as sft_api
from synthesis.sft.api_tools import ToolRuntimeContext
from synthesis.sft.qwen3_vl_template import (
    add_sft_image_placeholders,
    render_sft_qwen3_vl_text,
)

# rLLM imports
from rllm.engine.rollout import RolloutEngine
from rllm.utils.multimodal_debug import sequence_mode
from rllm.utils.model_format import (
    classify_tool_failure,
    format_feedback,
    validate_response,
)

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    SFTToolAdapter,
    build_sft_react_system_prompt,
)

# Constants from original DeepResearch
logger = logging.getLogger(__name__)

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 50
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = 4096


def _repair_unescaped_json_quotes(text: str) -> str:
    """Repair common model-generated JSON with unescaped inner quotes.

    Tool arguments are parsed with ``json5`` first. This fallback is
    intentionally conservative: it only changes a double quote encountered
    inside a double-quoted string when the following non-whitespace character
    cannot terminate a JSON/JSON5 value. For example, the quotes around
    ``ME!`` in ``"single "ME!" was released"`` are escaped, while normal
    object/string delimiters are left untouched.

    Single-quoted JSON5 strings are tracked separately, so embedded double
    quotes in those strings are not modified.
    """

    def next_non_whitespace(index: int) -> str | None:
        while index < len(text) and text[index].isspace():
            index += 1
        return text[index] if index < len(text) else None

    repaired: list[str] = []
    quote: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if quote is None:
            repaired.append(char)
            if char in {'"', "'"}:
                quote = char
            continue

        if escaped:
            repaired.append(char)
            escaped = False
            continue

        if char == "\\":
            repaired.append(char)
            escaped = True
            continue

        if char != quote:
            repaired.append(char)
            continue

        # A normal closing quote is followed by a JSON delimiter (or EOF).
        # Otherwise it is most likely an unescaped quote inside the value.
        following = next_non_whitespace(index + 1)
        if quote == '"' and following not in {None, ",", ":", "}", "]"}:
            repaired.append('\\"')
        else:
            repaired.append(char)
            quote = None

    return "".join(repaired)


def _parse_tool_call_payload(tool_call_text: str) -> Any:
    """Parse a model tool call, with a narrow malformed-quote fallback."""

    try:
        return json5.loads(tool_call_text)
    except Exception as original_error:  # noqa: BLE001
        repaired_text = _repair_unescaped_json_quotes(tool_call_text)
        if repaired_text == tool_call_text:
            raise

        try:
            parsed = json5.loads(repaired_text)
        except Exception:
            # Preserve the original parser failure and existing caller
            # behaviour when the fallback cannot make the payload valid.
            raise original_error

        logger.warning(
            "Repaired unescaped double quote(s) in a model tool call payload"
        )
        return parsed


def _extract_answer_tool_fallback(tool_args: Any) -> str:
    """Extract a final answer from a mistakenly emitted ``answer`` call."""
    if isinstance(tool_args, dict):
        for key in ("answer", "final_answer", "response", "content"):
            value = tool_args.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                return value.strip()
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value).strip()
    elif isinstance(tool_args, str):
        return tool_args.strip()
    return ""


DEEPRESEARCH_SYSTEM_PROMPT_TEXT = """You are an advanced **Visual Investigation Agent**. Your goal is to answer user questions with maximum precision by proactively using a suite of powerful image processing and retrieval tools. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

**CORE PHILOSOPHY: "Verify, Don't Guess"**
1. **Tool-First Mindset**: Do not rely solely on your internal visual encoder if a tool can provide a clearer view or exact text. If text is small, **Crop** it. If text is blurry, **Sharpen** it. If the image is tilted, **Correct Perspective**.
2. **Chain Your Tools**: Complex problems often require a sequence of actions (e.g., `perspective_correct` -> `crop` -> `layout_parsing`). Do not stop at the first step.
3. **Layout Parsing Workflow Rule**: For document images, use `layout_parsing` to extract structured text. You can optionally `crop` the document region first if needed, then use `layout_parsing` directly on the image reference (e.g., `img_1`).
4. **External Validation**: If a question involves specific entities, facts, or context not purely visible in the pixel data, you **MUST** use `text_search` to verify.

---

### 1. TOOL CALLING FORMAT

You may call one or more functions to assist with the user query. You are provided with function signatures within `<tools></tools>` XML tags.

**How to call a tool**: Return a JSON object with function name and arguments within `<tool_call></tool_call>` XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "crop", "arguments": {"image": "img_1", "x": 0, "y": 0, "width": 200, "height": 100}}
</tool_call>

---

### 2. YOUR TOOLBOX & TRIGGER CONDITIONS

**A. Visual Perception Tools**
* **`crop`**:
    * *Trigger*: The target (text/object) covers < 30% of the image, or multiple distinct sections need analysis.
    * *Benefit*: drastically improves OCR and recognition accuracy by removing noise.
    * *Params*: `{"image": "img_n", "x": int, "y": int, "width": int, "height": int}`

* **`layout_parsing`** (using Layout Parsing API):
    * *Trigger*: Document images with structured text (paragraphs, titles, footnotes). **NEVER transcribe text manually**; always use layout parsing for ground truth.
    * *Workflow*: `crop` (optional) -> `layout_parsing` (on the image reference)
    * *Params*: `{"image": "img_n", "use_chart_recognition": false, "use_doc_orientation_classify": false}` or `{"file_path": "/absolute/path/to/image.png", ...}` (file_path is optional, image reference is preferred)
    * *Output*: Returns detected text blocks with structured content. **IMPORTANT**: The layout parsing result will clearly show "Layout Parsing SUCCESS" if text is detected, followed by "ALL RECOGNIZED TEXT" section. **ALWAYS use the text from the layout parsing result** - do not ignore it or claim "no text detected" if layout parsing returns text. If layout parsing shows text, that is the ground truth.

**B. Image Enhancement Tools (The "Pre-processing Pipeline")**
* **`perspective_correct`**:
    * *Trigger*: Document is photographed at an angle, trapezoidal shapes, or text lines are not horizontal.
    * *Params*: `{"image": "img_n"}`
* **`super_resolution`**:
    * *Trigger*: Image is pixelated, low-res (e.g., < 500px width), or text strokes are broken.
    * *Params*: `{"image": "img_n", "scale": 4}`
* **`sharpen`**:
    * *Trigger*: Motion blur, out-of-focus text, or soft edges.
    * *Params*: `{"image": "img_n", "amount": 1.5}`

**C. Knowledge Retrieval Tools**
* **`text_search`** (Text Search with AI Summarization):
    * *Trigger*: Questions about "Who/What/When/Where", specific terminology, facts requiring external knowledge, or when you need up-to-date information not visible in the image.
    * *How it works*: This tool combines **Serper API** (web search), **JINA Reader** (webpage content extraction), and **Qwen3-32B** (AI summarization). It searches the web, extracts full webpage content, and generates query-focused summaries.
    * *Params*: `{"q": "search query", "hl": "en", "top_k": 5}`
        - `q` (required): The search query string
        - `hl` (optional): Language code (default: "en")
        - `top_k` (optional): Number of results to return and summarize (default: 5)
    * *Output*: Returns a list of summarized passages from top-k relevant webpages, each with title, URL, and AI-generated summary focused on your query. **Use these summaries as reliable sources** - they are already processed and condensed for relevance.
* **`image_search`** (Visual Search):
    * *Trigger*: Need to identify an unknown object, finding similar styles, or understanding scene context.
    * *Params*: `{"url": "image_url"}` (url can be an image reference like "img_1" or a direct URL)
    * *Output*: Returns AI-summarized results with only "title" and "source" fields, filtered by Qwen3-32B to remove irrelevant information.
    * **CRITICAL WORKFLOW RULE**: After using `image_search`, you **MUST** follow up with `text_search` to get detailed information about the identified entities. Image search only provides initial identification - text search provides the comprehensive facts you need for your answer.

---

### 3. THE THINKING PROTOCOL (<thinking>)

Before generating ANY tag, you must perform a structured analysis inside `<thinking>` tags. You must evaluate the **Image Quality** and **Information Gap**.

**Mandatory Thinking Structure:**
1.  **Analyze Request**: What is the user actually looking for?
2.  **Assess Image Quality**:
    * Is the text legible? -> If NO, plan `sharpen` or `super_resolution`.
    * Is the geometry flat? -> If NO, plan `perspective_correct`.
    * Is the target too small? -> If YES, plan `crop`.
3.  **Identify Information Gaps**: Do I need external facts? -> If YES, plan `text_search`.
4.  **Formulate Plan**: Decide the immediate next step.

**CRITICAL: Understanding Layout Parsing Results**
- When layout parsing returns text, **ALWAYS trust and use the layout parsing result** as ground truth.
- Layout parsing output will clearly show "Layout Parsing SUCCESS" if text is detected.
- Look for the "ALL RECOGNIZED TEXT" section - this contains the exact text recognized.
- **DO NOT** claim "layout parsing didn't detect any text" if the layout parsing result shows text blocks.
- If layout parsing returns text, use it directly in your answer - do not rely on visual observation when layout parsing has provided the text.

**CRITICAL: Understanding Image Search Results**
- Image search results are processed by Qwen3-32B to extract only relevant "title" and "source" information.
- The results are filtered to remove irrelevant details - only use what is provided.
- **After image_search, you MUST use text_search** to get detailed information about the identified entities.
- Image search provides initial identification, but text search provides the comprehensive facts needed for your answer.

**CRITICAL: Understanding Text Search Results**
- Text search returns **AI-generated summaries** from multiple webpages, not raw search results.
- Each result includes: Title, URL, and a Summary that is already focused on your query.
- **Trust the summaries** - they are generated by Qwen3-32B and filtered for relevance.
- If multiple passages contain relevant information, synthesize them in your final answer.
- Always cite the URLs when using information from text_search results.

---

### 4. COMMON WORKFLOW RECIPES (Examples)

**Scenario A: The "Unreadable Receipt/Document"**
* *Observation*: "The image is a receipt, but it's blurry and tilted."
* *Action 1*: `<tool_call>{"name": "perspective_correct", "arguments": {"image": "img_1"}}</tool_call>`
* *Action 2*: `<tool_call>{"name": "sharpen", "arguments": {"image": "img_2", "amount": 1.5}}</tool_call>` (on the new corrected image)
* *Action 3*: `<tool_call>{"name": "layout_parsing", "arguments": {"image": "img_3"}}</tool_call>` (on the sharpened image)

**Scenario B: The "Detailed Chart Analysis"**
* *Observation*: "There is a dense chart with a legend in the corner."
* *Action 1*: `<tool_call>{"name": "crop", "arguments": {"image": "img_1", "x": 0, "y": 0, "width": 200, "height": 100}}</tool_call>` (focus on the legend, creates img_2)
* *Action 2*: `<tool_call>{"name": "layout_parsing", "arguments": {"image": "img_2"}}</tool_call>` (read the legend text from the cropped image)
* *Action 3*: `<tool_call>{"name": "crop", "arguments": {"image": "img_1", "x": 200, "y": 100, "width": 400, "height": 300}}</tool_call>` (focus on the data bars, creates img_3)

**Scenario C: The "Entity Identification"**
* *Observation*: "I see a landmark but don't know its history."
* *Action 1*: `<tool_call>{"name": "image_search", "arguments": {"url": "img_1"}}</tool_call>` (to analyze the image and identify the name)
* *Action 2*: `<tool_call>{"name": "text_search", "arguments": {"q": "landmark name history", "hl": "en", "top_k": 5}}</tool_call>` (to get AI-summarized historical facts from top webpages using the name found)
* **MANDATORY**: After every `image_search`, you **MUST** call `text_search` with a query based on the identified entity/object to get comprehensive information.

---

### 5. OUTPUT RULES

1.  **Single Action Per Turn**: Output only ONE `<tool_call>` per turn. Wait for the result before proceeding.
2.  **Think First**: Never output a `<tool_call>` or `<answer>` without a preceding `<thinking>` block (or `<thinking>` tag).
3.  **Tool Call Format**: Always use `<tool_call>` XML tag with JSON format: `<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>`
4.  **Image References**: Start with `img_1`. Results from tools become `img_2`, `img_3`, etc. Always operate on the *latest* best version of the image.
5.  **Final Answer**: When you have sufficient info, output `<answer>...</answer>`.
    * **Visual Aids**: In your final response, if a diagram would help explain a concept (e.g., scientific process, machine part), insert `[Image of <query>]` tags naturally in the text.

---

### 6. EXECUTION FORMATS

**Case: Tool Use (Example)**
<thinking>
The user asks for the total on the invoice. The image (img_1) is taken from a side angle (skewed). Direct layout parsing will likely fail. I must first correct the perspective to make the text horizontal.
</thinking>
<tool_call>
{"name": "perspective_correct", "arguments": {"image": "img_1"}}
</tool_call>

**Case: Final Response (Example)**
<thinking>
I have cropped the chart (img_2) and used layout parsing on the values. The trend shows a 50% increase. I can now answer the user.
</thinking>
<answer>
Based on the analysis of the chart, the revenue increased by 50%.

boxed{50%}
</answer>

Current date: """


# Keep the historical prompt above for reference, but use the SFT schemas and
# tool names at runtime.  The SFT backend still uses its own exact dispatcher;
# the RL model-facing format remains ``<thinking>``/``<tool_call>``/``<answer>``.
DEEPRESEARCH_SYSTEM_PROMPT = build_sft_react_system_prompt()


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5):
    """
    Use N-grams to detect repetition in a string.

    Args:
        text (str): Input text to analyze.
        n (int): N-gram window size (default 10).
            - For long repetitive sequences, 10-20 is recommended.
        threshold (float): Distinct-N threshold (0~1).
            - Values below this indicate heavy repetition (default 0.5).

    Returns:
        bool: True if repetition is detected, False otherwise.
    """
    if not text or len(text) < n:
        print("text is too short, cannot analyze.")
        return False

    # 1. Generate N-grams (character-level sliding window).
    # List comprehension: slice from index i to i+n.
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]

    total_count = len(ngrams)
    if total_count == 0:
        return False

    # 2. Count frequencies.
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)

    # 3. Compute Distinct-N (unique count / total count).
    # Repetitive text is typically < 0.4.
    distinct_ratio = unique_count / total_count

    # 4. Determine repetition.
    is_repetitive = distinct_ratio < threshold

    return is_repetitive


def count_words(text: str) -> int:
    # Match segments that look like English words.
    # Rule: starts and ends with a letter, may contain letters, apostrophes, or hyphens.
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    words = pattern.findall(text)
    return len(words)


def build_text_completion_prompt(
    messages: list[dict], allow_special: bool = True
) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    del allow_special  # retained for compatibility with the old helper API
    return render_sft_qwen3_vl_text(messages)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        max_llm_calls: int | None = None,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        # SFT tools are synchronous and may sleep during their retry budget.
        # Run them in the workflow's shared executor instead of blocking the
        # async rollout event loop.
        self.executor = kwargs.get("executor")
        self._sft_context: ToolRuntimeContext | None = None
        self._pending_sft_images: list[Any] = []
        # Private trajectory metadata for reward shaping. It never enters the
        # model-visible tool response.
        self._search_trace: list[dict[str, Any]] = []
        # Configuration from original DeepResearch.  Keep the historical
        # default, but allow the workflow/configuration to override it for a
        # particular experiment.
        self.max_llm_calls = (
            int(max_llm_calls)
            if max_llm_calls is not None
            else MAX_LLM_CALL_PER_RUN
        )
        self.default_max_tries = default_max_tries

        # Token accounting is reset at the start of every trajectory.  The
        # rollout engine reports the fully rendered prompt length for each
        # model call, including the previous tool observations.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.prompt_tokens_by_call: list[int] = []
        self.completion_tokens_by_call: list[int] = []
        self.model_call_count = 0
        # Per-trajectory latency samples.  Keep the raw samples small and
        # local to the trajectory; the workflow aggregates them into batch
        # metrics before the DataProto is returned to the trainer.
        self._llm_latency_samples_s: list[float] = []
        self._llm_latency_failures = 0
        self._tool_latency_samples_s: dict[str, list[float]] = {}
        self._tool_latency_failures: dict[str, int] = {}
        self._tool_dispatch_latency_samples_s: dict[str, list[float]] = {
            "executor_wait_s": [],
            "worker_execute_s": [],
            "executor_result_wait_s": [],
            "event_loop_probe_wait_s": [],
            "worker_to_future_callback_s": [],
            "future_callback_to_resume_s": [],
            "event_loop_resume_postprocess_s": [],
            "tool_result_postprocess_s": [],
            "llm_response_postprocess_s": [],
        }
        # Private timing for the most recent SFT-dispatched tool call.  It is
        # emitted only by the opt-in per-call timing debug stream and never
        # enters the model-visible observation.
        self._last_tool_execution_timing: dict[str, Any] = {}
        # ``read_url`` reports scalar backend timing buckets through the SFT
        # dispatcher. Keep only totals/calls here; no tool payload is copied
        # into the trajectory latency metadata.
        self._read_url_stage_latency: dict[str, dict[str, float]] = {}
        self._wikimedia_download_stats: dict[str, float] = {
            "calls": 0.0,
            "download_success": 0.0,
            "cache_success": 0.0,
            "failure": 0.0,
            "total_s": 0.0,
        }
        # Count tool attempts at dispatch time, so timeout diagnostics also
        # include a tool that is still running when the workflow is cancelled.
        self._tool_call_counts: dict[str, int] = {}
        # Stage 1 keeps the processor output of every model call so the
        # stepwise DataProto path can carry visual payloads into update.  Do
        # not retain these large tensors in the default/legacy path.
        self._sequence_mode = sequence_mode()
        self.model_outputs_by_round: list[Any] = []
        self.rollout_token_records: list[dict] = []
        self.last_model_output: Any | None = None
        self._trajectory_id = ""
        self._current_round = 0

        self.max_prompt_tokens = MAX_PROMPT_LENGTH_PER_RUN
        self.max_response_tokens = MAX_RESPONSE_LENGTH_PER_RUN

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<thinking>" in content and "</thinking>" in content

    @staticmethod
    def _debug_text_without_images(content: Any) -> str:
        """Return only textual content for the answer-tool debug message.

        Debugging should show the prompt and model outputs without copying image
        objects or image placeholders into the terminal log.  The normal
        trajectory is not modified by this helper.
        """
        if isinstance(content, str):
            return re.sub(
                r"<image>|<\|vision_start\|>|<\|image_pad\|>|<\|vision_end\|>",
                "",
                content,
            ).strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    part_type = str(part.get("type", ""))
                    if part_type in {"image", "image_url", "input_image"}:
                        continue
                    if part_type in {"text", "input_text"}:
                        text = part.get("text", "")
                        if text:
                            text_parts.append(str(text))
            return MultiTurnReactAgent._debug_text_without_images("\n".join(text_parts))
        return ""

    def _build_answer_tool_debug_input(self, messages: list[dict]) -> list[dict[str, str]]:
        """Keep system prompt, original user prompt, and assistant turns only."""
        debug_messages: list[dict[str, str]] = []
        initial_user_seen = False
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            if role == "system":
                debug_messages.append(
                    {
                        "role": "system",
                        "content": self._debug_text_without_images(message.get("content", "")),
                    }
                )
            elif role == "user" and not initial_user_seen:
                debug_messages.append(
                    {
                        "role": "user",
                        "content": self._debug_text_without_images(message.get("content", "")),
                    }
                )
                initial_user_seen = True
            elif role == "assistant":
                debug_messages.append(
                    {
                        "role": "assistant",
                        "content": self._debug_text_without_images(message.get("content", "")),
                    }
                )
        return debug_messages

    def _log_step_error(
        self,
        *,
        reason: str,
        round_number: int,
        consecutive_errors: int | None = None,
        tool_name: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Print one bounded diagnostic line for an already-detected error.

        This is intentionally diagnostic-only: it does not change the error
        flag, retry counter, reward, or termination behavior.
        """
        if str(os.environ.get("RLLM_STEP_ERROR_DEBUG", "1")).strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return

        fields = [
            f"task={self._trajectory_id or 'unknown'}",
            f"round={round_number}",
            f"reason={reason}",
        ]
        if consecutive_errors is not None:
            fields.append(f"consecutive={consecutive_errors}")
        if tool_name:
            fields.append(f"tool={tool_name}")
        if detail:
            compact_detail = " ".join(str(detail).split())[:240]
            fields.append(f"detail={compact_detail}")
        print("[RL_STEP_ERROR] " + " ".join(fields), flush=True)

    async def call_server(
        self, messages: list[dict], max_tries: Optional[int] = None, **kwargs
    ):
        """Call rollout engine once; assumes XML ReAct format."""
        try:
            # Force per-round limits from DeepResearchAgent without local token estimation.
            if hasattr(self.rollout_engine, "max_prompt_length"):
                self.rollout_engine.max_prompt_length = int(self.max_prompt_tokens)
            if hasattr(self.rollout_engine, "max_response_length"):
                self.rollout_engine.max_response_length = int(self.max_response_tokens)

            kwargs.pop("max_new_tokens", None)
            kwargs["max_tokens"] = int(self.max_response_tokens)
            if getattr(self.rollout_engine, "continuous_tito", False):
                if not hasattr(self, "_continuous_token_state"):
                    self._continuous_token_state = {}
                kwargs["continuous_token_state"] = self._continuous_token_state
            response = await self.rollout_engine.get_model_response(
                messages=messages, **kwargs
            )

            return response
        except Exception as exc:  # noqa: BLE001
            if str(os.environ.get("RLLM_STEP_ERROR_DEBUG", "1")).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }:
                self._log_step_error(
                    reason="llm_request_failed",
                    round_number=self._current_round,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            raise

    def record_token_usage(self, response) -> None:
        """Record per-call and cumulative token usage for this trajectory.

        A later prompt contains the earlier assistant messages and tool
        observations.  Thus ``total_prompt_tokens`` is the cumulative request
        cost, while ``last_prompt_tokens`` is the final full conversation
        context length before the latest model response.
        """
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)

        if prompt_tokens is not None:
            try:
                self.last_prompt_tokens = int(prompt_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.last_prompt_tokens = 0

        if completion_tokens is not None:
            try:
                self.last_completion_tokens = int(completion_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.last_completion_tokens = 0

        self.total_prompt_tokens += self.last_prompt_tokens
        self.total_completion_tokens += self.last_completion_tokens
        self.prompt_tokens_by_call.append(self.last_prompt_tokens)
        self.completion_tokens_by_call.append(self.last_completion_tokens)
        self.model_call_count += 1

    @staticmethod
    def _percentile(samples: list[float], percentile: float) -> float:
        """Return a dependency-free percentile for a small sample list."""

        if not samples:
            return 0.0
        ordered = sorted(float(sample) for sample in samples)
        index = max(
            0,
            min(
                len(ordered) - 1,
                math.ceil((float(percentile) / 100.0) * len(ordered)) - 1,
            ),
        )
        return ordered[index]

    @classmethod
    def _summarize_latency_samples(
        cls,
        samples: list[float],
        *,
        failures: int = 0,
        include_samples: bool = True,
    ) -> dict[str, Any]:
        values = [max(0.0, float(sample)) for sample in samples]
        calls = len(values)
        summary: dict[str, Any] = {
            "calls": calls,
            "failures": int(failures),
            "total_s": round(sum(values), 6),
            "mean_s": round(sum(values) / calls, 6) if calls else 0.0,
            "p95_s": round(cls._percentile(values, 95), 6) if values else 0.0,
            "max_s": round(max(values), 6) if values else 0.0,
        }
        if include_samples:
            # These are scalar telemetry values, not prompts/tool payloads.
            # Retaining them enables exact batch-level p95 aggregation.
            summary["samples_s"] = [round(value, 6) for value in values]
        return summary

    def _record_llm_latency(self, elapsed_s: float, *, success: bool) -> None:
        self._llm_latency_samples_s.append(max(0.0, float(elapsed_s)))
        if not success:
            self._llm_latency_failures += 1

    def _record_tool_latency(
        self,
        tool_name: str,
        elapsed_s: float,
        *,
        success: bool,
    ) -> None:
        name = str(tool_name or "unknown")
        self._tool_latency_samples_s.setdefault(name, []).append(
            max(0.0, float(elapsed_s))
        )
        if not success:
            self._tool_latency_failures[name] = (
                self._tool_latency_failures.get(name, 0) + 1
            )

    def _print_tool_timing(
        self,
        tool_name: str,
        elapsed_s: float,
        *,
        success: bool,
    ) -> None:
        """Suppress per-tool timing output while retaining timing collection.

        The actual timing payload is stored by ``custom_call_tool`` and later
        aggregated into the every-50-rollout progress line.  Emitting one
        flushed JSON line for every tool call creates substantial Ray/stdout
        backpressure at high rollout concurrency.
        """
        return

    def _record_read_url_stage_latency(self, timing: Any) -> None:
        """Accumulate scalar ``read_url`` backend timing for this trajectory."""

        if not isinstance(timing, dict):
            return
        stages = timing.get("read_url_stages")
        if not isinstance(stages, dict):
            return
        for stage_name, elapsed_s in stages.items():
            try:
                elapsed = max(0.0, float(elapsed_s))
            except (TypeError, ValueError):
                continue
            if elapsed <= 0.0:
                continue
            summary = self._read_url_stage_latency.setdefault(
                str(stage_name), {"total_s": 0.0, "calls": 0.0}
            )
            summary["total_s"] += elapsed
            summary["calls"] += 1.0

        wikimedia_download = timing.get("wikimedia_download")
        if isinstance(wikimedia_download, dict):
            for key in (
                "calls",
                "download_success",
                "cache_success",
                "failure",
                "total_s",
            ):
                try:
                    self._wikimedia_download_stats[key] += max(
                        0.0, float(wikimedia_download.get(key) or 0.0)
                    )
                except (TypeError, ValueError):
                    continue

    def _build_latency_stats(self, trajectory_time_s: float) -> dict[str, Any]:
        tool_summaries: dict[str, dict[str, Any]] = {}
        all_tool_samples: list[float] = []
        for tool_name, samples in sorted(self._tool_latency_samples_s.items()):
            all_tool_samples.extend(samples)
            tool_summaries[tool_name] = self._summarize_latency_samples(
                samples,
                failures=self._tool_latency_failures.get(tool_name, 0),
            )

        tool_dispatch_summaries = {
            metric_name: self._summarize_latency_samples(
                samples,
                include_samples=True,
            )
            for metric_name, samples in self._tool_dispatch_latency_samples_s.items()
            if samples
        }

        read_url_stages: dict[str, dict[str, Any]] = {}
        for stage_name, aggregate in sorted(self._read_url_stage_latency.items()):
            total_s = max(0.0, float(aggregate.get("total_s") or 0.0))
            calls = int(aggregate.get("calls") or 0)
            read_url_stages[stage_name] = {
                "calls": calls,
                "total_s": round(total_s, 6),
                "mean_s": round(total_s / calls, 6) if calls else 0.0,
            }

        wikimedia_download = dict(self._wikimedia_download_stats)
        wikimedia_calls = int(wikimedia_download.get("calls") or 0)
        wikimedia_download["calls"] = wikimedia_calls
        wikimedia_download["download_success"] = int(
            wikimedia_download.get("download_success") or 0
        )
        wikimedia_download["cache_success"] = int(
            wikimedia_download.get("cache_success") or 0
        )
        wikimedia_download["failure"] = int(
            wikimedia_download.get("failure") or 0
        )
        wikimedia_download["total_s"] = round(
            max(0.0, float(wikimedia_download.get("total_s") or 0.0)), 6
        )
        wikimedia_download["mean_s"] = round(
            wikimedia_download["total_s"] / wikimedia_calls, 6
        ) if wikimedia_calls else 0.0

        return {
            "trajectory_time_s": round(max(0.0, float(trajectory_time_s)), 6),
            "llm": self._summarize_latency_samples(
                self._llm_latency_samples_s,
                failures=self._llm_latency_failures,
            ),
            "tool": self._summarize_latency_samples(
                all_tool_samples,
                failures=sum(self._tool_latency_failures.values()),
                include_samples=False,
            ),
            "tools": tool_summaries,
            "tool_dispatch": tool_dispatch_summaries,
            "read_url_stages": read_url_stages,
            "wikimedia_download": wikimedia_download,
        }

    def get_total_tokens_used(self) -> int:
        """Return cumulative prompt + completion tokens for this trajectory."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate prompt length for the next call using the rollout engine's tokenizer."""
        tokenizer = getattr(self.rollout_engine, "tokenizer", None)
        chat_parser = getattr(self.rollout_engine, "chat_parser", None)

        if tokenizer is None or chat_parser is None:
            return self.total_prompt_tokens

        try:
            prompt = chat_parser.parse(
                messages,
                add_generation_prompt=True,
                is_first_msg=True,
                tools=[],
                accumulate_reasoning=getattr(
                    self.rollout_engine, "accumulate_reasoning", False
                ),
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"[TokenEstimator] Failed to estimate prompt tokens: {exc}")
            return self.total_prompt_tokens

    def _build_result(
        self,
        *,
        question: str,
        answer: str | None,
        messages: list[dict],
        prediction: str,
        termination: str,
        rounds: int,
        start_time: float,
        # next_prompt_tokens: int | None = None,
    ) -> dict:
        """Assemble result payload with token usage metadata."""
        trajectory_prompt_tokens = self.total_prompt_tokens
        trajectory_completion_tokens = self.total_completion_tokens
        trajectory_total_tokens = trajectory_prompt_tokens + trajectory_completion_tokens
        # The latest request prompt contains the initial prompt, all previous
        # assistant turns, and all tool observations.  Adding the latest
        # completion gives the final trajectory context length without
        # repeatedly counting prefixes from earlier requests.
        trajectory_context_tokens = self.last_prompt_tokens + self.last_completion_tokens
        token_usage = {
            # ``prompt``/``completion`` are now cumulative over the trajectory.
            "prompt": trajectory_prompt_tokens,
            "completion": trajectory_completion_tokens,
            "total": trajectory_total_tokens,
            "trajectory_context": trajectory_context_tokens,
            "trajectory_prompt_requests": trajectory_prompt_tokens,
            "model_calls": self.model_call_count,
            "last_prompt": self.last_prompt_tokens,
            "last_completion": self.last_completion_tokens,
            "prompt_by_call": self.prompt_tokens_by_call,
            "completion_by_call": self.completion_tokens_by_call,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.perf_counter() - start_time,
            "token_usage": token_usage,
            "search_trace": list(self._search_trace),
        }
        result["latency"] = self._build_latency_stats(result["time_taken"])
        result["rollout_token_records"] = self.rollout_token_records
        if self._sequence_mode == "stepwise":
            # ModelOutput contains the processor-expanded prompt ids and the
            # corresponding pixel_values/image_grid_thw.  It is consumed by
            # DeepResearchWorkflow before the result leaves the worker.
            result["model_outputs"] = self.model_outputs_by_round
        elif self._sequence_mode == "cumulative" and self.last_model_output is not None:
            # Stage 2 only needs the final call's already-processed context;
            # keeping every round would duplicate large visual tensors.
            result["last_model_output"] = self.last_model_output
        # Release cached PIL images after rollout; the final ModelOutput already
        # owns the feature tensors and immutable sequence/mask needed by training.
        self._continuous_token_state = {}
        return result

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        Supports image-processing tools via an internal ``image_paths``
        dictionary that maps ``img_1``, ``img_2``, … to local paths / URLs.
        """
        start_time = time.perf_counter()
        # RL may inject a sample-scoped cache.  Remove it before forwarding
        # kwargs to the model client; it is only for tool execution.
        tool_cache = kwargs.pop("tool_cache", None)
        self._trajectory_id = str(kwargs.pop("trajectory_id", "") or "")

        # A workflow object may be reused by the execution engine.  Keep
        # token statistics scoped to this individual trajectory.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.prompt_tokens_by_call = []
        self.completion_tokens_by_call = []
        self.model_call_count = 0
        self._llm_latency_samples_s = []
        self._llm_latency_failures = 0
        self._tool_latency_samples_s = {}
        self._tool_latency_failures = {}
        self._tool_dispatch_latency_samples_s = {
            "executor_wait_s": [],
            "worker_execute_s": [],
            "executor_result_wait_s": [],
            "event_loop_probe_wait_s": [],
            "worker_to_future_callback_s": [],
            "future_callback_to_resume_s": [],
            "event_loop_resume_postprocess_s": [],
            "tool_result_postprocess_s": [],
            "llm_response_postprocess_s": [],
        }
        self._last_tool_execution_timing = {}
        self._read_url_stage_latency = {}
        self._wikimedia_download_stats = {
            "calls": 0.0,
            "download_success": 0.0,
            "cache_success": 0.0,
            "failure": 0.0,
            "total_s": 0.0,
        }
        self._tool_call_counts = {}
        self.model_outputs_by_round = []
        self.rollout_token_records = []
        self.last_model_output = None
        self._continuous_token_state = {}
        self._current_round = 0
        # Read this at run time as well as construction time so a workflow
        # object reused by a test/worker observes the current exported mode.
        self._sequence_mode = sequence_mode()

        effective_system_prompt = system_prompt or self.system_prompt or DEEPRESEARCH_SYSTEM_PROMPT

        # ---- image_paths management for visual tools ----
        self._image_paths: dict[str, str] = {}
        self._intermediate_dir = kwargs.pop(
            "intermediate_dir",
            os.path.join("/tmp", "vdr_tools", str(int(time.time() * 1000))),
        )
        if image_path:
            self._image_paths["img_1"] = image_path

        # One SFT ToolRuntimeContext is created per trajectory.  It owns the
        # search-result/resource registry and image registry, so IDs returned by
        # t2t/t2i/i2i_search remain resolvable by later read_url calls.
        self._sft_context = ToolRuntimeContext(
            working_dir=self._intermediate_dir,
            filename_prefix="rl",
            case_id="rl_session",
        )
        initial_image = None
        if images:
            initial_image = images[0]
        elif image_path:
            initial_image = image_path
        if initial_image is not None:
            self._sft_context.image_registry["img_1"] = initial_image
            self._sft_context._image_counter = 1
        self._pending_sft_images = []
        self._search_trace = []

        if images:
            user_message = {
                "role": "user",
                "content": add_sft_image_placeholders(question, len(images)),
                "images": images,
            }
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": effective_system_prompt},
            user_message,
        ]

        tool_schemas = {}
        for name, tool in self.tools.items():
            definition = getattr(tool, "json", {})
            if callable(definition):
                definition = definition()
            function = definition.get("function", definition) if isinstance(definition, dict) else {}
            tool_schemas[name] = function.get("parameters", {})
        if "PythonInterpreter" in self.tools:
            tool_schemas["PythonInterpreter"] = {
                "type": "object", "properties": {"code": {"type": "string"}},
                "required": ["code"],
            }

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""
        consecutive_bad_steps = 0

        while num_llm_calls_available > 0:
            round += 1
            self._current_round = round
            num_llm_calls_available -= 1

            # Get model response from rollout engine
            llm_started_at = time.perf_counter()
            llm_succeeded = False
            try:
                response = await self.call_server(messages, **kwargs)
                llm_succeeded = True
            except Exception as exc:  # noqa: BLE001
                prediction = "call_server failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )
            finally:
                self._record_llm_latency(
                    time.perf_counter() - llm_started_at,
                    success=llm_succeeded,
                )

            llm_response_received_at = time.perf_counter()
            self.record_token_usage(response)
            # Keep only lightweight generation tokens, not per-turn image tensors.
            self.rollout_token_records.append({
                "token_ids": getattr(response, "completion_ids", None),
                "log_probs": getattr(response, "logprobs", None),
            })
            if getattr(self.rollout_engine, "continuous_tito", False):
                from rllm.utils.rollout_consistency import token_fingerprint
                prompt_ids = getattr(response, "prompt_ids", None)
                if prompt_ids is not None:
                    self.rollout_token_records[-1]["prompt_fingerprint"] = token_fingerprint(prompt_ids)
                    self.rollout_token_records[-1]["prompt_length"] = len(prompt_ids)
            if self._sequence_mode in {"stepwise", "cumulative"}:
                self.last_model_output = response
            if self._sequence_mode == "stepwise":
                self.model_outputs_by_round.append(response)

            raw_content = (
                response.text if hasattr(response, "text") and response.text else ""
            )
            content = raw_content

            format_check = validate_response(content, tool_schemas)

            if not format_check.valid:
                messages.append({
                    "role": "assistant", "content": content.strip(),
                    "step_error": True, "error_type": format_check.error_type,
                    "error_detail": format_check.detail,
                })
                consecutive_bad_steps += 1
                self._log_step_error(
                    reason=format_check.error_type, round_number=round,
                    consecutive_errors=consecutive_bad_steps, detail=format_check.detail,
                )
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{format_feedback(format_check.detail)}\n</tool_response>",
                })
                if analyze_repetition_ngram(content) and count_words(content) > 2500:
                    prediction = "Repetition response"
                    termination = "repetition_detected"
                    break
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    break

            elif format_check.action == "tool_call":
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                    "error_type": "",
                }
                messages.append(assistant_message)
                tool_error = False
                tool_error_reason = ""
                tool_error_detail = ""
                tool_name = ""
                answer_tool_fallback = False
                tool_returned_at = 0.0

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]

                try:
                    tool_call = _parse_tool_call_payload(tool_call_text)
                except Exception as exc:
                    tool_call = None
                    result = "[Json Parse Error]: Tool call is not a valid JSON."
                    tool_error = True
                    tool_error_reason = "tool_call_json_parse"
                    tool_error_detail = f"{type(exc).__name__}: {exc}"
                if not isinstance(tool_call, dict):
                    if not tool_error:
                        result = "[Json Parse Error]: Tool call must be a JSON object."
                        tool_error_reason = "tool_call_not_object"
                    tool_error = True
                else:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    if False:  # Temporarily suppress high-volume raw-output logs.
                        # Keep the entire model output on one physical log line
                        # so it remains readable when many Ray workers print it.
                        raw_output_one_line = repr(str(raw_content))
                        debug_input_one_line = json.dumps(
                            self._build_answer_tool_debug_input(messages),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        print(
                            f"[RL_ANSWER_TOOL_DEBUG] task={self._trajectory_id or 'unknown'} "
                            f"round={round} input_messages={debug_input_one_line} "
                            f"raw_output={raw_output_one_line}",
                            flush=True,
                        )
                    if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(tool_args, dict):
                        result = "[Json Parse Error]: Tool call requires a name and an arguments object."
                        tool_error = True
                        tool_error_reason = "tool_call_schema"
                    else:
                        # Measure event-loop work after the LLM Future has
                        # completed and before the tool executor is entered:
                        # response bookkeeping, tool-call parsing, and
                        # message construction.  This is kept in the same
                        # low-volume aggregate stream as tool dispatch timing.
                        self._tool_dispatch_latency_samples_s[
                            "llm_response_postprocess_s"
                        ].append(
                            max(
                                0.0,
                                time.perf_counter() - llm_response_received_at,
                            )
                        )
                        counted_tool_name = tool_name or "<empty>"
                        self._tool_call_counts[counted_tool_name] = (
                            self._tool_call_counts.get(counted_tool_name, 0) + 1
                        )
                        if tool_name == "answer":
                            # ``answer`` is not a registered backend tool, but
                            # some checkpoints still emit it instead of using
                            # the <answer>...</answer> protocol.  Preserve the
                            # malformed step for format accounting, while
                            # salvaging its answer and terminating immediately.
                            prediction = _extract_answer_tool_fallback(tool_args)
                            answer_tool_fallback = True
                            tool_error = True
                            tool_error_reason = "answer_tool_fallback"
                            tool_error_detail = (
                                "The answer tool is unavailable; extracted the "
                                "final answer from its arguments and stopped."
                            )
                            result = (
                                "[Answer Tool Fallback] The answer tool is not "
                                "available. The answer was extracted from its "
                                "arguments."
                            )
                        elif tool_name == "PythonInterpreter":
                            # Keep the legacy compatibility path, but dispatch it
                            # only by the parsed tool name.  Checking whether the
                            # raw JSON contains the word "python" would wrongly
                            # intercept valid search queries such as "Python history".
                            self._last_tool_execution_timing = {}
                            tool_started_at = time.perf_counter()
                            try:
                                code_raw = str(tool_args.get("code") or "").strip()
                                if not code_raw:
                                    result = "[Python Interpreter Error]: Python code is required."
                                    tool_error = True
                                    tool_error_reason = "python_missing_code"
                                else:
                                    result = await self.execute_python(code_raw)
                                    if isinstance(result, str) and result.startswith(
                                        (
                                            "Python execution error:",
                                            "PythonInterpreter tool not available",
                                            "PythonInterpreter tool is not callable",
                                        )
                                    ):
                                        tool_error = True
                                        tool_error_reason = "python_execution_error"
                            except Exception as exc:
                                result = "[Python Interpreter Error]: Python code formatting error."
                                tool_error = True
                                tool_error_reason = "python_exception"
                                tool_error_detail = f"{type(exc).__name__}: {exc}"
                            finally:
                                self._record_tool_latency(
                                    tool_name,
                                    time.perf_counter() - tool_started_at,
                                    success=not tool_error,
                                )
                                tool_returned_at = time.perf_counter()
                        else:
                            self._last_tool_execution_timing = {}
                            self._last_tool_error_detail = ""
                            tool_started_at = time.perf_counter()
                            try:
                                result = await self.custom_call_tool(
                                    tool_name,
                                    tool_args,
                                    question_text=question,
                                    assistant_text=content,
                                    tool_goal=str(tool_args.get("goal") or ""),
                                    tool_cache=tool_cache,
                                )
                                if self._last_tool_error_detail:
                                    tool_error = True
                                    tool_error_reason = "tool_result_error"
                                    tool_error_detail = self._last_tool_error_detail
                            except Exception as exc:  # noqa: BLE001
                                result = f"Tool execution error: {type(exc).__name__}: {exc}"
                                tool_error = True
                                tool_error_reason = "tool_execution_exception"
                                tool_error_detail = f"{type(exc).__name__}: {exc}"
                            finally:
                                tool_elapsed_s = time.perf_counter() - tool_started_at
                                self._record_tool_latency(
                                    tool_name,
                                    tool_elapsed_s,
                                    success=not tool_error,
                                )
                                self._print_tool_timing(
                                    tool_name,
                                    tool_elapsed_s,
                                    success=not tool_error,
                                )
                                tool_returned_at = time.perf_counter()

                if tool_error:
                    assistant_message["step_error"] = True
                    if tool_error_reason.startswith("tool_call_") or tool_error_reason == "answer_tool_fallback":
                        assistant_message["error_type"] = "model_format_error"
                        result = format_feedback(tool_error_detail or str(result))
                    else:
                        assistant_message["error_type"] = classify_tool_failure(tool_error_detail or str(result))
                    assistant_message["error_detail"] = tool_error_detail or str(result)
                    if assistant_message["error_type"] == "tool_argument_error":
                        result = format_feedback(assistant_message["error_detail"])
                    elif assistant_message["error_type"] == "external_tool_error":
                        result = f"External tool error: {result}\nThe service failed; this is not a tool-call format error."

                observation_body = str(result)
                if self._pending_sft_images:
                    # SFT keeps the observation text and image placeholder in
                    # one user-turn content string.  Image objects are carried
                    # separately in the same order as the placeholders.
                    observation_body = (
                        f"{observation_body}\nThe image is shown below:\n"
                        + "\n".join(["<image>"] * len(self._pending_sft_images))
                    )
                tool_response = f"<tool_response>\n{observation_body}\n</tool_response>"
                observation_message = {"role": "user", "content": tool_response}
                if self._pending_sft_images:
                    observation_message["images"] = self._pending_sft_images
                messages.append(observation_message)
                self._pending_sft_images = []
                if assistant_message["step_error"]:
                    consecutive_bad_steps += 1
                    self._log_step_error(
                        reason=tool_error_reason or "tool_error",
                        round_number=round,
                        consecutive_errors=consecutive_bad_steps,
                        tool_name=tool_name,
                        detail=tool_error_detail or str(result),
                    )
                    if tool_error_reason in {
                        "tool_call_json_parse", "tool_call_not_object", "tool_call_schema"
                    }:
                        # Preserve response.text before tool-response truncation.
                        # JSON escaping keeps concurrent workers' records on one line.
                        print(
                            "[RL_TOOL_CALL_RAW_OUTPUT] " + json.dumps(
                                {
                                    "task": self._trajectory_id or "unknown",
                                    "round": round,
                                    "reason": tool_error_reason,
                                    "raw_output": raw_content,
                                    "parsed_payload": tool_call_text,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                else:
                    consecutive_bad_steps = 0

                # Time the synchronous event-loop work after a tool has
                # returned and before the next LLM request can be issued:
                # observation construction, message mutation, error-marker
                # handling, and image-reference handoff.  This is distinct
                # from the executor/Future callback measurements inside
                # custom_call_tool().
                if tool_returned_at > 0.0:
                    self._tool_dispatch_latency_samples_s[
                        "tool_result_postprocess_s"
                    ].append(
                        max(0.0, time.perf_counter() - tool_returned_at)
                    )
                if answer_tool_fallback:
                    termination = "answer_tool_fallback"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            elif format_check.action == "answer":
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": False,
                        "error_type": "",
                    }
                )
                prediction = format_check.payload
                termination = "answer"
                consecutive_bad_steps = 0
                break

            if num_llm_calls_available <= 0 and "<answer>" not in content:
                prediction = f"No answer found after {self.max_llm_calls} rounds."
                termination = f"answer not found after {self.max_llm_calls} rounds"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

        # Only a validated assistant answer can terminate successfully.
        # Feedback includes examples with <answer> tags and is never a prediction.
        if termination is None:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        result = self._build_result(
            question=question,
            answer=answer,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round,
            start_time=start_time,
        )

        # Temporarily suppress per-trajectory verbose output.  The compact
        # every-50-rollout summary in AgentWorkflowEngine retains timing and
        # termination statistics without blocking the event loop per sample.
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """Execute one tool call through the SFT dispatcher when applicable.

        SFT tools are synchronous and may sleep during their timeout retry
        budget.  They therefore run in the workflow's shared executor instead
        of blocking the async rollout event loop.
        """

        self._pending_sft_images = []

        if tool_name not in self.tools:
            available_tools = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available_tools}"

        tool = self.tools[tool_name]
        if isinstance(tool, SFTToolAdapter) or getattr(tool, "uses_sft_dispatcher", False):
            context = self._sft_context
            if context is None:
                raise RuntimeError(
                    "SFT ToolRuntimeContext has not been initialized for this trajectory"
                )
            if tool_name == "read_url":
                resource_id = str(tool_args.get("resource_id") or "").strip()
                direct_url = str(tool_args.get("url") or "").strip()
                if resource_id and not direct_url and context.resolve_resource_id(resource_id) is None:
                    self._last_tool_error_detail = f"resource_id not found: {resource_id}"
                elif not resource_id and not direct_url:
                    self._last_tool_error_detail = "URL is required."

            question_text = str(kwargs.get("question_text") or "")
            assistant_text = str(kwargs.get("assistant_text") or "")
            tool_goal = str(kwargs.get("tool_goal") or "")

            loop = asyncio.get_running_loop()
            submitted_at = time.perf_counter()
            worker_started_at = [0.0]
            worker_finished_at = [0.0]
            event_loop_probe_at = [0.0]
            future_done_callback_at = [0.0]

            def mark_event_loop_probe() -> None:
                # This callback is scheduled from the executor thread as soon
                # as the tool worker returns.  Its delay estimates how long
                # the shared asyncio loop takes to service a ready callback.
                event_loop_probe_at[0] = time.perf_counter()

            def mark_future_done(_future: asyncio.Future) -> None:
                # ``run_in_executor`` chains the executor future back onto the
                # asyncio future.  Recording this callback separately lets us
                # distinguish loop delivery from the coroutine's own wake-up.
                future_done_callback_at[0] = time.perf_counter()

            def execute_sft_tool():
                worker_started_at[0] = time.perf_counter()
                try:
                    return sft_api.execute_tool_call(
                        name=tool_name,
                        arguments=tool_args,
                        context=context,
                        question_text=question_text,
                        assistant_text=assistant_text,
                        tool_goal=tool_goal,
                        tool_cache=kwargs.get("tool_cache"),
                    )
                finally:
                    # This timestamp is set inside the executor thread.  The
                    # difference between it and the await-resume timestamp
                    # below measures event-loop/result-delivery delay.
                    worker_finished_at[0] = time.perf_counter()

                    # Keep this callback deliberately lightweight.  It is a
                    # probe only; no per-tool log is emitted here because
                    # stdout logging itself can create event-loop pressure.
                    try:
                        loop.call_soon_threadsafe(mark_event_loop_probe)
                    except RuntimeError:
                        # The loop may be closing while an error is unwinding.
                        pass

            execution_future = loop.run_in_executor(
                self.executor,
                execute_sft_tool,
            )
            execution_future.add_done_callback(mark_future_done)
            execution = await execution_future
            resumed_at = time.perf_counter()
            timing = dict(getattr(execution, "timing", {}) or {})
            if worker_started_at[0] > 0.0:
                timing["executor_wait_s"] = max(
                    0.0,
                    worker_started_at[0] - submitted_at,
                )
                if worker_finished_at[0] > 0.0:
                    timing["worker_execute_s"] = max(
                        0.0,
                        worker_finished_at[0] - worker_started_at[0],
                    )
                    timing["executor_result_wait_s"] = max(
                        0.0,
                        resumed_at - worker_finished_at[0],
                    )
                    if event_loop_probe_at[0] > 0.0:
                        timing["event_loop_probe_wait_s"] = max(
                            0.0,
                            event_loop_probe_at[0] - worker_finished_at[0],
                        )
                    if future_done_callback_at[0] > 0.0:
                        timing["worker_to_future_callback_s"] = max(
                            0.0,
                            future_done_callback_at[0] - worker_finished_at[0],
                        )
                        timing["future_callback_to_resume_s"] = max(
                            0.0,
                            resumed_at - future_done_callback_at[0],
                        )
                timing["sft_execution_s"] = max(
                    0.0,
                    resumed_at - worker_started_at[0],
                )
            self._last_tool_execution_timing = timing
            if isinstance(execution.output, dict) and execution.output.get("ok") is False:
                self._last_tool_error_detail = str(execution.output.get("error") or execution.output)
                self._log_step_error(
                    reason="tool_result_error",
                    round_number=self._current_round,
                    tool_name=tool_name,
                    detail=execution.output.get("error") or execution.output,
                )
            elif (
                tool_name == "read_url"
                and isinstance(execution.output, dict)
                and execution.output.get("content") == "Unable to read the requested page."
                and not self._last_tool_error_detail
            ):
                # The public reader observation intentionally removes backend
                # diagnostics and the `ok` flag. Its fixed failure content is
                # still distinguishable without exposing private URLs.
                self._last_tool_error_detail = "The read_url service could not read the requested page."
            self._record_read_url_stage_latency(getattr(execution, "timing", None))
            self._pending_sft_images = list((execution.new_images or {}).values())
            # Search redundancy is always measured, including observation-only
            # runs. The private trace is consumed and discarded at trajectory
            # completion, so it never reaches model-visible messages or the
            # training batch payload.
            self._record_t2t_search_trace(execution, context)

            # This is the synchronous work performed by the event-loop
            # coroutine after the executor Future has completed: extracting
            # timing metadata, recording stage counters, and preparing image
            # references for the next observation.  It must stay separate
            # from worker_to_future_callback_s so a large postprocess step is
            # not mistaken for callback-delivery delay.
            timing["event_loop_resume_postprocess_s"] = max(
                0.0,
                time.perf_counter() - resumed_at,
            )
            self._last_tool_execution_timing = timing
            for metric_name in (
                "executor_wait_s",
                "worker_execute_s",
                "executor_result_wait_s",
                "event_loop_probe_wait_s",
                "worker_to_future_callback_s",
                "future_callback_to_resume_s",
                "event_loop_resume_postprocess_s",
            ):
                metric_value = timing.get(metric_name)
                if isinstance(metric_value, (int, float)):
                    self._tool_dispatch_latency_samples_s[metric_name].append(
                        max(0.0, float(metric_value))
                    )
            return execution.output_text

        # Compatibility path for a caller that supplies a legacy non-SFT tool.
        try:
            if hasattr(tool, "call"):
                if asyncio.iscoroutinefunction(tool.call):
                    result = await tool.call(**tool_args)
                else:
                    result = await asyncio.get_running_loop().run_in_executor(
                        self.executor,
                        lambda: tool.call(**tool_args),
                    )
            elif callable(tool):
                result = await asyncio.get_running_loop().run_in_executor(
                    self.executor,
                    lambda: tool(**tool_args),
                )
            else:
                result = f"Tool {tool_name} is not callable"
                self._last_tool_error_detail = str(result)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            self._last_tool_error_detail = f"{type(exc).__name__}: {exc}"
            return f"Error calling tool {tool_name}: {exc}"

    def _record_t2t_search_trace(
        self, execution: Any, context: ToolRuntimeContext
    ) -> None:
        """Retain private t2t URL provenance for reward-time comparison."""

        if getattr(execution, "name", None) != "t2t_search":
            return
        arguments = getattr(execution, "arguments", {})
        output = getattr(execution, "output", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(output, dict):
            output = {}

        query = str(arguments.get("query") or arguments.get("q") or "").strip()
        urls: list[str] = []
        seen_urls: set[str] = set()
        results = output.get("results", [])
        if bool(output.get("ok")) and isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                resource_id = str(item.get("source_page_id") or "").strip()
                resource = context.resolve_resource_id(resource_id) if resource_id else None
                url = ""
                if resource is not None:
                    url = str(
                        getattr(resource, "primary_url", None)
                        or getattr(resource, "source_page_url", None)
                        or ""
                    ).strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    urls.append(url)

        self._search_trace.append(
            {
                "tool_name": "t2t_search",
                "round": int(getattr(self, "_current_round", 0) or 0),
                "query": query,
                "ok": bool(output.get("ok")),
                "result_urls": urls,
            }
        )

    async def execute_python(self, code: str) -> str:
        """
        Execute Python code using the PythonInterpreter tool.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        if "PythonInterpreter" in self.tools:
            try:
                # Use the PythonInterpreter tool
                tool = self.tools["PythonInterpreter"]
                if hasattr(tool, "call"):
                    if asyncio.iscoroutinefunction(tool.call):
                        result = await tool.call(code=code)
                    else:
                        result = tool.call(code=code)
                    return str(result)
                else:
                    return "PythonInterpreter tool is not callable"
            except Exception as e:
                return f"Python execution error: {e}"
        else:
            return "PythonInterpreter tool not available"

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._tool_call_counts = {}

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Result dictionary
        """
        # Reset token counters for each new run
        self.reset()
        return await self._run(
            question,
            answer,
            images,
            image_path,
            system_prompt=system_prompt,
            **kwargs,
        )


DeepResearchAgent = MultiTurnReactAgent
