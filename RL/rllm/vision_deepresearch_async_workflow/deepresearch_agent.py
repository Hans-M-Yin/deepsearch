from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any, List, Optional

from PIL import Image
import re
from collections import Counter
import json5

# rLLM imports
from rllm.engine.rollout import RolloutEngine

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 50
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = 4096

DEEPRESEARCH_SYSTEM_PROMPT_TEXT = """You are an advanced **Visual Investigation Agent**. Your goal is to answer user questions with maximum precision by proactively using a suite of powerful image processing and retrieval tools. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <response></response> tags.

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

### 3. THE THINKING PROTOCOL (<think>)

Before generating ANY tag, you must perform a structured analysis inside `<think>` tags. You must evaluate the **Image Quality** and **Information Gap**.

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
2.  **Think First**: Never output a `<tool_call>` or `<response>` without a preceding `<think>` block (or `<think>` tag).
3.  **Tool Call Format**: Always use `<tool_call>` XML tag with JSON format: `<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>`
4.  **Image References**: Start with `img_1`. Results from tools become `img_2`, `img_3`, etc. Always operate on the *latest* best version of the image.
5.  **Final Answer**: When you have sufficient info, output `<response>...</response>`.
    * **Visual Aids**: In your final response, if a diagram would help explain a concept (e.g., scientific process, machine part), insert `[Image of <query>]` tags naturally in the text.

---

### 6. EXECUTION FORMATS

**Case: Tool Use (Example)**
<think>
The user asks for the total on the invoice. The image (img_1) is taken from a side angle (skewed). Direct layout parsing will likely fail. I must first correct the perspective to make the text horizontal.
</think>
<tool_call>
{"name": "perspective_correct", "arguments": {"image": "img_1"}}
</tool_call>

**Case: Final Response (Example)**
<think>
I have cropped the chart (img_2) and used layout parsing on the values. The trend shows a 50% increase. I can now answer the user.
</think>
<response>
Based on the analysis of the chart, the revenue increased by 50%.

boxed{50%}
</response>

Current date: """


DEEPRESEARCH_SYSTEM_PROMPT = DEEPRESEARCH_SYSTEM_PROMPT_TEXT


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


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
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


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
        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.default_max_tries = default_max_tries

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.max_prompt_tokens = MAX_PROMPT_LENGTH_PER_RUN
        self.max_response_tokens = MAX_RESPONSE_LENGTH_PER_RUN

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

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
            response = await self.rollout_engine.get_model_response(
                messages=messages, **kwargs
            )

            if hasattr(response, "prompt_length") and hasattr(
                response, "completion_length"
            ):
                self.total_prompt_tokens += response.prompt_length
                self.total_completion_tokens += response.completion_length

            return response
        except Exception as exc:  # noqa: BLE001
            print(f"call_server failed: {exc}")
            raise

    def record_token_usage(self, response) -> None:
        """Record the latest prompt/completion token count from rollout engine."""
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)

        if prompt_tokens is not None:
            try:
                self.total_prompt_tokens = int(prompt_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_prompt_tokens = 0

        if completion_tokens is not None:
            try:
                self.total_completion_tokens = int(completion_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_completion_tokens = 0

    def get_total_tokens_used(self) -> int:
        """Return the latest prompt + completion token usage reported by the engine."""
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
        token_usage = {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": token_usage,
        }
        return result

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        Supports image-processing tools via an internal ``image_paths``
        dictionary that maps ``img_1``, ``img_2``, … to local paths / URLs.
        """
        start_time = time.time()

        system_prompt = (
            self.system_prompt or DEEPRESEARCH_SYSTEM_PROMPT
        ) + today_date()

        # ---- image_paths management for visual tools ----
        self._image_paths: dict[str, str] = {}
        self._intermediate_dir = kwargs.pop(
            "intermediate_dir",
            os.path.join("/tmp", "vdr_tools", str(int(time.time() * 1000))),
        )
        if image_path:
            self._image_paths["img_1"] = image_path

        if images:
            user_message = {
                "role": "user",
                "content": question,
                "images": images,
            }
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""
        consecutive_bad_steps = 0

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1

            # Get model response from rollout engine
            try:
                response = await self.call_server(messages, **kwargs)
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

            self.record_token_usage(response)

            content = (
                response.text if hasattr(response, "text") and response.text else ""
            )

            if "<tool_call>" in content:
                if "python" in content.lower() and "<code>" in content:
                    pass
                elif '"name":' in content:
                    try:
                        tool_text = content.split("<tool_call>")[1].split(
                            "</tool_call>"
                        )[0]
                        tool_data = json5.loads(tool_text)
                        tool_name = tool_data.get("name", "Unknown")
                    except Exception:
                        pass

            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            if "<tool_call>" in content and "</tool_call>" in content:
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                }
                messages.append(assistant_message)
                tool_error = False

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]
                if "python" in tool_call_text.lower():
                    try:
                        code_raw = (
                            content.split("<tool_call>")[1]
                            .split("</tool_call>")[0]
                            .split("<code>")[1]
                            .split("</code>")[0]
                            .strip()
                        )
                        result = await self.execute_python(code_raw)
                        if isinstance(result, str) and result.startswith(
                            (
                                "Python execution error:",
                                "PythonInterpreter tool not available",
                                "PythonInterpreter tool is not callable",
                            )
                        ):
                            tool_error = True
                    except Exception:
                        result = (
                            "[Python Interpreter Error]: Python code formatting error."
                        )
                        tool_error = True
                else:
                    try:
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        result = await self.custom_call_tool(tool_name, tool_args)
                    except Exception:
                        result = "[Json Parse Error]: Tool call is not a valid JSON."
                        tool_error = True

                if tool_error:
                    assistant_message["step_error"] = True

                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})
                if assistant_message["step_error"]:
                    consecutive_bad_steps += 1
                else:
                    consecutive_bad_steps = 0
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

            elif "<response>" in content and "</response>" in content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": False,
                    }
                )
                prediction = content.split("<response>")[1].split("</response>")[0].strip()
                termination = "answer"
                consecutive_bad_steps = 0
                break

            else:
                is_repetitive = analyze_repetition_ngram(content)
                is_overlong = count_words(content) > 2500
                if is_repetitive and is_overlong:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.strip(),
                            "step_error": True,
                        }
                    )
                    prediction = "Repetition response"
                    termination = "repetition_detected"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

                observation = "Error: Invalid content format. Content must contain <tool_call> or <response> tags. Let's try again."
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": True,
                    }
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
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

            if num_llm_calls_available <= 0 and "<answer>" not in content and "<response>" not in content:
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

        last_message_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        if last_message_content and "<response>" in last_message_content:
            prediction = last_message_content.split("<response>")[1].split("</response>")[0].strip()
            termination = "answer"
        elif last_message_content and "<answer>" in last_message_content:
            prediction = last_message_content.split("<answer>")[1].split("</answer>")[0].strip()
            termination = "answer"
        else:
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

        print("\n DeepResearch completed:")
        print(f"   Rounds: {round}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(
            "   Token usage: prompt={prompt}, completion={completion}, max_prompt={max_prompt}".format(
                prompt=self.total_prompt_tokens,
                completion=self.total_completion_tokens,
                max_prompt=self.max_prompt_tokens,
            )
        )
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """
        Execute tool calls with the available tools.

        Visual tools (crop, layout_parsing, image_search, perspective_correct,
        super_resolution, sharpen) receive extra context via **ctx so they can
        manage the shared ``image_paths`` dictionary.
        """
        VISUAL_TOOLS = {
            "crop", "layout_parsing", "text_search", "image_search",
            "web_search", "perspective_correct", "super_resolution", "sharpen",
        }

        if tool_name in self.tools:
            try:
                ctx = {}
                if tool_name in VISUAL_TOOLS:
                    ctx["image_paths"] = getattr(self, "_image_paths", {})
                    ctx["intermediate_dir"] = getattr(
                        self, "_intermediate_dir", "/tmp/vdr_tools"
                    )

                if hasattr(self.tools[tool_name], "call"):
                    if asyncio.iscoroutinefunction(self.tools[tool_name].call):
                        result = await self.tools[tool_name].call(**tool_args, **ctx)
                    else:
                        result = self.tools[tool_name].call(**tool_args, **ctx)
                elif callable(self.tools[tool_name]):
                    result = self.tools[tool_name](**tool_args, **ctx)
                else:
                    result = f"Tool {tool_name} is not callable"

                return str(result)

            except Exception as e:
                return f"Error calling tool {tool_name}: {e}"
        else:
            available_tools = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available_tools}"

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

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
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
        return await self._run(question, answer, images, image_path, **kwargs)


DeepResearchAgent = MultiTurnReactAgent
