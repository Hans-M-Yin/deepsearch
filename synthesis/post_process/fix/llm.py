"""Lazy model adapter; JSON requests are cached independently of the pipeline."""

import json
import logging
import re
from .storage import digest

JSON_REQUIREMENT = """Return exactly one valid JSON object matching the requested schema. Use double-quoted keys and strings, escape embedded quotes, backslashes and newlines inside strings, and include all required colons and commas. Do not include comments, trailing commas, Markdown fences or explanatory text outside the JSON object."""

logger = logging.getLogger(__name__)


class JsonModel:
    def __init__(
        self, cache, max_tokens=6000, gpt_reasoning_effort="xhigh", json_retries=2
    ):
        self.cache = cache
        self.max_tokens = max_tokens
        self.gpt_reasoning_effort = gpt_reasoning_effort
        if not isinstance(json_retries, int) or json_retries < 0:
            raise ValueError("json_retries must be a nonnegative integer")
        self.json_retries = json_retries

    def ask(self, alias, system, payload, images=()):
        system = system + "\n\n" + JSON_REQUIREMENT
        # File bytes participate in the cache key, not only mutable filenames.
        import hashlib
        from pathlib import Path

        fingerprints = [
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images
        ]
        inputs = {
            "model": alias,
            "system": system,
            "payload": payload,
            "images": fingerprints,
            "max_tokens": self.max_tokens,
            "gpt_reasoning_effort": self.gpt_reasoning_effort,
        }

        def compute():
            from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
            from synthesis.post_process.expand_trajectory_reasoning import (
                _image_content_part,
            )

            parts = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            for i, path in enumerate(images):
                part = _image_content_part(path)
                if part is None:
                    raise ValueError(f"Cannot encode image: {path}")
                parts.extend([{"type": "text", "text": f"IMAGE {i}:"}, part])
            messages = [
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=parts),
            ]
            for attempt in range(self.json_retries + 1):
                response = LLM_WORKER.generate(
                    ModelRequest(
                        model=alias,
                        messages=list(messages),
                        temperature=None,
                        max_tokens=self.max_tokens,
                        metadata={
                            "trace_label": "i2i_fix",
                            "session_id": digest(inputs)[:24],
                            "reasoning_effort": self.gpt_reasoning_effort,
                            "json_attempt": attempt + 1,
                        },
                    )
                )
                raw = response.content.strip()
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
                try:
                    obj = json.loads(raw)
                    if not isinstance(obj, dict):
                        raise ValueError("Model must return a JSON object")
                except ValueError as exc:
                    if attempt == self.json_retries:
                        raise
                    logger.warning(
                        "Invalid JSON from %s; retry %d/%d: %s",
                        alias,
                        attempt + 1,
                        self.json_retries,
                        exc,
                    )
                    # Repeat only this request, retaining images and task context.
                    # Do not accumulate potentially long malformed responses.
                    messages = messages[:2] + [
                        ModelMessage(
                            role="user",
                            content=f"The previous response could not be parsed as a JSON object: {exc}. Regenerate the complete response for the original task as valid JSON. {JSON_REQUIREMENT}",
                        )
                    ]
                    continue
                return obj

        return self.cache.get_or_compute("llm", inputs, compute)
