"""Reject immutable answers that contradict accepted same-entity evidence."""

import re
from .trajectory_index import messages, content

PROMPT = """Check an immutable original answer against verified image evidence.
The listed accepted candidates depict the SAME ENTITY but a DIFFERENT PHOTOGRAPH.
Determine whether the answer asserts that those candidates are the same/exact
photograph, merely a crop, or have identical poses/composition. Such claims conflict
with the supplied evidence even if the final short answer is correct. Do not
confuse claims about a separate later image with this initial identification.
Do not change the answer or rationalize a conflict. Identity-only statements are
compatible. Return JSON {"compatible":true|false,"reason":"...",
"conflicting_quotes":["exact conflicting answer text, if any"]}.
"""


def check_answer(record, plan, model, config):
    accepted = [
        c for c in plan.attempts if c.verdict == "YES" and c.category == "SAME_ENTITY"
    ]
    if not accepted:
        return {"compatible": True, "skipped": True}
    answers = [
        a
        for m in messages(record)
        for a in re.findall(r"<answer>(.*?)</answer>", content(m), re.S)
    ]
    result = model.ask(
        config.judge_model,
        PROMPT,
        {
            "answers": answers,
            "accepted_same_entity_candidates": [
                {"card": c.public, "evidence": c.evidence} for c in accepted
            ],
        },
    )
    if type(result.get("compatible")) is not bool or not result.get("reason"):
        raise ValueError("Invalid immutable-answer consistency assessment")
    return result
