"""Identify the original continuation entity and assess metadata convergence."""

from .trajectory_index import messages, content
import re

ANCHOR_PROMPT = """Audit an existing image-search trajectory. Identify the exact
one or two historical candidates supporting the SAME entity the continuation relies on. The
original answer is NOT evidence that this candidate is correct. Return JSON:
{"anchor_image_ids": ["one or two existing image IDs"], "target": "question-specific visual target",
 "entity": "adopted entity", "safe": true, "reason": "..."}.
safe=false if no supporting candidate can be identified, more than two indispensable sources are required for distinct facts, the question/answer is
clearly inconsistent, or the continuation depends on several competing entities.
Two explicitly referenced sources agreeing on the same work/entity are safe;
do not reject merely because the continuation cites both. Do not arbitrarily drop
one of two relied-on references. Do not change the question or solve it."""

ANCHOR_PROMPT += """
Distinguish indispensable dependencies from incidental corroboration. Counting
titles mentioned in prose is NOT counting required anchors. Prefer the candidate
whose page/image is actually opened and used to establish the identification.
If that one source carries the identification and other titles merely agree,
select that primary source alone; explain which others were only corroborative.
If no primary source is established and two captions jointly supply the adopted
attribution, retain both. If three or more titles repeat the SAME attribution
without distinct downstream facts or tool dependencies, choose the strongest one
or two sufficient sources, explaining why the rest are redundant; do NOT reject
solely because three titles were mentioned. Never drop a source supplying a unique
fact or required downstream resource. Existing continuation claims about removed
corroborators will be edited to reflect only the retained evidence.
Example: an opened archival source identifies a president and a library search
title merely agrees: only the opened archival source is required, not both.
Example: Wikipedia, a magazine and a poster title all name the same two people:
this is redundant corroboration, not automatically three indispensable anchors.
"""

ANCHOR_PROMPT += """
The target must describe the LOCAL visual objective of this particular search,
grounded primarily in the assistant's pre-search reasoning and search arguments.
Use the continuation only to disambiguate which visual target was adopted, not
to inject a future answer or the adopted entity's name into target. Express target
as an answer-free instruction, e.g. 'identify the left driver marked MAX', NOT
'find his first-victory podium sponsor'. Do not advance to downstream subquestions.
If pre-search intent and the adopted target conflict, return safe=false."""

ANCHOR_PROMPT += """
Also return dependency_image_ids: all historical image IDs whose titles, URL
keywords, snippets or associated pages supply indispensable terms/facts for ANY
retained downstream action, even if they are not identity anchors. Trace search
query terms to their actual evidence. Do not drop a source merely because another
source identifies the same person. This separate dependency list may exceed two.
Do not infer a dependency from world knowledge or the final answer.
"""

ANCHOR_PROMPT += """
In target, distinguish the semantic objective (person identity, object instance,
category or exact photograph) from the retrieval method. If an exact-photo lookup
is used to identify a person, state BOTH purposes; do not replace the identity
objective with an unconditional requirement to find the original photograph.
"""

CONVERGENCE_PROMPT = """Assess whether the ORIGINAL five reverse-image results
support the same question-relevant entity WITHOUT opening candidate images.
Repeated copies of one source are not independent support. A match to the whole
photo does not identify a different small target in it. Examine the reference
image and question. Return JSON {"supporting_image_ids": [],
"independent_sources": 0, "competing_entity": false,
"target_unambiguous": false, "reason": "..."}.
Only count titles/snippets that actually identify the target entity. Mere title
similarity, the old answer, and the old assistant's confidence are insufficient."""


def select_anchor(record, window, model, config, source):
    decision = model.ask(
        config.model,
        ANCHOR_PROMPT,
        {
            "question": window.question,
            "pre_search_reasoning_and_call": content(
                messages(record)[window.call_index]
            ),
            "search_call": window.call,
            "candidates": window.candidates,
            "continuation": [content(m) for m in messages(record)[window.start :]],
        },
        [source],
    )
    ids = {c.get("image_id") for c in window.candidates}
    selected = decision.get("anchor_image_ids", [decision.get("anchor_image_id")])
    if (
        decision.get("safe") is not True
        or not isinstance(selected, list)
        or not 1 <= len(selected) <= 2
        or any(not isinstance(x, str) or x not in ids for x in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("No safely identified historical anchor: " + str(decision))
    decision["anchor_image_ids"] = selected
    decision["anchor_image_id"] = selected[0]
    dependencies = decision.get("dependency_image_ids", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(x, str) or x not in ids for x in dependencies
    ):
        raise ValueError("Invalid historical evidence dependencies")
    continuation = "\n".join(content(m) for m in messages(record)[window.start :])
    # Explicit resource references are dependencies even if the LLM overlooks
    # them. Preserve the whole original card, including its supporting text.
    referenced = [
        c["image_id"]
        for c in window.candidates
        if any(
            isinstance(c.get(key), str)
            and c[key]
            and re.search(r"(?<![\w])" + re.escape(c[key]) + r"(?![\w])", continuation)
            for key in ("image_id", "source_page_id")
        )
    ]
    decision["required_reference_ids"] = list(
        dict.fromkeys(selected + dependencies + referenced)
    )
    if not decision.get("entity") or not decision.get("target"):
        raise ValueError("Missing entity/target")
    return decision


def assess_convergence(window, anchor, model, config, source):
    # Only zero-view records can take the unchanged/no-view branch.
    if window.reads:
        return {
            "eligible": False,
            "reason": "Existing candidate image check; zero-view branch ineligible",
            "skipped_model_call": True,
        }
    result = model.ask(
        config.model,
        CONVERGENCE_PROMPT,
        {
            "question": window.question,
            "target": anchor["target"],
            "entity": anchor["entity"],
            "candidates": window.candidates,
        },
        [source],
    )
    ids = {c.get("image_id") for c in window.candidates}
    supported = set(result.get("supporting_image_ids", []))
    sources = {
        str(c.get("source", "")).strip().casefold()
        for c in window.candidates
        if c.get("image_id") in supported and c.get("source")
    }
    result["eligible"] = bool(
        len(window.candidates) == 5
        and supported <= ids
        and len(supported) >= config.convergence_min
        and set(anchor.get("anchor_image_ids", [anchor["anchor_image_id"]]))
        <= supported
        and (len(anchor.get("anchor_image_ids", [])) != 2 or len(supported) == 5)
        and len(sources) >= 2
        and result.get("independent_sources", 0) >= 2
        and result.get("competing_entity") is False
        and result.get("target_unambiguous") is True
        and not window.reads
    )
    return result
