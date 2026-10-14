"""Audit original search progress and retain every relied-on reference."""

from .errors import SkipAugmentation
from .trajectory_index import (
    messages,
    content,
    candidate_read_occurrences,
    role,
    ASSISTANT,
)
import re

PROMPT = """Audit whether this reverse-image search actually advanced the existing
trajectory: identification of a particular person/object/place or an important
lead used downstream. Do not solve the question anew. Return JSON:
{"useful":true|false, "safe":true|false,
 "anchor_image_ids":["ALL historical image IDs actually relied on or cited"],
 "dependency_image_ids":["additional cards supplying essential downstream evidence; empty if none"],
 "viewed_before_conclusion_ids":["IDs actually viewed before adopting the conclusion"],
 "target":"answer-free local visual objective", "entity":"adopted entity",
 "reason":"explanation"}.
If no important progress came from these results, return useful=false and empty
IDs. Retain ALL relied-on references, including corroborating citations. Never
reduce three references to one or two. Rejected leads are not supporting sources.
A page URL or text is NOT a viewed image: there must be an actual image observation
before the conclusion. Only use actual_image_reads to identify possible image
checks, then check their chronology. The answer is not proof of correctness.
safe=false for contradictory/unresolvable dependencies. target describes the
pre-search objective without injecting identities or answers learned afterwards.
Distinguish identifying a subject from requiring exactly the same photograph.

FIELD CONTRACT -- these two lists answer DIFFERENT questions:
anchor_image_ids identifies SEARCH CARDS whose title, snippet, source-page text,
or actual image supplied an adopted lead. It does NOT mean images opened.
Copy the card's image_id even when ONLY its title or source page was used.
viewed_before_conclusion_ids identifies actual image observations before the
conclusion; this list MAY be empty when anchor_image_ids is nonempty.
For useful=true and safe=true, anchor_image_ids MUST be nonempty. Never leave
both lists empty merely because no candidate image was opened.
Example: card image_A links to page_A. The agent uses its title and reads page_A
to identify the target, without viewing an image: anchor_image_ids=["image_A"],
viewed_before_conclusion_ids=[]. Example: two titles jointly provide the adopted
identity: retain BOTH card image IDs even if neither image was opened.
Use candidate_reference_map to translate a cited page ID back to its card image
ID. Do not invent IDs or blindly select every result. If an image was opened
twice (e.g. overall composition, then signature), these are legitimate separate
observations: use their turn numbers and goals to assess chronology, not just
the last read. Include each candidate ID only once in the returned lists.

Always return dependency_image_ids, even when empty. Inspect the ENTIRE supplied
continuation, not only the first identification. Include additional cards whose
title, snippet, image_url_keywords, source_page_url_keywords or associated page
supplies essential terms or facts for a retained downstream search/action.
An identity anchor and an evidence dependency are different roles: another card
may identify the same entity yet uniquely supply a person's name, event, caption
or query phrase. Retain that evidence source too. For example, if an archive card
identifies the portrait but a second card's URL keywords supply the name used in
the next text search, include the second card in dependency_image_ids. Do not
invent missing evidence from world knowledge or the final answer. Do not include
every thematic lead indiscriminately; identify actual dependencies. Both lists
contain original card image IDs, never page IDs. Dependency protection has no
one/two-reference limit and takes priority over replacing cards with negatives.
"""


def assess_progress(record, window, source, model, config):
    payload = {
        "question": window.question,
        "history": [content(m) for m in messages(record)[: window.call_index + 1]],
        "search_call": window.call,
        "candidates": window.candidates,
        "candidate_reference_map": [
            {
                "image_id": c["image_id"],
                "title": c.get("title", ""),
                "page_ids": [c[k] for k in ("source_page_id", "page_id") if c.get(k)],
            }
            for c in window.candidates
        ],
        "actual_image_reads": {
            rid: {
                "turn": r["turn"],
                "observations": [
                    {"turn": o["turn"], "action": content(messages(record)[o["turn"]])}
                    for o in candidate_read_occurrences(r)
                ],
            }
            for rid, r in window.reads.items()
        },
        "continuation": [
            {"turn": i, "text": content(m)}
            for i, m in enumerate(messages(record))
            if i >= window.start
        ],
    }
    failures = []
    for attempt in range(config.progress_audit_retries + 1):
        result = model.ask(config.model, PROMPT, payload, [source])
        if isinstance(result, dict) and result.get("useful") is False:
            raise SkipAugmentation("no_search_progress", result)
        # A substantive unsafe decision is not a formatting error to retry away.
        if isinstance(result, dict) and result.get("safe") is False:
            raise ValueError("Unsafe progress audit: " + str(result))
        try:
            validate_progress(result, window)
            break
        except ValueError as exc:
            failures.append({"response": result, "error": str(exc)})
            if attempt == config.progress_audit_retries:
                raise ValueError(
                    "Invalid progress audit after bounded repair: " + str(failures)
                ) from exc
            payload = {
                **payload,
                "repair_attempt": attempt + 1,
                "invalid_response": result,
                "validation_error": str(exc),
                "repair_instruction": "Return the complete corrected JSON. Anchor IDs identify adopted search cards, NOT images viewed. If useful=true, trace the adopted title/page to the image_id in candidate_reference_map even without an image read. Keep viewed_before_conclusion_ids empty when no image was viewed. Do not invent dependencies or change useful/safe simply to evade validation.",
            }
    result["progress_audit_attempts"] = attempt + 1
    if failures:
        result["progress_audit_repairs"] = failures
    selected = result["anchor_image_ids"]
    # Actual downstream tool dependencies are not optional model judgments.
    required = set(selected) | set(result["dependency_image_ids"])
    # Also protect explicit prose citations, not only IDs used in tool calls.
    # Tool observations are excluded: a repeated result list is not a citation.
    continuation = "\n".join(
        content(m) for m in messages(record)[window.start :] if role(m) in ASSISTANT
    )
    # All original reads are retained by the new splicer.
    required.update(window.reads)
    for card in window.candidates:
        if any(
            card.get(k) in window.protected_ids
            or (
                isinstance(card.get(k), str)
                and card[k]
                and re.search(r"(?<!\w)" + re.escape(card[k]) + r"(?!\w)", continuation)
            )
            for k in ("image_id", "source_page_id", "page_id")
        ):
            required.add(card["image_id"])
    result["anchor_image_ids"] = [
        c["image_id"] for c in window.candidates if c["image_id"] in required
    ]
    result["anchor_image_id"] = result["anchor_image_ids"][0]
    # Both the recovery stage (anchor_image_ids) and composition stage
    # (required_reference_ids) must see the same complete preservation set.
    result["required_reference_ids"] = list(result["anchor_image_ids"])
    return result


def validate_progress(result, window):
    if not isinstance(result, dict):
        raise ValueError("Progress audit must return a JSON object")
    ids = {c["image_id"] for c in window.candidates}
    selected = result.get("anchor_image_ids")
    viewed = result.get("viewed_before_conclusion_ids")
    dependencies = result.get("dependency_image_ids")
    if (
        not isinstance(dependencies, list)
        or any(not isinstance(x, str) or x not in ids for x in dependencies)
        or len(set(dependencies)) != len(dependencies)
    ):
        raise ValueError(
            "dependency_image_ids must list unique existing card image IDs (or [])"
        )
    if result.get("useful") is True and selected == []:
        raise ValueError(
            "useful=true requires nonempty anchor_image_ids for adopted titles/pages, regardless of whether images were viewed"
        )
    if (
        result.get("useful") is not True
        or result.get("safe") is not True
        or not isinstance(selected, list)
        or not selected
        or any(not isinstance(x, str) or x not in ids for x in selected)
        or len(set(selected)) != len(selected)
        or not isinstance(viewed, list)
        or any(not isinstance(x, str) or x not in window.reads for x in viewed)
        or len(set(viewed)) != len(viewed)
        or not result.get("target")
        or not result.get("entity")
    ):
        raise ValueError("Invalid or unsafe progress audit: " + str(result))
