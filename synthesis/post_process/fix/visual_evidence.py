"""Evidence labels are private planning data, never search-result fields."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from .timing import measure
from .evidence_policy import TARGET_EVIDENCE

CLASSIFICATION = """Classify visual content relative to IMAGE 0 and its local target:
CONSISTENT: same photograph/content and same target; resizing, cropping or image
quality changes are allowed, provided the target remains visible and verifiable.
SIMILAR: card text cannot safely rule out usefulness for the current search,
and the image has high visual-content similarity, but is a different target.
Give shared visual features and discriminative evidence for the nonmatch.
RELATED: card text cannot safely rule out usefulness for the current search,
but inspection shows low visual-content similarity and a different target.
UNRELATED: the visible card text ALONE already provides sufficient reason to
exclude the result as irrelevant to the question/current local search objective.
This is a text-based exclusion, NOT merely a different event/place/person name.
Never use the adopted answer or image inspection to invent a pre-view exclusion.
If text is vague or insufficient, use RELATED or SIMILAR for verified negatives.
Judge visual similarity from shapes, layout, appearance and image content, not
semantic titles or inferred historical identities. Rotated/cropped/resized views
of the same photograph with a verifiable target are CONSISTENT, even if the title
is misleading. Positive visual correspondence takes precedence over text exclusion.
If no card text is supplied, UNRELATED cannot be established; classify verified
negatives by visual similarity as RELATED or SIMILAR instead.
Different pose, background, clothes or viewpoint alone does NOT prove a different
person/object. Use auxiliary state SAME_ENTITY for another photograph of a
visually confirmed same target. Use DIFFERENT_PHOTO when clearly not the same
photograph, but identity cannot be established. Neither state is an identity
negative. Use UNRESOLVED if even photograph correspondence cannot be judged,
detail is inadequate or the target is omitted. No numerical confidence scores.
Metadata and names are not visual proof; do not infer labels from desired branches.
Return category and a nonempty evidence list of concrete visible comparisons."""

CLASSIFICATION += TARGET_EVIDENCE

PROMPT = (
    "Compare IMAGE 0 with the candidate image(s). "
    + CLASSIFICATION
    + """
Return JSON {"category":"CONSISTENT|SIMILAR|RELATED|UNRELATED|SAME_ENTITY|DIFFERENT_PHOTO|UNRESOLVED",
"evidence":["visible comparisons"]}."""
)


def verify(
    candidate,
    source,
    question,
    target,
    entity,
    backend,
    model,
    config,
    region=None,
    trajectory_history=None,
):
    backend.read(candidate)
    from PIL import Image

    for path in candidate.image_paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if min(image.size) < 24:
                raise ValueError("Candidate image too small to verify")
    result = model.ask(
        config.verifier_model,
        PROMPT,
        {
            "question": question,
            "target": target,
            "trajectory_history": trajectory_history or [],
            "query_region_xyxy_1000": region,
            "candidate_title": candidate.public.get("title", ""),
            "candidate_card": candidate.public,
        },
        [source, *candidate.image_paths],
    )
    category = result.get("category")
    evidence = result.get("evidence", [])
    labels = {
        "CONSISTENT": "YES",
        "SIMILAR": "NO",
        "RELATED": "NO",
        "UNRELATED": "NO",
        "UNRESOLVED": "UNCERTAIN",
        "SAME_ENTITY": "CONTINUE",
        "DIFFERENT_PHOTO": "CONTINUE",
    }
    if category not in labels:
        raise ValueError("Invalid visual category")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(x, str) and x.strip() for x in evidence)
    ):
        raise ValueError("Missing visual comparison evidence")
    candidate.category = category
    candidate.verdict = (
        "YES"
        if category == "SAME_ENTITY" and candidate.originally_viewed
        else labels[category]
    )
    candidate.evidence = evidence
    return candidate


def verify_pool(candidates, source, window, anchor, backend, model, config):
    # Do not convert unavailable images into negatives.
    def check(candidate):
        with measure("visual_slot_wait", candidate=candidate.key):
            slots = getattr(backend, "visual_slots", None)
            if slots is not None:
                slots.acquire()
        try:
            return check_acquired(candidate)
        finally:
            if slots is not None:
                slots.release()

    def check_acquired(candidate):
        try:
            with measure("visual_candidate", candidate=candidate.key):
                verify(
                    candidate,
                    source,
                    window.question,
                    anchor["target"],
                    anchor["entity"],
                    backend,
                    model,
                    config,
                    window.call["arguments"].get("region"),
                    anchor.get("pre_search_context", []),
                )
        except Exception as exc:
            candidate.verdict = "UNCERTAIN"
            candidate.category = "UNRESOLVED"
            return {"candidate": candidate.key, "error": str(exc)}
        return None

    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=config.visual_check_workers) as executor:
        # Every worker gets its own Context, sharing only the thread-safe timer.
        futures = [executor.submit(copy_context().run, check, c) for c in candidates]
        # Preserve input order regardless of completion order.
        return [
            failure for future in futures if (failure := future.result()) is not None
        ]
