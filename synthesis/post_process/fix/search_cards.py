"""Build a bounded search environment, not a prescribed inspection sequence."""

from .schema import Plan
from .candidate_relevance import visible_card
from .errors import SkipAugmentation
from .storage import digest
import random

POSITIVE = {"CONSISTENT", "SAME_ENTITY"}
SELECTABLE = POSITIVE | {"RELATED", "SIMILAR"}

COMPOSE = """Build a real reverse-image search result list of at most five cards.
Keep EVERY required key. Fill available slots from supplied real candidates;
Required UNRESOLVED cards are retained historical text/page references whose images are unavailable. Keep their text unchanged; they are neither verified positives nor selectable images and must not replace the required verified positive.
prefer varied plausible leads over redundant copies when available, but never
fabricate entries or an inspection order. Include at least one confirmed target,
preferring CONSISTENT over SAME_ENTITY if space permits. UNRELATED may remain as
background results. Do not force wrong inspections. Return JSON
{"candidate_keys":["keys in display order"], "reason":"composition rationale"}.
Use all available slots unless there are fewer distinct usable cards. Private
categories are for building this environment only, never public search text.
Copy candidate keys EXACTLY from the supplied candidates; never reconstruct IDs.
Respect positive_limit and minimum_negative_count when provided. Maximize RELATED/SIMILAR slots; use UNRELATED only as background filler when useful negatives are insufficient. Required references always take precedence over these preferences.
"""


def validate_composition(result, by_key, required_keys, count):
    """Reject malformed choices explicitly; never guess an intended resource ID."""
    keys = result.get("candidate_keys") if isinstance(result, dict) else None
    if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
        raise ValueError("candidate_keys must be a list of strings")
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        raise ValueError(f"Unknown candidate keys (copy exact supplied IDs): {unknown}")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate candidate keys")
    if len(keys) != count:
        raise ValueError(f"Expected {count} candidate keys, got {len(keys)}")
    missing = set(required_keys) - set(keys)
    if missing:
        raise ValueError(f"Missing required references: {sorted(missing)}")
    if not any(by_key[k].category in POSITIVE for k in keys):
        raise ValueError("Composed search list lacks a verified positive")
    return [by_key[k] for k in keys]


def compose_candidates(payload, by_key, model, config):
    """Locally repair malformed composition without repeating search or vision."""
    repairs = []
    for attempt in range(config.composition_retries + 1):
        result = model.ask(config.model, COMPOSE, payload)
        try:
            selected = validate_composition(
                result, by_key, payload["required_keys"], payload["result_count"]
            )
            if sum(c.category in POSITIVE for c in selected) > payload.get(
                "positive_limit", 5
            ):
                raise ValueError("Too many positive candidates")
            if sum(
                c.category in {"RELATED", "SIMILAR"} for c in selected
            ) < payload.get("minimum_negative_count", 0):
                raise ValueError(
                    "Prefer available RELATED/SIMILAR over redundant positives or UNRELATED"
                )
            return selected, repairs
        except ValueError as exc:
            repairs.append({"response": result, "error": str(exc)})
            if attempt == config.composition_retries:
                raise ValueError(
                    f"Composition invalid after {attempt + 1} attempts: {exc}"
                ) from exc
            payload = {
                **payload,
                "repair_attempt": attempt + 1,
                "previous_invalid_response": result,
                "validation_error": str(exc),
                "allowed_candidate_keys": list(by_key),
                "repair_instruction": "Return the complete corrected candidate_keys list. Copy exact allowed IDs, keep every required reference and a positive candidate. Do not invent or approximate keys.",
            }


def build_search_plan(historical, pool, anchor, window, model, config):
    required_ids = anchor.get("required_reference_ids", anchor["anchor_image_ids"])
    required = [
        c
        for c in historical
        if any(c.key.endswith("old_" + rid) for rid in required_ids)
    ]
    if len(required) != len(required_ids):
        raise SkipAugmentation("missing_required_reference")
    viewed = set(anchor.get("viewed_before_conclusion_ids", []))
    for c in required:
        unresolved = not c.image_paths or c.category not in SELECTABLE | {"UNRELATED"}
        was_viewed = c.originally_viewed or any(
            c.key.endswith("old_" + rid) for rid in viewed
        )
        if unresolved and was_viewed:
            raise SkipAugmentation("unresolved_required_reference")
        # Preserve the real historical text/page, not a fabricated image binding.
        # Failed recovery does not invalidate a reference that was only read as text.
        c.text_reference_only = unresolved
        if unresolved:
            c.category = "UNRESOLVED"
            c.verdict = "UNCERTAIN"
    # Historical dependencies win over replay duplicates with the same image ID.
    candidates = list(required)
    seen = {c.public["image_id"] for c in candidates}
    if len(seen) != len(candidates):
        raise SkipAugmentation("required_reference_id_collision")
    for c in pool:
        if (
            c.category in SELECTABLE | {"UNRELATED"}
            and c.image_paths
            and c.public["image_id"] not in seen
        ):
            candidates.append(c)
            seen.add(c.public["image_id"])
    if any(
        (c.category not in SELECTABLE | {"UNRELATED"} or not c.image_paths)
        and not c.text_reference_only
        for c in required
    ):
        raise SkipAugmentation("unresolved_required_reference")
    if len(required) > config.result_count:
        raise SkipAugmentation("too_many_required_references")
    if not any(c.category in POSITIVE for c in candidates):
        raise SkipAugmentation("no_verified_positive")
    # Measure natural convergence before imposing an artificial positive cap.
    natural_convergent = len(candidates) >= 5 and all(
        c.category in POSITIVE for c in candidates
    )
    required_keys = {c.key for c in required}
    required_positive = sum(c.category in POSITIVE for c in required)
    positive_limit = max(2, required_positive)
    if not natural_convergent:
        extra_positives = sorted(
            (
                c
                for c in candidates
                if c.key not in required_keys and c.category in POSITIVE
            ),
            key=lambda c: (c.category != "CONSISTENT", c.key),
        )[: max(0, positive_limit - required_positive)]
        keep_positive = required_keys | {c.key for c in extra_positives}
        candidates = [
            c
            for c in candidates
            if c.category not in POSITIVE or c.key in keep_positive
        ]
    by_key = {c.key: c for c in candidates}
    count = min(config.result_count, len(candidates))
    required_negative = sum(c.category in {"RELATED", "SIMILAR"} for c in required)
    available_negative = sum(
        c.category in {"RELATED", "SIMILAR"} and c.key not in required_keys
        for c in candidates
    )
    negative_minimum = required_negative + min(
        available_negative, max(0, count - len(required) - int(required_positive == 0))
    )
    composition_repairs = []
    if len(candidates) <= count:
        selected = candidates
    else:
        selected, composition_repairs = compose_candidates(
            {
                "question": window.question,
                "local_visual_target": anchor["target"],
                "trajectory_history": anchor.get("pre_search_context", []),
                "required_keys": [c.key for c in required],
                "result_count": count,
                "positive_limit": config.result_count
                if natural_convergent
                else positive_limit,
                "minimum_negative_count": negative_minimum,
                "candidates": [
                    {"key": c.key, "card": visible_card(c), "category": c.category}
                    for c in candidates
                ],
            },
            by_key,
            model,
            config,
        )
    if not any(c.category in POSITIVE for c in selected):
        raise ValueError("Composed search list lacks a verified positive")
    # The composer sees private labels; its ranking must not leak those labels
    # through a systematically positive-first public result list.
    order_seed = digest(
        {
            "seed": config.seed,
            "question": window.question,
            "keys": sorted(c.key for c in selected),
        }
    )
    selected = sorted(selected, key=lambda c: c.key)
    random.Random(order_seed).shuffle(selected)
    for c in selected:
        c.public = visible_card(c)
        if c.category == "SAME_ENTITY":
            c.verdict = "YES"
    return Plan(
        "direct",
        selected,
        [],
        metadata={
            "flow_version": "selector_writer_v1",
            "composition_mode": "natural",
            "composition_repairs": composition_repairs,
            "natural_convergent": natural_convergent,
            "positive_limit": positive_limit,
            "display_order": "seeded_label_independent_shuffle",
            "retained_reference_keys": [c.key for c in required],
            "text_only_reference_keys": [
                c.key for c in required if c.text_reference_only
            ],
            "local_visual_target": anchor["target"],
        },
    )


def is_convergent(plan):
    """Exactly five independently classified cards all match the local target."""
    return len(plan.first) == 5 and all(c.category in POSITIVE for c in plan.first)
