"""Deterministic randomization constrained by available visual evidence."""

import random
from itertools import combinations
from .schema import Plan
from .storage import digest
from .matching import same_metadata


def rng_for(seed, record_hash, call_index):
    return random.Random(int(digest([seed, record_hash, call_index])[:16], 16))


def choose(rng, weights, allowed):
    names = [name for name in weights if name in allowed and weights[name] > 0]
    if not names:
        raise ValueError("No feasible enabled branch")
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]


def deduplicate(candidates):
    result, seen = [], set()
    for c in candidates:
        urls = [
            r.get("image_url") or r.get("primary_url")
            for r in c.resources
            if r.get("kind") == "image"
        ]
        signature = urls[0] if urls else c.public.get("image_id")
        if signature not in seen:
            seen.add(signature)
            result.append(c)
    return result


def inspectable_negative(candidate):
    return (
        candidate.verdict == "NO"
        and candidate.category in {"SIMILAR", "RELATED"}
        and candidate.text_competitive is not False
    )


def sample_negatives(candidates, count, rng, config):
    """Prefer confusing images without imposing a fixed trajectory inspection order."""
    groups = {
        name: [c for c in candidates if c.category == name and inspectable_negative(c)]
        for name in ("SIMILAR", "RELATED")
    }
    result = []
    for _ in range(count):
        name = (
            "SIMILAR"
            if rng.random() < config.similar_negative_probability
            else "RELATED"
        )
        if not groups[name]:
            name = next((n for n in ("SIMILAR", "RELATED") if groups[n]), name)
        if not groups[name]:
            raise ValueError("Not enough classified negatives")
        result.append(groups[name].pop(rng.randrange(len(groups[name]))))
    return result


def controlled_set(historical, pool, positive, required, config, rng):
    """Choose only feasible replacement/count combinations; never fill with extra positives."""
    choices = []
    others = [c for c in historical if c.key != positive.key]
    for count in config.replacement_counts:
        if count == 0:
            if all(
                c.verdict in {"YES", "NO"} and c.image_paths for c in historical
            ) and all(c.key in {h.key for h in historical} for c in required):
                choices.append((0, None, historical, [], [], 0, 0))
            continue
        for subset in combinations(others, 4 - count):
            kept = [positive, *subset]
            if any(c.verdict not in {"YES", "NO"} for c in kept):
                continue
            ids = {c.public["image_id"] for c in kept}
            available = deduplicate(
                [c for c in pool if c.public["image_id"] not in ids]
            )
            good = [c for c in available if c.verdict == "YES"]
            bad = [c for c in available if inspectable_negative(c)]
            forced = [c for c in required if c.public["image_id"] not in ids]
            forced_ids = {c.public["image_id"] for c in forced}
            if not forced_ids <= {c.public["image_id"] for c in bad}:
                continue
            bad = forced + [c for c in bad if c.public["image_id"] not in forced_ids]
            for slots in config.positive_slots:
                ng = slots - sum(c.verdict == "YES" for c in kept)
                nb = count - ng
                if 0 <= ng <= len(good) and len(forced) <= nb <= len(bad):
                    choices.append((count, slots, kept, good, bad, ng, nb))
    if not choices:
        # Keep a real five-card observation without manufacturing extra negatives.
        usable = deduplicate(
            [positive]
            + [
                c
                for c in historical + pool
                if c.key != positive.key
                and c.image_paths
                and c.verdict in {"YES", "NO", "CONTINUE"}
            ]
        )
        rest = [
            c for c in usable if c.public["image_id"] != positive.public["image_id"]
        ]
        negatives = [c for c in rest if inspectable_negative(c)]
        selected = sample_negatives(negatives, min(2, len(negatives)), rng, config)
        remainder = [c for c in rest if c.key not in {s.key for s in selected}]
        if positive.verdict != "YES" or not positive.image_paths:
            raise ValueError("No feasible verified candidate composition")
        first = (
            [positive]
            + selected
            + rng.sample(remainder, min(len(remainder), 4 - len(selected)))
        )
        rng.shuffle(first)
        return first, {
            "composition_mode": "lightweight",
            "fallback_reason": "insufficient controlled candidates",
        }
    # Sample replacement count first, avoiding combinatorial bias toward one count.
    count = rng.choice(sorted({choice[0] for choice in choices}))
    options = [c for c in choices if c[0] == count]
    count, slots, kept, good, bad, ng, nb = rng.choice(options)
    if count == 0:
        return list(kept), {"replacement_count": 0, "composition_mode": "unchanged"}
    forced = [
        c
        for c in required
        if c.public["image_id"] not in {k.public["image_id"] for k in kept}
    ]
    extra_bad = [
        c
        for c in bad
        if c.public["image_id"] not in {f.public["image_id"] for f in forced}
    ]
    result = (
        kept
        + rng.sample(good, ng)
        + forced
        + sample_negatives(extra_bad, nb - len(forced), rng, config)
    )
    rng.shuffle(result)
    return result, {
        "replacement_count": count,
        "composition_mode": "controlled",
        "positive_count": slots,
    }


def build_plan(window, historical, pool, anchor, convergence, config, rng, second=None):
    if len(anchor.get("anchor_image_ids", [])) == 2:
        from .dual_reference import build_dual_plan

        return build_dual_plan(window, historical, pool, anchor, config, rng)
    positive = next(
        c for c in historical if c.key.endswith("old_" + anchor["anchor_image_id"])
    )
    good = deduplicate([positive] + [c for c in pool if c.verdict == "YES"])
    good = [c for c in good if c.verdict == "YES"]
    negatives = deduplicate([c for c in pool if inspectable_negative(c)])
    # Shuffle before sorting so equally rated negatives do not impose a fixed order.
    rng.shuffle(negatives)
    hard = [c for c in negatives if c.category in {"SIMILAR", "RELATED"}]
    allowed = set()
    if convergence.get("eligible"):
        allowed.add("convergent")
    if positive.verdict == "YES" and historical:
        allowed.add("direct")
        if hard:
            allowed.add("contrast")
        if len(negatives) >= 2:
            allowed.add("recovery")
    branch = choose(rng, config.branch_weights, allowed)
    if branch == "convergent":
        return Plan(
            branch,
            historical,
            [],
            metadata={"convergence": convergence, "replacement_count": 0},
        )
    protected = [
        c
        for c in historical
        if any(
            c.public.get(k) in window.protected_ids
            for k in ("image_id", "page_id", "source_page_id")
        )
    ]
    for c in protected:
        if c.key != positive.key:
            raise ValueError("Continuation depends on another historical candidate")
    if branch == "recovery":
        attempts = hard[:2]
        first = attempts + sample_negatives(
            [c for c in negatives if c.key not in {a.key for a in attempts}],
            min(3, len(negatives) - 2),
            rng,
            config,
        )
        rng.shuffle(first)
        return Plan(
            branch,
            first,
            attempts,
            retry_call=window.call,
            retry_after=2,
            positive_key=positive.key,
            metadata={"needs_second_search": True},
        )
    wrong_n = 0
    if branch == "contrast":
        possible = {k for k in config.wrong_count_weights if int(k) <= len(hard)}
        wrong_n = int(choose(rng, config.wrong_count_weights, possible))
    attempts = rng.sample(hard[: max(4, wrong_n)], wrong_n) if wrong_n else []
    first, composition = controlled_set(
        historical, pool, positive, attempts, config, rng
    )
    return Plan(
        branch,
        first,
        attempts + [positive],
        positive_key=positive.key,
        metadata={**composition, "wrong_count": wrong_n},
    )


def finish_recovery(plan, second_pool, positive, rng, config=None):
    """The second observation must be backed by a second actual tool response."""
    matching = [
        c
        for c in second_pool
        if c.verdict == "YES" and same_metadata(c.public, positive.public)
    ]
    if not matching:
        raise ValueError("Retry did not recover the historical continuation candidate")
    recovered = matching[0]
    # Keep the original page reference; the image check uses the real retry image.
    for field in ("page_id", "source_page_id"):
        if positive.public.get(field):
            recovered.public[field] = positive.public[field]
    from .schema import Config

    config = config or Config()
    rest = deduplicate([c for c in second_pool if c.key != recovered.key])
    good = [c for c in rest if c.verdict == "YES"]
    bad = [c for c in rest if inspectable_negative(c)]
    feasible = [
        n for n in config.positive_slots if len(good) >= n - 1 and len(bad) >= 5 - n
    ]
    if not feasible:
        usable = [
            c
            for c in rest
            if c.image_paths
            and c.verdict in {"YES", "NO", "CONTINUE"}
            and c.public["image_id"] != recovered.public["image_id"]
        ]
        plan.second = [recovered] + rng.sample(usable, min(4, len(usable)))
        rng.shuffle(plan.second)
        plan.attempts.append(recovered)
        plan.positive_key = recovered.key
        plan.metadata.update(
            needs_second_search=False, retry_composition_mode="lightweight"
        )
        plan.metadata.pop("retry_positive_count", None)
        return plan
    slots = rng.choice(feasible)
    plan.second = (
        [recovered]
        + rng.sample(good, slots - 1)
        + sample_negatives(bad, 5 - slots, rng, config)
    )
    rng.shuffle(plan.second)
    plan.attempts.append(recovered)
    plan.positive_key = recovered.key
    plan.metadata["needs_second_search"] = False
    plan.metadata["retry_positive_count"] = slots
    return plan
