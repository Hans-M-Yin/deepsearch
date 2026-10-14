"""Composition for two preserved, independently verified historical references."""

from itertools import combinations
from .schema import Plan


def build_dual_plan(window, historical, pool, anchor, config, rng):
    from .candidate_sets import inspectable_negative, choose, sample_negatives

    ids = anchor["anchor_image_ids"]
    refs = [next(c for c in historical if c.key.endswith("old_" + rid)) for rid in ids]
    if any(c.verdict != "YES" or not c.image_paths for c in refs):
        raise ValueError(
            "Both historical references must have verified matching images"
        )
    rest = [c for c in historical if c not in refs]
    choices = []
    for count in config.replacement_counts:
        if not 0 <= count <= 3:
            continue
        for kept in combinations(rest, 3 - count):
            if any(
                not c.image_paths or c.verdict not in {"YES", "NO", "CONTINUE"}
                for c in kept
            ):
                continue
            retained = refs + list(kept)
            # Never discard an original resource used by the continuation.
            if any(
                any(
                    c.public.get(k) in window.protected_ids
                    for k in ("image_id", "page_id", "source_page_id")
                )
                for c in rest
                if c not in kept
            ):
                continue
            seen = {c.public["image_id"] for c in retained}
            available = []
            for candidate in pool:
                rid = candidate.public["image_id"]
                if (
                    rid not in seen
                    and candidate.image_paths
                    and candidate.verdict in {"YES", "NO", "CONTINUE"}
                ):
                    available.append(candidate)
                    seen.add(rid)
            if len(available) >= count:
                choices.append((count, retained, available))
    if not choices:
        from .candidate_sets import deduplicate

        retained = refs + [
            c
            for c in rest
            if any(
                c.public.get(k) in window.protected_ids
                for k in ("image_id", "page_id", "source_page_id")
            )
        ]
        if any(
            not c.image_paths or c.verdict not in {"YES", "NO", "CONTINUE"}
            for c in retained
        ):
            raise ValueError("Required historical resource has no usable image")
        available = deduplicate(
            [
                c
                for c in rest + pool
                if c not in retained
                and c.image_paths
                and c.verdict in {"YES", "NO", "CONTINUE"}
                and c.public["image_id"] not in {x.public["image_id"] for x in retained}
            ]
        )
        choices.append((min(5 - len(retained), len(available)), retained, available))
    count = rng.choice(sorted({x[0] for x in choices}))
    _, retained, available = rng.choice([x for x in choices if x[0] == count])
    negatives = [c for c in available if inspectable_negative(c)]
    branch = choose(
        rng,
        config.branch_weights,
        {"direct", "contrast"} if negatives and count else {"direct"},
    )
    forced = (
        sample_negatives(negatives, min(count, len(negatives), 2), rng, config)
        if branch == "contrast"
        else []
    )
    rest_available = [c for c in available if c not in forced]
    first = retained + forced + rng.sample(rest_available, count - len(forced))
    rng.shuffle(first)
    return Plan(
        branch,
        first,
        [],
        positive_key=refs[-1].key,
        metadata={
            "composition_mode": "dual_reference",
            "replacement_count": count,
            "required_reference_keys": [c.key for c in refs],
            "anchor_image_ids": ids,
        },
    )
