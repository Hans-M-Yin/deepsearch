"""Backend-only title search; preserve true resources in the common Candidate shape."""

from dataclasses import asdict
from .schema import Candidate
from .storage import digest


def recover_anchor(historical, window, anchor, record, backend, model, config, source):
    """Try title search only for an unavailable historical anchor; verify before binding."""
    from copy import deepcopy

    if len(anchor.get("anchor_image_ids", [])) > 1:
        results = [
            recover_anchor(
                historical,
                window,
                {**anchor, "anchor_image_id": rid, "anchor_image_ids": [rid]},
                record,
                backend,
                model,
                config,
                source,
            )
            for rid in anchor["anchor_image_ids"]
        ]
        return {
            "status": "unresolved"
            if any(r["status"] == "unresolved" for r in results)
            else "recovered"
            if any(r["status"] == "recovered" for r in results)
            else "not_needed",
            "references": results,
        }
    from .provenance import page_alignment
    from .visual_evidence import verify_pool

    target = next(
        c for c in historical if c.key.endswith("old_" + anchor["anchor_image_id"])
    )
    if target.image_paths or target.resources:
        return {"status": "not_needed"}
    candidates = search_title(backend, target.public["title"])
    aligned = [c for c in candidates if page_alignment(target.public, c, record)]
    aligned.sort(key=lambda c: c.public["image_id"] != target.public["image_id"])
    # Prefer the old source, but a mirror may host the identical photograph.
    # Verify every returned candidate; provenance alone must not exclude a match.
    checked = aligned + [c for c in candidates if c not in aligned]
    failures = verify_pool(checked, source, window, anchor, backend, model, config)
    viewed = anchor.get("viewed_before_conclusion_ids", [])

    def acceptable(c):
        if not c.image_paths:
            return False
        if c.category == "CONSISTENT" and c.verdict == "YES":
            return True
        # A different photo may only be bound to the old page when its real
        # provenance matches. Text-only references need not depict the target.
        if c not in aligned:
            return False
        if c.category == "SAME_ENTITY":
            return True
        return anchor["anchor_image_id"] not in viewed and c.category in {
            "RELATED",
            "SIMILAR",
            "UNRELATED",
        }

    accepted = next(
        (c for c in checked if acceptable(c)),
        None,
    )
    audit = {
        "status": "recovered" if accepted else "unresolved",
        "query": target.public["title"],
        "returned": len(candidates),
        "source_aligned": len(aligned),
        "checked": [asdict(c) for c in checked],
        "failures": failures,
    }
    if accepted is not None:
        recovered = deepcopy(accepted)
        recovered.key = "recovered_" + target.key
        recovered.historical = True
        # Restore the historical card/page, using a verified replacement image.
        # Private resources retain the actual retrieval URL; never relabel that URL
        # as belonging to the old page. Existing page observations remain unchanged.
        recovered.public = {**target.public, "image_id": accepted.public["image_id"]}
        historical[historical.index(target)] = recovered
        audit["selected_key"] = accepted.key
        audit["binding_basis"] = (
            "source_aligned_reference_image"
            if accepted in aligned and accepted.category != "CONSISTENT"
            else "source_and_visual_match"
            if accepted in aligned
            else "visual_match_original_page_retained"
        )
        audit["retrieved_card"] = accepted.public
        audit["retained_card"] = recovered.public
    return audit


def search_title(backend, title, top_k=10):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Missing historical title")
    inputs = {"query": title.strip(), "top_k": top_k, "version": 1}

    def compute():
        from synthesis.sft.api_tools import execute_tool_call

        identity = "title_recovery_" + digest(inputs)[:20]
        context = backend._context(identity)
        result = execute_tool_call(
            "t2i_search",
            {
                "query": inputs["query"],
                "top_k": top_k,
            },
            context,
        )
        if result.output.get("ok") is False:
            raise RuntimeError("Title recovery search failed")
        candidates = []
        for index, item in enumerate(result.output.get("results", [])):
            public = {
                key: item[key]
                for key in ("title", "source", "image_id", "source_page_id")
                if key in item
            }
            if not all(public.get(k) for k in ("image_id", "source_page_id")):
                raise ValueError("Title result lacks image/page resource IDs")
            resources = []
            for key in ("image_id", "source_page_id"):
                resource = context.resolve_resource_id(public[key])
                if resource is None:
                    raise ValueError("Unresolved title search resource")
                resources.append(asdict(resource))
            candidates.append(
                asdict(Candidate(f"{identity}_{index}", public, resources))
            )
        return {"candidates": candidates, "query": inputs["query"]}

    saved = backend.cache.get_or_compute("title_recovery", inputs, compute)
    return [Candidate(**candidate) for candidate in saved["candidates"]]
