"""Mechanical assembly; no LLM is allowed to author tool observations."""

import copy
import json
import re
from .storage import digest
from .candidate_relevance import visible_card
from .trajectory_index import (
    messages,
    content,
    set_content,
    image_bindings,
    parse_observation,
    RESOURCE_FIELDS,
    call,
    role,
    ASSISTANT,
    ACTION,
)


def assistant(reasoning, tool=None):
    text = f"<thinking>\n{reasoning}\n</thinking>"
    if tool:
        text += (
            "\n<tool_call>\n" + json.dumps(tool, ensure_ascii=False) + "\n</tool_call>"
        )
    return {"from": "gpt", "value": text}


def public_results(candidates, namespace, preserve):
    """Namespace new IDs without touching historical page/image dependencies."""
    items, mapping = [], {}
    for rank, c in enumerate(candidates, 1):
        item = visible_card(c)
        for field in RESOURCE_FIELDS:
            old = item.get(field)
            if old and old not in preserve:
                mapping.setdefault(old, f"{namespace}_{field}_{digest(old)[:12]}")
                item[field] = mapping[old]
        if "rank" in item:
            item["rank"] = rank
        items.append(item)
    return items, mapping


def remap_text(text, mapping):
    # Exact identifier boundaries avoid replacing img_1 inside img_10.
    if not mapping:
        return text
    pattern = (
        r"(?<![\w])("
        + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        + r")(?![\w])"
    )
    return re.sub(pattern, lambda m: mapping[m.group(0)], text)


def assemble(record, window, plan, steps, patches):
    if plan.metadata.get("context_edit_scope") == "inserted_only" and (
        patches or plan.metadata.get("transition_includes_final_check")
    ):
        raise ValueError("Insertion-only editing cannot replace original reasoning")
    if plan.branch == "convergent":
        return copy.deepcopy(record), {"inserted": 0, "id_mapping": {}}
    output = copy.deepcopy(record)
    turns = messages(output)
    removed = {r["turn"] + delta for r in window.reads.values() for delta in (0, 1)}
    _, by_turn = image_bindings(record)
    namespace = "fix_" + digest([record, window.call_index])[:12]
    preserve = {c[k] for c in window.candidates for k in RESOURCE_FIELDS if c.get(k)}
    # URL-derived IDs can recur in later searches (e.g. the NGA page in sample 4).
    # Keep all original IDs stable, not just those in the replaced observation;
    # otherwise new cards rename a resource while retained observations do not.
    for message in messages(record):
        preserve.update(
            re.findall(r"\b(?:image|page|img)_[A-Za-z0-9_]+\b", content(message))
        )
        for candidate in plan.first + plan.second:
            for field in RESOURCE_FIELDS:
                rid = candidate.public.get(field)
                if rid and re.search(
                    r"(?<!\w)" + re.escape(rid) + r"(?!\w)", content(message)
                ):
                    preserve.add(rid)
    first, mapping = public_results(plan.first, namespace + "_s1", preserve)
    second, second_map = public_results(plan.second, namespace + "_s2", preserve)
    # Repeated raw URL IDs may recur across searches: keep a stable mapping.
    for old, new in second_map.items():
        if old in mapping:
            second = [
                {k: mapping[old] if v == new else v for k, v in item.items()}
                for item in second
            ]
        else:
            mapping[old] = new
    observation = copy.deepcopy(window.observation)
    field = "matches" if "matches" in observation else "results"
    observation[field] = first
    if "top_k" in observation:
        observation["top_k"] = len(first)
    # Search results must not acquire private verdicts or synthetic tool claims.
    turns[window.observation_index] = set_content(
        turns[window.observation_index], json.dumps(observation, ensure_ascii=False)
    )
    inserted, new_images, accepted_image_id = [], [], None
    confirmed_views = {}
    candidate_by_key = {c.key: c for c in plan.attempts}
    pending = ""
    for step in steps:
        before = "\n".join(x for x in (pending, step["before"]) if x)
        if step["kind"] == "retry":
            inserted.append(assistant(before, plan.retry_call))
            retry_observation = {"ok": True, "top_k": len(second), field: second}
            inserted.append(
                {
                    "from": "observation",
                    "value": json.dumps(retry_observation, ensure_ascii=False),
                }
            )
            pending = ""
            continue
        candidate = candidate_by_key[step["candidate_key"]]
        rid = mapping.get(candidate.public["image_id"], candidate.public["image_id"])
        inserted.append(
            assistant(
                before,
                {
                    "name": "read_url",
                    "arguments": {
                        "resource_id": rid,
                        "goal": "Compare the candidate with the target in the original image",
                    },
                },
            )
        )
        if len(candidate.image_paths) != 1:
            raise ValueError("Expected one image per candidate read")
        # Use the same public schema as real read_url output, with a unique image ID.
        image_id = namespace + "_view_" + str(len(new_images))
        inserted.append(
            {
                "from": "observation",
                "value": json.dumps({"image_id": image_id}) + "\n<image>",
            }
        )
        new_images.extend(candidate.image_paths)
        if candidate.verdict == "YES":
            confirmed_views[candidate.key] = image_id
        if candidate.key == plan.positive_key:
            accepted_image_id = image_id
        pending = step["after"]
    for i, patch in patches.items():
        turns[i] = patch
    # The transition is written using the final visual verdict; keep its evidence
    # explicitly present even if the continuation editor summarizes too aggressively.
    transition = content(turns[window.end])
    tag = "thinking" if "<thinking>" in transition else "think"
    if not plan.metadata.get("transition_includes_final_check"):
        transition = transition.replace(f"<{tag}>", f"<{tag}>\n{pending}\n", 1)
    turns[window.end] = set_content(turns[window.end], transition)
    removed_refs = {}
    changed_view_refs = set()
    for rid, read in window.reads.items():
        obs = parse_observation(read["observation"])
        old_ref = obs.get("image_id")
        if old_ref:
            matched = next(
                (
                    c
                    for c in plan.attempts
                    if c.key.endswith("old_" + rid) and c.verdict == "YES"
                ),
                None,
            )
            if rid != plan.metadata.get("anchor_image_id") and matched is None:
                # No later call may depend on a discarded original candidate view.
                if any(
                    old_ref in content(m)
                    for i, m in enumerate(turns)
                    if i >= window.end and i not in removed
                ):
                    raise ValueError(
                        "Continuation references a removed non-positive image"
                    )
            elif accepted_image_id:
                accepted = matched or next(
                    c for c in plan.attempts if c.key == plan.positive_key
                )
                removed_refs[old_ref] = confirmed_views.get(
                    accepted.key, accepted_image_id
                )
                from .storage import file_digest
                from .trajectory_index import resolve_path

                # Same entity does not imply the same photograph/crop coordinates.
                old_paths = (
                    plan.metadata.get("original_anchor_paths")
                    if rid == plan.metadata.get("anchor_image_id")
                    else None
                ) or [resolve_path(p, ".") for p in read["paths"]]
                if [file_digest(p) for p in old_paths] != [
                    file_digest(p) for p in accepted.image_paths
                ]:
                    changed_view_refs.add(old_ref)
    for i in range(window.end, len(turns)):
        if i in removed or role(turns[i]) not in ASSISTANT:
            continue
        action = call(content(turns[i]))
        if action and any(ref in json.dumps(action) for ref in changed_view_refs):
            raise ValueError(
                "Downstream tool depends on the original photograph, not just its entity"
            )
    for i in range(window.end, len(turns)):
        turns[i] = set_content(turns[i], remap_text(content(turns[i]), removed_refs))
    source_repairs = {}
    bindings, _ = image_bindings(record)
    for i in range(window.end, len(turns)):
        action = call(content(turns[i])) if role(turns[i]) in ASSISTANT else None
        if (
            action
            and action.get("name") == "i2i_search"
            and not (action["arguments"].get("image") or action["arguments"].get("url"))
        ):
            last_visible = max(
                (j for j, paths in by_turn.items() if j < i and paths), default=-1
            )
            for original_turn in messages(record)[last_visible + 1 : i]:
                original_call = (
                    call(content(original_turn))
                    if role(original_turn) in ASSISTANT
                    else None
                )
                if (
                    original_call
                    and original_call.get("name") == "i2i_search"
                    and original_call["arguments"].get("region")
                    and original_call["arguments"]["region"] != [0, 0, 1000, 1000]
                ):
                    raise ValueError(
                        "Downstream implicit i2i depends on an unrecorded crop"
                    )
            prior_paths = [p for j, paths in by_turn.items() if j < i for p in paths]
            ref = next(
                (
                    k
                    for k, p in bindings.items()
                    if prior_paths and p == prior_paths[-1]
                ),
                None,
            )
            if ref is None:
                raise ValueError("Cannot preserve implicit downstream i2i image source")
            if ref in changed_view_refs:
                raise ValueError(
                    "Downstream crop/search requires the original photograph"
                )
            action["arguments"]["image"] = removed_refs.get(ref, ref)
            source_repairs[str(i)] = action
            turns[i] = set_content(
                turns[i],
                ACTION.sub(
                    lambda m: (
                        f"<{m.group(1)}>\n{json.dumps(action, ensure_ascii=False)}\n</{m.group(1)}>"
                    ),
                    content(turns[i]),
                ),
            )
    # Namespace references in writer-generated text as well as generated tool calls.
    inserted = [set_content(m, remap_text(content(m), mapping)) for m in inserted]
    for i in patches:
        turns[i] = set_content(turns[i], remap_text(content(turns[i]), mapping))
    continuation_map = {}
    retained = []
    for i in range(window.start, len(turns)):
        if i not in removed:
            continuation_map[str(i)] = window.start + len(inserted) + len(retained)
            retained.append(turns[i])
    turns[window.start :] = inserted + retained
    prior = [p for i, paths in by_turn.items() if i < window.start for p in paths]
    tail = [
        p
        for i, paths in by_turn.items()
        if i >= window.end and i not in removed
        for p in paths
    ]
    output["images"] = prior + new_images + tail
    if plan.branch == "recovery" or source_repairs:
        # The runtime supports explicit image references; expose that capability
        # in legacy dataset tool schemas whose i2i schema only advertised region.
        from .tool_schema import expose_image_argument

        output["system"] = expose_image_argument(output.get("system", ""))
    return output, {
        "inserted": len(inserted),
        "replaced": len(removed),
        "removed_indices": sorted(removed),
        "continuation_map": continuation_map,
        "id_mapping": {**mapping, **removed_refs},
        "namespace": namespace,
        "source_repairs": source_repairs,
    }
