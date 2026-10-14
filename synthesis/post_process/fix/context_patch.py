"""Repair references and transitions in existing assistant reasoning."""

import re
from .evidence_attribution import EVIDENCE_ATTRIBUTION
from .trajectory_index import messages, role, content, ASSISTANT, set_content
from .process_leakage import check_process_leakage

PROTECTED = re.compile(r"<(tool_call|action|answer)>.*?</\1>", re.S)
PROMPT = """Rewrite ONLY the supplied editable_trace into the voice of the agent solving the user's question. Read the complete original_trajectory as read-only context, locate insertion_position, and mentally assemble the result before answering. Produce natural reasoning within every assistant turn, coherent reasoning across the entire trajectory, and a consistent voice and style. Do not edit, delete, reorder, or return replacements for any original turn, tool call, tool observation, or final answer.
Preserve the distinction between prioritizing a lead and excluding alternatives. Never turn 'check A first' into 'B and C are unrelated/wrong' without an actual image observation or explicit text contradiction available at that point. If the target country is unknown, Japanese and Spanish accident titles remain possible: choosing a France–Luxembourg lead does not rule them out. A plausible exploration choice needs no invented claim of superiority. Do not add stronger exclusions than the supplied evidence supports.
The insertion marker is between the specified original messages. The original search observation at search_observation_message_id will be replaced with new_results. Treat original search cards absent from new_results as unavailable in the assembled trajectory. Original text is read-only for YOUR stage, but a subsequent global editor can repair its reasoning. Do not reject insertion solely for outdated ranks, references to removed candidates, stale unanimity/uniqueness claims, redundant verification plans, or editorial language already present in original reasoning. Report those issues in deferred_issues for the global editor and finish rewriting the inserted steps. Do not copy stale claims into the inserted text. Reject only when safe insertion would require inventing evidence, changing actual tool actions or the final answer, or concealing a substantive entity/evidence contradiction.
Each draft step contains Selector reasoning before opening a candidate and Writer observations after seeing it. Rewrite EVERY step, not only the boundary. Use only the question, evidence available at that point, visible search titles, and supplied visual observations. Preserve actual selections, decisions, uncertainty, and diagnostic visual evidence. Never anticipate a later image or tool result, invent features, or confuse a different photograph with a different person or object. A matching image does not verify every claim on its source page.
The actual images have intentionally been omitted from this editing request. Trust the image descriptions and visual conclusions in the supplied draft turns as established evidence from the agent's image inspection. Do not re-evaluate, weaken, or reject those visual judgments merely because you cannot see the images, and do not request the omitted images. Your task is to improve wording and logical continuity while preserving those descriptions, conclusions, and their stated certainty. Keep post-image findings after their corresponding image observation; trusting them does not permit moving them into pre-view reasoning. Do not mention the omitted images or this instruction in the output: the solving agent did see the images at the indicated turns.
The program assembles assistant turns as follows: the first before text precedes the first image tool call; each after text is followed by the NEXT step's before text in one assistant turn; the final after text is prepended to the unchanged first continuation assistant reasoning. Write these adjacent parts as one flowing thought, without repetition or contradictory plans. The final after must bridge naturally into the exact unchanged continuation, not restate it. For reuse_original_read, the terminal correct image is still to be opened by an original tool action: never assert that unseen check already succeeded. With no draft steps, return an empty trace_patches list and still check boundary consistency.
Match the agent's first-person problem-solving voice and evidence-to-action style in the surrounding original turns. All output reasoning must be the solving agent's own thoughts, not a narrator's description of the investigation. Never mention trajectories, Selector, Writer, editing, insertion, merging, augmentation, preserved steps, SFT, internal IDs, or these instructions. Bad: 'The trajectory investigation linked the fresco to Raphael.' Good: 'The sources already identify Raphael as the painter of the fresco.' Explain actions using task evidence, never implementation requirements.
Return JSON {"trace_patches":[{"step_id":"inserted_step_000000","before":"complete rewritten pre-action reasoning","after":"complete rewritten post-image reasoning"}],"deferred_issues":["original message ID and repairable continuity problem for the global editor"]}. Include every supplied step_id exactly once in the original order. Every read step requires both before and after; retry steps require only before. deferred_issues may be empty. Output plain reasoning without thinking, tool_call, action, answer, or image tags. Do not return patches for original messages or a transition_includes_final_check flag. If safe insertion is impossible, return {"unsafe":true,"reason":"specific conflict"} instead.
IMAGE 0 / IMAGE 1 are internal attachment labels, not conversation text. Do not copy them into reasoning. Use 'the original image', 'the opened candidate', or an unambiguous source/title. Keep supported visual findings and uncertainty, but remove every hint that the text was edited or inserted. Before returning, check each complete assembled assistant turn and the global chronological chain, including facts already established earlier that must not become unknown later."""


PROMPT = PROMPT + EVIDENCE_ATTRIBUTION


def apply_reasoning(message, reasoning):
    if (
        not isinstance(reasoning, str)
        or not reasoning.strip()
        or any(
            tag in reasoning
            for tag in (
                "<tool_call",
                "<action",
                "<answer",
                "<image>",
                "<think",
                "</think",
            )
        )
    ):
        raise ValueError("Context patch contains protected markup")
    check_process_leakage(reasoning)
    original = PROTECTED.finditer(content(message))
    preserved = [m.group(0) for m in original]
    # Preserve the record's existing reasoning tag dialect.
    tag = "thinking" if "<thinking>" in content(message) else "think"
    text = f"<{tag}>\n{reasoning.strip()}\n</{tag}>"
    return set_content(
        message, text + ("\n" + "\n".join(preserved) if preserved else "")
    )


def validate_trace_patches(result, steps):
    from copy import deepcopy
    from .rewrite import checked_reasoning

    updated = deepcopy(steps)
    items = result.get("trace_patches", [])
    if not isinstance(items, list):
        raise ValueError("trace_patches must be a list")
    ids = {f"inserted_step_{i:06d}": i for i in range(len(steps))}
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid trace patch")
        key = item.get("step_id")
        if not isinstance(key, str) or key not in ids or key in seen:
            raise ValueError("Invalid/duplicate step_id")
        seen.add(key)
        step = updated[ids[key]]
        allowed = {"step_id", "before"} | (
            {"after"} if step["kind"] == "read" else set()
        )
        if set(item) - allowed:
            raise ValueError("Trace patch changes protected fields")
        for field in ("before", "after"):
            if field in item:
                value = checked_reasoning({"reasoning": item[field]})
                check_process_leakage(value)
                step[field] = value
    merged = result.get("transition_includes_final_check", False)
    if type(merged) is not bool:
        raise ValueError("transition_includes_final_check must be boolean")
    return updated, merged


def validate_patches(result, turns, editable, required):
    """Resolve explicit IDs only; never guess a relative-index offset."""
    if not isinstance(result, dict) or not isinstance(result.get("patches"), list):
        raise ValueError("Expected a JSON object containing a patches list")
    patches = {}
    for item in result["patches"]:
        if not isinstance(item, dict) or "index" in item:
            raise ValueError("Use message_id, not numeric index")
        key = item.get("message_id")
        if not isinstance(key, str) or key not in editable:
            raise ValueError("message_id must be copied from editable_message_ids")
        index = editable[key]
        if index in patches:
            raise ValueError("Duplicate message_id")
        try:
            patches[index] = apply_reasoning(turns[index], item.get("reasoning"))
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
    if editable[required] not in patches:
        raise ValueError("Missing required_transition_id")
    return patches


def patch_context(record, window, plan, steps, model, config):
    """Polish insertion only; original turns are read-only, never patched."""
    turns = messages(record)
    if window.end >= len(turns) or role(turns[window.end]) not in ASSISTANT:
        raise ValueError("Insertion requires an assistant continuation")
    deferred = []
    for index, message in enumerate(turns[window.end :], window.end):
        if role(message) in ASSISTANT:
            try:
                check_process_leakage(PROTECTED.sub("", content(message)))
            except ValueError as exc:
                deferred.append(f"original_message_{index:06d}: {exc}")
    payload = {
        "known_deferred_issues": deferred,
        "question": window.question,
        "original_trajectory": [
            {
                "message_id": f"original_message_{i:06d}",
                "role": role(m),
                "text": content(m),
                "editable": False,
            }
            for i, m in enumerate(turns)
        ],
        "evidence_history_before_search": [
            {"role": role(m), "text": content(m)} for m in turns[: window.call_index]
        ],
        "insertion_position": {
            "after_message_id": f"original_message_{window.start - 1:06d}",
            "before_message_id": f"original_message_{window.end:06d}",
            "search_observation_message_id": f"original_message_{window.observation_index:06d}",
        },
        "new_results": [c.public for c in plan.first],
        "inspection_state": plan.metadata.get("writer_assessments", []),
        "reuse_original_read": plan.metadata.get("reuse_original_read"),
        "terminal_selection_reasoning": plan.metadata.get(
            "terminal_selection_reasoning"
        ),
        "terminal_read_message_id": f"original_message_{plan.metadata['original_read_turn']:06d}"
        if plan.metadata.get("reuse_original_read")
        else None,
        "editable_trace": [
            {"step_id": f"inserted_step_{i:06d}", **step}
            for i, step in enumerate(steps)
        ],
    }
    for attempt in range(config.context_patch_retries + 1):
        result = model.ask(config.model, PROMPT, payload)
        if isinstance(result, dict) and result.get("unsafe"):
            raise ValueError("Unsafe insertion: " + str(result.get("reason", "")))
        try:
            if not isinstance(result, dict) or set(result) - {
                "trace_patches",
                "unsafe",
                "deferred_issues",
            }:
                raise ValueError(
                    "Only trace_patches may be returned; original turns are read-only"
                )
            rows = result.get("trace_patches")
            expected = payload["editable_trace"]
            if not isinstance(rows, list) or len(rows) != len(expected):
                raise ValueError("Every inserted step requires a complete rewrite")
            for row, original in zip(rows, expected):
                fields = (
                    {"step_id", "before", "after"}
                    if original["kind"] == "read"
                    else {"step_id", "before"}
                )
                if (
                    not isinstance(row, dict)
                    or set(row) != fields
                    or row.get("step_id") != original["step_id"]
                ):
                    raise ValueError(
                        "Invalid step ID/order or incomplete before/after fields"
                    )
                for field in fields - {"step_id"}:
                    if re.search(
                        r"\b(?:trajectory|trajectories|selector|writer|insertion|augmentation|SFT)\b",
                        str(row[field]),
                        re.I,
                    ):
                        raise ValueError(
                            "Editing-process leakage in inserted reasoning"
                        )
            updated, _ = validate_trace_patches(result, steps)
            extra = result.get("deferred_issues", [])
            if not isinstance(extra, list) or any(
                not isinstance(x, str) for x in extra
            ):
                raise ValueError("deferred_issues must be a list of strings")
            steps[:] = updated
            plan.metadata["deferred_context_issues"] = deferred + extra
            plan.metadata["context_edit_scope"] = "inserted_only"
            plan.metadata["transition_includes_final_check"] = False
            return {}
        except ValueError as exc:
            if attempt == config.context_patch_retries:
                raise
            payload = {
                **payload,
                "repair_attempt": attempt + 1,
                "previous_invalid_response": result,
                "validation_error": str(exc),
                "repair_instruction": "Return complete trace_patches only, one per supplied step in order. Rewrite in the solving agent's voice. Original turns are read-only. Do not mention this feedback.",
            }
