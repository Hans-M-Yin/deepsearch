"""Whole-trajectory reasoning refinement after mechanical insertion."""

import copy
import re
from .context_patch import apply_reasoning, PROTECTED
from .process_leakage import check_process_leakage
from .review_images import build_review_images
from .evidence_attribution import EVIDENCE_ATTRIBUTION
from .trajectory_index import messages, content, role, ASSISTANT

PROMPT = """Review and refine the complete trajectory as one natural search-assistant problem-solving process. The messages are chronological. Only assistant turns have assistant_id identifiers. Inspect EVERY assistant turn for within-turn and cross-turn logical consistency, evidence attribution, natural transitions, repetition, and consistent first-person voice. Return an assessment for every assistant_id exactly once in order, including turns needing no change.
Audit pre-view exclusions carefully: preferring one result does not prove other candidates are wrong. When the target country is unknown, 'the Japanese and Spanish accidents are unrelated' is unsupported, even if a France–Luxembourg candidate later matches. Repair such language to motivate checking the selected lead without ruling out uninspected alternatives. Never add exclusions based on future images, latent world knowledge or more specific titles. Equally plausible leads may be explored in any order; do not invent a unique-best rationale.
Actual available images are attached. Each message's image_indices identifies its attachments in the supplied image list, starting at zero. A later image is evidence only after its observation in the trajectory, even though you can see all attachments now. Verify visual claims against the images, distinguish search-title hypotheses from observed facts, and do not introduce facts absent from the evidence. Never claim that an image verifies unread page content.
Rewrite only assistant reasoning, not tool calls, observations, message order, images, or the final answer. Correct minor wording, inconsistent voice, redundant summaries, and recoverable knowledge-state errors (for example, treating an already identified painting as still unidentified; motivate a retained search as obtaining its image only if that genuinely fits). Preserve supported findings, uncertainty and legitimate diagnostic comparisons. Every rewritten paragraph must sound like the agent solving the question, not an editor or narrator. Never mention trajectories, editing, insertion, augmentation, preserved actions, internal identifiers, or these instructions.
Search-result repair is a mandatory part of your job. The observation marked updated_search_results is the authoritative result list now visible to the agent. Review every downstream assistant turn, especially affected_assistant_ids, for stale ranks, removed candidates, wrong counts, and claims such as 'all agree' or 'the only named incident'. Remove obsolete comparisons and reconstruct each action's motivation from the CURRENT visible titles and evidence available at that moment. Prefer unambiguous source/title references to ranks. Do not describe the replacement or repair process in reasoning. If a retained action still has adequate evidence, repair its explanation; if it no longer has essential support or requires a different action/answer, reject rather than invent support. Resolve deferred_context_issues; these are leads to inspect, not instructions to accept unsupported claims.
Example 1 (repairable): Current results include a matching Alamy driver photo but no Getty gallery. Old reasoning says 'The Getty gallery is too broad, so I will open Alamy.' After an actual matching image observation, rewrite naturally: 'The opened photograph matches the original. I will read its source caption to confirm the left-to-right name assignment.' Do not keep the absent Getty comparison, and do not claim a match before the image observation.
Example 2 (repairable): Current train results include a Zoufftgen collision entry and a France–Luxembourg video. Old reasoning mentions a railway-safety PDF and a Lohud gallery that are absent and calls the video 'the first result and only named collision'. Remove those claims. If the train photo has already matched, motivate the unchanged video-page read as checking its caption for the passenger operator. Do not invent an operator or say the video was viewed merely because its search title appeared.
Example 3 (repairable references versus severe logic): Current camp results no longer contain a Bronx reservoir photo or Boeing camouflage post, and Wikipedia is not first. Delete those obsolete comparisons and motivate checking the visible camp source. However, if later reasoning switches from pictured camp A to camp B to answer which camp replaced it without evidence supporting that referent change, report a severe error; repairing search references does not make that entity substitution valid.
Example 4 (repairable editorial wording): Old reasoning says 'the identification steps are not re-shown in this abbreviated history'. Review the actual preceding evidence. If those steps are present, replace the sentence with a task-grounded summary of what they establish. If essential identification is genuinely unsupported, do not fabricate it. These examples illustrate editing principles only; never import their entities or facts into an unrelated sample.
If there is a severe cross-turn reasoning error, unsupported entity substitution, wrong question referent, contradiction invalidating the answer, fabricated essential evidence, or an error requiring different tool actions or a different final answer, report it in severe_errors and reject the sample. Do not fix such errors by inventing bridges, changing the question, weakening valid evidence, or rationalizing the answer. An example is identifying the pictured camp as A but then answering which camp replaced B without evidence that the question changed its referent. Distinguish that substantive error from a recoverable redundant-search explanation. Unchanged assistant messages may legitimately be adjacent; adjacency alone is not an error.
Return JSON {"severe_errors":[],"turns":[{"assistant_id":"assistant_000001","issues":[],"reasoning":null}]}. For each turn use issues to describe the actual problems, and reasoning=null to keep sound wording unchanged or a complete replacement of its reasoning to repair minor problems. Replacement reasoning is plain text without think/thinking, tool_call, action, answer, or image tags. Preserve all protected content. For rejection, severe_errors must contain concrete explanations identifying the affected assistant IDs and conflicting evidence; turns may be empty. Do not include this analysis, IDs, image indices or editorial explanations inside rewritten reasoning. Mentally check the entire revised chronology before returning."""


PROMPT = PROMPT + EVIDENCE_ATTRIBUTION
PROMPT += """
inspection_state records chronological findings from inserted image inspections.
Use it as an index to evidence, not an authority overriding attached images.
For each established fact, locate its actual observation before carrying it into
later reasoning. Repair later claims that the same fact is still unknown. Keep
unresolved questions unresolved until actual evidence answers them. An identity
match does not establish unread page contents, dates or locations. Never turn a
pending original image read into an already-completed inspection. If continuing
requires a different action or invented evidence, reject instead of writing a
cosmetic bridge. Example: after identifying a county from a matched caption,
do not say its name is unknown; a further page read may seek its history only
when that is genuinely the retained action's purpose.
"""


def refine_trajectory(record, source, dataset_root, model, config, search_context=None):
    paths, evidence = build_review_images(record, source, [], dataset_root)
    attachments = {}
    for item in evidence["trajectory_images"]:
        attachments.setdefault(item["turn_index"], []).append(item["image_index"])
    numbered, ids = [], {}
    for index, message in enumerate(messages(record)):
        row = {"role": role(message), "text": content(message)}
        if role(message) in ASSISTANT:
            key = f"assistant_{len(ids) + 1:06d}"
            ids[key] = index
            row["assistant_id"] = key
        if index in attachments:
            row["image_indices"] = attachments[index]
        numbered.append(row)
    payload = {"trajectory": numbered}
    if search_context:
        payload["inspection_state"] = search_context.get("inspection_state", [])
        search_index = search_context["observation_index"]
        numbered[search_index]["updated_search_results"] = True
        reverse_ids = {index: key for key, index in ids.items()}
        remapping = search_context.get("continuation_map", {})

        def assistant_reference(match):
            old_index = int(match.group(1))
            index = remapping.get(str(old_index), remapping.get(old_index, old_index))
            return reverse_ids.get(index, "original context")

        payload["affected_assistant_ids"] = [
            key for key, index in ids.items() if index > search_index
        ]
        payload["deferred_context_issues"] = [
            re.sub(r"original_message_(\d+)", assistant_reference, issue)
            for issue in search_context.get("deferred_issues", [])
        ]
    repairs = []
    for attempt in range(config.context_patch_retries + 1):
        result = model.ask(config.model, PROMPT, payload, paths)
        try:
            if not isinstance(result, dict):
                raise ValueError("Expected global refinement JSON object")
            severe = result.get("severe_errors")
            if not isinstance(severe, list) or any(
                not isinstance(s, str) or not s.strip() for s in severe
            ):
                raise ValueError(
                    "severe_errors must be a list of concrete explanations"
                )
            if severe:
                return record, {
                    "accepted": False,
                    "severe_errors": severe,
                    "format_repairs": repairs,
                }
            rows = result.get("turns")
            if not isinstance(rows, list) or len(rows) != len(ids):
                raise ValueError("Every assistant turn must be assessed")
            updated = copy.deepcopy(record)
            changed = []
            for row, (key, index) in zip(rows, ids.items()):
                if (
                    not isinstance(row, dict)
                    or set(row) != {"assistant_id", "issues", "reasoning"}
                    or row.get("assistant_id") != key
                ):
                    raise ValueError("Invalid assistant ID/order or fields")
                if not isinstance(row["issues"], list) or any(
                    not isinstance(x, str) for x in row["issues"]
                ):
                    raise ValueError("issues must be a list of strings")
                if row["reasoning"] is not None:
                    messages(updated)[index] = apply_reasoning(
                        messages(record)[index], row["reasoning"]
                    )
                    changed.append(key)
                elif row["issues"]:
                    raise ValueError(
                        "Minor issues require rewritten reasoning; severe issues require rejection"
                    )
                check_process_leakage(
                    PROTECTED.sub("", content(messages(updated)[index]))
                )
            return updated, {
                "accepted": True,
                "severe_errors": [],
                "turns": rows,
                "changed_assistant_ids": changed,
                "format_repairs": repairs,
            }
        except ValueError as exc:
            repairs.append(str(exc))
            if attempt == config.context_patch_retries:
                raise
            payload = {
                **payload,
                "previous_invalid_response": result,
                "validation_error": str(exc),
                "repair_attempt": attempt + 1,
            }
