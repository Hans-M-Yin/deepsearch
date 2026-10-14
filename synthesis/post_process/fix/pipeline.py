"""Natural inspection loop orchestration; no preassigned contrast/recovery branch."""

from dataclasses import asdict, replace
from collections import Counter
from copy import deepcopy

from .trajectory_index import index_record, resolve_path
from .progress import assess_progress
from .replay_i2i import recover_historical
from .visual_evidence import verify_pool
from .search_cards import build_search_plan, is_convergent, POSITIVE
from .rollout import run_loop
from .selection import selection_history
from .context_patch import patch_context
from .global_refine import refine_trajectory
from .splice import assemble
from .validate import validate_structure, semantic_review
from .storage import write_json
from .errors import ReviewRejected, StructureRejected, SkipAugmentation
from .timing import timed_call, mark, since


def process_record(record, dataset_root, root, config, backend, model, cache):
    current = record
    audit = {"status": "unchanged", "flow_version": "selector_writer_v1", "windows": []}
    windows = timed_call("index", index_record, record)
    for window in reversed(windows[: config.max_windows_per_record]):
        item = {"call_index": window.call_index, "status": "started"}
        audit["windows"].append(item)
        start = mark()
        plan = None
        stage = "source"
        try:
            source = timed_call(
                "source", resolve_path, window.source_image, dataset_root
            )
            stage = "progress"
            anchor = timed_call(
                "progress", assess_progress, current, window, source, model, config
            )
            item["anchor"] = anchor
            anchor["pre_search_context"] = selection_history(current, window)
            stage = "search"
            pool = timed_call(
                "search",
                backend.search,
                source,
                window.call,
                record.get("metadata"),
                nonce="first",
            )
            historical = timed_call(
                "recover_historical",
                recover_historical,
                window,
                pool,
                dataset_root,
                current,
            )
            item["replay_count"] = len(pool)
            stage = "evidence"
            item["visual_failures"] = timed_call(
                "visual_verification",
                verify_pool,
                historical + pool,
                source,
                window,
                anchor,
                backend,
                model,
                config,
            )
            from .title_recovery import recover_anchor

            required = set(anchor["anchor_image_ids"])
            for c in historical:
                if (
                    any(c.key.endswith("old_" + rid) for rid in required)
                    and not c.image_paths
                ):
                    c.resources = []
            stage = "title_recovery"
            item["title_recovery"] = timed_call(
                "title_recovery",
                recover_anchor,
                historical,
                window,
                anchor,
                current,
                backend,
                model,
                config,
                source,
            )
            # Composition distinguishes required image observations from text-only
            # references; an unresolved unviewed image must not reject the record.
            item["visual_candidates"] = [asdict(c) for c in historical + pool]
            item["visual_categories"] = dict(
                Counter(c.category for c in historical + pool)
            )
            stage = "composition"
            plan = timed_call(
                "composition",
                build_search_plan,
                historical,
                pool,
                anchor,
                window,
                model,
                config,
            )
            plan.metadata["original_correct_views"] = {
                c.key: rid
                for c in historical
                for rid in anchor["viewed_before_conclusion_ids"]
                if c.key.endswith("old_" + rid) and c.category in POSITIVE
            }
            convergence = {
                "eligible": is_convergent(plan),
                "basis": "five_final_cards_match_local_target",
            }
            item["convergence"] = convergence
            if convergence["eligible"]:
                item.update(status="convergent", branch="convergent", wrong_count=0)
                continue
            stage = "selection"
            steps = timed_call(
                "selection_writer_loop",
                run_loop,
                current,
                window,
                plan,
                anchor,
                source,
                model,
                config,
            )
            if plan.metadata.get("reuse_original_read"):
                plan.metadata["original_read_turn"] = window.reads[
                    plan.metadata["reuse_original_read"]
                ]["turn"]
            from .answer_consistency import check_answer

            stage = "answer_consistency"
            item["answer_consistency"] = timed_call(
                "answer_consistency", check_answer, current, plan, model, config
            )
            if item["answer_consistency"].get("compatible") is not True:
                raise ReviewRejected(
                    "Immutable answer conflicts with visual evidence: "
                    + str(item["answer_consistency"])
                )
            # Retain all original tool/image observations, including an existing
            # correct check: do not delete and regenerate it.
            splice_window = replace(window, end=window.start, reads={})
            stage = "context"
            patches = timed_call(
                "context",
                patch_context,
                current,
                splice_window,
                plan,
                steps,
                model,
                config,
            )
            stage = "structure"
            updated, splice_audit = timed_call(
                "splice", assemble, current, splice_window, plan, steps, patches
            )
            errors = timed_call(
                "structural_validation",
                validate_structure,
                current,
                updated,
                splice_window,
                plan,
                splice_audit,
            )
            if errors:
                raise StructureRejected("Structural validation: " + "; ".join(errors))
            stage = "global_refine"
            pre_refinement = deepcopy(updated)
            updated, item["global_refinement"] = timed_call(
                "global_refine",
                refine_trajectory,
                updated,
                source,
                dataset_root,
                model,
                config,
                search_context={
                    "observation_index": splice_window.observation_index,
                    "continuation_map": splice_audit["continuation_map"],
                    "deferred_issues": plan.metadata.get("deferred_context_issues", []),
                    "inspection_state": plan.metadata.get("writer_assessments", []),
                },
            )
            if not item["global_refinement"]["accepted"]:
                audit["discard"] = True
                raise ReviewRejected(
                    "Severe global logic error: "
                    + str(item["global_refinement"]["severe_errors"])
                )
            errors = timed_call(
                "post_refine_structural_validation",
                validate_structure,
                current,
                updated,
                splice_window,
                plan,
                splice_audit,
                pre_refinement=pre_refinement,
            )
            if errors:
                raise StructureRejected(
                    "Post-refinement validation: " + "; ".join(errors)
                )
            stage = "review"
            review = timed_call(
                "review",
                semantic_review,
                current,
                updated,
                splice_window,
                plan,
                source,
                model,
                config,
                dataset_root=dataset_root,
            )
            item.update(
                splice=splice_audit,
                review=review,
                steps=steps,
                judge_uses_distinct_alias=config.judge_model
                not in {config.model, config.verifier_model},
            )
            if review.get("pass") is not True or review.get("issues"):
                raise ReviewRejected(
                    "Independent review rejected rewrite: " + str(review)
                )
            current = updated
            item.update(
                status="rewritten",
                branch=plan.branch,
                wrong_count=plan.metadata["wrong_count"],
                inconclusive_count=0,
                positive_rank=next(
                    i + 1
                    for i, c in enumerate(plan.first)
                    if c.key == plan.positive_key
                ),
            )
            audit["status"] = "rewritten"
        except SkipAugmentation as exc:
            item.update(status="skipped", skip_reason=exc.reason, details=exc.details)
        except Exception as exc:
            item.update(
                status="quarantined",
                failure_stage=stage,
                failure_kind="global_logic_error"
                if audit.get("discard")
                else "review_rejected"
                if isinstance(exc, ReviewRejected)
                else "structure_rejected"
                if isinstance(exc, StructureRejected)
                else "selection_exhausted"
                if stage == "selection"
                else "candidate_composition"
                if stage == "composition"
                else "model_or_tool_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            if audit["status"] != "rewritten":
                audit["status"] = "quarantined"
            if audit.get("discard"):
                audit["status"] = "quarantined"
                current = record
                break
        finally:
            if plan is not None:
                item["plan"] = plan.to_dict()
                write_json(root / f"window_{window.call_index}_plan.json", item)
            item["timing"] = since(start)
            write_json(root / f"window_{window.call_index}_audit.json", item)
    return current, audit
