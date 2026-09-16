#!/usr/bin/env python3
"""Run the partial-credit judge against saved DeepResearch rollout dumps.

Saved rollout JSON omits task metadata and serializes only model-visible
messages.  This utility joins each rollout back to an RL parquet by its exact
original question, reconstructs the ``Step`` sequence used by the live
workflow, and calls the existing scorer.  It is diagnostic-only: no rollout,
dataset, reward, or training state is modified. The report includes one short
judge rationale per gold statement; it contains no prompt or retrieved page
content.

Example:
  set -a; source RL/.env_remote; set +a
  PYTHONPATH=RL/rllm python debug/score_saved_partial_credit.py \
    --rollout-dir RL/rollouts_saved/open_mm_search_8b_4node_vllm/train_step_30_epoch_1 \
    --dataset-parquet runs/.../train_partial_credit_v2.parquet \
    --samples 6 --mode reward --output /tmp/partial_credit_pilot.json
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

# Match the training launchers: importing ``rllm`` also imports project-level
# synthesis helpers, so both roots must be present when this utility is run
# directly from an arbitrary working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RLLM_ROOT = _PROJECT_ROOT / "RL" / "rllm"
for _path in (str(_PROJECT_ROOT), str(_RLLM_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rllm.agents.agent import Step
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    _extract_action_from_response,
    _get_next_observation,
)
from vision_deepresearch_async_workflow.partial_credit_reward import (
    PartialCreditConfig,
    resolve_partial_credit_statements,
    score_statement_partial_credit,
)


def _question_key(value: Any) -> str:
    return " ".join(str(value or "").split())


def _find_question(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if "<tool_response>" not in content:
            return content
    return ""


def _rebuild_steps(messages: Any) -> list[Step]:
    if not isinstance(messages, list):
        return []
    normalized_messages = [
        dict(message) for message in messages if isinstance(message, Mapping)
    ]
    steps: list[Step] = []
    for index, message in enumerate(normalized_messages):
        if message.get("role") != "assistant":
            continue
        response = str(message.get("content") or "")
        steps.append(
            Step(
                chat_completions=normalized_messages[: index + 1],
                model_response=response,
                action=_extract_action_from_response(response),
                observation=_get_next_observation(normalized_messages, index),
            )
        )
    return steps


def _iter_rollouts(directory: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(directory.glob("episode_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[partial-credit-pilot] skip unreadable {path}: {exc}")
            continue
        if isinstance(payload, dict):
            yield path, payload


def _load_tasks(dataset_path: Path) -> dict[str, dict[str, Any]]:
    dataframe = pd.read_parquet(dataset_path)
    tasks: dict[str, dict[str, Any]] = {}
    for row in dataframe.to_dict("records"):
        key = _question_key(row.get("question"))
        if key:
            tasks.setdefault(key, row)
    return tasks


def _select_candidates(
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    count: int,
) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Pick distinct questions, balanced between correct and incorrect runs."""

    buckets: dict[bool, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {
        True: [],
        False: [],
    }
    seen_questions: dict[bool, set[str]] = {True: set(), False: set()}
    for candidate in candidates:
        rollout = candidate[1]
        is_correct = bool(rollout.get("is_correct"))
        question = _question_key(candidate[2].get("question"))
        if question in seen_questions[is_correct]:
            continue
        seen_questions[is_correct].add(question)
        buckets[is_correct].append(candidate)

    selected: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    desired_per_bucket = count // 2
    for is_correct in (True, False):
        selected.extend(buckets[is_correct][:desired_per_bucket])
    if len(selected) < count:
        for candidate in buckets[True][desired_per_bucket:] + buckets[False][desired_per_bucket:]:
            if len(selected) >= count:
                break
            selected.append(candidate)
    return selected


def _task_with_single_statement(
    task: Mapping[str, Any], statement: Mapping[str, str]
) -> dict[str, Any]:
    """Return an isolated task copy for one statement--trajectory audit pair."""

    task_copy = dict(task)
    extra_info = task_copy.get("extra_info")
    extra_copy = dict(extra_info) if isinstance(extra_info, Mapping) else {}
    partial_credit = extra_copy.get("partial_credit")
    partial_credit_copy = (
        dict(partial_credit) if isinstance(partial_credit, Mapping) else {}
    )
    partial_credit_copy["statements"] = [dict(statement)]
    extra_copy["partial_credit"] = partial_credit_copy
    task_copy["extra_info"] = extra_copy
    return task_copy


def _select_statement_pairs(
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    count: int,
    *,
    seed: int,
) -> list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, str], int]]:
    """Select distinct statement--trajectory pairs with outcome/hop balance.

    A pair contains one saved rollout and exactly one gold intermediate
    statement.  Alternating final-outcome buckets and hop positions makes a
    100-pair audit more informative than taking the first 100 rows, which can
    otherwise over-represent early hops or only failed trajectories.
    """

    if count < 1:
        raise ValueError("--statement-pairs must be positive")

    buckets: dict[tuple[bool, int], list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, str], int]]] = {}
    for path, rollout, task in candidates:
        statements, _, _ = resolve_partial_credit_statements(task)
        for statement_index, statement in enumerate(statements):
            key = (bool(rollout.get("is_correct")), statement_index)
            buckets.setdefault(key, []).append(
                (path, rollout, task, statement, statement_index)
            )

    available = sum(len(bucket) for bucket in buckets.values())
    if available < count:
        raise RuntimeError(
            f"Requested {count} statement--trajectory pairs but only {available} are available"
        )

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Round-robin outcome first, then statement position. This preserves a
    # near-50/50 correct-vs-incorrect split while avoiding domination by hop 0.
    keys = sorted(buckets, key=lambda item: (not item[0], item[1]))
    selected: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, str], int]] = []
    positions = {key: 0 for key in keys}
    while len(selected) < count:
        made_progress = False
        for key in keys:
            position = positions[key]
            if position >= len(buckets[key]):
                continue
            selected.append(buckets[key][position])
            positions[key] = position + 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError("Statement-pair sampler exhausted unexpectedly")
    return selected


def _public_result(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the report compact and never retain a judge prompt or raw pages."""

    keep = {
        "mode",
        "scored",
        "reason",
        "source_statement_count",
        "eligible_statement_count",
        "trace_turn_count",
        "trace_chars",
        "trace_chunk_steps",
        "trace_chunk_count",
        "trace_chunk_scored_count",
        "trace_chunk_failed_count",
        "judge_request_s",
        "judge_queue_wait_s",
        "judge_valid_verdict_count",
        "judge_rejected_verdict_count",
        "verified_statement_count",
        "chunk_valid_verdict_count",
        "chunk_judge_positive_occurrence_count",
        "chunk_judge_positive_statement_count",
        "chunk_judge_duplicate_positive_statement_count",
        "chunk_judge_duplicate_positive_occurrence_count",
        "chunk_judge_duplicate_positive_statement_rate",
        "chunk_valid_duplicate_statement_count",
        "chunk_valid_duplicate_occurrence_count",
        "chunk_valid_duplicate_statement_rate",
        "raw_bonus",
        "bonus",
        "events",
        "judge_statement_assessments",
        "chunk_results",
        "judge_error_type",
        "statement_limit_hit",
    }
    result = {key: value for key, value in record.items() if key in keep}
    # The live workflow intentionally drops judge errors from persisted
    # rollout metadata.  This one-off diagnostic needs the HTTP status/body
    # summary to distinguish a gateway rejection from a scorer bug, but keeps
    # it bounded and still never writes a prompt, retrieved page, or API key.
    judge_error = str(record.get("judge_error") or "").strip()
    if judge_error:
        result["judge_error"] = judge_error[:2_000]
    return result


async def _score(
    selected: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    config: PartialCreditConfig,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path, rollout, task in selected:
        steps = _rebuild_steps(rollout.get("messages"))
        statements, _, statement_reason = resolve_partial_credit_statements(task)
        score = await score_statement_partial_credit(
            task=task,
            steps=steps,
            config=config,
        )
        results.append(
            {
                "rollout_file": str(path),
                "episode_id": str(rollout.get("episode_id") or ""),
                "is_correct": bool(rollout.get("is_correct")),
                "rollout_reward": float(
                    (rollout.get("trajectories") or [{}])[0].get("reward", 0.0)
                ),
                "question": str(task.get("question") or ""),
                "gold_statements": statements,
                "metadata_reason": statement_reason,
                "reconstructed_step_count": len(steps),
                "partial_credit": _public_result(score),
            }
        )
        print(
            "[partial-credit-pilot]"
            f" correct={bool(rollout.get('is_correct'))}"
            f" statements={len(statements)} steps={len(steps)}"
            f" reason={score.get('reason')}"
            f" verified={score.get('verified_statement_count', 0)}"
            f" raw_bonus={score.get('raw_bonus', 0.0):.3f}"
            f" file={path.name}"
        )
    return results


async def _score_statement_pairs(
    selected: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, str], int]],
    config: PartialCreditConfig,
) -> list[dict[str, Any]]:
    """Score selected statement--trajectory pairs one statement at a time."""

    results: list[dict[str, Any]] = []
    for path, rollout, task, statement, statement_index in selected:
        steps = _rebuild_steps(rollout.get("messages"))
        score = await score_statement_partial_credit(
            task=_task_with_single_statement(task, statement),
            steps=steps,
            config=config,
        )
        results.append(
            {
                "rollout_file": str(path),
                "episode_id": str(rollout.get("episode_id") or ""),
                "is_correct": bool(rollout.get("is_correct")),
                "rollout_reward": float(
                    (rollout.get("trajectories") or [{}])[0].get("reward", 0.0)
                ),
                "question": str(task.get("question") or ""),
                "statement_pair": {
                    "statement_index": statement_index,
                    "statement_id": str(statement.get("statement_id") or ""),
                    "statement": str(statement.get("statement") or ""),
                },
                "reconstructed_step_count": len(steps),
                "partial_credit": _public_result(score),
            }
        )
        print(
            "[partial-credit-pilot]"
            " pair=1"
            f" correct={bool(rollout.get('is_correct'))}"
            f" hop={statement_index}"
            f" statement_id={statement.get('statement_id', '')}"
            f" steps={len(steps)}"
            f" reason={score.get('reason')}"
            f" verified={score.get('verified_statement_count', 0)}"
            f" raw_bonus={score.get('raw_bonus', 0.0):.3f}"
            f" file={path.name}"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--dataset-parquet", required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--statement-pairs",
        type=int,
        default=0,
        help=(
            "Audit exactly this many statement--trajectory pairs. When positive, "
            "this supersedes --samples and judges one gold statement per request."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260916,
        help="Deterministic sampling seed for --statement-pairs.",
    )
    parser.add_argument("--mode", choices=("shadow", "reward"), default="reward")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_dir = Path(args.rollout_dir).resolve()
    dataset_path = Path(args.dataset_parquet).resolve()
    output_path = Path(args.output).resolve()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if not rollout_dir.is_dir():
        raise FileNotFoundError(rollout_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    tasks = _load_tasks(dataset_path)
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    unmatched = 0
    missing_statements = 0
    for path, rollout in _iter_rollouts(rollout_dir):
        question_key = _question_key(_find_question(rollout.get("messages")))
        task = tasks.get(question_key)
        if task is None:
            unmatched += 1
            continue
        statements, _, _ = resolve_partial_credit_statements(task)
        if not statements:
            missing_statements += 1
            continue
        candidates.append((path, rollout, task))

    statement_pair_mode = args.statement_pairs > 0
    selected = (
        _select_statement_pairs(
            candidates,
            args.statement_pairs,
            seed=args.seed,
        )
        if statement_pair_mode
        else _select_candidates(candidates, args.samples)
    )
    if not selected:
        raise RuntimeError("No saved rollout matched a task with partial-credit statements")

    base_config = PartialCreditConfig.from_env()
    # The live scorer stores scalar metrics only. This diagnostic run asks for
    # a one-sentence rationale for every statement so false negatives can be
    # audited without exporting the model-visible trace or retrieved pages.
    config = replace(
        base_config,
        mode=args.mode,
        judge_concurrency=1,
        include_judge_reasons=True,
    )
    results = asyncio.run(
        _score_statement_pairs(selected, config)
        if statement_pair_mode
        else _score(selected, config)
    )
    chunk_judge_positive_statement_count_total = sum(
        int(
            item["partial_credit"].get(
                "chunk_judge_positive_statement_count", 0
            )
        )
        for item in results
    )
    chunk_judge_duplicate_positive_statement_count_total = sum(
        int(
            item["partial_credit"].get(
                "chunk_judge_duplicate_positive_statement_count", 0
            )
        )
        for item in results
    )
    chunk_valid_unique_statement_count_total = sum(
        int(item["partial_credit"].get("verified_statement_count", 0))
        for item in results
    )
    chunk_valid_duplicate_statement_count_total = sum(
        int(
            item["partial_credit"].get(
                "chunk_valid_duplicate_statement_count", 0
            )
        )
        for item in results
    )
    summary = {
        "rollout_dir": str(rollout_dir),
        "dataset_parquet": str(dataset_path),
        "candidate_count": len(candidates),
        "audit_unit": "statement_trajectory_pair" if statement_pair_mode else "trajectory",
        "unmatched_rollout_count": unmatched,
        "matched_without_statements_count": missing_statements,
        "selected_count": len(results),
        "selected_statement_pair_count": len(results) if statement_pair_mode else 0,
        "selected_unique_trajectory_count": len(
            {str(item.get("episode_id") or "") for item in results}
        ),
        "selected_correct_count": sum(item["is_correct"] for item in results),
        "selected_incorrect_count": sum(not item["is_correct"] for item in results),
        "scored_count": sum(bool(item["partial_credit"].get("scored")) for item in results),
        "verified_statement_count": sum(
            int(item["partial_credit"].get("verified_statement_count", 0))
            for item in results
        ),
        "raw_bonus_total": sum(
            float(item["partial_credit"].get("raw_bonus", 0.0)) for item in results
        ),
        "trace_chunk_count_total": sum(
            int(item["partial_credit"].get("trace_chunk_count", 0))
            for item in results
        ),
        "trace_chunk_scored_count_total": sum(
            int(item["partial_credit"].get("trace_chunk_scored_count", 0))
            for item in results
        ),
        "chunk_valid_verdict_count_total": sum(
            int(item["partial_credit"].get("chunk_valid_verdict_count", 0))
            for item in results
        ),
        "chunk_judge_positive_statement_count_total": chunk_judge_positive_statement_count_total,
        "chunk_judge_duplicate_positive_statement_count_total": chunk_judge_duplicate_positive_statement_count_total,
        "chunk_judge_duplicate_positive_occurrence_count_total": sum(
            int(
                item["partial_credit"].get(
                    "chunk_judge_duplicate_positive_occurrence_count", 0
                )
            )
            for item in results
        ),
        "chunk_judge_duplicate_positive_statement_rate": (
            chunk_judge_duplicate_positive_statement_count_total
            / chunk_judge_positive_statement_count_total
            if chunk_judge_positive_statement_count_total
            else 0.0
        ),
        "chunk_valid_unique_statement_count_total": chunk_valid_unique_statement_count_total,
        "chunk_valid_duplicate_statement_count_total": chunk_valid_duplicate_statement_count_total,
        "chunk_valid_duplicate_occurrence_count_total": sum(
            int(
                item["partial_credit"].get(
                    "chunk_valid_duplicate_occurrence_count", 0
                )
            )
            for item in results
        ),
        "chunk_valid_duplicate_statement_rate": (
            chunk_valid_duplicate_statement_count_total
            / chunk_valid_unique_statement_count_total
            if chunk_valid_unique_statement_count_total
            else 0.0
        ),
    }
    report = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[partial-credit-pilot] report={output_path}")


if __name__ == "__main__":
    main()
