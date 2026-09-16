#!/usr/bin/env python3
"""Audit whether generated node chains provide useful partial-credit signals.

This is deliberately an *offline*, read-only diagnostic.  It joins saved RL
episodes to the registered RL dataset by exact question text, then resolves a
synthetic example's ``source_sample_id`` to the source VQA ``samples.jsonl``.
Those records contain the constructed ``hop_chain`` and give us an independent
set of intermediate nodes to inspect.

The script uses conservative lexical matching of a gold node in an assistant
action turn.  A hit means only that the rollout explicitly mentioned a gold
node before its final answer; it is not proof that the node was correctly
inferred, supported by evidence, or necessary (a rollout may have used a
valid shortcut).  Consequently, missing hits must never be interpreted as a
negative supervision signal without a second-stage judge/human audit.

Examples:
  python debug/analyze_rollout_intermediate_credit.py \
    --rollout-root RL/rollouts_saved/open_mm_search_8b_4node_vllm \
    --dataset-parquet RL/rllm/rllm/data/datasets/Vision-DeepResearch-QA-rl-quality-filtered-benchmark-6010/train.parquet \
    --max-episodes 3100 --report-json /tmp/intermediate_credit_report.json

  python debug/analyze_rollout_intermediate_credit.py \
    --rollout-root RL/rollouts_saved/open_mm_search_8b_4node_vllm \
    --dataset-parquet RL/rllm/rllm/data/datasets/Vision-DeepResearch-QA-rl-quality-filtered-benchmark-6010/train.parquet \
    --max-episodes 600 --seed 3407 --print-cases
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NON_ALNUM_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class EpisodeAudit:
    path: str
    training_step: int | None
    task_hash: str | None
    correct: bool
    reward: float | None
    termination_reason: str | None
    raw_termination: str | None
    dataset_id: str
    source_path_id: str
    chain_length: int
    assistant_action_turns: int
    tool_calls: int
    search_tool_calls: int
    read_url_calls: int
    final_answer: str
    question: str
    targets: list[str]
    target_first_hit_turn: list[int | None]
    prefinal_node_hits: int
    all_message_node_hits: int

    @property
    def prefinal_coverage(self) -> float:
        return self.prefinal_node_hits / self.chain_length if self.chain_length else 0.0

    @property
    def all_message_coverage(self) -> float:
        return self.all_message_node_hits / self.chain_length if self.chain_length else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        required=True,
        action="append",
        help=(
            "Registered dataset parquet; may be supplied multiple times to "
            "cover a run that crossed dataset versions. Earlier arguments "
            "win on duplicate question text."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional uniform sample size after deterministic shuffling.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--print-cases",
        action="store_true",
        help="Print one representative case for each diagnostic bucket.",
    )
    parser.add_argument(
        "--case-text-limit",
        type=int,
        default=1200,
        help="Maximum characters printed for a question/final answer.",
    )
    return parser.parse_args()


def normalize(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.casefold().replace("_", " ")
    text = NON_ALNUM_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def target_aliases(target: str) -> list[str]:
    """Return conservative textual aliases for a chain node.

    The full target is always retained.  A comma/parenthetical prefix is added
    only when it is at least four normalized characters, e.g. ``Durham,
    England`` -> ``Durham``.  This improves recall for place qualifiers while
    avoiding aliases such as ``the`` or a single letter.
    """

    candidates = [target]
    for separator in (",", "("):
        if separator in target:
            candidates.append(target.split(separator, 1)[0])
    aliases: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = normalize(candidate)
        if len(value) >= 4 and value not in seen:
            seen.add(value)
            aliases.append(value)
    return aliases


def contains_alias(text: str, aliases: Iterable[str]) -> bool:
    normalized = f" {normalize(text)} "
    return any(f" {alias} " in normalized for alias in aliases)


def extract_tool_name(text: str) -> str | None:
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    return str(name) if isinstance(name, str) else None


def extract_final_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def question_from_episode(episode: dict[str, Any]) -> str | None:
    """Find the initial user question, excluding tool observations."""

    for message in episode.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and "<tool_response>" not in content:
            # The rollout formatter appends one ``<image>`` marker per input
            # image to the user message.  Registered dataset questions store
            # only the human question, so remove *terminal* placeholders for
            # the join but leave any literal in-question text untouched.
            return re.sub(r"(?:\s*<image>\s*)+$", "", content).strip()
    return None


def load_dataset_index(dataset_path: Path) -> dict[str, dict[str, Any]]:
    dataframe = pd.read_parquet(dataset_path)
    required = {"id", "question", "extra_info"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    result: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in dataframe[["id", "question", "extra_info"]].to_dict("records"):
        question = row["question"]
        if not isinstance(question, str) or not question:
            continue
        if question in result:
            duplicates += 1
            continue
        extra = row["extra_info"] if isinstance(row["extra_info"], dict) else {}
        result[question] = {
            "dataset_id": str(row["id"]),
            "source_run_dir": extra.get("source_run_dir"),
            "source_sample_id": extra.get("source_sample_id"),
            "source_path_id": str(extra.get("source_path_id") or ""),
        }
    if duplicates:
        print(f"[warning] duplicate question text in dataset: {duplicates}", file=sys.stderr)
    return result


def load_chains(dataset_index: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """Load only chain targets referenced by the registered dataset."""

    wanted_by_run: dict[str, set[str]] = defaultdict(set)
    for metadata in dataset_index.values():
        source_run_dir = metadata.get("source_run_dir")
        source_sample_id = metadata.get("source_sample_id")
        if isinstance(source_run_dir, str) and isinstance(source_sample_id, str):
            wanted_by_run[source_run_dir].add(source_sample_id)

    chains: dict[tuple[str, str], list[str]] = {}
    for source_run_dir, wanted_sample_ids in sorted(wanted_by_run.items()):
        samples_path = Path(source_run_dir) / "samples.jsonl"
        if not samples_path.exists():
            continue
        with samples_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    sample = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                sample_id = sample.get("sample_id")
                if sample_id not in wanted_sample_ids:
                    continue
                raw_hops = sample.get("hop_chain")
                if not isinstance(raw_hops, list):
                    continue
                targets = [
                    str(hop.get("target")).strip()
                    for hop in raw_hops
                    if isinstance(hop, dict)
                    and isinstance(hop.get("target"), str)
                    and target_aliases(hop["target"])
                ]
                if targets:
                    chains[(source_run_dir, sample_id)] = targets
    return chains


def iter_episode_paths(root: Path, max_episodes: int | None, seed: int) -> list[Path]:
    paths = sorted(root.rglob("episode_*.json"))
    if max_episodes is not None and max_episodes < len(paths):
        random.Random(seed).shuffle(paths)
        paths = sorted(paths[:max_episodes])
    return paths


def audit_episode(
    path: Path,
    dataset_index: dict[str, dict[str, Any]],
    chains: dict[tuple[str, str], list[str]],
) -> tuple[EpisodeAudit | None, str]:
    try:
        episode = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable_episode"

    question = question_from_episode(episode)
    if question is None:
        return None, "missing_question"
    metadata = dataset_index.get(question)
    if metadata is None:
        return None, "question_not_in_dataset"

    source_run_dir = metadata.get("source_run_dir")
    source_sample_id = metadata.get("source_sample_id")
    if not isinstance(source_run_dir, str) or not isinstance(source_sample_id, str):
        return None, "missing_source_metadata"
    targets = chains.get((source_run_dir, source_sample_id))
    if not targets:
        return None, "no_gold_hop_chain"

    assistant_messages = [
        str(message.get("content") or "")
        for message in episode.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistant_messages:
        return None, "missing_assistant_messages"

    action_messages = [message for message in assistant_messages if "<answer>" not in message]
    # Normally only the last assistant message is final.  This keeps the
    # statistic sensible for interrupted/error trajectories with no answer.
    if not action_messages and "<answer>" not in assistant_messages[-1]:
        action_messages = assistant_messages

    aliases = [target_aliases(target) for target in targets]
    first_hit_turn: list[int | None] = []
    for aliases_for_target in aliases:
        first = next(
            (
                turn_index + 1
                for turn_index, message in enumerate(action_messages)
                if contains_alias(message, aliases_for_target)
            ),
            None,
        )
        first_hit_turn.append(first)

    prefinal_hits = sum(turn is not None for turn in first_hit_turn)
    all_hits = sum(
        any(contains_alias(message, aliases_for_target) for message in assistant_messages)
        for aliases_for_target in aliases
    )
    tool_names = [extract_tool_name(message) for message in action_messages]
    tool_names = [name for name in tool_names if name]
    final_message = assistant_messages[-1]
    trajectories = episode.get("trajectories")
    reward: float | None = None
    if isinstance(trajectories, list) and trajectories and isinstance(trajectories[0], dict):
        raw_reward = trajectories[0].get("reward")
        if isinstance(raw_reward, (int, float)):
            reward = float(raw_reward)

    return (
        EpisodeAudit(
            path=str(path),
            training_step=episode.get("training_step") if isinstance(episode.get("training_step"), int) else None,
            task_hash=str(episode.get("task_hash")) if episode.get("task_hash") is not None else None,
            correct=bool(episode.get("is_correct")),
            reward=reward,
            termination_reason=str(episode.get("termination_reason") or ""),
            raw_termination=str(episode.get("raw_termination") or ""),
            dataset_id=metadata["dataset_id"],
            source_path_id=metadata["source_path_id"],
            chain_length=len(targets),
            assistant_action_turns=len(action_messages),
            tool_calls=len(tool_names),
            search_tool_calls=sum(name in {"t2t_search", "t2i_search", "i2i_search", "tmost_search"} for name in tool_names),
            read_url_calls=sum(name == "read_url" for name in tool_names),
            final_answer=extract_final_answer(final_message),
            question=question,
            targets=targets,
            target_first_hit_turn=first_hit_turn,
            prefinal_node_hits=prefinal_hits,
            all_message_node_hits=all_hits,
        ),
        "ok",
    )


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else None


def bucket_name(record: EpisodeAudit) -> str:
    if record.prefinal_node_hits == 0:
        return "no_prefinal_gold_node_hit"
    if record.prefinal_node_hits == record.chain_length:
        return "all_gold_nodes_hit_prefinal"
    first_hit = record.target_first_hit_turn[0]
    if first_hit is not None and record.prefinal_node_hits < record.chain_length:
        return "entry_hit_but_later_node_missing"
    return "partial_non_entry_node_hit"


def summarize(records: list[EpisodeAudit], skip_reasons: Counter[str]) -> dict[str, Any]:
    by_correctness: dict[str, list[EpisodeAudit]] = {
        "correct": [record for record in records if record.correct],
        "incorrect": [record for record in records if not record.correct],
    }
    correctness_summary: dict[str, Any] = {}
    for name, subset in by_correctness.items():
        buckets = Counter(bucket_name(record) for record in subset)
        correctness_summary[name] = {
            "episodes": len(subset),
            "mean_reward": mean(record.reward for record in subset if record.reward is not None),
            "mean_chain_length": mean(record.chain_length for record in subset),
            "mean_prefinal_gold_node_coverage": mean(record.prefinal_coverage for record in subset),
            "mean_all_message_gold_node_coverage": mean(record.all_message_coverage for record in subset),
            "mean_assistant_action_turns": mean(record.assistant_action_turns for record in subset),
            "mean_tool_calls": mean(record.tool_calls for record in subset),
            "mean_search_tool_calls": mean(record.search_tool_calls for record in subset),
            "mean_read_url_calls": mean(record.read_url_calls for record in subset),
            "node_hit_buckets": dict(sorted(buckets.items())),
            "node_hit_bucket_rates": {
                key: round(value / len(subset), 4) if subset else None
                for key, value in sorted(buckets.items())
            },
        }

    # This isolates the exact cases for which an auxiliary local signal is
    # potentially useful: final reward is bad, yet the trajectory explicitly
    # reached some gold structure before the terminal failure.
    incorrect = by_correctness["incorrect"]
    failure_partial_progress = [record for record in incorrect if record.prefinal_node_hits > 0]
    failure_entry_then_miss = [
        record
        for record in incorrect
        if record.target_first_hit_turn[0] is not None and record.prefinal_node_hits < record.chain_length
    ]
    failure_all_nodes = [
        record for record in incorrect if record.prefinal_node_hits == record.chain_length
    ]

    by_chain_length: dict[str, dict[str, Any]] = {}
    for chain_length in sorted({record.chain_length for record in records}):
        row: dict[str, Any] = {}
        for label, subset in by_correctness.items():
            conditioned = [record for record in subset if record.chain_length == chain_length]
            row[label] = {
                "episodes": len(conditioned),
                "mean_prefinal_gold_node_coverage": mean(record.prefinal_coverage for record in conditioned),
                "all_nodes_hit_prefinal_rate": round(
                    sum(record.prefinal_node_hits == record.chain_length for record in conditioned)
                    / len(conditioned),
                    4,
                )
                if conditioned
                else None,
                "mean_assistant_action_turns": mean(record.assistant_action_turns for record in conditioned),
            }
        by_chain_length[str(chain_length)] = row

    # Same-question comparisons remove much of the difficulty/chain-length
    # confound.  A group is useful only when the sampled episode logs contain
    # at least one correct and one incorrect rollout for the same task.
    by_dataset_id: dict[str, list[EpisodeAudit]] = defaultdict(list)
    for record in records:
        by_dataset_id[record.dataset_id].append(record)
    mixed_diffs: list[float] = []
    mixed_correct_coverages: list[float] = []
    mixed_incorrect_coverages: list[float] = []
    mixed_groups = 0
    for group in by_dataset_id.values():
        group_correct = [record.prefinal_coverage for record in group if record.correct]
        group_incorrect = [record.prefinal_coverage for record in group if not record.correct]
        if not group_correct or not group_incorrect:
            continue
        mixed_groups += 1
        correct_coverage = statistics.fmean(group_correct)
        incorrect_coverage = statistics.fmean(group_incorrect)
        mixed_correct_coverages.append(correct_coverage)
        mixed_incorrect_coverages.append(incorrect_coverage)
        mixed_diffs.append(correct_coverage - incorrect_coverage)

    return {
        "audited_episodes": len(records),
        "skipped_episodes": dict(sorted(skip_reasons.items())),
        "correctness": correctness_summary,
        "failure_partial_progress": {
            "episodes": len(failure_partial_progress),
            "rate_among_incorrect": round(len(failure_partial_progress) / len(incorrect), 4) if incorrect else None,
            "entry_hit_but_later_node_missing": len(failure_entry_then_miss),
            "all_gold_nodes_hit_prefinal_but_final_incorrect": len(failure_all_nodes),
        },
        "by_chain_length": by_chain_length,
        "same_question_mixed_rollout_groups": {
            "groups": mixed_groups,
            "mean_correct_prefinal_coverage": mean(mixed_correct_coverages),
            "mean_incorrect_prefinal_coverage": mean(mixed_incorrect_coverages),
            "mean_within_question_coverage_difference_correct_minus_incorrect": mean(mixed_diffs),
            "share_groups_correct_coverage_higher": round(
                sum(difference > 0 for difference in mixed_diffs) / len(mixed_diffs), 4
            )
            if mixed_diffs
            else None,
        },
        "termination_by_correctness": {
            name: dict(Counter(record.termination_reason for record in subset))
            for name, subset in by_correctness.items()
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Intermediate-credit audit ===")
    print(f"audited chain-backed episodes: {summary['audited_episodes']}")
    print(f"skipped: {summary['skipped_episodes']}")
    for label, row in summary["correctness"].items():
        print(f"\n[{label}] episodes={row['episodes']}")
        for key in (
            "mean_reward",
            "mean_chain_length",
            "mean_prefinal_gold_node_coverage",
            "mean_all_message_gold_node_coverage",
            "mean_assistant_action_turns",
            "mean_tool_calls",
            "mean_search_tool_calls",
            "mean_read_url_calls",
        ):
            print(f"  {key}={row[key]}")
        print(f"  node_hit_buckets={row['node_hit_buckets']}")
        print(f"  node_hit_bucket_rates={row['node_hit_bucket_rates']}")
    print("\n[failure_partial_progress]", summary["failure_partial_progress"])
    print("[same_question_mixed_rollout_groups]", summary["same_question_mixed_rollout_groups"])
    print("[by_chain_length]")
    for chain_length, row in summary["by_chain_length"].items():
        print(f"  chain_length={chain_length}: {row}")


def case_payload(record: EpisodeAudit, text_limit: int) -> dict[str, Any]:
    payload = asdict(record)
    payload["question"] = record.question[:text_limit]
    payload["final_answer"] = record.final_answer[:text_limit]
    payload["bucket"] = bucket_name(record)
    return payload


def print_cases(records: list[EpisodeAudit], text_limit: int) -> list[dict[str, Any]]:
    wanted = (
        ("correct_all_nodes", lambda record: record.correct and bucket_name(record) == "all_gold_nodes_hit_prefinal"),
        ("incorrect_entry_then_miss", lambda record: not record.correct and bucket_name(record) == "entry_hit_but_later_node_missing"),
        ("incorrect_all_nodes_terminal_failure", lambda record: not record.correct and bucket_name(record) == "all_gold_nodes_hit_prefinal"),
        ("incorrect_no_node_many_turns", lambda record: not record.correct and bucket_name(record) == "no_prefinal_gold_node_hit"),
    )
    selected: list[dict[str, Any]] = []
    for label, predicate in wanted:
        candidates = [record for record in records if predicate(record)]
        if label == "incorrect_no_node_many_turns":
            candidates.sort(key=lambda record: record.assistant_action_turns, reverse=True)
        else:
            candidates.sort(key=lambda record: (record.training_step or -1, record.path))
        if not candidates:
            print(f"\n=== case: {label} (not found) ===")
            continue
        payload = case_payload(candidates[0], text_limit)
        payload["case_label"] = label
        selected.append(payload)
        print(f"\n=== case: {label} ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return selected


def main() -> int:
    args = parse_args()
    if not args.rollout_root.is_dir():
        raise FileNotFoundError(f"Rollout root not found: {args.rollout_root}")
    missing_datasets = [path for path in args.dataset_parquet if not path.is_file()]
    if missing_datasets:
        raise FileNotFoundError(f"Dataset parquet(s) not found: {missing_datasets}")

    dataset_index: dict[str, dict[str, Any]] = {}
    cross_dataset_duplicates = 0
    for dataset_path in args.dataset_parquet:
        for question, metadata in load_dataset_index(dataset_path).items():
            if question in dataset_index:
                cross_dataset_duplicates += 1
                continue
            dataset_index[question] = metadata
    if cross_dataset_duplicates:
        print(
            f"[warning] duplicate question text across supplied datasets: {cross_dataset_duplicates}",
            file=sys.stderr,
        )
    chains = load_chains(dataset_index)
    paths = iter_episode_paths(args.rollout_root, args.max_episodes, args.seed)
    print(
        f"episodes_found={len(paths)} dataset_questions={len(dataset_index)} "
        f"gold_chains_loaded={len(chains)}",
        file=sys.stderr,
    )

    records: list[EpisodeAudit] = []
    skipped: Counter[str] = Counter()
    for index, path in enumerate(paths, start=1):
        record, status = audit_episode(path, dataset_index, chains)
        if record is None:
            skipped[status] += 1
        else:
            records.append(record)
        if index % 500 == 0:
            print(f"audited={index}/{len(paths)} retained={len(records)}", file=sys.stderr)

    summary = summarize(records, skipped)
    print_summary(summary)
    selected_cases = print_cases(records, args.case_text_limit) if args.print_cases else []
    if args.report_json:
        report = {
            "config": {
                "rollout_root": str(args.rollout_root),
                "dataset_parquet": [str(path) for path in args.dataset_parquet],
                "max_episodes": args.max_episodes,
                "seed": args.seed,
                "metric_definition": (
                    "prefinal gold-node hit = conservative lexical match of a gold hop target "
                    "inside a non-final assistant action turn; it is a proxy, not correctness."
                ),
            },
            "summary": summary,
            "representative_cases": selected_cases,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_json={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
