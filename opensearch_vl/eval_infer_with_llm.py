#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate OpenSearch-VL inference trajectories with an LLM judge.

Examples:
    python opensearch_vl/eval_infer_with_llm.py --traj-dir /path/to/output_dir
    python opensearch_vl/eval_infer_with_llm.py --traj-dir /path/to/output_dir --answer-file /path/to/dataset.parquet
    python opensearch_vl/eval_infer_with_llm.py --traj-dir /path/to/output_dir \
        --prompt-style official --rerun
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm
from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest


JUDGE_MODEL_ALIAS = os.environ.get("JUDGE_MODEL_ALIAS", "gpt54_internal_azure")
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "1024"))
DEFAULT_JUDGE_PROMPT_STYLE = os.environ.get("JUDGE_PROMPT_STYLE", "official").strip().lower()


LOCAL_JUDGE_PROMPT = (
    "You are an impartial evaluator for a visual question answering benchmark.\n\n"
    "[Question]\n{question}\n\n"
    "[Reference Answer]\n{correct_answer}\n\n"
    "[Model Answer]\n{response}\n\n"
    "Task: Determine whether the model provides the correct answer to the question.\n\n"
    "Evaluation rules:\n"
    "1. Treat the reference answer as the ground truth. Do not fact-check, reinterpret, or challenge it.\n"
    "2. Evaluate the model's final substantive answer, not the quality of its reasoning process.\n"
    "3. Ignore search descriptions, citations, intermediate reasoning, and extra context unless they "
    "directly change or contradict the final answer.\n"
    "4. Accept paraphrases, aliases, reordered names, capitalization differences, punctuation differences, "
    "unit conversions, and equivalent date formats.\n"
    "5. If the model gives a more specific answer that logically includes or entails the reference answer, "
    "mark it correct. For example, '6,500 light-years' satisfies 'over 2,900 light-years'.\n"
    "6. Do not require exact numerical string matching when the answer is an approximation, bound, range, "
    "or equivalent measurement. Reject materially different numerical values when exactness is required.\n"
    "7. Additional relevant information does not make an answer incorrect. Mark it incorrect only if the "
    "additional information contradicts the reference answer or changes the answer to the question.\n"
    "8. If the model gives a clear final answer but its surrounding explanation contains unrelated or minor "
    "factual errors, judge the final answer itself. Mark it incorrect only if the explanation creates a direct "
    "contradiction or makes the intended answer ambiguous.\n"
    "9. If the question is invalid or based on a false premise, the model must identify that invalidity; "
    "confidently answering it as an ordinary factual question is incorrect.\n"
    "10. Mark an answer incorrect if it gives the wrong entity, wrong attribute, wrong relation, materially "
    "wrong number/date, incomplete required items, or an answer that is too vague to answer the question.\n"
    "11. If the model answer is empty, evasive, or says the answer cannot be determined, mark it incorrect "
    "unless the reference answer itself indicates an invalid or unanswerable question.\n"
    "12. Provide a brief reason focused on the final answer.\n\n"
    "Output format:\n"
    "correct: [yes/no]\n"
    "reasoning: [your explanation]"
)


# Canonical prompt used by the official OpenSearch-VL evaluator:
# https://github.com/shawn0728/OpenSearch-VL/blob/main/opensearch_vl/eval_with_gpt4o.py
OFFICIAL_JUDGE_PROMPT = (
    "You are an impartial judge evaluating whether a deep research report contains the correct answer.\n\n"
    "[Question]\n{question}\n\n"
    "[Correct Answer]\n{correct_answer}\n\n"
    "[Deep Research Report]\n{response}\n\n"
    "Task: Determine if the deep research report contains the correct answer anywhere in its content.\n\n"
    "Instructions:\n"
    "1. Read through the entire research report carefully\n"
    "2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n"
    "3. Check if the information in the report is consistent with the correct answer\n"
    "4. The answer does NOT need to be in a specific format or labeled as \"final answer\"\n"
    "5. Provide your reasoning\n"
    "6. Answer with \"yes\" if the report contains the correct answer, \"no\" if it doesn't or contradicts it\n\n"
    "Output format:\n"
    "correct: [yes/no]\n"
    "reasoning: [your explanation]"
)


JUDGE_PROMPTS = {
    "local": LOCAL_JUDGE_PROMPT,
    "official": OFFICIAL_JUDGE_PROMPT,
}


def get_judge_prompt(prompt_style: str) -> str:
    style = str(prompt_style or "").strip().lower()
    if style not in JUDGE_PROMPTS:
        choices = ", ".join(sorted(JUDGE_PROMPTS))
        raise ValueError(f"Unknown judge prompt style {prompt_style!r}; choose one of: {choices}")
    return JUDGE_PROMPTS[style]


def call_judge(
    prompt: str,
    *,
    judge_model_alias: str = JUDGE_MODEL_ALIAS,
    judge_max_tokens: int = JUDGE_MAX_TOKENS,
) -> str:
    response = LLM_WORKER.generate(
        ModelRequest(
            model=judge_model_alias,
            messages=[ModelMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=judge_max_tokens,
            metadata={"trace_label": "eval_infer_with_llm_judge"},
        )
    )
    return str(response.content or "")


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    answer_tag = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()

    boxed_idx = text.find("\\boxed{")
    if boxed_idx != -1:
        start = boxed_idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        boxed = text[start : i - 1].strip()
        boxed = re.sub(r"\\text\{([^}]*)\}", r"\1", boxed)
        boxed = boxed.replace("\\#", "#").replace("\\%", "%")
        if boxed:
            return boxed

    response_tag = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if response_tag:
        return response_tag.group(1).strip()

    parts = re.split(r"</think>|</thinking>", text)
    if len(parts) > 1:
        last_part = parts[-1].strip()
        last_part = re.sub(r"<tool_call>.*?</tool_call>", "", last_part, flags=re.DOTALL).strip()
        if last_part:
            return last_part[:2000]

    return text[-2000:] if len(text) > 2000 else text


def parse_judge_response(raw: str) -> dict[str, Any]:
    acc = 0
    reasoning = ""
    correct_match = re.search(r"correct:\s*(yes|no)", raw, re.IGNORECASE)
    if correct_match:
        acc = 1 if correct_match.group(1).lower() == "yes" else 0
    reasoning_match = re.search(r"reasoning:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    return {"acc": acc, "reasoning": reasoning, "raw_judge": raw}


def judge_answer(
    question: str,
    correct_answer: str,
    response: str,
    *,
    judge_model_alias: str = JUDGE_MODEL_ALIAS,
    judge_max_tokens: int = JUDGE_MAX_TOKENS,
    prompt_style: str = DEFAULT_JUDGE_PROMPT_STYLE,
) -> dict[str, Any]:
    prompt = get_judge_prompt(prompt_style).format(
        question=question,
        correct_answer=correct_answer,
        response=response[:4000],
    )
    raw = call_judge(
        prompt,
        judge_model_alias=judge_model_alias,
        judge_max_tokens=judge_max_tokens,
    )
    return parse_judge_response(raw)


def load_trajectory(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_answer_map(parquet_path: str) -> dict[str, str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        answer = str(row.get("answer", "") or "")
        for key in ("question_id", "sample_id", "data_id", "id"):
            value = str(row.get(key, "") or "").strip()
            if value:
                mapping[value] = answer
    return mapping


def _extract_question(traj: dict[str, Any]) -> str:
    prompt_msgs = traj.get("prompt", [])
    if isinstance(prompt_msgs, list):
        for msg in prompt_msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = str(msg.get("content", "") or "").strip()
                if content:
                    return content

    original = traj.get("original_data", {})
    return str(
        original.get("question")
        or original.get("final_question")
        or original.get("polished_question")
        or original.get("draft_question")
        or ""
    ).strip()


def _extract_reference_answer(
    traj: dict[str, Any],
    answer_map: dict[str, str],
) -> str:
    original = traj.get("original_data", {})

    direct = str(original.get("answer", "") or "").strip()
    if direct:
        return direct

    answers = original.get("answers", [])
    if isinstance(answers, list) and answers:
        return ", ".join(str(item) for item in answers if str(item).strip())

    for key in ("question_id", "sample_id", "data_id", "case_id", "id"):
        value = str(
            original.get(key)
            or traj.get(key)
            or ""
        ).strip()
        if value and value in answer_map:
            return answer_map[value]

    return ""


def process_single_trajectory(
    path: Path,
    answer_map: dict[str, str],
    judge_model_alias: str,
    judge_max_tokens: int,
    prompt_style: str = DEFAULT_JUDGE_PROMPT_STYLE,
) -> dict[str, Any]:
    prompt_style = str(prompt_style or DEFAULT_JUDGE_PROMPT_STYLE).strip().lower()
    traj = load_trajectory(path)
    case_id = str(traj.get("case_id") or path.stem)
    question = _extract_question(traj)
    correct_answer = _extract_reference_answer(traj, answer_map)
    final_text = str(traj.get("final_response_text", "") or "")
    model_answer = extract_final_answer(final_text)

    judge_result = judge_answer(
        question,
        correct_answer,
        model_answer,
        judge_model_alias=judge_model_alias,
        judge_max_tokens=judge_max_tokens,
        prompt_style=prompt_style,
    )
    return {
        "case_id": case_id,
        "question": question,
        "correct_answer": correct_answer,
        "model_answer": model_answer,
        "judge_prompt_style": prompt_style,
        **judge_result,
    }


def _trajectory_case_id(path: Path) -> str:
    name = path.name
    suffix = "_trajectory.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _load_existing_results(details_path: Path) -> dict[str, dict[str, Any]]:
    if not details_path.exists():
        return {}
    results_by_case: dict[str, dict[str, Any]] = {}
    with details_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = str(item.get("case_id", "") or "").strip()
            if case_id:
                results_by_case[case_id] = item
    return results_by_case


def _order_results(
    traj_files: list[Path],
    results_by_case: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in traj_files:
        case_id = _trajectory_case_id(path)
        item = results_by_case.get(case_id)
        if item is not None:
            ordered.append(item)
            seen.add(case_id)
    for case_id, item in results_by_case.items():
        if case_id not in seen:
            ordered.append(item)
    return ordered


def run_eval(
    traj_dir: str,
    *,
    answer_file: str | None = None,
    output_path: str | None = None,
    max_workers: int = 8,
    limit: int = 0,
    judge_model_alias: str = JUDGE_MODEL_ALIAS,
    judge_max_tokens: int = JUDGE_MAX_TOKENS,
    prompt_style: str = DEFAULT_JUDGE_PROMPT_STYLE,
    rerun: bool = False,
) -> dict[str, Any]:
    prompt_style = str(prompt_style or "").strip().lower()
    get_judge_prompt(prompt_style)  # Validate before starting any judge requests.

    traj_files = sorted(Path(traj_dir).glob("*_trajectory.json"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {traj_dir}")

    if limit > 0:
        traj_files = traj_files[:limit]

    if output_path is None:
        output_path = os.path.join(traj_dir, "llm_eval_report.json")
    details_path = Path(output_path.replace(".json", "_details.jsonl"))

    answer_map = load_answer_map(answer_file) if answer_file else {}
    if rerun:
        # Explicitly discard the old result index. This reruns judging only; trajectories
        # are read from traj_dir and no inference/model rollout is started here.
        results_by_case: dict[str, dict[str, Any]] = {}
        pending_files = list(traj_files)
        print(
            f"[rerun] Ignoring existing evaluation results and judging "
            f"{len(pending_files)} trajectory file(s) from scratch"
        )
    else:
        results_by_case = _load_existing_results(details_path)
        completed_case_ids = {
            case_id
            for case_id, item in results_by_case.items()
            if not item.get("error")
        }
        pending_files = [
            path for path in traj_files if _trajectory_case_id(path) not in completed_case_ids
        ]

        if completed_case_ids:
            print(
                f"[resume] Reusing {len(completed_case_ids)} completed evaluation result(s) "
                f"from {details_path}"
            )

    if pending_files:
        details_mode = "w" if rerun else "a"
        with details_path.open(details_mode, encoding="utf-8") as append_handle:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        process_single_trajectory,
                        path,
                        answer_map,
                        judge_model_alias,
                        judge_max_tokens,
                        prompt_style,
                    ): path
                    for path in pending_files
                }
                for future in tqdm(as_completed(future_map), total=len(pending_files), desc="Judge"):
                    path = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "case_id": _trajectory_case_id(path),
                            "acc": 0,
                            "reasoning": "",
                            "raw_judge": "",
                            "judge_prompt_style": prompt_style,
                            "error": str(exc),
                        }
                    results_by_case[str(result.get("case_id", ""))] = result
                    append_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    append_handle.flush()
    else:
        print("[resume] No pending trajectories to evaluate.")

    results = _order_results(traj_files, results_by_case)

    total = len(results)
    correct = sum(int(item.get("acc", 0)) for item in results)
    accuracy = (correct / total * 100.0) if total else 0.0
    error_count = sum(1 for item in results if item.get("error"))
    acc_counter = Counter(int(item.get("acc", 0)) for item in results)

    report = {
        "traj_dir": traj_dir,
        "total": total,
        "correct": correct,
        "accuracy": f"{accuracy:.2f}%",
        "judge_model": judge_model_alias,
        "judge_prompt_style": prompt_style,
        "rerun": rerun,
        "error_count": error_count,
        "label_distribution": dict(acc_counter),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with details_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", required=True, help="Directory containing *_trajectory.json files.")
    parser.add_argument("--answer-file", default=None, help="Optional parquet file containing reference answers.")
    parser.add_argument("--output", default=None, help="Output report path.")
    parser.add_argument("--max-workers", type=int, default=8, help="Judge parallelism.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N trajectories.")
    parser.add_argument(
        "--judge-model-alias",
        default=JUDGE_MODEL_ALIAS,
        help="Registered synthesis/models.json alias used for the LLM judge.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=JUDGE_MAX_TOKENS,
        help="Max tokens for the judge model response.",
    )
    parser.add_argument(
        "--prompt-style",
        "--judge-prompt",
        dest="prompt_style",
        choices=sorted(JUDGE_PROMPTS),
        default=DEFAULT_JUDGE_PROMPT_STYLE,
        help="Judge prompt to use: local (current custom prompt) or official (OpenSearch-VL canonical prompt).",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore existing *_details.jsonl results and re-score every trajectory from scratch; does not run inference.",
    )
    args = parser.parse_args()

    report = run_eval(
        traj_dir=args.traj_dir,
        answer_file=args.answer_file,
        output_path=args.output,
        max_workers=args.max_workers,
        limit=args.limit,
        judge_model_alias=args.judge_model_alias,
        judge_max_tokens=args.judge_max_tokens,
        prompt_style=args.prompt_style,
        rerun=args.rerun,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
