#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate OpenSearch-VL inference trajectories with an LLM judge.

Examples:
    python opensearch_vl/eval_infer_with_llm.py --traj-dir /path/to/output_dir
    python opensearch_vl/eval_infer_with_llm.py --traj-dir /path/to/output_dir --answer-file /path/to/dataset.parquet
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


API_VERSION = os.environ.get("JUDGE_API_VERSION", "v2.03")
BASE_URL = os.environ.get("JUDGE_API_BASE_URL", "")
APP_ID = os.environ.get("JUDGE_APP_ID", "")
APP_KEY = os.environ.get("JUDGE_APP_KEY", "")
MODEL_MARKER = os.environ.get("JUDGE_MODEL_MARKER", "api_openai_gpt-4o")


JUDGE_PROMPT = (
    "You are an impartial judge evaluating whether a model answer matches the "
    "reference answer for a visual question answering task.\n\n"
    "[Question]\n{question}\n\n"
    "[Reference Answer]\n{correct_answer}\n\n"
    "[Model Answer]\n{response}\n\n"
    "Task: Determine whether the model answer is correct.\n\n"
    "Instructions:\n"
    "1. Judge semantic correctness, not surface form.\n"
    "2. Accept paraphrases that preserve the same meaning.\n"
    "3. Reject answers that are incomplete, contradictory, or unsupported.\n"
    "4. Provide a short reason.\n\n"
    "Output format:\n"
    "correct: [yes/no]\n"
    "reasoning: [your explanation]"
)


def get_simple_auth(source: str, secret_id: str, secret_key: str) -> tuple[str, str]:
    date_time = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    auth = 'hmac id="' + secret_id + '", algorithm="hmac-sha1", headers="date source", signature="'
    sign_str = "date: " + date_time + "\n" + "source: " + source
    sign = hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    return auth + sign + '"', date_time


def call_judge(prompt: str, max_retries: int = 5, timeout: int = 120) -> str:
    if not (BASE_URL and APP_ID and APP_KEY):
        raise RuntimeError(
            "Judge API is not configured. Please set JUDGE_API_BASE_URL, "
            "JUDGE_APP_ID and JUDGE_APP_KEY."
        )

    for attempt in range(max_retries):
        try:
            sign, date_time = get_simple_auth("gpt-54-eval", APP_ID, APP_KEY)
            headers = {
                "Apiversion": API_VERSION,
                "Authorization": sign,
                "Date": date_time,
                "Source": "gpt-54-eval",
                "Content-Type": "application/json",
            }
            body = {
                "request_id": str(uuid.uuid4()),
                "model_marker": MODEL_MARKER,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "value": prompt}],
                    }
                ],
                "system": "",
                "params": {"stream": False, "temperature": 0.0},
                "timeout": timeout,
            }
            response = requests.post(
                f"{BASE_URL}/api/v1/data_eval",
                headers=headers,
                json=body,
                timeout=timeout + 30,
            )
            if response.status_code != 200:
                time.sleep(2 ** attempt)
                continue

            payload = response.json()
            if "answer" in payload and isinstance(payload["answer"], list) and payload["answer"]:
                return payload["answer"][0].get("value", "")
            if "choices" in payload and payload["choices"]:
                return payload["choices"][0].get("message", {}).get("content", "")
            if "data" in payload and "choices" in payload["data"] and payload["data"]["choices"]:
                return payload["data"]["choices"][0].get("message", {}).get("content", "")
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            time.sleep(2 ** attempt)
    return ""


def extract_boxed_content(text: str) -> str | None:
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    content = text[start : i - 1].strip()
    content = re.sub(r"\\text\{([^}]*)\}", r"\1", content)
    return content.replace("\\#", "#").replace("\\%", "%")


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    boxed = extract_boxed_content(text)
    if boxed:
        return boxed

    answer_tag = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()

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


def judge_answer(question: str, correct_answer: str, response: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response[:4000],
    )
    raw = call_judge(prompt)
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


def process_single_trajectory(path: Path, answer_map: dict[str, str]) -> dict[str, Any]:
    traj = load_trajectory(path)
    case_id = str(traj.get("case_id") or path.stem)
    question = _extract_question(traj)
    correct_answer = _extract_reference_answer(traj, answer_map)
    final_text = str(traj.get("final_response_text", "") or "")
    model_answer = extract_final_answer(final_text)

    judge_result = judge_answer(question, correct_answer, model_answer)
    return {
        "case_id": case_id,
        "question": question,
        "correct_answer": correct_answer,
        "model_answer": model_answer,
        **judge_result,
    }


def run_eval(
    traj_dir: str,
    *,
    answer_file: str | None = None,
    output_path: str | None = None,
    max_workers: int = 8,
    limit: int = 0,
) -> dict[str, Any]:
    traj_files = sorted(Path(traj_dir).glob("*_trajectory.json"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {traj_dir}")

    if limit > 0:
        traj_files = traj_files[:limit]

    answer_map = load_answer_map(answer_file) if answer_file else {}
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_single_trajectory, path, answer_map): path
            for path in traj_files
        }
        for future in tqdm(as_completed(future_map), total=len(traj_files), desc="Judge"):
            path = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "case_id": path.stem,
                        "acc": 0,
                        "reasoning": "",
                        "raw_judge": "",
                        "error": str(exc),
                    }
                )

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
        "judge_model": MODEL_MARKER,
        "error_count": error_count,
        "label_distribution": dict(acc_counter),
    }

    if output_path is None:
        output_path = os.path.join(traj_dir, "llm_eval_report.json")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    details_path = output_path.replace(".json", "_details.jsonl")
    with open(details_path, "w", encoding="utf-8") as handle:
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
    args = parser.parse_args()

    report = run_eval(
        traj_dir=args.traj_dir,
        answer_file=args.answer_file,
        output_path=args.output,
        max_workers=args.max_workers,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
