#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate LLM_WORKER on image-description uniqueness examples.

The gold set is recovered from ``synthesis/ignore/DiscussioN_about_uniqueness.txt``:
raw ``image_*`` records provide the English descriptions and expert answer tables
provide the labels.  If the expert later revises an answer, the last uniqueness
label for that image ID wins (for example, broad championship celebrations that
were revised from unique to semi-unique).

Examples:
    # Inspect/export the recovered gold set without calling an API.
    python debug/eval_image_uniqueness.py --dataset-only

    # Evaluate every example, one LLM request per image.
    python debug/eval_image_uniqueness.py --model gpt54_internal_azure

    # Small smoke run and a resumable named output directory.
    python debug/eval_image_uniqueness.py \
      --model gpt54_internal_azure --limit 3 \
      --output-dir debug/outputs/image_uniqueness_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest


DEFAULT_DISCUSSION = ROOT / "synthesis/ignore/DiscussioN_about_uniqueness.txt"
DEFAULT_MODEL = os.environ.get("IMAGE_UNIQUENESS_EVAL_MODEL", "gpt54_internal_azure")
LABELS = ("唯一性", "半唯一性", "不唯一性")
SHORT_TO_CANONICAL = {
    "唯一": "唯一性",
    "唯一性": "唯一性",
    "unique": "唯一性",
    "半唯一": "半唯一性",
    "半唯一性": "半唯一性",
    "semi-unique": "半唯一性",
    "semi unique": "半唯一性",
    "semi_unique": "半唯一性",
    "不唯一": "不唯一性",
    "不唯一性": "不唯一性",
    "non-unique": "不唯一性",
    "non unique": "不唯一性",
    "non_unique": "不唯一性",
}

# Keep the user-supplied benchmark prompt in one constant so runs are auditable.
SYSTEM_PROMPT = """你是一个图片描述唯一性判断器。我会给你若干条数据，每条数据是「一张图片的英文描述短语/句子」（可能附带图片 URL、来源标签）。你的任务是：对每一条描述，判断它属于「唯一性」「半唯一性」还是「不唯一性」，并给出简短理由。

判断的基准思路：想象我拿这条描述去 Google 搜索图片，会返回什么样的图片池？据此归类。

# 三个类别的定义

## 唯一性（Unique）
满足以下任一条件即为唯一性：
1. 图片代表一个具体历史事件 / 现象发生时的某一特定瞬间（如某场决赛的夺冠庆祝、某次火箭发射、某人跳伞出舱的瞬间）。
   - 允许不同拍摄角度，但必须是同一时刻，或内容相同的一小段时间内。
2. 图片代表一个固定图案的物体本身：画作、专辑封面、电影海报、书籍装帧、雕塑定型作品、电影某一帧定格画面等。
3. 描述明确指向「同一张特定目标图片」，即使不对应唯一事件或艺术品。
   - 例："梅西 2022 世界杯后在 ins 上发布的搂着大力神杯睡觉的照片"——指向性唯一。
判据：用该描述搜图，返回的基本是同一张图，仅在尺寸/清晰度上有差异。

## 半唯一性（Semi-unique）
描述指向一个「地球上独一无二的实体本身」——某建筑、某地标、某景点、某雕塑、某人物（在某一时期）本身。
关键特征：不同人、不同时间、不同场合、不同角度都可能拍摄这个目标。
判据：用该描述搜图，能搜到正确的目标，但会是不同时间/角度/天气下拍摄的多张不同照片。

## 不唯一性（Non-unique）
满足以下任一即为不唯一：
1. 描述没有指向特定事件，而是指向一个无显著性、历史上多次发生的泛化事件。
2. 描述虽然提到特定时间，但无法定位到某一具体瞬间/具体画面（如"2025 年 CVPR 会议"——可能是官网、场馆、talk、晚宴等各种画面）。
3. 凭你使用 Google 的经验，认为搜索该描述返回的图片内容会各异、无法收敛。

# 判断步骤
1. 先判断描述指向的是「事件/瞬间」「固定图案物体」「独一无二实体」还是「泛化概念」。
2. 事件/瞬间 + 可定位到具体时刻 → 唯一性；固定图案物体 → 唯一性；明确指向某张特定图片 → 唯一性。
3. 独一无二实体本身（可被反复、多角度拍摄）→ 半唯一性。
4. 泛化事件 / 无法定位到具体瞬间 / 搜索结果会发散 → 不唯一性。

# 易错案例（务必内化）
- 【易错1｜"实体本身" vs "该实体的某次事件"】
  "The Scotiabank Saddledome"（这座场馆本身）→ 半唯一性。
  但 "View of the Scotiabank Saddledome's arena bowl filled with floodwater up to the 10th row in June 2013"（2013 年洪水淹到第 10 排的画面）→ 唯一性，因为它锁定了一个具体历史瞬间，而非场馆本身。
  ⚠️ 判断时要区分：描述的是「实体」还是「实体经历的某个特定时刻」。
- 【易错2｜"看似具体的会议/活动" 却无法定位瞬间】
  "2025 年 CVPR 会议" → 不唯一性。它有确切时间，容易被误判为唯一，但它无法收敛到某一张具体图片（官网、会场、演讲、晚宴都符合）。
  ⚠️ 有"确切时间"不等于唯一，必须能收敛到「同一张图/同一瞬间」。
- 【补充提示｜绘画/封面/海报一律唯一性】
  只要是画作、专辑封面、电影海报、书籍装帧等"固定图案物体"，即便它描绘的是一个泛化题材（如"描绘工人的画作"），图片本身仍是唯一的定型作品 → 唯一性。

# 输出格式
逐条输出：
[图片标识]  分类：唯一性/半唯一性/不唯一性  ｜  理由：（一句话）"""

# One record is sent per request. This suffix makes the requested analysis and
# final answer mechanically separable while retaining the user's final-line format.
OUTPUT_CONTRACT = """

本次只判断用户消息中的一条数据。请严格输出两行，不要输出 Markdown 表格或代码块：
分析：用简短中文说明搜索图片池为什么收敛或发散。
[图片标识]  分类：唯一性/半唯一性/不唯一性  ｜  理由：（一句话）
第二行必须使用输入中的真实图片标识，且分类只能三选一。"""


@dataclass(frozen=True)
class GoldExample:
    image_id: str
    description: str
    source_text_id: str
    source_label: str
    image_url: str
    gold_label: str
    expert_reason: str
    description_line: int
    gold_line: int


@dataclass
class EvalResult:
    image_id: str
    description: str
    source_text_id: str
    source_label: str
    image_url: str
    gold_label: str
    expert_reason: str
    predicted_label: str | None
    correct: bool
    parse_error: str | None
    analysis: str
    model_reason: str
    raw_response: str
    model_alias: str
    served_model: str | None
    usage: dict[str, Any] | None
    elapsed_seconds: float
    attempt_count: int
    error: str | None


_IMAGE_LINE_RE = re.compile(
    r"^(image_[0-9a-f]+)\s+(.*?)\s*(text_[0-9a-f]+)\s*\|\s*(.*)$"
)
_IMAGE_ID_RE = re.compile(r"image_[0-9a-f]+")
_URL_RE = re.compile(r"https?://\S+")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _strip_markdown(value: str) -> str:
    return _clean(value.replace("**", "").replace("`", ""))


def _canonical_label(value: str) -> str | None:
    normalized = _strip_markdown(value).strip(" ：:|。.;（）()[]").casefold()
    normalized = re.sub(r"（.*?）|\(.*?\)", "", normalized).strip()
    return SHORT_TO_CANONICAL.get(normalized)


def _parse_description_line(line: str) -> tuple[str, str, str, str, str] | None:
    match = _IMAGE_LINE_RE.match(line.strip())
    if not match:
        return None
    image_id, description, source_text_id, tail = match.groups()
    url_match = _URL_RE.search(tail)
    image_url = url_match.group(0).rstrip(".,") if url_match else ""
    source_label = tail[: url_match.start()] if url_match else tail
    return (
        image_id,
        _clean(description),
        source_text_id,
        _clean(source_label),
        image_url,
    )


def _parse_gold_table_row(line: str) -> tuple[str, str, str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or "image_" not in stripped:
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return None
    id_match = _IMAGE_ID_RE.search(cells[0])
    if not id_match:
        return None

    # Standard rows have [id, summary, label, reason]. Revision rows have
    # [id, old_label, new_label, reason]. Taking the last uniqueness label cell
    # correctly applies the expert's revised judgment.
    label_index = -1
    label: str | None = None
    for index, cell in enumerate(cells[1:], start=1):
        candidate = _canonical_label(cell)
        if candidate:
            label_index = index
            label = candidate
    if label is None:
        return None
    reason = _strip_markdown(cells[label_index + 1]) if label_index + 1 < len(cells) else ""
    return id_match.group(0), label, reason


def load_gold_examples(path: Path) -> list[GoldExample]:
    """Recover descriptions and the expert's final labels from the discussion."""

    text = path.read_text(encoding="utf-8")
    descriptions: dict[str, tuple[str, str, str, str, int]] = {}
    final_labels: dict[str, tuple[str, str, int]] = {}

    for line_no, line in enumerate(text.splitlines(), start=1):
        parsed_description = _parse_description_line(line)
        if parsed_description:
            image_id, description, source_text_id, source_label, image_url = parsed_description
            previous = descriptions.get(image_id)
            current = (description, source_text_id, source_label, image_url, line_no)
            if previous and previous[:4] != current[:4]:
                raise ValueError(f"Conflicting raw records for {image_id} at line {line_no}")
            descriptions.setdefault(image_id, current)

        parsed_gold = _parse_gold_table_row(line)
        if parsed_gold:
            image_id, label, reason = parsed_gold
            final_labels[image_id] = (label, reason, line_no)

    missing_labels = sorted(set(descriptions) - set(final_labels))
    missing_descriptions = sorted(set(final_labels) - set(descriptions))
    if missing_labels or missing_descriptions:
        raise ValueError(
            "Discussion extraction mismatch: "
            f"missing_labels={missing_labels}, missing_descriptions={missing_descriptions}"
        )

    examples: list[GoldExample] = []
    for image_id, (description, source_text_id, source_label, image_url, description_line) in descriptions.items():
        gold_label, expert_reason, gold_line = final_labels[image_id]
        examples.append(
            GoldExample(
                image_id=image_id,
                description=description,
                source_text_id=source_text_id,
                source_label=source_label,
                image_url=image_url,
                gold_label=gold_label,
                expert_reason=expert_reason,
                description_line=description_line,
                gold_line=gold_line,
            )
        )
    return examples


def build_user_prompt(
    example: GoldExample,
    *,
    include_source_label: bool,
    include_url: bool,
) -> str:
    lines = [f"图片标识：{example.image_id}", f"英文图片描述：{example.description}"]
    if include_source_label:
        lines.append(f"来源标签：{example.source_text_id} | {example.source_label}")
    if include_url:
        lines.append(f"图片 URL：{example.image_url}")
    return "\n".join(lines)


def parse_model_response(raw: str, *, expected_image_id: str) -> dict[str, str | None]:
    text = str(raw or "").strip()
    analysis_match = re.search(r"(?:^|\n)\s*分析\s*[：:]\s*(.+?)(?=\n\s*(?:image_[0-9a-f]+|\[?图片标识\]?|最终判断)\b|\Z)", text, re.S | re.I)
    analysis = _clean(analysis_match.group(1)) if analysis_match else ""

    final_candidates: list[tuple[str, str, str]] = []
    final_pattern = re.compile(
        r"(?P<id>image_[0-9a-f]+).*?分类\s*[：:]\s*"
        r"(?P<label>半唯一性|不唯一性|唯一性|半唯一|不唯一|唯一|semi[- _]?unique|non[- _]?unique|unique)"
        r"(?:\s*[｜|]\s*理由\s*[：:]\s*(?P<reason>[^\n]+))?",
        re.I,
    )
    for match in final_pattern.finditer(text):
        label = _canonical_label(match.group("label"))
        if label:
            final_candidates.append(
                (match.group("id"), label, _clean(match.group("reason") or ""))
            )

    if final_candidates:
        matching = [item for item in final_candidates if item[0] == expected_image_id]
        image_id, label, reason = (matching or final_candidates)[-1]
        return {
            "label": label,
            "analysis": analysis,
            "reason": reason,
            "parse_error": None if image_id == expected_image_id else f"response_id_mismatch:{image_id}",
        }

    # Conservative fallback for models that obey the labels but not the line format.
    explicit = re.findall(
        r"(?:最终判断|最终分类|分类|label)\s*[：:]\s*"
        r"(半唯一性|不唯一性|唯一性|半唯一|不唯一|唯一|semi[- _]?unique|non[- _]?unique|unique)",
        text,
        re.I,
    )
    if explicit:
        label = _canonical_label(explicit[-1])
        return {"label": label, "analysis": analysis, "reason": "", "parse_error": "fallback_label_parse"}
    return {"label": None, "analysis": analysis, "reason": "", "parse_error": "missing_final_label"}


def _request_one(
    example: GoldExample,
    *,
    model_alias: str,
    include_source_label: bool,
    include_url: bool,
    retries: int,
    retry_backoff: float,
) -> EvalResult:
    user_prompt = build_user_prompt(
        example,
        include_source_label=include_source_label,
        include_url=include_url,
    )
    last_error: str | None = None
    start = time.perf_counter()
    for attempt in range(1, retries + 2):
        try:
            response = LLM_WORKER.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=SYSTEM_PROMPT + OUTPUT_CONTRACT),
                        ModelMessage(role="user", content=user_prompt),
                    ],
                    metadata={
                        "trace_label": "eval_image_uniqueness",
                        "image_id": example.image_id,
                    },
                )
            )
            raw = str(response.content or "")
            parsed = parse_model_response(raw, expected_image_id=example.image_id)
            predicted = parsed["label"]
            return EvalResult(
                image_id=example.image_id,
                description=example.description,
                source_text_id=example.source_text_id,
                source_label=example.source_label,
                image_url=example.image_url,
                gold_label=example.gold_label,
                expert_reason=example.expert_reason,
                predicted_label=predicted,
                correct=bool(predicted == example.gold_label),
                parse_error=parsed["parse_error"],
                analysis=str(parsed["analysis"] or ""),
                model_reason=str(parsed["reason"] or ""),
                raw_response=raw,
                model_alias=model_alias,
                served_model=response.model,
                usage=response.usage,
                elapsed_seconds=round(time.perf_counter() - start, 4),
                attempt_count=attempt,
                error=None,
            )
        except Exception as exc:  # network/API failures are recorded per item
            last_error = f"{exc.__class__.__name__}: {exc}"
            if attempt <= retries:
                time.sleep(retry_backoff * (2 ** (attempt - 1)))

    return EvalResult(
        image_id=example.image_id,
        description=example.description,
        source_text_id=example.source_text_id,
        source_label=example.source_label,
        image_url=example.image_url,
        gold_label=example.gold_label,
        expert_reason=example.expert_reason,
        predicted_label=None,
        correct=False,
        parse_error="request_failed",
        analysis="",
        model_reason="",
        raw_response="",
        model_alias=model_alias,
        served_model=None,
        usage=None,
        elapsed_seconds=round(time.perf_counter() - start, 4),
        attempt_count=retries + 1,
        error=last_error,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _load_existing_results(path: Path) -> dict[str, EvalResult]:
    results: dict[str, EvalResult] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
                result = EvalResult(**payload)
            except (json.JSONDecodeError, TypeError) as exc:
                print(f"warning: ignoring malformed result {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            results[result.image_id] = result
    return results


def compute_metrics(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    parsed = [item for item in results if item.predicted_label in LABELS]
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for item in parsed:
        confusion[item.gold_label][str(item.predicted_label)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        support = sum(1 for item in results if item.gold_label == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    correct = sum(item.correct for item in results)
    macro_f1 = sum(float(per_class[label]["f1"]) for label in LABELS) / len(LABELS)
    usage_totals: Counter[str] = Counter()
    for item in results:
        if isinstance(item.usage, dict):
            for key, value in item.usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] += value

    return {
        "total": total,
        "parsed": len(parsed),
        "parse_rate": round(len(parsed) / total, 6) if total else 0.0,
        "correct": correct,
        "accuracy_all": round(correct / total, 6) if total else 0.0,
        "accuracy_parsed": round(correct / len(parsed), 6) if parsed else 0.0,
        "macro_f1": round(macro_f1, 6),
        "gold_distribution": dict(Counter(item.gold_label for item in results)),
        "prediction_distribution": dict(Counter(item.predicted_label or "UNPARSED" for item in results)),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "request_errors": sum(bool(item.error) for item in results),
        "format_warnings": sum(bool(item.parse_error) for item in results),
        "elapsed_seconds_sum": round(sum(item.elapsed_seconds for item in results), 4),
        "usage_totals": dict(usage_totals),
    }


def _write_csv(path: Path, results: list[EvalResult]) -> None:
    fields = [
        "image_id", "gold_label", "predicted_label", "correct", "parse_error", "error",
        "description", "expert_reason", "analysis", "model_reason", "raw_response",
        "source_text_id", "source_label", "image_url", "model_alias", "served_model",
        "elapsed_seconds", "attempt_count", "usage",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["usage"] = json.dumps(row["usage"], ensure_ascii=False) if row["usage"] else ""
            writer.writerow({field: row.get(field) for field in fields})


def _write_markdown_report(path: Path, results: list[EvalResult], metrics: dict[str, Any]) -> None:
    lines = [
        "# Image uniqueness evaluation",
        "",
        f"- Total: {metrics['total']}",
        f"- Parsed: {metrics['parsed']} ({metrics['parse_rate']:.2%})",
        f"- Correct: {metrics['correct']}",
        f"- Accuracy (all): {metrics['accuracy_all']:.2%}",
        f"- Accuracy (parsed): {metrics['accuracy_parsed']:.2%}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Request errors: {metrics['request_errors']}",
        "",
        "## Confusion matrix",
        "",
        "| gold \\ predicted | 唯一性 | 半唯一性 | 不唯一性 |",
        "|---|---:|---:|---:|",
    ]
    confusion = metrics["confusion_matrix"]
    for gold in LABELS:
        lines.append(f"| {gold} | {confusion[gold]['唯一性']} | {confusion[gold]['半唯一性']} | {confusion[gold]['不唯一性']} |")
    lines.extend(["", "## Errors and disagreements", ""])
    disagreements = [item for item in results if not item.correct]
    if not disagreements:
        lines.append("No disagreements.")
    else:
        lines.extend([
            "| image_id | gold | prediction | description | expert reason | model analysis/reason |",
            "|---|---|---|---|---|---|",
        ])
        for item in disagreements:
            model_text = item.analysis or item.model_reason or item.error or item.raw_response
            values = [
                item.image_id,
                item.gold_label,
                item.predicted_label or "UNPARSED",
                item.description,
                item.expert_reason,
                model_text,
            ]
            escaped = [str(value or "").replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(escaped) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discussion", type=Path, default=DEFAULT_DISCUSSION, help="Discussion text containing raw examples and expert labels.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Registered LLM_WORKER model alias.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Reusing it resumes completed image IDs by default.")
    parser.add_argument("--workers", type=int, default=4, help="Maximum concurrent requests (default: 4).")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first failed request.")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Initial exponential retry delay in seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N selected examples; <=0 means all.")
    parser.add_argument("--image-id", action="append", default=[], help="Evaluate only this image ID; may be repeated.")
    parser.add_argument("--dataset-only", action="store_true", help="Only extract/export the gold set; do not call LLM_WORKER.")
    parser.add_argument("--omit-url", action="store_true", help="Do not include the image URL in each user request.")
    parser.add_argument("--omit-source-label", action="store_true", help="Do not include the source text label in each user request.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior results.jsonl in the output directory.")
    parser.add_argument("--rerun-failures", action="store_true", help="When resuming, rerun request failures and unparsed responses.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    discussion = args.discussion.expanduser().resolve()
    if not discussion.is_file():
        raise SystemExit(f"error: discussion file does not exist: {discussion}")
    if args.workers <= 0:
        raise SystemExit("error: --workers must be positive")

    examples = load_gold_examples(discussion)
    if args.image_id:
        wanted = set(args.image_id)
        known = {item.image_id for item in examples}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"error: unknown --image-id values: {unknown}")
        examples = [item for item in examples if item.image_id in wanted]
    if args.limit > 0:
        examples = examples[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or ROOT / "debug/outputs" / f"image_uniqueness_eval_{timestamp}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "gold_examples.jsonl"
    _write_jsonl(gold_path, (asdict(item) for item in examples))
    (output_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + OUTPUT_CONTRACT + "\n", encoding="utf-8")

    gold_counts = Counter(item.gold_label for item in examples)
    print(f"[dataset] extracted={len(examples)} labels={dict(gold_counts)}")
    print(f"[dataset] snapshot={gold_path}")
    if args.dataset_only:
        return 0

    results_path = output_dir / "results.jsonl"
    existing = {} if args.no_resume else _load_existing_results(results_path)
    if args.no_resume and results_path.exists():
        results_path.unlink()

    pending: list[GoldExample] = []
    for example in examples:
        previous = existing.get(example.image_id)
        if previous is None:
            pending.append(example)
        elif args.rerun_failures and (previous.error or previous.predicted_label not in LABELS):
            pending.append(example)
        elif previous.model_alias != args.model:
            print(
                f"warning: resuming {example.image_id} produced by model={previous.model_alias!r}, "
                f"current --model={args.model!r}; use --no-resume to rerun it",
                file=sys.stderr,
            )

    run_config = {
        "created_at": datetime.now().isoformat(),
        "discussion": str(discussion),
        "model_alias": args.model,
        "workers": args.workers,
        "retries": args.retries,
        "include_url": not args.omit_url,
        "include_source_label": not args.omit_source_label,
        "selected_examples": len(examples),
        "pending_examples": len(pending),
        "system_prompt_file": "system_prompt.txt",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[eval] model={args.model} selected={len(examples)} resume={len(existing)} pending={len(pending)}")
    write_lock = threading.Lock()
    completed_now = 0
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _request_one,
                    example,
                    model_alias=args.model,
                    include_source_label=not args.omit_source_label,
                    include_url=not args.omit_url,
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                ): example
                for example in pending
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # defensive: _request_one already captures request errors
                    print(f"[error] {example.image_id}: {exc}", file=sys.stderr)
                    continue
                existing[result.image_id] = result
                _append_jsonl(results_path, asdict(result), write_lock)
                completed_now += 1
                prediction = result.predicted_label or "UNPARSED"
                marker = "OK" if result.correct else "MISS"
                print(
                    f"[{completed_now}/{len(pending)}] {marker} {result.image_id} "
                    f"gold={result.gold_label} pred={prediction} time={result.elapsed_seconds:.2f}s",
                    flush=True,
                )

    selected_ids = {item.image_id for item in examples}
    ordered_results = [existing[item.image_id] for item in examples if item.image_id in existing and item.image_id in selected_ids]
    if len(ordered_results) != len(examples):
        missing = [item.image_id for item in examples if item.image_id not in existing]
        print(f"warning: no result produced for {len(missing)} examples: {missing}", file=sys.stderr)

    # Compact away duplicate resumed/rerun JSONL entries and make ordering deterministic.
    _write_jsonl(results_path, (asdict(item) for item in ordered_results))
    metrics = compute_metrics(ordered_results)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "results.csv", ordered_results)
    _write_markdown_report(output_dir / "report.md", ordered_results, metrics)

    print(
        "[summary] "
        f"accuracy={metrics['accuracy_all']:.2%} parsed={metrics['parse_rate']:.2%} "
        f"macro_f1={metrics['macro_f1']:.4f} errors={metrics['request_errors']}"
    )
    print(f"[output] {output_dir}")
    return 0 if len(ordered_results) == len(examples) else 2


if __name__ == "__main__":
    raise SystemExit(main())
