#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify graph image nodes by image-description uniqueness.

Visual-plan images are judged independently from their persisted ``search_query``
using ``LLM_WORKER``. Wikipedia-inline images do not require a model call and are
assigned ``unique_state = "wiki_inline"``.

No sampling/generation parameters are passed by this script; all such settings
come from the selected model alias in ``synthesis/models.json``.

Examples:
    python synthesis/post_process/classify_image_unique_state.py \
      --graph-dir runs/my_graph \
      --judge-model gpt54_internal_azure

    python synthesis/post_process/classify_image_unique_state.py \
      --graph-dir runs/my_graph \
      --judge-model gpt54_internal_azure \
      --dry-run --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelWorkerClient
from synthesis.store import JsonlGraphStore


UNIQUE_STATE_FIELD = "unique_state"
VALID_UNIQUE_STATES = {"unique", "semi-unique", "no-unique", "wiki_inline"}
VISUAL_PLAN_SOURCE_TYPES = {"image_search", "image_search_bundle"}
WIKI_INLINE_SOURCE_TYPES = {"wikipedia_inline_image"}
CHINESE_TO_STATE = {
    "唯一性": "unique",
    "半唯一性": "semi-unique",
    "不唯一性": "no-unique",
}
_SHORT_TO_CHINESE = {
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
    "no-unique": "不唯一性",
    "no unique": "不唯一性",
    "no_unique": "不唯一性",
}

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

OUTPUT_CONTRACT = """

本次只判断用户消息中的一条数据。请严格输出两行，不要输出 Markdown 表格或代码块：
分析：用简短中文说明搜索图片池为什么收敛或发散。
[图片标识]  分类：唯一性/半唯一性/不唯一性  ｜  理由：（一句话）
第二行必须使用输入中的真实图片标识，且分类只能三选一。"""

_IMAGE_ID_RE = re.compile(r"[A-Za-z0-9_.:-]+")


@dataclass
class JudgeResult:
    image_node_id: str
    search_query: str
    unique_state: str | None
    chinese_label: str | None
    analysis: str
    reason: str
    raw_response: str
    judge_model_alias: str
    served_model: str | None
    usage: dict[str, Any] | None
    elapsed_seconds: float
    attempt_count: int
    parse_warning: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _canonical_chinese_label(value: str) -> str | None:
    normalized = _clean(value).replace("**", "").replace("`", "")
    normalized = normalized.strip(" ：:|。.;（）()[]").casefold()
    normalized = re.sub(r"（.*?）|\(.*?\)", "", normalized).strip()
    return _SHORT_TO_CHINESE.get(normalized)


def image_origin(
    node: dict[str, Any],
    incoming_edges: Iterable[dict[str, Any]] = (),
    *,
    nodes_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Classify origin, giving wiki-inline markers precedence over all fallbacks."""

    metadata = _as_dict(node.get("metadata"))
    source = _as_dict(node.get("source"))
    source_type = _clean(source.get("source_type")).lower()
    origin = _clean(metadata.get("image_origin")).lower()
    variant_sources = {
        _clean(_as_dict(variant).get("source")).lower()
        for variant in node.get("image_variants") or []
        if isinstance(variant, dict)
    }
    if (
        source_type in WIKI_INLINE_SOURCE_TYPES
        or origin in {"wikipedia_inline", "wiki_inline"}
        or "wikipedia_inline" in variant_sources
    ):
        return "wiki_inline"
    if source_type in VISUAL_PLAN_SOURCE_TYPES or origin == "visual_plan":
        return "visual_plan"
    for edge in incoming_edges:
        if str(edge.get("src_node_type") or "") == "text":
            return "visual_plan"
        if nodes_by_id is not None:
            source_node = nodes_by_id.get(str(edge.get("src_node_id") or ""))
            if source_node is not None and source_node.get("node_type") == "text":
                return "visual_plan"
    return "other"


def image_search_query(node: dict[str, Any]) -> str:
    metadata = _as_dict(node.get("metadata"))
    return _clean(metadata.get("search_query") or metadata.get("query"))


def build_user_prompt(image_node_id: str, search_query: str) -> str:
    return f"图片标识：{image_node_id}\n英文图片描述：{search_query}"


def parse_judge_response(raw: str, *, expected_image_id: str) -> dict[str, str | None]:
    text = str(raw or "").strip()
    analysis_match = re.search(
        r"(?:^|\n)\s*分析\s*[：:]\s*(.+?)(?=\n\s*[^\s]+\s+分类\s*[：:]|\Z)",
        text,
        re.S | re.I,
    )
    analysis = _clean(analysis_match.group(1)) if analysis_match else ""

    final_pattern = re.compile(
        r"(?P<id>[^\s\[\]]+)\s+分类\s*[：:]\s*"
        r"(?P<label>半唯一性|不唯一性|唯一性|半唯一|不唯一|唯一|semi[- _]?unique|non[- _]?unique|no[- _]?unique|unique)"
        r"(?:\s*[｜|]\s*理由\s*[：:]\s*(?P<reason>[^\n]+))?",
        re.I,
    )
    candidates: list[tuple[str, str, str]] = []
    for match in final_pattern.finditer(text):
        label = _canonical_chinese_label(match.group("label"))
        if label:
            candidates.append((match.group("id"), label, _clean(match.group("reason") or "")))
    if candidates:
        matching = [item for item in candidates if item[0] == expected_image_id]
        response_id, label, reason = (matching or candidates)[-1]
        return {
            "label": label,
            "analysis": analysis,
            "reason": reason,
            "parse_warning": None if response_id == expected_image_id else f"response_id_mismatch:{response_id}",
        }

    explicit = re.findall(
        r"(?:最终判断|最终分类|分类|label)\s*[：:]\s*"
        r"(半唯一性|不唯一性|唯一性|半唯一|不唯一|唯一|semi[- _]?unique|non[- _]?unique|no[- _]?unique|unique)",
        text,
        re.I,
    )
    if explicit:
        return {
            "label": _canonical_chinese_label(explicit[-1]),
            "analysis": analysis,
            "reason": "",
            "parse_warning": "fallback_label_parse",
        }
    return {"label": None, "analysis": analysis, "reason": "", "parse_warning": "missing_final_label"}


def judge_image_node(
    *,
    image_node_id: str,
    search_query: str,
    judge_model_alias: str,
    model_client: ModelWorkerClient = LLM_WORKER,
    retries: int = 2,
    retry_backoff: float = 1.0,
) -> JudgeResult:
    """Judge one visual-plan query without overriding model sampling parameters."""

    started = time.perf_counter()
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            response = model_client.generate(
                ModelRequest(
                    model=judge_model_alias,
                    messages=[
                        ModelMessage(role="system", content=SYSTEM_PROMPT + OUTPUT_CONTRACT),
                        ModelMessage(role="user", content=build_user_prompt(image_node_id, search_query)),
                    ],
                    metadata={
                        "trace_label": "classify_image_unique_state",
                        "image_node_id": image_node_id,
                    },
                )
            )
            raw = str(response.content or "")
            parsed = parse_judge_response(raw, expected_image_id=image_node_id)
            chinese_label = parsed["label"]
            unique_state = CHINESE_TO_STATE.get(str(chinese_label)) if chinese_label else None
            error = None if unique_state else "unparseable_judge_response"
            return JudgeResult(
                image_node_id=image_node_id,
                search_query=search_query,
                unique_state=unique_state,
                chinese_label=chinese_label,
                analysis=str(parsed["analysis"] or ""),
                reason=str(parsed["reason"] or ""),
                raw_response=raw,
                judge_model_alias=judge_model_alias,
                served_model=response.model,
                usage=response.usage,
                elapsed_seconds=round(time.perf_counter() - started, 4),
                attempt_count=attempt,
                parse_warning=parsed["parse_warning"],
                error=error,
            )
        except Exception as exc:  # network/API errors are recorded per image
            last_error = f"{exc.__class__.__name__}: {exc}"
            if attempt <= retries:
                time.sleep(retry_backoff * (2 ** (attempt - 1)))

    return JudgeResult(
        image_node_id=image_node_id,
        search_query=search_query,
        unique_state=None,
        chinese_label=None,
        analysis="",
        reason="",
        raw_response="",
        judge_model_alias=judge_model_alias,
        served_model=None,
        usage=None,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        attempt_count=retries + 1,
        parse_warning=None,
        error=last_error or "judge_request_failed",
    )


def _load_checkpoint(path: Path) -> dict[str, JudgeResult]:
    results: dict[str, JudgeResult] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                result = JudgeResult(**payload)
            except (json.JSONDecodeError, TypeError) as exc:
                print(f"warning: ignoring invalid checkpoint line {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            results[result.image_node_id] = result
    return results


def _append_checkpoint(path: Path, result: JudgeResult, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _rewrite_checkpoint(path: Path, results: Iterable[JudgeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def classify_graph(
    *,
    graph_dir: Path,
    judge_model_alias: str,
    model_client: ModelWorkerClient = LLM_WORKER,
    workers: int = 4,
    retries: int = 2,
    retry_backoff: float = 1.0,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_partial: bool = False,
    results_jsonl: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    graph_dir = graph_dir.expanduser().resolve()
    store = JsonlGraphStore(graph_dir)
    nodes = store.list_nodes()
    edges = store.list_edges()
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incoming[str(edge.get("dst_node_id") or "")].append(edge)

    image_nodes = sorted(
        (node for node in nodes if node.get("node_type") == "image" and node.get("node_id")),
        key=lambda node: str(node.get("node_id")),
    )
    checkpoint_path = (results_jsonl or graph_dir / "image_unique_state_results.jsonl").expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path) if resume else {}

    counters: Counter[str] = Counter()
    origins: dict[str, str] = {}
    visual_queries: dict[str, str] = {}
    all_nodes_by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id")}
    nodes_by_id = {str(node.get("node_id")): node for node in image_nodes}

    for node in image_nodes:
        node_id = str(node["node_id"])
        origin = image_origin(
            node,
            incoming.get(node_id, []),
            nodes_by_id=all_nodes_by_id,
        )
        origins[node_id] = origin
        counters[f"origin_{origin}"] += 1
        existing_state = str(node.get(UNIQUE_STATE_FIELD) or "").strip()
        if not overwrite and existing_state in VALID_UNIQUE_STATES:
            counters["already_classified"] += 1
            continue
        if origin == "wiki_inline":
            counters["wiki_inline_to_assign"] += 1
            continue
        if origin != "visual_plan":
            counters["other_skipped"] += 1
            continue
        query = image_search_query(node)
        if not query:
            counters["missing_search_query"] += 1
            continue
        visual_queries[node_id] = query

    judge_results: dict[str, JudgeResult] = {}
    pending: list[tuple[str, str]] = []
    for node_id, query in visual_queries.items():
        cached = checkpoint.get(node_id)
        if (
            cached is not None
            and cached.search_query == query
            and cached.judge_model_alias == judge_model_alias
            and cached.unique_state in {"unique", "semi-unique", "no-unique"}
            and not overwrite
        ):
            judge_results[node_id] = cached
            counters["checkpoint_reused"] += 1
        else:
            pending.append((node_id, query))

    checkpoint_lock = threading.Lock()
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    judge_image_node,
                    image_node_id=node_id,
                    search_query=query,
                    judge_model_alias=judge_model_alias,
                    model_client=model_client,
                    retries=retries,
                    retry_backoff=retry_backoff,
                ): node_id
                for node_id, query in pending
            }
            completed = 0
            for future in as_completed(futures):
                node_id = futures[future]
                result = future.result()
                judge_results[node_id] = result
                checkpoint[node_id] = result
                _append_checkpoint(checkpoint_path, result, checkpoint_lock)
                completed += 1
                state = result.unique_state or "ERROR"
                print(
                    f"[{completed}/{len(pending)}] {node_id} unique_state={state} "
                    f"time={result.elapsed_seconds:.2f}s",
                    flush=True,
                )

    failed_results = [result for result in judge_results.values() if result.unique_state is None]
    missing_query_ids = [
        node_id
        for node_id, origin in origins.items()
        if origin == "visual_plan"
        and not image_search_query(nodes_by_id[node_id])
        and (overwrite or str(nodes_by_id[node_id].get(UNIQUE_STATE_FIELD) or "") not in VALID_UNIQUE_STATES)
    ]
    blocking_failures = len(failed_results) + len(missing_query_ids)

    mutations: dict[str, str] = {}
    for node_id, origin in origins.items():
        node = nodes_by_id[node_id]
        existing_state = str(node.get(UNIQUE_STATE_FIELD) or "").strip()
        if not overwrite and existing_state in VALID_UNIQUE_STATES:
            continue
        if origin == "wiki_inline":
            mutations[node_id] = "wiki_inline"
        elif origin == "visual_plan":
            result = judge_results.get(node_id)
            if result and result.unique_state:
                mutations[node_id] = result.unique_state

    graph_written = False
    if not dry_run and (allow_partial or blocking_failures == 0):
        for node_id, state in mutations.items():
            node = dict(nodes_by_id[node_id])
            node[UNIQUE_STATE_FIELD] = state
            store.upsert_node(node)
        if store.has_pending_writes():
            store.flush()
            graph_written = True
    elif not dry_run and blocking_failures:
        counters["atomic_write_blocked"] = 1

    final_checkpoint_results = [checkpoint[node_id] for node_id in sorted(checkpoint)]
    if final_checkpoint_results:
        _rewrite_checkpoint(checkpoint_path, final_checkpoint_results)

    assigned_distribution = Counter(mutations.values())
    counters["judge_succeeded"] = sum(result.unique_state is not None for result in judge_results.values())
    counters["judge_failed"] = len(failed_results)
    counters["would_assign"] = len(mutations)
    counters["written"] = len(mutations) if graph_written else 0

    return {
        "graph_dir": str(graph_dir),
        "judge_model_alias": judge_model_alias,
        "unique_state_field": UNIQUE_STATE_FIELD,
        "dry_run": dry_run,
        "allow_partial": allow_partial,
        "overwrite": overwrite,
        "graph_written": graph_written,
        "checkpoint_path": str(checkpoint_path),
        "image_node_count": len(image_nodes),
        "counters": dict(counters),
        "assigned_distribution": dict(assigned_distribution),
        "failed_image_node_ids": [result.image_node_id for result in failed_results],
        "missing_search_query_image_node_ids": missing_query_ids,
        "unsupported_origin_image_node_ids": [node_id for node_id, origin in origins.items() if origin == "other"],
        "mutations": mutations,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory containing graph JSONL tables.")
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("IMAGE_UNIQUENESS_JUDGE_MODEL", ""),
        help="Registered LLM_WORKER model alias; defaults to IMAGE_UNIQUENESS_JUDGE_MODEL.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Maximum concurrent judge requests.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first failed request.")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Initial exponential retry delay in seconds.")
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Judge checkpoint path; default: <graph-dir>/image_unique_state_results.jsonl.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse a compatible judge checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Rejudge and replace existing valid unique_state values.")
    parser.add_argument("--dry-run", action="store_true", help="Run judges and report mutations without modifying nodes.jsonl.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write successful classifications even if another visual-plan image failed or lacks search_query.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the final JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graph_dir = args.graph_dir.expanduser().resolve()
    if not graph_dir.is_dir():
        raise SystemExit(f"error: graph directory does not exist: {graph_dir}")
    if not args.judge_model:
        raise SystemExit("error: provide --judge-model or set IMAGE_UNIQUENESS_JUDGE_MODEL")
    if args.workers <= 0:
        raise SystemExit("error: --workers must be positive")

    summary = classify_graph(
        graph_dir=graph_dir,
        judge_model_alias=args.judge_model,
        workers=args.workers,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        allow_partial=args.allow_partial,
        results_jsonl=args.results_jsonl,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    has_failures = bool(summary["failed_image_node_ids"] or summary["missing_search_query_image_node_ids"])
    return 2 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
