"""Inspect Wikipedia text-neighbor ranking and LLM filtering for one URL.

Run from the repository root, for example:

    python synthesis/debug_text_neighbors.py \
      --url https://en.wikipedia.org/wiki/Kobe_Bryant
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
from synthesis.run_min_graph import DEFAULT_ENV_PATH, check_reader_service, load_env_file
from synthesis.wiki_text_builder import (
    PROMPT_FILTER_WIKI_NEIGHBORS,
    EnhancedReaderClient,
    WikiLinkCandidate,
    WIKI_PAGE_PREFIXES_TO_SKIP,
    WikiTextBuilder,
)


PROMPT_JUDGE_TEXT_RELATION = """You are judging whether a graph relation is a valid source-to-target statement.

You will be given a source Wikipedia node, a candidate target Wikipedia node,
the proposed relation, the hyperlink anchor, and local context from the source
page.

Return whether the relation is a valid directed statement from the source to the
target. A valid relation should describe the target from the source side, be
supported by the local context, and not mainly describe a third-party person,
event, or background fact. Penalize relations that are standalone target clues,
too broad, unrelated, or require multiple implicit hops from source to target.

Return valid JSON exactly with these fields:
{
  "valid": true,
  "directness": "direct|indirect|unrelated|too_broad|unsupported",
  "reason": "short reason"
}
"""


def _truncate(text: str | None, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _print_candidates(
    title: str,
    candidates: list[WikiLinkCandidate],
    *,
    limit: int,
    context_chars: int,
) -> None:
    print(f"\n=== {title} ({min(limit, len(candidates))}/{len(candidates)}) ===")
    for index, candidate in enumerate(candidates[:limit], start=1):
        reasons = ", ".join(candidate.quality_reasons) if candidate.quality_reasons else "-"
        print(
            f"{index:>2}. score={candidate.score:.2f} raw_rank={candidate.rank} "
            f"title={candidate.title!r} anchor={candidate.anchor_text!r}"
        )
        print(f"    url={candidate.url}")
        print(f"    relation={candidate.relation or '-'}")
        if candidate.relation_info:
            method = candidate.relation_info.get("method") or "-"
            evidence = _truncate(candidate.relation_info.get("evidence"), context_chars)
            print(f"    relation_method={method}")
            if evidence:
                print(f"    relation_evidence={evidence}")
        print(f"    reasons={reasons}")
        if context_chars > 0:
            print(f"    context={_truncate(candidate.context, context_chars)}")


def _build_rule_candidates(
    builder: WikiTextBuilder,
    *,
    page_url: str,
    link_markdown: str,
) -> list[WikiLinkCandidate]:
    rule_candidates: list[WikiLinkCandidate] = []
    seen_urls: set[str] = set()
    for rank, (anchor_text, href, start, end) in enumerate(builder._iter_markdown_links(link_markdown), start=1):
        url = builder._wiki_url_from_href(href, source_url=page_url)
        if not url or url in seen_urls:
            continue
        title = builder._title_from_url(url)
        if not title or title.startswith(WIKI_PAGE_PREFIXES_TO_SKIP):
            continue
        context = builder._context(link_markdown, start, end)
        score, reasons = builder._score_link_candidate(
            title=title,
            anchor_text=anchor_text,
            context=context,
            rank=rank,
        )
        if score <= 0:
            continue
        seen_urls.add(url)
        rule_candidates.append(
            WikiLinkCandidate(
                title=title,
                url=url,
                anchor_text=anchor_text.strip(),
                source_url=page_url,
                context=context,
                rank=rank,
                start_char=start,
                end_char=end,
                window_id=builder._window_id(start),
                score=score,
                quality_reasons=reasons,
            )
        )
    rule_candidates = builder._uniformly_sample_candidates(rule_candidates, builder.max_raw_links)
    return builder._attach_relations_to_candidates(
        source_title=builder._title_from_url(page_url) or page_url,
        candidates=rule_candidates,
    )


def _run_llm_filter_debug(
    builder: WikiTextBuilder,
    *,
    source_url: str,
    candidates: list[WikiLinkCandidate],
) -> dict[str, Any]:
    model_alias = os.environ.get("WIKI_NEIGHBOR_MODEL")
    ranked_candidates = sorted(candidates, key=lambda item: (-item.score, item.rank or 10**9))
    prompt_candidates = ranked_candidates[: max(1, builder.max_llm_neighbor_candidates)]
    if not model_alias or not prompt_candidates:
        return {
            "enabled": False,
            "model": model_alias,
            "ranked_candidates": ranked_candidates,
            "prompt_candidates": prompt_candidates,
            "system_prompt": PROMPT_FILTER_WIKI_NEIGHBORS,
            "prompt_input": None,
            "rows": [],
            "kept": [],
            "raw_output": None,
            "error": None,
            "fallback": "model_disabled" if not model_alias else "no_candidates",
        }

    source_title = builder._title_from_url(source_url) or source_url
    rule_scores = {candidate.url: candidate.score for candidate in prompt_candidates}
    prompt_input = builder._neighbor_filter_prompt_input(source_title, source_url, prompt_candidates)

    started_at = time.perf_counter()
    try:
        response = builder.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_FILTER_WIKI_NEIGHBORS),
                    ModelMessage(role="user", content=prompt_input),
                ],
                max_tokens=2048,
                metadata={"trace_label": f"neighbor_filter_debug:{source_title}"},
            )
        )
        elapsed_s = time.perf_counter() - started_at
        decisions = builder._parse_neighbor_filter_response(response.content)
        if not decisions:
            raise ValueError("empty_neighbor_filter_decisions")
    except Exception as exc:
        return {
            "enabled": True,
            "model": model_alias,
            "ranked_candidates": ranked_candidates,
            "prompt_candidates": prompt_candidates,
            "system_prompt": PROMPT_FILTER_WIKI_NEIGHBORS,
            "prompt_input": prompt_input,
            "rows": [],
            "kept": [],
            "raw_output": None,
            "error": f"{exc.__class__.__name__}: {exc}",
            "fallback": "llm_error",
        }

    kept: list[WikiLinkCandidate] = []
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(prompt_candidates, start=1):
        decision = decisions.get(index)
        llm_score = builder._parse_neighbor_score(decision.get("score")) if decision else None
        final_score = llm_score if llm_score is not None else candidate.score
        rows.append(
            builder._neighbor_debug_row(
                index=index,
                candidate=candidate,
                rule_score=rule_scores.get(candidate.url, candidate.score),
                decision=decision,
                final_score=final_score,
            )
        )
        if decision and decision.get("keep") == "yes" and final_score >= 3.0:
            kept_candidate = WikiLinkCandidate(**candidate.to_dict())
            kept_candidate.score = final_score
            kept_candidate.quality_reasons = list(candidate.quality_reasons)
            kept_candidate.quality_reasons.append("llm_neighbor_keep")
            if decision.get("relation"):
                kept_candidate.quality_reasons.append(f"llm_relation:{decision['relation']}")
            if decision.get("reason"):
                kept_candidate.quality_reasons.append(f"llm_reason:{decision['reason'][:120]}")
            kept.append(kept_candidate)

    fallback = None if kept else "llm_kept_none"
    return {
        "enabled": True,
        "model": model_alias,
        "ranked_candidates": ranked_candidates,
        "prompt_candidates": prompt_candidates,
        "system_prompt": PROMPT_FILTER_WIKI_NEIGHBORS,
        "prompt_input": prompt_input,
        "rows": rows,
        "kept": kept,
        "raw_output": response.content,
        "error": None,
        "elapsed_s": elapsed_s,
        "fallback": fallback,
    }


def _print_llm_debug(result: dict[str, Any], *, context_chars: int) -> None:
    print("\n=== LLM Filter ===")
    if not result["enabled"]:
        model = result.get("model") or "<disabled>"
        print(f"LLM filter disabled. WIKI_NEIGHBOR_MODEL={model}")
        return
    if result.get("error"):
        print(f"LLM call failed: {result['error']}")
        return

    print(
        f"model={result['model']} prompt_candidates={len(result['prompt_candidates'])} "
        f"kept={len(result['kept'])} elapsed_s={result.get('elapsed_s', 0.0):.2f}"
    )
    if result.get("fallback") == "llm_kept_none":
        print("LLM kept no candidates; pipeline returns no LLM-kept neighbors.")

    system_prompt = result.get("system_prompt")
    prompt_input = result.get("prompt_input")
    raw_output = result.get("raw_output")
    if system_prompt:
        print("\n--- PROMPT_FILTER_WIKI_NEIGHBORS / system ---")
        print(system_prompt)
    if prompt_input:
        print("\n--- PROMPT_FILTER_WIKI_NEIGHBORS / user ---")
        print(prompt_input)
    if raw_output is not None:
        print("\n--- PROMPT_FILTER_WIKI_NEIGHBORS / raw output ---")
        print(raw_output)

    for row in result["rows"]:
        print(
            f"{row['index']:>2}. keep={row['keep']!r} llm_score={row['llm_score'] or '-'} "
            f"rule_score={row['rule_score']:.2f} final_score={row['final_score']:.2f} "
            f"title={row['title']!r}"
        )
        print(f"    relation={row['relation'] or '-'}")
        print(f"    reason={row['reason'] or '-'}")
        print(f"    context={_truncate(row['context'], context_chars)}")


def _run_qa_penalty_debug(
    builder: WikiTextBuilder,
    *,
    source_url: str,
    candidates: list[WikiLinkCandidate],
    relations_by_url: dict[str, str],
) -> dict[str, Any]:
    answer_model = os.environ.get("WIKI_NEIGHBOR_QA_MODEL") or os.environ.get("WIKI_NEIGHBOR_MODEL")
    judge_model = (
        os.environ.get("WIKI_NEIGHBOR_QA_JUDGE_MODEL")
        or os.environ.get("WIKI_NEIGHBOR_QA_MODEL")
        or os.environ.get("WIKI_NEIGHBOR_MODEL")
    )
    ranked_candidates = sorted(candidates, key=lambda item: (-item.score, item.rank or 10**9))
    if builder.max_qa_neighbor_candidates and builder.max_qa_neighbor_candidates > 0:
        qa_candidates = ranked_candidates[: builder.max_qa_neighbor_candidates]
    else:
        qa_candidates = ranked_candidates
    if not answer_model or not judge_model or not qa_candidates:
        return {
            "enabled": False,
            "answer_model": answer_model,
            "judge_model": judge_model,
            "qa_candidates": qa_candidates,
            "rows": [],
            "error": None,
            "fallback": "model_disabled" if not (answer_model and judge_model) else "no_candidates",
        }

    source_title = builder._title_from_url(source_url) or source_url
    before_scores = {candidate.url: candidate.score for candidate in qa_candidates}
    relations_used = {
        candidate.url: (relations_by_url.get(candidate.url) or candidate.anchor_text or "").strip()
        for candidate in qa_candidates
    }
    started_at = time.perf_counter()
    debug_records_by_url: dict[str, dict[str, Any]] = {}
    try:
        builder._apply_neighbor_familiarity_penalty(
            source_title=source_title,
            candidates=candidates,
            relations_by_url=relations_by_url,
            debug_records_by_url=debug_records_by_url,
        )
        elapsed_s = time.perf_counter() - started_at
    except Exception as exc:
        return {
            "enabled": True,
            "answer_model": answer_model,
            "judge_model": judge_model,
            "qa_candidates": qa_candidates,
            "rows": [],
            "error": f"{exc.__class__.__name__}: {exc}",
            "fallback": "qa_error",
        }

    rows: list[dict[str, Any]] = []
    for candidate in qa_candidates:
        before = before_scores.get(candidate.url, candidate.score)
        after = candidate.score
        penalty = max(0.0, before - after)
        debug_record = debug_records_by_url.get(candidate.url) or {}
        rows.append(
            {
                "title": candidate.title,
                "url": candidate.url,
                "relation": debug_record.get("relation") or relations_used.get(candidate.url) or "-",
                "before_score": before,
                "after_score": after,
                "penalty": penalty,
                "question": debug_record.get("question") or "",
                "answers": list(debug_record.get("answers") or []),
                "correct_count": int(debug_record.get("correct_count") or 0),
                "context": re.sub(r"\s+", " ", candidate.context or "").strip()[:220],
            }
        )

    return {
        "enabled": True,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "qa_candidates": qa_candidates,
        "rows": rows,
        "error": None,
        "elapsed_s": elapsed_s,
        "fallback": None,
    }


def _judge_relation(
    builder: WikiTextBuilder,
    *,
    source_title: str,
    candidate: WikiLinkCandidate,
    model_alias: str | None,
) -> dict[str, Any]:
    relation = candidate.relation or candidate.anchor_text or ""
    if not model_alias:
        return {
            "enabled": False,
            "valid": None,
            "directness": "not_judged",
            "reason": "missing_judge_model",
        }
    payload = (
        f"Source node: {source_title}\n"
        f"Target node: {candidate.title}\n"
        f"Proposed relation: {relation}\n"
        f"Anchor text: {candidate.anchor_text}\n"
        f"Local context:\n{candidate.context or ''}\n"
    )
    try:
        response = builder.model_client.generate(
            ModelRequest(
                model=model_alias,
                response_format={"type": "json_object"},
                messages=[
                    ModelMessage(role="system", content=PROMPT_JUDGE_TEXT_RELATION),
                    ModelMessage(role="user", content=payload),
                ],
                max_tokens=512,
                metadata={"trace_label": f"text_relation_judge:{source_title}:{candidate.title}"},
            )
        )
        parsed = json.loads(response.content)
    except Exception as exc:
        return {
            "enabled": True,
            "valid": None,
            "directness": "judge_error",
            "reason": f"{exc.__class__.__name__}: {exc}",
        }
    return {
        "enabled": True,
        "valid": bool(parsed.get("valid")),
        "directness": str(parsed.get("directness") or "").strip() or "unknown",
        "reason": str(parsed.get("reason") or "").strip(),
        "raw": parsed,
    }


def _relations_by_url_from_candidates(candidates: list[WikiLinkCandidate]) -> dict[str, str]:
    return {
        candidate.url: (candidate.relation or candidate.anchor_text or "").strip()
        for candidate in candidates
    }


def _select_final_input(llm_result: dict[str, Any], rule_candidates: list[WikiLinkCandidate]) -> list[WikiLinkCandidate]:
    if not llm_result.get("enabled"):
        return rule_candidates
    if llm_result.get("error"):
        return []
    return list(llm_result.get("kept") or [])


def _process_url(
    builder: WikiTextBuilder,
    reader: EnhancedReaderClient,
    *,
    url: str,
    judge_model: str | None,
) -> dict[str, Any]:
    document = reader.read(url)
    page_url = builder._normalize_wikipedia_url(document.url or url)
    page_title = document.title or builder._title_from_url(page_url) or page_url
    builder._validate_article_page(page_url=page_url, page_title=page_title, content=document.content)
    link_markdown = builder._safe_truncate_markdown(
        document.raw_markdown or document.content,
        builder.max_link_markdown_chars,
    )
    rule_candidates = _build_rule_candidates(builder, page_url=page_url, link_markdown=link_markdown)
    ranked_candidates = sorted(rule_candidates, key=lambda item: (-item.score, item.rank or 10**9))
    llm_result = _run_llm_filter_debug(builder, source_url=page_url, candidates=rule_candidates)
    final_input = _select_final_input(llm_result, rule_candidates)
    relations_by_url = _relations_by_url_from_candidates(final_input)
    qa_result = _run_qa_penalty_debug(
        builder,
        source_url=page_url,
        candidates=final_input,
        relations_by_url=relations_by_url,
    )
    final_candidates = builder._select_position_diverse_links(final_input)
    judge_rows = []
    for candidate in final_candidates:
        judge = _judge_relation(
            builder,
            source_title=page_title,
            candidate=candidate,
            model_alias=judge_model,
        )
        judge_rows.append({"candidate": candidate, "judge": judge})
    return {
        "url": page_url,
        "title": page_title,
        "raw_markdown_chars": len(document.raw_markdown or document.content),
        "rule_candidates": rule_candidates,
        "ranked_candidates": ranked_candidates,
        "llm_result": llm_result,
        "qa_result": qa_result,
        "final_candidates": final_candidates,
        "judge_rows": judge_rows,
    }


def _print_qa_debug(result: dict[str, Any], *, context_chars: int) -> None:
    print("\n=== QA Penalty ===")
    if not result["enabled"]:
        print(
            "QA penalty disabled. "
            f"WIKI_NEIGHBOR_QA_MODEL={result.get('answer_model') or '<disabled>'} "
            f"WIKI_NEIGHBOR_QA_JUDGE_MODEL={result.get('judge_model') or '<disabled>'}"
        )
        return
    if result.get("error"):
        print(f"QA penalty failed: {result['error']}")
        return

    print(
        f"answer_model={result['answer_model']} judge_model={result['judge_model']} "
        f"qa_candidates={len(result['qa_candidates'])} elapsed_s={result.get('elapsed_s', 0.0):.2f}"
    )
    for index, row in enumerate(result["rows"], start=1):
        print(
            f"{index:>2}. penalty={row['penalty']:.2f} "
            f"before={row['before_score']:.2f} after={row['after_score']:.2f} "
            f"title={row['title']!r}"
        )
        print(f"    relation={row['relation']}")
        print(f"    question={row['question'] or '-'}")
        answers = list(row.get("answers") or [])
        for answer_index in range(3):
            answer_text = answers[answer_index] if answer_index < len(answers) else "UNKNOWN"
            print(f"    answer_{answer_index + 1}={answer_text}")
        print(f"    judged_correct_count={row.get('correct_count', 0)}")
        print(f"    context={_truncate(row['context'], context_chars)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug text-neighbor ranking and LLM filtering for one Wikipedia URL.")
    parser.add_argument("--url", action="append", default=[], help="Wikipedia page URL to inspect. Repeat for batch mode.")
    parser.add_argument("--urls-file", default="", help="Optional text file with one Wikipedia URL per line for batch mode.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Optional env file with model/reader settings.")
    parser.add_argument("--override-env", action="store_true", help="Override existing environment variables from the env file.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip the preflight reader availability check.")
    parser.add_argument("--reader-check-timeout", type=float, default=60.0, help="Seconds to wait for reader preflight.")
    parser.add_argument("--max-links", type=int, default=60, help="Maximum links retained after position diversity.")
    parser.add_argument("--max-raw-links", type=int, default=0, help="Maximum raw Wikipedia links to score before filtering. <=0 uses builder default.")
    parser.add_argument("--max-llm-candidates", type=int, default=60, help="Maximum rule-ranked candidates shown to the LLM.")
    parser.add_argument("--max-qa-candidates", type=int, default=0, help="Maximum reranked candidates sent through QA penalty. <=0 means use all kept neighbors.")
    parser.add_argument("--show-rule-top", type=int, default=60, help="How many rule-ranked candidates to print.")
    parser.add_argument("--show-final-top", type=int, default=30, help="How many final candidates to print.")
    parser.add_argument("--context-chars", type=int, default=180, help="Characters of local context shown per candidate.")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional model alias for relation validity judging. Defaults to WIKI_RELATION_JUDGE_MODEL, then WIKI_NEIGHBOR_MODEL.",
    )
    parser.add_argument("--no-judge", action="store_true", help="Disable relation validity judging.")
    parser.add_argument("--show-invalid-limit", type=int, default=50, help="Maximum invalid judged relations to print in batch mode.")
    args = parser.parse_args(argv)

    env_path = Path(args.env_file).expanduser()
    if env_path.exists():
        load_env_file(env_path, override=args.override_env)

    urls = list(args.url or [])
    if args.urls_file:
        urls_path = Path(args.urls_file).expanduser()
        for raw_line in urls_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    urls = list(dict.fromkeys(urls))
    if not urls:
        parser.error("provide at least one --url or --urls-file")

    if not args.skip_reader_check:
        ok, message = check_reader_service(
            args.reader_base_url,
            test_url=urls[0],
            timeout_s=args.reader_check_timeout,
        )
        if not ok:
            print(f"[preflight] Enhanced Reader unavailable at {args.reader_base_url}: {message}", file=sys.stderr)
            return 2

    reader = EnhancedReaderClient(base_url=args.reader_base_url)
    builder = WikiTextBuilder(
        reader=reader,
        store=None,
        model_client=LLM_WORKER,
        max_links=args.max_links,
        max_raw_links=args.max_raw_links if args.max_raw_links > 0 else None,
        max_llm_neighbor_candidates=args.max_llm_candidates,
        max_qa_neighbor_candidates=args.max_qa_candidates,
    )
    judge_model = None if args.no_judge else (args.judge_model or os.environ.get("WIKI_RELATION_JUDGE_MODEL") or os.environ.get("WIKI_NEIGHBOR_MODEL"))

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for url in urls:
        try:
            results.append(_process_url(builder, reader, url=url, judge_model=judge_model))
        except Exception as exc:
            errors.append({"url": url, "error": f"{exc.__class__.__name__}: {exc}"})

    if len(urls) > 1:
        total_final = sum(len(item["final_candidates"]) for item in results)
        invalid_cases: list[dict[str, Any]] = []
        judged_count = 0
        for item in results:
            for row in item["judge_rows"]:
                judge = row["judge"]
                if judge.get("enabled"):
                    judged_count += 1
                if judge.get("valid") is False:
                    invalid_cases.append({"source": item, "candidate": row["candidate"], "judge": judge})
        print("=== batch text neighbor debug ===")
        print(f"input_url_count: {len(urls)}")
        print(f"processed_url_count: {len(results)}")
        print(f"error_count: {len(errors)}")
        print(f"total_expanded_text_nodes: {total_final}")
        print(f"judge_model: {judge_model or '<disabled>'}")
        print(f"judged_relation_count: {judged_count}")
        print(f"invalid_relation_count: {len(invalid_cases)}")
        print("\n=== per source summary ===")
        for item in results:
            invalid_for_source = sum(1 for row in item["judge_rows"] if row["judge"].get("valid") is False)
            print(
                f"- {item['title']} | final={len(item['final_candidates'])} "
                f"rule={len(item['rule_candidates'])} invalid={invalid_for_source} url={item['url']}"
            )
        if errors:
            print("\n=== errors ===")
            for item in errors:
                print(f"- {item['url']}: {item['error']}")
        print("\n=== invalid relation cases ===")
        if not invalid_cases:
            print("(none)")
        for index, case in enumerate(invalid_cases[: max(0, args.show_invalid_limit)], start=1):
            source = case["source"]
            candidate = case["candidate"]
            judge = case["judge"]
            print(f"{index}. source={source['title']!r}")
            print(f"   source_url={source['url']}")
            print(f"   target={candidate.title!r}")
            print(f"   target_url={candidate.url}")
            print(f"   relation={candidate.relation or '-'}")
            print(f"   directness={judge.get('directness') or '-'}")
            print(f"   judge_reason={judge.get('reason') or '-'}")
            print(f"   anchor={candidate.anchor_text or '-'}")
            print(f"   context={_truncate(candidate.context, args.context_chars)}")
        return 0 if not errors else 1

    if not results:
        for item in errors:
            print(f"{item['url']}: {item['error']}", file=sys.stderr)
        return 1

    result = results[0]
    page_url = result["url"]
    page_title = result["title"]
    rule_candidates = result["rule_candidates"]
    ranked_candidates = result["ranked_candidates"]
    llm_result = result["llm_result"]
    qa_result = result["qa_result"]
    final_candidates = result["final_candidates"]

    print(f"URL: {page_url}")
    print(f"Title: {page_title}")
    print(f"raw_markdown_chars: {result['raw_markdown_chars']}")
    print(f"rule_candidates: {len(rule_candidates)}")
    print(f"llm_enabled: {'yes' if llm_result['enabled'] else 'no'}")
    print(f"qa_enabled: {'yes' if qa_result['enabled'] else 'no'}")
    print(f"final_candidates_after_diversity: {len(final_candidates)}")

    _print_candidates(
        "Rule-Ranked Candidates",
        ranked_candidates,
        limit=args.show_rule_top,
        context_chars=args.context_chars,
    )
    _print_llm_debug(llm_result, context_chars=args.context_chars)
    _print_qa_debug(qa_result, context_chars=args.context_chars)
    _print_candidates(
        "Final Candidates After Diversity",
        final_candidates,
        limit=args.show_final_top,
        context_chars=args.context_chars,
    )
    print("\n=== Relation Judge ===")
    if args.no_judge or not judge_model:
        print("disabled")
    else:
        for index, row in enumerate(result["judge_rows"], start=1):
            candidate = row["candidate"]
            judge = row["judge"]
            print(
                f"{index:>2}. valid={judge.get('valid')} directness={judge.get('directness') or '-'} "
                f"title={candidate.title!r}"
            )
            print(f"    relation={candidate.relation or '-'}")
            print(f"    reason={judge.get('reason') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
