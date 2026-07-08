"""Inspect Wikipedia text-neighbor ranking and LLM filtering for one URL.

Run from the repository root, for example:

    python synthesis/debug_text_neighbors.py \
      --url https://en.wikipedia.org/wiki/Kobe_Bryant
"""

from __future__ import annotations

import argparse
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
        print(f"    reasons={reasons}")
        if context_chars > 0:
            print(f"    context={_truncate(candidate.context, context_chars)}")


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
                temperature=0.0,
                max_tokens=2048,
                metadata={"trace_label": f"neighbor_filter_debug:{source_title}"},
            )
        )
        elapsed_s = time.perf_counter() - started_at
        decisions = builder._parse_neighbor_filter_response(response.content)
    except Exception as exc:
        return {
            "enabled": True,
            "model": model_alias,
            "ranked_candidates": ranked_candidates,
            "prompt_candidates": prompt_candidates,
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
        print("LLM kept no candidates; pipeline would fall back to the rule-ranked list.")

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
    qa_candidates = ranked_candidates[: max(1, builder.max_qa_neighbor_candidates)]
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
    parser.add_argument("--url", required=True, help="Wikipedia page URL to inspect.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Optional env file with model/reader settings.")
    parser.add_argument("--override-env", action="store_true", help="Override existing environment variables from the env file.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip the preflight reader availability check.")
    parser.add_argument("--reader-check-timeout", type=float, default=60.0, help="Seconds to wait for reader preflight.")
    parser.add_argument("--max-links", type=int, default=5, help="Maximum links retained after position diversity.")
    parser.add_argument("--max-raw-links", type=int, default=0, help="Maximum raw Wikipedia links to score before filtering. <=0 uses builder default.")
    parser.add_argument("--max-llm-candidates", type=int, default=60, help="Maximum rule-ranked candidates shown to the LLM.")
    parser.add_argument("--max-qa-candidates", type=int, default=20, help="Maximum reranked candidates sent through QA penalty.")
    parser.add_argument("--show-rule-top", type=int, default=60, help="How many rule-ranked candidates to print.")
    parser.add_argument("--show-final-top", type=int, default=30, help="How many final candidates to print.")
    parser.add_argument("--context-chars", type=int, default=180, help="Characters of local context shown per candidate.")
    args = parser.parse_args(argv)

    env_path = Path(args.env_file).expanduser()
    if env_path.exists():
        load_env_file(env_path, override=args.override_env)

    if not args.skip_reader_check:
        ok, message = check_reader_service(
            args.reader_base_url,
            test_url=args.url,
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

    document = reader.read(args.url)
    page_url = builder._normalize_wikipedia_url(document.url or args.url)
    page_title = document.title or builder._title_from_url(page_url) or page_url
    builder._validate_article_page(page_url=page_url, page_title=page_title, content=document.content)
    link_markdown = builder._safe_truncate_markdown(
        document.raw_markdown or document.content,
        builder.max_link_markdown_chars,
    )

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
        if len(rule_candidates) >= builder.max_raw_links:
            break

    ranked_candidates = sorted(rule_candidates, key=lambda item: (-item.score, item.rank or 10**9))
    llm_result = _run_llm_filter_debug(builder, source_url=page_url, candidates=rule_candidates)
    final_input = llm_result["kept"] if llm_result.get("kept") else rule_candidates
    relations_by_url = {
        row["url"]: row["relation"]
        for row in llm_result.get("rows", [])
        if row.get("url")
    }
    qa_result = _run_qa_penalty_debug(
        builder,
        source_url=page_url,
        candidates=final_input,
        relations_by_url=relations_by_url,
    )
    final_candidates = builder._select_position_diverse_links(final_input)

    print(f"URL: {page_url}")
    print(f"Title: {page_title}")
    print(f"raw_markdown_chars: {len(document.raw_markdown or document.content)}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
