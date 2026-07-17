"""Inspect repository-verifier bundles and optionally run verification.

Examples:
    python -m synthesis.vqa.debug.debug_repository_verifier --vqa-dir /path/to/vqa --sample-id sample_x
    python -m synthesis.vqa.debug.debug_repository_verifier --vqa-dir /path/to/vqa --question-id q_000001 --run-verification --answer-model-alias gpt4o --judge-model-alias gpt4o
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.vqa.debug"

from synthesis.store import JsonlGraphStore
from synthesis.vqa.graph_view import GraphView
from synthesis.vqa.repository_verifier import (
    OfflineGraphRepositoryVerifier,
    RepositoryAssembler,
    RepositoryVerificationConfig,
    _extract_question_input_image_url,
    _infer_graph_dir,
    _load_jsonl,
    build_question_only_shortcut_request,
    build_repository_answer_judge_request,
    build_repository_solver_request,
    format_repository_bundle,
    format_verification_record,
)


SEPARATOR = "=" * 96


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", required=True, help="Directory containing questions.jsonl and samples.jsonl.")
    parser.add_argument("--graph-dir", default=None, help="Graph directory. Inferred from vqa_dir when omitted.")
    parser.add_argument("--sample-id", action="append", default=[], help="Only inspect the given sample_id. Repeatable.")
    parser.add_argument("--question-id", action="append", default=[], help="Only inspect the given question_id. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to print.")
    parser.add_argument("--width", type=int, default=100, help="Wrap width for long text fields.")
    parser.add_argument("--run-verification", action="store_true", help="Actually call the answer/judge models and print the verification result.")
    parser.add_argument("--answer-model-alias", default=None, help="Required when --run-verification is set.")
    parser.add_argument("--judge-model-alias", default=None, help="Required when --run-verification is set.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-relevant-docs-per-edge", type=int, default=2)
    parser.add_argument("--max-sibling-doc-distractors-per-edge", type=int, default=1)
    parser.add_argument("--max-random-doc-distractors", type=int, default=2)
    parser.add_argument("--max-sibling-image-distractors-per-image", type=int, default=1)
    parser.add_argument("--max-random-image-distractors", type=int, default=1)
    parser.add_argument("--min-reasoning-steps", type=int, default=1)
    parser.add_argument("--min-unique-citations", type=int, default=2)
    parser.add_argument("--question-only-answer-max-tokens", type=int, default=256)
    parser.add_argument("--hide-hidden", action="store_true", help="Hide internal relevant/distractor labels and source ids.")
    return parser


def _load_records(vqa_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    questions = _load_jsonl(vqa_dir / "questions.jsonl")
    samples = _load_jsonl(vqa_dir / "samples.jsonl")
    samples_by_id = {
        str(record.get("sample_id")): record
        for record in samples
        if record.get("sample_id") is not None
    }
    return questions, samples_by_id


def _select_pairs(
    questions: list[dict],
    samples_by_id: dict[str, dict],
    *,
    sample_ids: set[str],
    question_ids: set[str],
    limit: int | None,
) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    for question_record in questions:
        sample_id = str(question_record.get("sample_id") or "")
        question_id = str(question_record.get("question_id") or "")
        if sample_ids and sample_id not in sample_ids:
            continue
        if question_ids and question_id not in question_ids:
            continue
        sample_record = samples_by_id.get(sample_id)
        if sample_record is None:
            continue
        pairs.append((question_record, sample_record))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    vqa_dir = Path(args.vqa_dir).expanduser().resolve()
    graph_dir = Path(args.graph_dir).expanduser().resolve() if args.graph_dir else _infer_graph_dir(vqa_dir)
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl does not exist: {samples_path}")

    questions, samples_by_id = _load_records(vqa_dir)
    selected = _select_pairs(
        questions,
        samples_by_id,
        sample_ids={str(item) for item in args.sample_id if str(item).strip()},
        question_ids={str(item) for item in args.question_id if str(item).strip()},
        limit=args.limit,
    )
    if not selected:
        print(f"No matching question/sample pairs found in {vqa_dir}")
        return 1

    store = JsonlGraphStore(graph_dir)
    graph = GraphView(store)
    config = RepositoryVerificationConfig(
        random_seed=args.random_seed,
        max_relevant_docs_per_edge=args.max_relevant_docs_per_edge,
        max_sibling_doc_distractors_per_edge=args.max_sibling_doc_distractors_per_edge,
        max_random_doc_distractors=args.max_random_doc_distractors,
        max_sibling_image_distractors_per_image=args.max_sibling_image_distractors_per_image,
        max_random_image_distractors=args.max_random_image_distractors,
        min_reasoning_steps=args.min_reasoning_steps,
        min_unique_citations=args.min_unique_citations,
        question_only_answer_max_tokens=args.question_only_answer_max_tokens,
    )
    assembler = RepositoryAssembler(graph=graph, config=config)

    verifier = None
    if args.run_verification:
        if not args.answer_model_alias or not args.judge_model_alias:
            raise SystemExit("--run-verification requires both --answer-model-alias and --judge-model-alias")
        from synthesis.model_worker import LLM_WORKER

        verifier = OfflineGraphRepositoryVerifier(
            assembler=assembler,
            model_client=LLM_WORKER,
            answer_model_alias=args.answer_model_alias,
            judge_model_alias=args.judge_model_alias,
        )

    outputs: list[str] = []
    for index, (question_record, sample_record) in enumerate(selected, start=1):
        bundle = assembler.build_bundle(question_record=question_record, sample_record=sample_record)
        outputs.append(SEPARATOR)
        outputs.append(f"Case {index} | question_id={question_record.get('question_id')} | sample_id={question_record.get('sample_id')}")
        outputs.append(format_repository_bundle(bundle, width=args.width, include_hidden=not args.hide_hidden))
        solver_request = build_repository_solver_request(
            bundle=bundle,
            answer_model_alias=args.answer_model_alias,
            answer_max_tokens=config.answer_max_tokens,
            user_content=assembler.build_solver_user_content(bundle=bundle),
        )
        outputs.append("Answer Model Request")
        outputs.append(json.dumps(solver_request.to_dict(), ensure_ascii=False, indent=2))
        question_only_request = build_question_only_shortcut_request(
            question=str(bundle.question or ""),
            answer_model_alias=args.answer_model_alias,
            answer_max_tokens=config.question_only_answer_max_tokens,
            question_id=str(question_record.get("question_id") or question_record.get("sample_id") or f"case_{index}"),
            image_url=_extract_question_input_image_url(
                question_record=question_record,
                sample_record=sample_record,
            ),
        )
        outputs.append("Question-Only Shortcut Request")
        outputs.append(json.dumps(question_only_request.to_dict(), ensure_ascii=False, indent=2))
        if verifier is not None:
            fingerprint = verifier._question_fingerprint(question_record=question_record, sample_record=sample_record)
            record = verifier.verify_question_record(
                question_record=question_record,
                sample_record=sample_record,
                question_index=index,
                question_fingerprint=fingerprint,
            )
            outputs.append(format_verification_record(record, width=args.width))
            outputs.append("Answer Model Raw Output")
            outputs.append(json.dumps((record.get("solver_result") or {}).get("raw") or {}, ensure_ascii=False, indent=2))
            outputs.append("Question-Only Shortcut Raw Output")
            outputs.append(json.dumps((record.get("question_only_solver_result") or {}).get("raw") or {}, ensure_ascii=False, indent=2))
            judge_request = build_repository_answer_judge_request(
                question=record.get("question") or question_record.get("question") or "",
                gold_answer=record.get("gold_answer") or question_record.get("answer") or "",
                predicted_answer=((record.get("solver_result") or {}).get("answer") or ""),
                judge_model_alias=args.judge_model_alias,
                judge_max_tokens=config.judge_max_tokens,
                question_id=str(question_record.get("question_id") or question_record.get("sample_id") or f"case_{index}"),
            )
            outputs.append("Judge Model Request")
            outputs.append(json.dumps(judge_request.to_dict(), ensure_ascii=False, indent=2))
            outputs.append("Judge Model Raw Output")
            outputs.append(json.dumps((((record.get("checks") or {}).get("answer_judgment") or {}).get("raw")) or {}, ensure_ascii=False, indent=2))
            question_only_judge_request = build_repository_answer_judge_request(
                question=record.get("question") or question_record.get("question") or "",
                gold_answer=record.get("gold_answer") or question_record.get("answer") or "",
                predicted_answer=((record.get("question_only_solver_result") or {}).get("answer") or ""),
                judge_model_alias=args.judge_model_alias,
                judge_max_tokens=config.judge_max_tokens,
                question_id=str(question_record.get("question_id") or question_record.get("sample_id") or f"case_{index}") + ":question_only",
            )
            outputs.append("Question-Only Judge Request")
            outputs.append(json.dumps(question_only_judge_request.to_dict(), ensure_ascii=False, indent=2))
            outputs.append("Question-Only Judge Raw Output")
            outputs.append(json.dumps(((((record.get("checks") or {}).get("question_only_shortcut") or {}).get("answer_judgment") or {}).get("raw")) or {}, ensure_ascii=False, indent=2))
    outputs.append(SEPARATOR)
    print("\n\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
