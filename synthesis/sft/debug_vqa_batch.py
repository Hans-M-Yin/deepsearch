"""Debug and inspect SFT trajectories over one question or a VQA batch directory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import traceback
from typing import Any

from .pipeline import (
    build_agent_config,
    build_runtime_context,
    check_hop_chain_coverage,
    extract_answer,
    format_messages,
    judge,
    run_agent_loop,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _extract_image_urls_from_vqa_records(
    question_record: dict[str, Any],
    sample_record: dict[str, Any] | None,
) -> list[str]:
    candidates: list[Any] = [
        question_record.get("image_url"),
        question_record.get("input_image_url"),
    ]
    sample_record = sample_record or {}
    candidates.extend(
        [
            sample_record.get("input_image_url"),
            ((sample_record.get("metadata") or {}).get("input_image_url") if isinstance(sample_record.get("metadata"), dict) else None),
        ]
    )

    writer_outputs = sample_record.get("writer_outputs") or {}
    if isinstance(writer_outputs, dict):
        for stage_name in ("obfuscated", "polished", "draft"):
            stage = writer_outputs.get(stage_name) or {}
            stage_metadata = stage.get("metadata") or {}
            if isinstance(stage_metadata, dict):
                candidates.extend(
                    [
                        stage_metadata.get("starting_image_url"),
                        stage_metadata.get("polish_starting_image_url"),
                    ]
                )

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _load_vqa_records(vqa_dir: Path) -> list[dict[str, Any]]:
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl does not exist: {samples_path}")

    question_records = _load_jsonl(questions_path)
    sample_records = _load_jsonl(samples_path)
    samples_by_id = {
        str(record.get("sample_id")): record
        for record in sample_records
        if record.get("sample_id") is not None
    }

    merged_records: list[dict[str, Any]] = []
    for question_record in question_records:
        sample = samples_by_id.get(str(question_record.get("sample_id") or ""))
        merged_records.append(
            {
                "question_id": question_record.get("question_id"),
                "sample_id": question_record.get("sample_id"),
                "path_id": question_record.get("path_id"),
                "question": question_record.get("final_question") or question_record.get("question") or "",
                "gold_answer": question_record.get("answer") or "",
                "hop_chain": list((sample or {}).get("hop_chain") or []),
                "image_paths": [],
                "image_urls": _extract_image_urls_from_vqa_records(question_record, sample),
                "sample_record": sample or {},
                "question_record": question_record,
            }
        )
    return merged_records


def _single_question_record(
    *,
    question: str,
    gold_answer: str,
    hop_chain_json: str | None,
    image_paths: list[str] | None = None,
    image_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    hop_chain = json.loads(hop_chain_json) if hop_chain_json else []
    if not isinstance(hop_chain, list):
        raise ValueError("--hop-chain-json must decode to a JSON list.")
    return [
        {
            "question_id": "single_question",
            "sample_id": None,
            "path_id": None,
            "question": question,
            "gold_answer": gold_answer,
            "hop_chain": hop_chain,
            "image_paths": list(image_paths or []),
            "image_urls": list(image_urls or []),
            "sample_record": {},
            "question_record": {
                "question": question,
                "answer": gold_answer,
            },
        }
    ]


def _print_record_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"question_id: {result.get('question_id')}")
    if result.get("sample_id") is not None:
        print(f"sample_id: {result.get('sample_id')}")
    if result.get("path_id") is not None:
        print(f"path_id: {result.get('path_id')}")
    print(f"question: {result.get('question')}")
    print(f"gold_answer: {result.get('gold_answer')}")
    if result.get("input_images"):
        print("input_images:")
        print(json.dumps(result.get("input_images") or [], ensure_ascii=False, indent=2))
    print(f"extracted_answer: {result.get('extracted_answer')}")
    print("answer_judge:")
    print(json.dumps(result.get("answer_judge") or {}, ensure_ascii=False, indent=2))
    if result.get("hop_chain"):
        print("hop_chain_coverage:")
        print(json.dumps(result.get("hop_chain_coverage") or {}, ensure_ascii=False, indent=2))
    print("\n--- Trajectory Text ---")
    print((result.get("formatted_trajectory") or {}).get("text") or "")


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _message_text_for_transcript(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "")
    content = _message_content_to_text(message.get("content")).strip()
    if role == "tool":
        tool_name = str(message.get("name") or "").strip()
        if tool_name:
            return f"[{tool_name}]\n{content}" if content else f"[{tool_name}]"
    return content


def _build_source_metadata(record: dict[str, Any], *, vqa_dir: str | None) -> dict[str, Any]:
    return {
        "vqa_dir": vqa_dir,
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question_record": record.get("question_record") or {},
        "sample_record": record.get("sample_record") or {},
    }


def _build_raw_trajectory_record(
    *,
    record: dict[str, Any],
    input_images: list[dict[str, str]],
    messages: list[dict[str, Any]],
    extracted_answer: str,
    answer_judge: dict[str, Any],
    hop_chain_coverage: dict[str, Any] | None,
    vqa_dir: str | None,
) -> dict[str, Any]:
    return {
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "input_images": input_images,
        "source_metadata": _build_source_metadata(record, vqa_dir=vqa_dir),
        "raw_messages": messages,
        "extracted_answer": extracted_answer,
        "answer_judge": answer_judge,
        "hop_chain": list(record.get("hop_chain") or []),
        "hop_chain_coverage": hop_chain_coverage,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", help="Directory produced by synthesis.vqa.run_batch.")
    parser.add_argument("--question", help="Single question to debug.")
    parser.add_argument("--gold-answer", default="", help="Gold answer for single-question mode.")
    parser.add_argument("--hop-chain-json", help="JSON list for single-question hop chain.")
    parser.add_argument("--image", action="append", help="Attach a local image path to the user input.")
    parser.add_argument("--image-url", action="append", help="Attach a remote image URL to the user input.")
    parser.add_argument("--limit", type=int, default=5, help="How many questions to run in batch mode.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in batch mode.")
    parser.add_argument("--workers", type=int, default=1, help="How many records to process concurrently.")
    parser.add_argument("--workdir", default=os.path.join(os.getcwd(), "synthesis_sft_runs"))
    parser.add_argument("--output-jsonl", help="Optional path to save raw trajectory records.")
    parser.add_argument(
        "--raw-trajectories-jsonl",
        help="Optional path to save raw formatted trajectories.",
    )
    parser.add_argument(
        "--repair-model",
        default=os.environ.get("SFT_REPAIR_MODEL") or "",
        help="Registered model alias for incorrect-trajectory diagnosis and repair with LLM_WORKER.",
    )
    parser.add_argument(
        "--repair-max-tokens",
        type=int,
        default=_optional_env_int("SFT_REPAIR_MAX_TOKENS") or 2048,
        help="Max tokens for the LLM_WORKER-based incorrect-trajectory repair stages.",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--model",
        default=os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "",
        help="Primary answer model. Prefer a registered alias from synthesis/models.json.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--api-mode",
        choices=("manual_react", "chat_completions", "responses"),
        default=os.environ.get("SFT_OPENAI_API_MODE") or "manual_react",
        help="Primary trajectory collection mode. Defaults to manual_react.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=(
            os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
    )
    parser.add_argument("--api-version", default=os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview")
    parser.add_argument("--max-tokens", type=int, default=_optional_env_int("SFT_OPENAI_MAX_TOKENS"))
    parser.add_argument(
        "--temperature",
        type=float,
        default=(float(os.environ["SFT_OPENAI_TEMPERATURE"]) if os.environ.get("SFT_OPENAI_TEMPERATURE") else None),
    )
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("SFT_OPENAI_MAX_TURNS", "8")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("SFT_OPENAI_TIMEOUT_S", "120")))
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--headers-json", default=os.environ.get("SFT_OPENAI_HEADERS_JSON"))
    parser.add_argument("--extra-body-json", default=os.environ.get("SFT_OPENAI_EXTRA_BODY_JSON"))
    parser.add_argument(
        "--expert-model",
        default=os.environ.get("SFT_JUDGE_MODEL"),
        help="Expert judge model. Prefer a registered alias from synthesis/models.json.",
    )
    parser.add_argument("--expert-api-key", default=os.environ.get("SFT_JUDGE_API_KEY"))
    parser.add_argument("--expert-azure-endpoint", default=os.environ.get("SFT_JUDGE_AZURE_ENDPOINT"))
    parser.add_argument("--expert-api-version", default=os.environ.get("SFT_JUDGE_API_VERSION") or os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview")
    parser.add_argument("--expert-max-tokens", type=int, default=_optional_env_int("SFT_JUDGE_MAX_TOKENS"))
    parser.add_argument(
        "--expert-temperature",
        type=float,
        default=(float(os.environ["SFT_JUDGE_TEMPERATURE"]) if os.environ.get("SFT_JUDGE_TEMPERATURE") else None),
    )
    return parser


def _parse_json_flag(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _config_from_model_arg(
    *,
    model_arg: str | None,
    api_key: str | None,
    api_mode: str,
    azure_endpoint: str | None,
    api_version: str | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout_s: float,
    system_prompt: str | None,
    headers_json: str | None,
    extra_body_json: str | None,
    max_turns: int,
    print_rounds: bool,
) -> Any:
    return build_agent_config(
        model=model_arg,
        api_key=api_key,
        client_type="azure_openai",
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        api_mode=api_mode,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
        system_prompt=system_prompt,
        headers=_parse_json_flag(headers_json),
        extra_body=_parse_json_flag(extra_body_json),
        max_turns=max_turns,
        print_rounds=print_rounds,
    )


def format_hop_chain_for_user_prompt(question_sample: dict[str, Any]) -> str:
    """Format the sample hop chain block appended to the SFT user prompt.

    This intentionally preserves the existing SFT-generation behavior: use the
    sample's raw ``hop_chain`` field and expose only each hop's ``statement`` as
    intermediate verification facts.  Keeping this as a separate interface makes
    it easier to swap in a question-facing hop chain later without changing the
    rest of prompt construction.
    """

    statements = [
        str(hop.get("statement") or "").strip()
        for hop in (question_sample.get("hop_chain") or [])
        if isinstance(hop, dict) and str(hop.get("statement") or "").strip()
    ]
    if not statements:
        return ""
    statements_lines = "\n".join(statements)
    return (
        "\nPrivate reference facts for verification only:\n"
        "The following facts describe one possible reasoning route used when constructing the question. "
        "They are not necessarily the only or best route. Do not reveal them directly or cite them as evidence. "
        "Use them only to check whether your tool-based solution is on the right track.\n"
        f"{statements_lines}"
    )


def _build_user_prompt_text(record: dict[str, Any]) -> str:
    question_text = str(record.get("question") or "").strip()
    gold_answer = str(record.get("gold_answer") or "").strip()
    statements_block = format_hop_chain_for_user_prompt(record)
    if gold_answer:
        return f"Question: {question_text}\nAnswer: {gold_answer}{statements_block}"
    return f"{question_text}{statements_block}"


def _build_user_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    image_paths = [str(item).strip() for item in (record.get("image_paths") or []) if str(item).strip()]
    image_urls = [str(item).strip() for item in (record.get("image_urls") or []) if str(item).strip()]
    if not image_paths and not image_urls:
        return None

    content: list[dict[str, Any]] = [{"type": "text", "text": _build_user_prompt_text(record)}]
    for path in image_paths:
        content.append({"type": "image_path", "path": path})
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _process_record(
    *,
    index: int,
    record: dict[str, Any],
    agent_config: Any,
    expert_config: Any,
    workdir: str,
    vqa_dir: str | None,
) -> dict[str, Any]:
    context = build_runtime_context(
        working_dir=os.path.join(workdir, f"debug_{index:04d}_{record.get('question_id') or 'question'}"),
        case_id=str(record.get("question_id") or f"debug_{index:04d}"),
        metadata={
            "question_id": record.get("question_id"),
            "sample_id": record.get("sample_id"),
            "path_id": record.get("path_id"),
            "question": record.get("question"),
            "gold_answer": record.get("gold_answer"),
            "hop_chain": list(record.get("hop_chain") or []),
        },
    )
    input_images: list[dict[str, str]] = []
    for image_path in record.get("image_paths") or []:
        normalized_path = os.path.abspath(str(image_path))
        context.register_image(normalized_path)
        input_images.append({"image_path": normalized_path})
    for image_url in record.get("image_urls") or []:
        normalized_url = str(image_url).strip()
        if normalized_url:
            context.register_image(normalized_url)
            input_images.append({"image_url": normalized_url})

    input_messages = _build_user_messages(record)
    messages = run_agent_loop(
        prompt=None if input_messages is not None else _build_user_prompt_text(record),
        messages=input_messages,
        config=agent_config,
        context=context,
    )
    extracted_answer = extract_answer(messages)
    answer_judge = judge(
        question=str(record.get("question") or ""),
        answer=str(record.get("gold_answer") or ""),
        extracted_answer=extracted_answer,
    )
    formatted_trajectory = format_messages(messages)
    hop_chain = list(record.get("hop_chain") or [])
    hop_chain_coverage = (
        check_hop_chain_coverage(messages, hop_chain, config=expert_config)
        if hop_chain and expert_config is not None
        else None
    )

    result_record = {
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "input_images": input_images,
        "extracted_answer": extracted_answer,
        "answer_judge": answer_judge,
        "hop_chain": hop_chain,
        "hop_chain_coverage": hop_chain_coverage,
        "formatted_trajectory": formatted_trajectory,
        "messages": messages,
    }
    raw_record = _build_raw_trajectory_record(
        record=record,
        input_images=input_images,
        messages=messages,
        extracted_answer=extracted_answer,
        answer_judge=answer_judge,
        hop_chain_coverage=hop_chain_coverage,
        vqa_dir=vqa_dir,
    )
    is_correct = bool((answer_judge or {}).get("is_correct"))
    return {
        "index": index,
        "result_record": result_record,
        "raw_record": raw_record,
        "is_correct": is_correct,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.vqa_dir) == bool(args.question):
        parser.error("Use exactly one of --vqa-dir or --question.")
    if args.question and not args.model:
        parser.error("--model is required in single-question mode unless SFT_OPENAI_MODEL / OPENAI_MODEL is set.")
    if args.vqa_dir and not args.model:
        parser.error("--model is required in batch mode unless SFT_OPENAI_MODEL / OPENAI_MODEL is set.")
    if args.workers <= 0:
        parser.error("--workers must be positive.")

    if args.vqa_dir:
        all_records = _load_vqa_records(Path(args.vqa_dir))
        records = all_records[args.offset : args.offset + args.limit]
    else:
        records = _single_question_record(
            question=args.question,
            gold_answer=args.gold_answer,
            hop_chain_json=args.hop_chain_json,
            image_paths=args.image,
            image_urls=args.image_url,
        )

    if args.vqa_dir and (args.image or args.image_url):
        for record in records:
            record["image_paths"] = list(args.image or [])
            record["image_urls"] = list(args.image_url or [])

    agent_config = _config_from_model_arg(
        model_arg=args.model,
        api_key=args.api_key,
        api_mode=args.api_mode,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        system_prompt=args.system_prompt,
        headers_json=args.headers_json,
        extra_body_json=args.extra_body_json,
        max_turns=args.max_turns,
        print_rounds=args.verbose,
    )
    expert_config = None
    if args.expert_model:
        expert_config = _config_from_model_arg(
            model_arg=args.expert_model,
            api_key=args.expert_api_key or args.api_key,
            api_mode="chat_completions",
            azure_endpoint=args.expert_azure_endpoint or args.azure_endpoint,
            api_version=args.expert_api_version,
            max_tokens=args.expert_max_tokens,
            temperature=args.expert_temperature,
            timeout_s=args.timeout_s,
            system_prompt=(
                "You are a strict trajectory auditor. "
                "You inspect whether an agent trajectory truly covers each intended reasoning hop."
            ),
            headers_json=args.headers_json,
            extra_body_json=None,
            max_turns=args.max_turns,
            print_rounds=False,
        )

    raw_output_path: Path | None = None
    if args.raw_trajectories_jsonl:
        raw_output_path = Path(args.raw_trajectories_jsonl)
    elif args.output_jsonl:
        raw_output_path = Path(args.output_jsonl)

    raw_output_handle = None
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_handle = raw_output_path.open("w", encoding="utf-8")
    total_count = 0
    correct_count = 0
    incorrect_count = 0
    failed_count = 0

    try:
        resolved_vqa_dir = str(Path(args.vqa_dir).resolve()) if args.vqa_dir else None
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_context = {
                executor.submit(
                    _process_record,
                    index=index,
                    record=record,
                    agent_config=agent_config,
                    expert_config=expert_config,
                    workdir=args.workdir,
                    vqa_dir=resolved_vqa_dir,
                ): {
                    "index": index,
                    "record": record,
                }
                for index, record in enumerate(records, start=1)
            }
            for future in as_completed(future_to_context):
                task_context = future_to_context[future]
                record = task_context["record"]
                try:
                    payload = future.result()
                except Exception as exc:
                    failed_count += 1
                    total_count += 1
                    print("\n" + "=" * 100)
                    print(f"question_id: {record.get('question_id')}")
                    if record.get("sample_id") is not None:
                        print(f"sample_id: {record.get('sample_id')}")
                    if record.get("path_id") is not None:
                        print(f"path_id: {record.get('path_id')}")
                    print("status: failed")
                    print(f"error: {exc.__class__.__name__}: {exc}")
                    print("traceback:")
                    print("".join(traceback.format_exception(exc)).rstrip())
                    continue

                result_record = payload["result_record"]
                raw_record = payload["raw_record"]
                is_correct = bool(payload["is_correct"])
                _print_record_result(result_record)
                if raw_output_handle is not None:
                    _write_jsonl_record(raw_output_handle, raw_record)

                total_count += 1
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1
    finally:
        if raw_output_handle is not None:
            raw_output_handle.close()

    print("\n" + "=" * 100)
    print("Trajectory Judge Summary")
    print(f"total: {total_count}")
    print(f"correct: {correct_count}")
    print(f"incorrect: {incorrect_count}")
    print(f"failed: {failed_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
