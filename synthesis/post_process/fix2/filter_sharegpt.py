"""Standalone full-trajectory ShareGPT quality gate; never edits input records."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import time

from synthesis.model_worker import LLM_WORKER, ModelRequest
from synthesis.post_process.fix.parallel import completed_records
from synthesis.post_process.fix.storage import digest, file_digest, read_json, write_json
from . import quality


def convert_record(record, index, image_root):
    """Preserve chronological text and bind every ShareGPT image placeholder."""
    images = record.get("images") or []
    cursor = 0
    messages, originals = [], []
    question = None
    tool_name = "unknown_tool"
    for turn in record.get("conversations", []):
        role = {"human": "user", "gpt": "assistant", "observation": "tool"}.get(turn["from"], turn["from"])
        value = turn["value"]
        if not isinstance(value, str):
            raise ValueError("ShareGPT turn value must be text")
        first_user = role == "user" and question is None
        if first_user:
            question = value.replace("<image>", "").strip()
        parts = []
        chunks = value.split("<image>")
        for n, chunk in enumerate(chunks):
            parts.append({"type": "text", "text": chunk})
            if n < len(chunks) - 1:
                if cursor >= len(images):
                    raise ValueError("More image placeholders than paths")
                value_path = images[cursor]
                if not isinstance(value_path, str):
                    raise ValueError("Image paths must be strings")
                if not value_path.startswith(("data:image/", "http://", "https://")):
                    path = Path(value_path)
                    value_path = str(path if path.is_absolute() else image_root / path)
                    if not Path(value_path).is_file():
                        raise ValueError(f"Missing image: {value_path}")
                parts.append({"type": "image_url", "image_url": {"url": value_path}})
                if first_user:
                    originals.append(value_path)
                cursor += 1
        row = {"role": role, "content": parts if len(chunks) > 1 else value}
        if role == "assistant":
            for call in re.findall(r"<(?:tool_call|action)>(.*?)</(?:tool_call|action)>", value, re.S):
                try:
                    parsed = json.loads(call)
                    tool_name = parsed.get("name") or parsed.get("tool_name") or "unknown_tool"
                except (ValueError, AttributeError):
                    pass
        if role == "tool":
            row["name"] = tool_name
        messages.append(row)
    if cursor != len(images):
        raise ValueError("Unmapped image paths remain")
    if not question:
        raise ValueError("Missing question")
    answers = [a for m in messages if m["role"] == "assistant"
               for a in re.findall(r"<answer>(.*?)</answer>", quality.message_text(m["content"]), re.S)]
    if not answers:
        raise ValueError("Missing final <answer>")
    return {"case_id": str(index), "question": question, "input_images": originals,
            "raw_messages": messages, "extracted_answer": answers[-1].strip()}


def judge_record(record, index, image_root, model_alias, max_tokens):
    started = time.monotonic()
    try:
        converted = convert_record(record, index, image_root)
        response = LLM_WORKER.generate(ModelRequest(
            model=model_alias,
            messages=quality.build_quality_reviewer_messages(converted, full_transcript=True),
            response_format={"type": "json_object"}, temperature=0.0,
            max_tokens=max_tokens, metadata={"trace_label": "sharegpt_final_quality"},
        ))
        parsed = quality._parse_json_object(response.content or "")
        if parsed is None:
            raise ValueError("Malformed reviewer JSON")
        result, invalid = quality._normalize_review(parsed)
        if invalid:
            raise ValueError(invalid)
        result["status"] = "ok"
        keep, reason = quality.validate_trajectory_quality_review(result)
        result.update(disposition="keep" if keep else "filter", filter_reason=reason)
    except Exception as error:
        result = {"status": "error", "disposition": "retry_pending",
                  "error": f"{type(error).__name__}: {error}"}
    return {**result, "index": index, "record_id": record.get("id"),
            "model_alias": model_alias, "elapsed_seconds": round(time.monotonic() - started, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--image-root", type=Path, help="Base for relative image paths; default: input directory")
    p.add_argument("--model-alias", default="gemini38flash_internal_azure")
    p.add_argument("--max-tokens", type=int, default=2400)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.workers < 1 or args.max_tokens < 1 or (args.limit is not None and args.limit < 0):
        p.error("workers/max-tokens must be positive; limit must be nonnegative")
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.input.resolve() == (args.workdir / "trajectories_sharegpt.json").resolve():
        p.error("Output must not overwrite input")
    with (args.workdir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("Another filter process is using this workdir")
        return run(args)


def run(args):
    image_root = (args.image_root or args.input.parent).resolve()
    manifest = {"input": str(args.input.resolve()), "input_digest": file_digest(args.input),
                "image_root": str(image_root), "model_alias": args.model_alias,
                "max_tokens": args.max_tokens, "full_transcript": True,
                "review_code": digest({name: file_digest(Path(__file__).with_name(name))
                    for name in ("filter_sharegpt.py", "quality.py", "trajectory.py", "visual.py")})}
    manifest_path = args.workdir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Existing workdir: use --resume or a new directory")
        if read_json(manifest_path) != manifest:
            raise ValueError("Input/model/reviewer configuration changed; use a new workdir")
    else:
        if (args.workdir / "records").exists():
            raise ValueError("Checkpoints without manifest; use a new workdir")
        write_json(manifest_path, manifest)
    records = read_json(args.input)
    if not isinstance(records, list):
        raise ValueError("Input must be a ShareGPT JSON array")
    selected = records if args.limit is None else records[:args.limit]
    reviews, pending = {}, []
    for index in range(len(selected)):
        path = args.workdir / "records" / f"{index}.json"
        if args.resume and path.exists():
            saved = read_json(path)
            if saved.get("status") == "ok":
                reviews[index] = saved
                continue
        pending.append(index)
    print(json.dumps({"selected": len(selected), "reused": len(reviews), "to_review": len(pending)}), flush=True)

    def process(index):
        review = judge_record(selected[index], index, image_root, args.model_alias, args.max_tokens)
        write_json(args.workdir / "records" / f"{index}.json", review)
        return review

    for index, review in completed_records(process, pending, args.workers):
        reviews[index] = review
        print(json.dumps({"index": index, "disposition": review["disposition"],
                          "elapsed_seconds": review["elapsed_seconds"]}), flush=True)
    counts = Counter(r["disposition"] for r in reviews.values())
    accepted = []
    for i, record in enumerate(selected):
        if reviews[i]["disposition"] != "keep":
            continue
        # Output lives in a new directory: preserve image references by resolving
        # relative paths, while leaving the actual trajectory text untouched.
        exported = dict(record)
        if "images" in record:
            exported["images"] = [value if value.startswith(("data:image/", "http://", "https://")) or Path(value).is_absolute()
                                  else str(image_root / value) for value in record["images"]]
        accepted.append(exported)
    write_json(args.workdir / "trajectories_sharegpt.json", accepted)
    write_json(args.workdir / "rejected.json", [reviews[i] for i in range(len(selected)) if reviews[i]["disposition"] == "filter"])
    write_json(args.workdir / "retry_pending.json", [reviews[i] for i in range(len(selected)) if reviews[i]["disposition"] == "retry_pending"])
    write_json(args.workdir / "reviews.json", [reviews[i] for i in range(len(selected))])
    summary = {"input_records": len(records), "selected": len(selected), "unprocessed": len(records) - len(selected),
               "dispositions": dict(counts), "export_records": counts["keep"]}
    write_json(args.workdir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
