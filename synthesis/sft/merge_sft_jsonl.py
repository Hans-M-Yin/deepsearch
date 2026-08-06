"""Merge multiple SFT JSONL files into one JSONL file.

Example:

    python -m synthesis.sft.merge_sft_jsonl \
        --input-jsonl \
        runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/0804_test_2_0_to_500.jsonl \
        runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/0804_test_2_500_to_1000.jsonl \
        runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/0804_test_2_1000_to_1500.jsonl \
        --output-jsonl runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/0804_test_2_merged.jsonl

Records are appended in the same order as the input files. The script validates
that every non-empty line is valid JSON and writes the result atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        required=True,
        help="Input JSONL files. They are merged in the order given.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Path of the merged JSONL file. Its parent directory is created if needed.",
    )
    return parser


def merge_jsonl(input_paths: list[Path], output_path: Path) -> list[int]:
    resolved_inputs = [path.resolve() for path in input_paths]
    resolved_output = output_path.resolve()
    if resolved_output in resolved_inputs:
        raise ValueError("The output path must be different from every input path.")

    missing = [path for path in resolved_inputs if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Input JSONL file(s) do not exist:\n{missing_text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for input_path in resolved_inputs:
                record_count = 0
                with input_path.open("r", encoding="utf-8") as input_file:
                    for line_number, line in enumerate(input_file, start=1):
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON in {input_path}:{line_number}: {exc.msg}"
                            ) from exc
                        output_file.write(line.rstrip("\r\n") + "\n")
                        record_count += 1
                counts.append(record_count)

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return counts


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = [Path(path) for path in args.input_jsonl]
    output_path = Path(args.output_jsonl)
    try:
        counts = merge_jsonl(input_paths, output_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"merge failed: {exc}") from exc

    print("Merged SFT JSONL files")
    for path, count in zip(input_paths, counts):
        print(f"  {path}: {count} records")
    print(f"Total records: {sum(counts)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
