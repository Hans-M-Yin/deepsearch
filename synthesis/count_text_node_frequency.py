#!/usr/bin/env python3
"""Count text/wiki nodes occurring in ShareGPT reasoning paths.

The input ShareGPT file and the graph JSONL files can be very large.  This
script therefore:

1. Streams the ShareGPT top-level JSON array and keeps only its sample IDs.
2. Streams samples.jsonl once, retaining paths only for those sample IDs.
3. Streams nodes.jsonl once, retaining names only for node IDs referenced by
   those paths.
4. Writes a two-column CSV (name,count), counting one occurrence per dataset
   record and per path node. Image nodes are excluded.

The default graph directory is inferred from a vqa directory laid out as
``.../vqa/<batch>``: its parent's parent is used as the graph directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


JSON_CHUNK_SIZE = 8 * 1024 * 1024


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def iter_json_array(path: Path, chunk_size: int = JSON_CHUNK_SIZE) -> Iterator[Any]:
    """Yield elements from a large top-level JSON array incrementally."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        first = True
        finished = False

        while not finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            elif position >= len(buffer):
                break

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if first:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"Expected a top-level JSON array in {path}")
                    position += 1
                    first = False

                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    position += 1
                    finished = True
                    break
                if buffer[position] == ",":
                    position += 1
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position >= len(buffer):
                        break

                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    # The next object may be split across chunks.
                    break

                yield item
                position = end

            # Retain only the unconsumed suffix.  This bounds the parser's
            # buffer by one chunk plus the largest individual record.
            if position:
                buffer = buffer[position:]
                position = 0

            if not chunk:
                if buffer.strip() not in ("", "]"):
                    raise ValueError(f"Unexpected trailing data in {path}")
                break

        if not finished:
            raise ValueError(f"Top-level JSON array in {path} was not closed")


def as_nonempty_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def infer_node_type(node_id: str) -> Optional[str]:
    if node_id.startswith("image_"):
        return "image"
    if node_id.startswith("text_"):
        return "text"
    return None


def extract_path(sample: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Return (node_id, path_node_type) pairs in reasoning-chain order."""

    path = sample.get("path")
    if not isinstance(path, dict):
        raise ValueError(f"sample {sample.get('sample_id')!r} has no path object")

    node_ids = path.get("node_ids")
    node_types = path.get("node_types")
    if not isinstance(node_ids, list) or not node_ids:
        raise ValueError(f"sample {sample.get('sample_id')!r} has no path.node_ids")

    if node_types is None:
        node_types = [infer_node_type(str(node_id)) for node_id in node_ids]
    if not isinstance(node_types, list) or len(node_ids) != len(node_types):
        raise ValueError(
            f"sample {sample.get('sample_id')!r} has mismatched path.node_ids/node_types"
        )

    result: List[Tuple[str, str]] = []
    for node_id, node_type in zip(node_ids, node_types):
        node_id_text = as_nonempty_string(node_id)
        node_type_text = as_nonempty_string(node_type)
        if node_id_text is None:
            raise ValueError(f"sample {sample.get('sample_id')!r} has an empty node ID")
        if node_type_text is None:
            node_type_text = infer_node_type(node_id_text) or "unknown"
        result.append((node_id_text, node_type_text.lower()))
    return tuple(result)


def text_node_name(node: Dict[str, Any]) -> Optional[str]:
    """Get a stable human-readable name from a graph text-node record."""

    for key in ("title", "name", "label"):
        value = as_nonempty_string(node.get(key))
        if value:
            return value

    canonical_id = as_nonempty_string(node.get("canonical_id"))
    if canonical_id:
        if canonical_id.startswith("wikipedia:"):
            canonical_id = canonical_id[len("wikipedia:") :].strip()
        if canonical_id:
            return canonical_id

    summary = as_nonempty_string(node.get("summary"))
    if summary:
        first_line = summary.splitlines()[0].strip()
        if first_line.startswith("#"):
            first_line = first_line.lstrip("#").strip()
        if first_line:
            return first_line
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="ShareGPT dataset directory containing trajectories_sharegpt.json",
    )
    parser.add_argument(
        "--vqa-dir",
        required=True,
        type=Path,
        help="Directory containing the source samples.jsonl",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        help="Directory containing nodes.jsonl; inferred from --vqa-dir if omitted",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="Print JSONL progress every N lines (default: 250000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    vqa_dir = args.vqa_dir.expanduser().resolve()
    graph_dir = (
        args.graph_dir.expanduser().resolve()
        if args.graph_dir is not None
        else vqa_dir.parent.parent
    )
    output = args.output.expanduser().resolve()

    dataset_path = dataset_dir / "trajectories_sharegpt.json"
    samples_path = vqa_dir / "samples.jsonl"
    nodes_path = graph_dir / "nodes.jsonl"
    for required_path in (dataset_path, samples_path, nodes_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    started_at = time.monotonic()
    log(f"[1/4] Streaming dataset records from {dataset_path}")
    dataset_rows: List[Tuple[str, Optional[str], Optional[str]]] = []
    dataset_unique_ids = set()
    duplicate_dataset_ids = 0
    missing_question_ids = 0
    for record_index, record in enumerate(iter_json_array(dataset_path), start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {record_index} is not a JSON object")
        sample_id = as_nonempty_string(record.get("sample_id"))
        if sample_id is None:
            raise ValueError(f"Dataset record {record_index} has no sample_id")
        question_id = as_nonempty_string(record.get("question_id"))
        path_id = as_nonempty_string(record.get("path_id"))
        if question_id is None:
            missing_question_ids += 1
        if sample_id in dataset_unique_ids:
            duplicate_dataset_ids += 1
        dataset_unique_ids.add(sample_id)
        dataset_rows.append((sample_id, question_id, path_id))
        if record_index % 2_000 == 0:
            log(f"  dataset records={record_index:,}, unique sample_id={len(dataset_unique_ids):,}")

    log(
        f"  dataset records={len(dataset_rows):,}, unique sample_id={len(dataset_unique_ids):,}, "
        f"duplicate rows={duplicate_dataset_ids:,}, missing question_id={missing_question_ids:,}"
    )

    log(f"[2/4] Streaming source samples from {samples_path}")
    paths_by_sample_id: Dict[str, Tuple[Tuple[str, str], ...]] = {}
    sample_lines = 0
    matched_sample_lines = 0
    duplicate_source_ids = 0
    for line in samples_path.open("r", encoding="utf-8"):
        sample_lines += 1
        if not line.strip():
            continue
        sample = json.loads(line)
        if not isinstance(sample, dict):
            continue
        sample_id = as_nonempty_string(sample.get("sample_id"))
        if sample_id not in dataset_unique_ids:
            if sample_lines % args.progress_every == 0:
                log(f"  samples lines={sample_lines:,}, matched unique={len(paths_by_sample_id):,}")
            continue

        path = extract_path(sample)
        previous = paths_by_sample_id.get(sample_id)
        if previous is not None:
            duplicate_source_ids += 1
            if previous != path:
                raise ValueError(f"Conflicting paths found for source sample_id={sample_id}")
        else:
            paths_by_sample_id[sample_id] = path
            matched_sample_lines += 1
        if sample_lines % args.progress_every == 0:
            log(f"  samples lines={sample_lines:,}, matched unique={len(paths_by_sample_id):,}")

    missing_sample_ids = dataset_unique_ids.difference(paths_by_sample_id)
    if missing_sample_ids:
        examples = sorted(missing_sample_ids)[:5]
        raise ValueError(
            f"{len(missing_sample_ids):,} dataset sample_id values were not found in samples.jsonl; "
            f"examples={examples}"
        )
    log(
        f"  samples lines={sample_lines:,}, matched unique={matched_sample_lines:,}, "
        f"duplicate source rows={duplicate_source_ids:,}"
    )

    text_candidate_ids = set()
    for sample_id, _, _ in dataset_rows:
        for node_id, path_node_type in paths_by_sample_id[sample_id]:
            # A few source paths label an image as ``unknown``.  The node ID
            # prefix is still authoritative for excluding those image nodes.
            if path_node_type != "image" and infer_node_type(node_id) != "image":
                text_candidate_ids.add(node_id)
    log(f"  candidate non-image node IDs={len(text_candidate_ids):,}")

    log(f"[3/4] Streaming graph nodes from {nodes_path}")
    node_names: Dict[str, str] = {}
    node_types: Dict[str, str] = {}
    node_lines = 0
    matched_node_lines = 0
    missing_names = 0
    for line in nodes_path.open("r", encoding="utf-8"):
        node_lines += 1
        if not line.strip():
            continue
        node = json.loads(line)
        if not isinstance(node, dict):
            continue
        node_id = as_nonempty_string(node.get("node_id"))
        if node_id not in text_candidate_ids:
            if node_lines % args.progress_every == 0:
                log(f"  graph lines={node_lines:,}, matched target nodes={matched_node_lines:,}")
            continue

        node_type = (as_nonempty_string(node.get("node_type")) or "unknown").lower()
        node_types[node_id] = node_type
        if node_type == "text":
            name = text_node_name(node)
            if name is not None:
                node_names[node_id] = name
            else:
                missing_names += 1
        matched_node_lines += 1
        if node_lines % args.progress_every == 0:
            log(f"  graph lines={node_lines:,}, matched target nodes={matched_node_lines:,}")

    unresolved_node_ids = text_candidate_ids.difference(node_types)
    if unresolved_node_ids:
        examples = sorted(unresolved_node_ids)[:5]
        raise ValueError(
            f"{len(unresolved_node_ids):,} candidate node IDs were not found in nodes.jsonl; "
            f"examples={examples}"
        )
    if missing_names:
        raise ValueError(f"{missing_names:,} target text nodes have no usable name")

    log(
        f"  graph lines={node_lines:,}, matched target nodes={matched_node_lines:,}, "
        f"text names={len(node_names):,}"
    )

    log("[4/4] Aggregating occurrences and writing CSV")
    counts: Counter[str] = Counter()
    skipped_unknown = 0
    skipped_graph_images = 0
    counted_occurrences = 0
    for sample_id, _, _ in dataset_rows:
        for node_id, path_node_type in paths_by_sample_id[sample_id]:
            if path_node_type == "image" or infer_node_type(node_id) == "image":
                continue
            graph_node_type = node_types[node_id]
            if graph_node_type == "image":
                skipped_graph_images += 1
                continue
            if graph_node_type != "text":
                skipped_unknown += 1
                continue
            name = node_names[node_id]
            counts[name] += 1
            counted_occurrences += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("name", "count"))
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                writer.writerow((name, count))
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    elapsed = time.monotonic() - started_at
    log(f"Wrote {len(counts):,} names and {counted_occurrences:,} text-node occurrences to {output}")
    log(
        f"Skipped graph-image occurrences={skipped_graph_images:,}, "
        f"unknown-node occurrences={skipped_unknown:,}, elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
