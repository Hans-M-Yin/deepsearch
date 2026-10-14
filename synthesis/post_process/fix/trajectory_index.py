"""ShareGPT parsing and conservative, resource-aware splice boundaries."""

import json
import re
from pathlib import Path
from .schema import Window

ASSISTANT = {"gpt", "assistant", "model"}
OBSERVATION = {"observation", "tool"}
ACTION = re.compile(r"<(tool_call|action)>\s*(.*?)\s*</\1>", re.S)
RESOURCE_FIELDS = ("image_id", "page_id", "source_page_id")


def messages(record):
    return record.get("conversations", record.get("messages", []))


def role(message):
    return message.get("from", message.get("role", message.get("speaker", ""))).lower()


def content(message):
    value = message.get("value", message.get("content", ""))
    if not isinstance(value, str):
        raise ValueError(
            "Only text ShareGPT messages with <image> placeholders are supported"
        )
    return value


def set_content(message, value):
    return {**message, "value" if "value" in message else "content": value}


def call(text):
    found = ACTION.findall(text)
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("Multiple tool calls in a turn are unsupported")
    result = json.loads(found[0][1])
    if not isinstance(result.get("arguments"), dict):
        raise ValueError("Tool arguments must be an object")
    return result


def resource_id(tool):
    args = tool.get("arguments", {})
    return str(
        args.get("resource_id") or args.get("image_id") or args.get("page_id") or ""
    )


def parse_observation(text):
    text = text.strip()
    payload, end = json.JSONDecoder().raw_decode(text)
    suffix = text[end:].strip()
    # Accept only the known image-rendering wrapper, not arbitrary trailing data.
    if suffix and not re.fullmatch(
        r"(?:The image is shown below:\s*)?(?:<image>\s*)+", suffix
    ):
        raise ValueError("Unexpected text after observation JSON")
    if not isinstance(payload, dict):
        raise ValueError("Observation JSON must be an object")
    return payload


def image_bindings(record):
    paths = record.get("images", [])
    offset, bindings, by_turn = 0, {}, {}
    for i, message in enumerate(messages(record)):
        count = content(message).count("<image>")
        selected = paths[offset : offset + count]
        if len(selected) != count:
            raise ValueError("Image placeholders do not match images")
        by_turn[i] = selected
        if count:
            if role(message) in OBSERVATION:
                try:
                    payload = parse_observation(content(message))
                except (ValueError, TypeError):
                    payload = {}
                if count == 1 and payload.get("image_id"):
                    bindings[payload["image_id"]] = selected[0]
            elif role(message) in {"human", "user"}:
                for j, path in enumerate(selected):
                    bindings.setdefault(f"img_{offset + j + 1}", path)
        offset += count
    if offset != len(paths):
        raise ValueError("Unused images in ShareGPT record")
    return bindings, by_turn


def resolve_path(path, dataset_root):
    source = Path(path)
    options = (
        [source]
        if source.is_absolute()
        else [Path(dataset_root) / source, Path.cwd() / source]
    )
    for option in options:
        if option.is_file():
            return str(option.resolve())
    raise FileNotFoundError(path)


def record_candidate_read(reads, rid, observation):
    """Keep the earliest view as the reuse anchor; retain later inspections too."""
    if rid not in reads:
        reads[rid] = {**observation, "later_reads": []}
    else:
        reads[rid]["later_reads"].append(observation)


def candidate_read_occurrences(read):
    return [
        {k: v for k, v in read.items() if k != "later_reads"},
        *read.get("later_reads", []),
    ]


def index_record(record):
    turns = messages(record)
    bindings, by_turn = image_bindings(record)
    question = "\n".join(content(m) for m in turns if role(m) in {"human", "user"})
    result = []
    for i, message in enumerate(turns):
        if role(message) not in ASSISTANT:
            continue
        tool = call(content(message))
        if not tool or tool.get("name") != "i2i_search":
            continue
        if i + 1 >= len(turns) or role(turns[i + 1]) not in OBSERVATION:
            continue
        observation = parse_observation(content(turns[i + 1]))
        candidates = observation.get("matches", observation.get("results", []))
        if not isinstance(candidates, list) or not candidates:
            continue
        args = tool["arguments"]
        ref = args.get("image") or args.get("url")
        # Reconstruct only images available before the call, never a future image.
        prior = {p for idx, paths in by_turn.items() if idx < i for p in paths}
        if ref:
            source = bindings.get(ref, ref if ref in prior else None)
            if source not in prior:
                continue
        else:
            available = [p for idx, paths in by_turn.items() if idx < i for p in paths]
            source = available[-1] if available else None
            last_visible = max(
                (idx for idx, paths in by_turn.items() if idx < i and paths), default=-1
            )
            # The runtime can register an uploaded crop without emitting an image
            # observation. A later implicit search could target that hidden crop.
            # Do not silently substitute the last visible image in this case.
            hidden_crop = False
            for prior_turn in turns[last_visible + 1 : i]:
                prior_call = (
                    call(content(prior_turn)) if role(prior_turn) in ASSISTANT else None
                )
                if (
                    prior_call
                    and prior_call.get("name") == "i2i_search"
                    and prior_call["arguments"].get("region")
                    and prior_call["arguments"]["region"] != [0, 0, 1000, 1000]
                ):
                    hidden_crop = True
            if hidden_crop:
                continue
        if not source:
            continue
        candidate_ids = {c.get("image_id") for c in candidates}
        image_for_resource = {
            c[k]: c["image_id"]
            for c in candidates
            for k in RESOURCE_FIELDS
            if c.get(k) and c.get("image_id")
        }
        end, reads = i + 2, {}
        # Locate the first retained action after leading image checks.
        while end + 1 < len(turns) and role(turns[end]) in ASSISTANT:
            next_call = call(content(turns[end]))
            if not next_call or next_call.get("name") != "read_url":
                break
            resource = resource_id(next_call)
            rid = image_for_resource.get(resource)
            if (
                rid not in candidate_ids
                or not by_turn[end + 1]
                or role(turns[end + 1]) not in OBSERVATION
            ):
                break
            record_candidate_read(
                reads,
                rid,
                {
                    "paths": by_turn[end + 1],
                    "observation": content(turns[end + 1]),
                    "turn": end,
                    "resource_id": resource,
                },
            )
            end += 2
        if end >= len(turns) or role(turns[end]) not in ASSISTANT:
            continue
        # A page read may precede the original candidate image read. Reuse that
        # image too, but retain all intervening page calls and observations.
        for cursor in range(end, len(turns) - 1):
            if role(turns[cursor]) not in ASSISTANT:
                continue
            later = call(content(turns[cursor]))
            if not later:
                continue
            if later.get("name") == "i2i_search":
                break
            resource = resource_id(later)
            rid = image_for_resource.get(resource)
            if (
                later.get("name") == "read_url"
                and rid in candidate_ids
                and role(turns[cursor + 1]) in OBSERVATION
                and by_turn[cursor + 1]
            ):
                record_candidate_read(
                    reads,
                    rid,
                    {
                        "paths": by_turn[cursor + 1],
                        "observation": content(turns[cursor + 1]),
                        "turn": cursor,
                        "resource_id": resource,
                    },
                )
        removed = {r["turn"] + delta for r in reads.values() for delta in (0, 1)}
        all_ids = {str(c[k]) for c in candidates for k in RESOURCE_FIELDS if c.get(k)}
        protected = set()
        for tail_index, tail in enumerate(turns[end:], end):
            if tail_index in removed:
                continue
            if role(tail) in ASSISTANT:
                action = call(content(tail))
                if action:
                    encoded = json.dumps(action)
                    protected.update(r for r in all_ids if r in encoded)
        result.append(
            Window(
                i,
                i + 1,
                i + 2,
                end,
                tool,
                observation,
                candidates,
                reads,
                question,
                source,
                sorted(protected),
            )
        )
    return result
