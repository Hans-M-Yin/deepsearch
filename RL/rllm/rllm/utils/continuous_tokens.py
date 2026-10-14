"""Opt-in Qwen image/text TITO ledger. Never tokenize generated text.

Keep unexpanded request IDs and expanded training IDs separately. Only new
context is processed; images/features are retained in insertion order.
Preparation is transactional: a failed/cancelled generation must not commit it.
"""
from copy import deepcopy

import torch


def batch_widths(prompts, responses, model_limit, response_limit):
    """Pad TITO rows to actual batch maxima; never truncate a sampled prefix."""
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("Invalid continuous TITO batch rows")
    for prompt, response in zip(prompts, responses):
        if len(prompt) + len(response) > model_limit or len(response) > response_limit:
            raise ValueError("Continuous TITO batch exceeds context/response budget; refusing truncation")
    return max(len(p) for p in prompts), max(len(r) for r in responses)


def prepare(parser, tokenizer, processor, messages, tools, state):
    if parser.__class__.__name__ != "QwenChatTemplateParser":
        raise ValueError("continuous TITO currently supports QwenChatTemplateParser only")
    previous = state.get("messages")
    if previous is None:
        if any(m["role"] == "assistant" for m in messages):
            raise ValueError("continuous TITO must start before the first assistant turn")
        delta = messages
        raw, expanded, mask, images, features = [], [], [], [], {}
        boundary = []
    else:
        if messages[:len(previous)] != previous:
            raise ValueError("continuous TITO history was edited; cannot reconstruct sampled context")
        appended = messages[len(previous):]
        if not appended or appended[0]["role"] != "assistant":
            raise ValueError("continuous TITO expects the preceding generated assistant message")
        delta = appended[1:]
        if not delta or any(m["role"] == "assistant" for m in delta):
            raise ValueError("continuous TITO expects new tool/user context after one assistant")
        raw, expanded, mask = list(state["raw"]), list(state["expanded"]), list(state["mask"])
        images, features = list(state["images"]), dict(state["features"])
        # End markers omitted by a length-limited generation are synthetic,
        # just like the newline after a sampled im_end. Both have loss mask 0.
        eot = tokenizer.encode(parser.eot_token, add_special_tokens=False)
        overlap = next((n for n in range(min(len(eot), len(raw)), 0, -1)
                        if raw[-n:] == eot[:n]), 0)
        boundary = eot[overlap:]
    if any(m.get("images") and m["role"] != "user" for m in delta):
        raise ValueError("continuous TITO expects image observations in user messages")
    if any(m.get("videos") or m.get("audios") for m in delta):
        raise ValueError("continuous TITO currently supports images, not video/audio")
    rendered = parser.parse(delta, add_generation_prompt=True,
                            is_first_msg=previous is None, tools=tools,
                            accumulate_reasoning=False)
    delta_raw = list(tokenizer.encode(rendered, add_special_tokens=False))
    declared = sum(len(m.get("images") or []) for m in delta)
    new_images = parser.process_image_data(delta) if declared else []
    if len(new_images) != declared:
        raise ValueError("continuous TITO image count changed during preprocessing")
    new_features = {}
    if new_images:
        if processor is None:
            raise ValueError("continuous TITO images require a processor")
        result = dict(processor(text=[rendered], images=new_images, return_tensors="pt"))
        delta_expanded = result.pop("input_ids")[0].tolist()
        result.pop("attention_mask", None)
        # Explicit allowlist: unknown sequence-dependent processor outputs must
        # not be concatenated as though they were per-image tensors.
        if set(result) != {"pixel_values", "image_grid_thw"}:
            raise ValueError(f"Unsupported continuous TITO processor fields: {sorted(result)}")
        grid = result["image_grid_thw"]
        if grid.ndim != 2 or grid.shape != (declared, 3):
            raise ValueError("Invalid continuous TITO image_grid_thw")
        merge = int(processor.image_processor.merge_size)
        counts = []
        for t, h, w in grid.tolist():
            if min(t, h, w) <= 0 or h % merge or w % merge:
                raise ValueError("Invalid continuous TITO image patch layout")
            counts.append(t * (h // merge) * (w // merge))
        image_id = tokenizer.convert_tokens_to_ids(parser.image_token)
        expected, index = [], 0
        for token in delta_raw:
            if token == image_id:
                if index >= len(counts):
                    raise ValueError("Unexpected image placeholder in continuous TITO context")
                expected.extend([token] * counts[index])
                index += 1
            else:
                expected.append(token)
        if index != declared or expected != delta_expanded:
            raise ValueError("Processor changed non-image IDs or image expansion disagrees")
        if result["pixel_values"].shape[0] != int(grid.prod(dim=-1).sum()):
            raise ValueError("Image pixels and grids disagree")
        new_features = result
    else:
        delta_expanded = delta_raw
        image_id = tokenizer.convert_tokens_to_ids(parser.image_token)
        if image_id in delta_raw:
            raise ValueError("Image placeholder without image payload")
    for key, value in new_features.items():
        features[key] = torch.cat([features[key], value], dim=0) if key in features else value
    raw.extend(boundary + delta_raw)
    expanded.extend(boundary + delta_expanded)
    mask.extend([0] * (len(boundary) + len(delta_expanded)))
    return dict(raw=raw, expanded=expanded, mask=mask,
                images=images + list(new_images), features=features,
                messages=deepcopy(messages),
                initial_length=state.get("initial_length", len(expanded)))


def commit(candidate, completion_ids, tokenizer, parser):
    ids = [int(x) for x in completion_ids]
    # This implementation supports image *inputs*, not generated image tokens.
    if tokenizer.convert_tokens_to_ids(parser.image_token) in ids:
        raise ValueError("Generated image placeholder unsupported in continuous TITO")
    candidate["raw"] = candidate["raw"] + ids
    candidate["expanded"] = candidate["expanded"] + ids
    candidate["mask"] = candidate["mask"] + [1] * len(ids)
    return candidate
