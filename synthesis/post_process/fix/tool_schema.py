"""Locate real tool definitions without mistaking documentation for JSON."""

import json
import re


def expose_image_argument(system):
    matches = []
    for block in re.finditer(r"<tools>\s*(.*?)\s*</tools>", system, re.S):
        try:
            specs = json.loads(block.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs):
            continue
        functions = [s.get("function") for s in specs]
        targets = [
            f
            for f in functions
            if isinstance(f, dict) and f.get("name") == "i2i_search"
        ]
        if targets:
            matches.append((block, specs, targets))
    if len(matches) != 1 or len(matches[0][2]) != 1:
        raise ValueError(
            "Expected one unambiguous tools JSON definition containing i2i_search; found "
            + str(len(matches))
            + " matching blocks"
        )
    block, specs, targets = matches[0]
    parameters = targets[0].get("parameters")
    if not isinstance(parameters, dict) or not isinstance(
        parameters.get("properties"), dict
    ):
        raise ValueError("Invalid i2i_search parameters/properties in tools JSON")
    parameters["properties"].setdefault(
        "image",
        {
            "type": "string",
            "description": "Explicit input image reference; img_1 is the first user image. Other image IDs come from image observations. Omit to search the latest image.",
        },
    )
    return (
        system[: block.start(1)]
        + json.dumps(specs, ensure_ascii=False, indent=2)
        + system[block.end(1) :]
    )
