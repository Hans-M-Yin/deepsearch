"""Attach trajectory images to semantic review with explicit turn provenance."""

from .trajectory_index import (
    image_bindings,
    messages,
    content,
    parse_observation,
    resolve_path,
)


def build_review_images(updated, source, attempts, dataset_root):
    paths, positions = [], {}

    def attach(path):
        resolved = resolve_path(path, dataset_root)
        if resolved not in positions:
            positions[resolved] = len(paths)
            paths.append(resolved)
        return positions[resolved]

    reference = attach(source)
    checks = [
        {"candidate_key": c.key, "image_indices": [attach(p) for p in c.image_paths]}
        for c in attempts
    ]
    _, by_turn = image_bindings(updated)
    turns = messages(updated)
    mapping = []
    for turn_index, turn_paths in by_turn.items():
        if not turn_paths:
            continue
        try:
            resource_id = parse_observation(content(turns[turn_index])).get("image_id")
        except (ValueError, TypeError):
            resource_id = None
        for placeholder_index, path in enumerate(turn_paths):
            mapping.append(
                {
                    "turn_index": turn_index,
                    "placeholder_index": placeholder_index,
                    "resource_image_id": resource_id,
                    "image_index": attach(path),
                }
            )
    return paths, {
        "reference_image_index": reference,
        "candidate_checks": checks,
        "trajectory_images": mapping,
    }
