"""Resume migrations with immutable data/semantic config and tunable concurrency."""

CONCURRENCY_FIELDS = frozenset({
    "record_workers", "visual_check_workers", "writer_workers", "global_visual_workers",
})


def resume_manifest(
    previous, current, *, resume, allow_code_change, run_id, started_at
):
    if allow_code_change and not resume:
        raise ValueError("--resume-allow-code-change requires --resume")
    if previous is None:
        return dict(current)
    identity = ("version", "input", "input_digest")
    changed = [key for key in identity if previous.get(key) != current.get(key)]
    old_config, new_config = previous.get("config"), current.get("config")
    concurrency_changes = {}
    if isinstance(old_config, dict) and isinstance(new_config, dict):
        for key in sorted(old_config.keys() | new_config.keys()):
            if key in old_config and key in new_config and old_config[key] == new_config[key]:
                continue
            if resume and key in CONCURRENCY_FIELDS and key in old_config and key in new_config:
                if type(new_config[key]) is not int or new_config[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
                concurrency_changes[key] = {"from": old_config[key], "to": new_config[key]}
            else:
                changed.append("config." + key)
    elif old_config != new_config:
        changed.append("config")
    if changed:
        raise ValueError(
            "Workdir input/config/schema mismatch; cannot resume: " + ", ".join(changed)
        )
    old_code = previous.get("code_digest")
    if not isinstance(old_code, str) or not old_code:
        raise ValueError("Workdir manifest has no valid code_digest")
    result = dict(previous)
    if concurrency_changes:
        result["config_history"] = [*previous.get("config_history", []), {
            "run_id": run_id, "started_at": started_at,
            "policy": "concurrency_only", "changes": concurrency_changes,
        }]
    if old_code != current["code_digest"]:
        if not (resume and allow_code_change):
            raise ValueError(
                "Workdir code differs; use a new workdir or explicitly --resume --resume-allow-code-change"
            )
        # Old checkpoints without individual provenance always belong to the
        # initial code, even across multiple subsequent migrations or crashes.
        result.setdefault("legacy_code_digest", old_code)
        history = list(previous.get("code_history", []))
        history.append(
            {
                "from": old_code,
                "to": current["code_digest"],
                "run_id": run_id,
                "started_at": started_at,
                "policy": "reuse_terminal_checkpoints_without_revalidation",
            }
        )
        result["code_history"] = history
    result.update(current)
    return result
