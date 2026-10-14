"""Blindly check selected images before their observations enter the trajectory."""

from .visual_evidence import CLASSIFICATION

PROMPT = (
    """Blindly compare IMAGE 0 and IMAGE 1 for the local visual target.
No candidate title, adopted identity, prior label or future answer is supplied.
"""
    + CLASSIFICATION
    + """
Return JSON {"category":"CONSISTENT|SIMILAR|RELATED|UNRELATED|SAME_ENTITY|DIFFERENT_PHOTO|UNRESOLVED",
"evidence":["concrete visible comparisons"]}."""
)


def check_selected(candidate, window, target, source, model, config):
    result = model.ask(
        config.judge_model,
        PROMPT,
        {"question": window.question, "local_visual_target": target},
        [source, *candidate.image_paths],
    )
    labels = {
        "CONSISTENT": "YES",
        "SIMILAR": "NO",
        "RELATED": "NO",
        "UNRELATED": "NO",
        "SAME_ENTITY": "CONTINUE",
        "DIFFERENT_PHOTO": "CONTINUE",
    }
    category, evidence = result.get("category"), result.get("evidence")
    if (
        category not in labels
        or not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(e, str) and e.strip() for e in evidence)
    ):
        raise ValueError("Selected image lacks usable blind visual evidence")
    candidate.category, candidate.verdict, candidate.evidence = (
        category,
        "YES"
        if category == "SAME_ENTITY" and candidate.originally_viewed
        else labels[category],
        evidence,
    )
    return result
