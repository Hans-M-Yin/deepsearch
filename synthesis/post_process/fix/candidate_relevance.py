"""Blind, pre-inspection competition assessment; no candidate images or labels."""


def visible_card(candidate):
    card = {
        k: candidate.public.get(k, "")
        for k in ("title", "source", "image_id", "source_page_id")
    }
    # These are real public search fields, not hidden verifier information.
    # Both selectors and the spliced observation must retain the same evidence.
    for key in ("snippet", "image_url_keywords", "source_page_url_keywords"):
        if key in candidate.public:
            card[key] = candidate.public[key]
    return card


PROMPT = """Assess which search cards are genuinely competitive next images to
inspect, based ONLY on the supplied original image, question, history and card
text. Do not use world knowledge, unseen candidate images or guessed answers.
Compare cards against each other, not merely against a broad topic. A generic
social title or a weak topic overlap is not competitive if other cards give much
more specific matching clues. Return JSON {"cards":[{"image_id":"...",
"competitive":true|false,"reason":"specific visible/textual reason"}]} for
EVERY supplied ID exactly once. Do not deliberately prefer incorrect candidates.
"""


def assess_competition(candidates, window, source, history, model, config):
    cards = {c.public["image_id"]: visible_card(c) for c in candidates}
    result = model.ask(
        config.model,
        PROMPT,
        {
            "question": window.question,
            "trajectory_history": history,
            "search_results": list(cards.values()),
        },
        [source],
    )
    rows = result.get("cards", [])
    judged = {}
    for row in rows:
        rid = row.get("image_id")
        if (
            rid not in cards
            or rid in judged
            or type(row.get("competitive")) is not bool
            or not row.get("reason")
        ):
            raise ValueError("Invalid candidate competition assessment")
        judged[rid] = row
    if set(judged) != set(cards):
        raise ValueError("Competition assessment omitted candidates")
    for candidate in candidates:
        candidate.text_competitive = judged[candidate.public["image_id"]]["competitive"]
    return result
