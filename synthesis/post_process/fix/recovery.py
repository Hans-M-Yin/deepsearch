"""Evidence-conditioned retry query, planned before observing retry results."""

PROMPT = """Choose a reverse-image-search retry strategy using the original image,
question and visual failures observed so far. You do not know future results.
Return JSON {"strategy":"same|full|target_crop", "region":[x1,y1,x2,y2],
"reason":"why this retry can help"}. Only use an allowed strategy. For same,
preserve the original region. For full use [0,0,1000,1000]. For target_crop use
normalized 0..1000 coordinates enclosing the question target with enough context;
avoid tiny or irrelevant regions. Do not assume the retry will succeed."""


def retry_query(window, plan, source, image_ref, model, config):
    result = model.ask(
        config.model,
        PROMPT,
        {
            "question": window.question,
            "original_region": window.call["arguments"].get("region"),
            "allowed_strategies": config.retry_strategies,
            "observed_failures": [
                {"card": c.public, "evidence": c.evidence} for c in plan.attempts
            ],
        },
        [source],
    )
    strategy = result.get("strategy")
    if strategy not in config.retry_strategies or not result.get("reason"):
        raise ValueError("Invalid retry strategy")
    args = {**window.call["arguments"], "image": image_ref}
    args.pop("url", None)
    if strategy == "full":
        args["region"] = [0, 0, 1000, 1000]
    elif strategy == "target_crop":
        box = result.get("region")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(x) in (int, float) and 0 <= x <= 1000 for x in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
            or (box[2] - box[0]) * (box[3] - box[1]) < 40000
        ):
            raise ValueError("Invalid or too-small retry crop")
        args["region"] = box
    plan.metadata["retry_strategy"] = strategy
    plan.metadata["retry_reason"] = result["reason"]
    return {"name": "i2i_search", "arguments": args}
