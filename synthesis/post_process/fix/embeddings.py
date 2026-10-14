"""Optional OpenAI-compatible title embeddings; visual evidence remains decisive."""

import math
import os
import re


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Invalid embedding dimensions")
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def score_titles(candidates, anchor_title, config, cache):
    texts = [anchor_title] + [c.public.get("title", "") for c in candidates]
    if config.embedding_url:

        def fetch():
            import requests

            headers = {}
            if os.environ.get("FIX_EMBEDDING_API_KEY"):
                headers["Authorization"] = (
                    "Bearer " + os.environ["FIX_EMBEDDING_API_KEY"]
                )
            response = requests.post(
                config.embedding_url,
                json={"input": texts, "model": config.embedding_model},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda x: x["index"])
            if len(rows) != len(texts):
                raise ValueError("Embedding count mismatch")
            return [r["embedding"] for r in rows]

        vectors = cache.get_or_compute(
            "embeddings",
            {
                "url": config.embedding_url,
                "model": config.embedding_model,
                "texts": texts,
            },
            fetch,
        )
        scores = [cosine(vectors[0], v) for v in vectors[1:]]
    else:
        tokens = [set(re.findall(r"\w+", t.casefold())) for t in texts]
        scores = [len(tokens[0] & t) / max(1, len(tokens[0] | t)) for t in tokens[1:]]
    for c, score in zip(candidates, scores):
        c.title_similarity = score
