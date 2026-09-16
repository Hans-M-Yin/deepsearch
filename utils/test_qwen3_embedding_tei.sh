#!/usr/bin/env bash
# Compare several duplicate-search query sets and input formats against TEI.
set -euo pipefail

MODEL_SIZE="${EMBEDDING_MODEL_SIZE:-0.6B}"
case "${MODEL_SIZE,,}" in
  0.6b|0p6b) DEFAULT_PORT=8086 ;;
  4b) DEFAULT_PORT=8087 ;;
  *)
    echo "EMBEDDING_MODEL_SIZE must be 0.6B or 4B, got: ${MODEL_SIZE}" >&2
    exit 1
    ;;
esac
EMBEDDING_URL="${SEARCH_SIMILARITY_EMBEDDING_URL:-http://127.0.0.1:${DEFAULT_PORT}/embed}"
TEST_TIMEOUT_S="${SEARCH_SIMILARITY_TEST_TIMEOUT_S:-30}"

python - "${EMBEDDING_URL}" "${TEST_TIMEOUT_S}" <<'PY'
import json
import math
import sys
import time
import urllib.request

url = sys.argv[1]
timeout_s = float(sys.argv[2])

# Pairs within a group are deliberately chosen to cover the cases used by the
# reward: near-duplicate reformulations, natural query refinements, and false
# positives that share an entity or vocabulary but express another intent.
query_groups = {
    "duplicate_rewrites_expected_high": [
        "Van Gogh birthplace",
        "Vincent van Gogh birthplace",
        "Where was Vincent van Gogh born?",
        "Van Gogh birthplace Netherlands",
    ],
    "same_task_refinements_expected_high": [
        "Van Gogh paintings in Paris",
        "Vincent van Gogh Paris paintings",
        "Van Gogh artwork Paris",
    ],
    "same_entity_different_intent_expected_lower": [
        "Van Gogh birthplace",
        "Van Gogh paintings in Paris",
        "Van Gogh self portraits",
        "Van Gogh death date",
    ],
    "lexical_overlap_different_intent_expected_lower": [
        "France capital city",
        "France population",
        "France president",
        "Paris population",
    ],
    # These are intentionally *not* all labelled as duplicates.  They model a
    # realistic finance-search chain: adding a year resolves an ambiguity, and
    # narrowing Alphabet to share class GOOGL can be useful.  Inspect these
    # scores before applying any hard penalty to a query refinement.
    "alphabet_date_refinements_inspect_not_auto_duplicate": [
        "Alphabet stock price July 24",
        "Alphabet stock price July 24 2024",
        "GOOGL quote July 24",
    ],
    # Within each company-name/ticker pair, the information need is normally
    # equivalent.  Cross-pairs are still useful to inspect because Alphabet
    # can refer to multiple share classes while GOOGL is specific.
    "alphabet_history_aliases_inspect_share_class_precision": [
        "Alphabet GOOGL historical prices",
        "Alphabet stock price history",
        "GOOGL historical prices",
        "GOOGL stock price history",
    ],
}

# The first format is the intended production format.  The others are useful
# calibration baselines: if scores move materially, reward-time formatting must
# be fixed and not changed silently between runs.
settings = {
    "semantic_instruction": (
        "Given a web search query, represent its information need for semantic "
        "similarity and duplicate-search detection."
    ),
    "duplicate_instruction": (
        "Represent this web search query for detecting whether two searches ask "
        "for the same information, despite paraphrasing."
    ),
    "bare_query_baseline": None,
}

queries = []
group_indices = {}
for group_name, group_queries in query_groups.items():
    start = len(queries)
    queries.extend(group_queries)
    group_indices[group_name] = list(range(start, start + len(group_queries)))

def cosine(left, right):
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm)

def encode_query(query, instruction):
    if instruction is None:
        return query
    return f"Instruct: {instruction}\nQuery: {query}"

def embed(inputs):
    request = urllib.request.Request(
        url,
        data=json.dumps({"inputs": inputs}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        vectors = json.load(response)
        status = response.status
    elapsed_s = time.perf_counter() - started
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        length = len(vectors) if isinstance(vectors, list) else "n/a"
        raise RuntimeError(
            "Unexpected TEI /embed response: "
            f"type={type(vectors)!r}, len={length}"
        )
    return vectors, status, elapsed_s

print(f"endpoint={url}")
print(f"request_timeout_s={timeout_s:g}")
print(f"query_count={len(queries)}")
print()

for setting_name, instruction in settings.items():
    inputs = [encode_query(query, instruction) for query in queries]
    vectors, status, request_s = embed(inputs)
    print(f"setting={setting_name}")
    print(f"  http_status={status}")
    print(f"  embedding_dim={len(vectors[0])}")
    print(f"  batch_request_s={request_s:.6f}")
    print(f"  mean_ms_per_query={request_s * 1000 / len(queries):.3f}")

    for group_name, indices in group_indices.items():
        scores = [
            cosine(vectors[left], vectors[right])
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        ]
        print(
            f"  group={group_name} pairs={len(scores)} "
            f"min={min(scores):.6f} mean={sum(scores) / len(scores):.6f} "
            f"max={max(scores):.6f}"
        )
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                print(
                    f"    cos={cosine(vectors[left], vectors[right]):.6f} "
                    f"left={queries[left]!r} right={queries[right]!r}"
                )
    print()
PY
