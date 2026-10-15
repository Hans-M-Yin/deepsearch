# Evaluate t2i search alignment

`eval_t2i_alignment.py` sits beside `run_infer.py`. It judges whether each
executed `t2i_search.query` seeks an image represented by a node in the saved
construction chain. It evaluates **query intent against the reference path**,
not whether searching was necessary, timely, successful, or led to a correct
answer. Alternative reasoning paths are outside this metric.

## Run

From the project root, inspect the complete population before making requests:

```bash
python opensearch_vl/eval_t2i_alignment.py --traj-dir opensearch_infer/outputs/Qwen3-VL-8B-SFT13_5-RL5-step110/searchweave_1005 --dry-run
```

Evaluate with a registered model-worker alias:

```bash
python opensearch_vl/eval_t2i_alignment.py --traj-dir opensearch_infer/outputs/Qwen3-VL-8B-SFT13_5-RL5-step110/searchweave_1005 --judge-model-alias gpt54_internal_azure --max-workers 8
```

To additionally merge repeated queries that seek the same scene/event within a
case, add `--deduplicate-intents`:

```bash
python opensearch_vl/eval_t2i_alignment.py --traj-dir opensearch_infer/outputs/Qwen3-VL-8B-SFT13_5-RL5-step110/searchweave_1005 --judge-model-alias gpt54_2_internal_azure --max-workers 8 --deduplicate-intents
```

The **default is the full directory**. For a small pilot add `--limit-cases 5`.
Run the same command again to resume. Add `--rerun` to ignore cached judgments;
`--cache-only` recomputes reports without API requests. `--dry-run` never imports
the judge backend and writes separate `dry_run_*` files, leaving existing
classifications/reports intact. Dry-run alignment values are bounds for
unjudged queries, not evaluation scores.

Useful options:

| Option | Meaning |
|---|---|
| `--output-dir DIR` | Default is `TRAJ_DIR/t2i_alignment_eval` |
| `--judge-model-alias ALIAS` | Default `gpt54_internal_azure`, or `JUDGE_MODEL_ALIAS` env var |
| `--max-workers 8` | Maximum concurrent unique query judgments |
| `--max-attempts 3` | Total judge generations per unique query; includes network and invalid-JSON retries |
| `--retry-delay 2` | Seconds between attempts |
| `--deduplicate-intents` | Adds a separate LLM grouping pass; reports per-call and deduplicated intent metrics side by side |
| `--category image-end` | Repeat to select multiple category labels |
| `--completed-only` | Optional; defaults to including failed inference trajectories |
| `--data-path FILE.parquet` | Fallback question/chain metadata for traces that do not contain a saved chain |
| `--json` | Print the entire JSON report rather than the console readout |

The judge uses `synthesis.model_worker.LLM_WORKER`. Alias configuration controls
request timeout, completion budget and any sampling overrides. The evaluator
follows the registered alias's sampling settings (and omits temperature when
the alias does not set one), disables worker-level retries, and bounds its own
generation attempts with `--max-attempts`. It never prints API credentials.

## Population and judge protocol

- Scans root `*_trajectory.json`, root `*_failure.json`, and `failures/*.json`.
  Selects one trajectory per `case_id`: completed takes precedence over failed,
  then a trajectory file over a failure file, then newest modification time.
- Includes executed calls in failed/max-turn trajectories. Stale failure files
  superseded by successful retries are not separate samples.
- Prefers `original_data.source_metadata.search_chain`. Missing references and
  questions, malformed chains, explicitly ineligible placeholders and unreadable
  files are reported; they are not silently scored as off-chain.
- Optional parquet fallback only applies when the saved chain is absent. If
  both questions exist they must agree after whitespace normalization. Existing
  saved chains are not replaced by a newer benchmark.
- Counts nonempty string queries with execution evidence. Mentions in thinking,
  unexecuted proposals, and forced-finalization prompts do not count. Executed
  attempts with invalid/missing queries are counted separately.
- Native Responses `tool_results` are authoritative when present, preventing
  double counting of the same call in legacy compatibility fields. Parallel
  native calls are evaluated individually.
- Gives the judge the question, compact ordered nodes/relations with explicit
  `IMAGE` markers, and **one current query**. No agent identity, final answer,
  previous/future reasoning or search result is supplied.
- Requires `aligned`, `off_chain`, or `uncertain`, matched IMAGE node IDs and a
  brief reason. Alias/description matches are allowed. Broad ambiguous queries
  are uncertain; explicit wrong entities/versions are off-chain.
- Validates node IDs and label consistency. Malformed judge responses retry;
  exhausted requests become `judge_error`, never off-chain or uncertain.
- With `--deduplicate-intents`, a second prompt groups queries **within each
  case** by the image scene/event stated in the queries. The grouper sees all
  executed t2i queries, but not the question, construction chain or call
  verdicts, so it cannot merge merely because several queries match the same
  reference node or answer the same question.
  It ignores wording/detail refinements for the same scene, keeps different
  people/events/places separate, and leaves ambiguous pairs in separate groups.
- Intent-group labels are derived from the existing per-call verdicts. If all
  members agree on label and matched node IDs, the group inherits that result;
  conflicting member labels or node IDs become `mixed` and remain unresolved.
  Raw calls and their original labels are always retained.

## Metrics

Every reported rate contains its numerator, denominator and fractional value
in JSON; a zero denominator gives `null` / `N/A`. Overall, category and inference
status summaries use the same definitions.

| Metric | Definition |
|---|---|
| `call_alignment_rate_confirmed` | Aligned calls / all valid executed t2i queries; includes matches to initial input images |
| `call_alignment_rate_among_decided` | Aligned / (aligned + off-chain), reported separately to expose uncertainty effects |
| `new_image_call_alignment_rate_confirmed` | Calls matching a non-initial image / all t2i queries |
| `macro_call_alignment_rate_among_searching_cases` | Average per-case aligned-call fraction over cases with t2i calls |
| `search_trigger_rate_on_new_image_cases` | Cases with any t2i / cases containing non-initial image nodes |
| `goal_aligned_search_rate_confirmed` | Cases with a matched non-initial image / cases containing non-initial images |
| `node_coverage_micro_confirmed` | Distinct matched non-initial nodes summed over cases / all non-initial image nodes |
| `node_coverage_macro_confirmed` | Average per-case non-initial node coverage, including zero-call cases as zero |
| `full_node_coverage_case_rate_confirmed` | Cases covering every non-initial image / cases with non-initial images |
| `extra_search_rate_on_input_only_cases` | Cases with t2i / cases whose sole image is initial; descriptive, not an error rate |
| `uncertain_rate`, `judge_error_rate`, `pending_rate` | Each label count / all t2i queries |
| `judge_completion_rate` | Valid aligned/off-chain/uncertain judgments / all t2i queries |

Also reports off-chain rate, upper bounds, initial-image match counts, distinct
queries, invalid t2i attempts and initial/intermediate/final node coverage. Repeated
queries count as repeated calls but cover each reference node only once.

With `--deduplicate-intents`, the evaluator makes one additional grouping
request per case containing t2i calls that is not already in the grouping
cache. `intent_overall` reports unique scene/event groups
separately from raw-call metrics. It includes aligned/off-chain/uncertain/mixed
group counts, unique-intent alignment, non-initial image alignment, node
coverage, calls per intent group, and the redundant-call rate
`(grouped calls - intent groups) / grouped calls`. If grouping fails for a case,
its calls remain visible but are not silently treated as separate confirmed
intents; `intent_grouping_completion_rate` and grouping error/pending counts
show the incomplete coverage. Grouping decisions have their own resumable cache.

Coverage is over **literal reference node IDs**: duplicate depictions in separate
construction nodes are not automatically merged by semantic identity. The first
path node, if IMAGE, is treated as initial input by construction even when saved
image payloads have been stripped. Other IMAGE nodes are coverage targets.

Uncertain/error/pending calls stay in the main denominator. Confirmed rates and
coverage are therefore lower bounds when such calls exist. Upper bounds assume
an unresolved call could match and, for node coverage, could cover all missing
nodes in its case. These are conservative possibility bounds, **not statistical
confidence intervals**. An operational error should be resumed before model
comparisons.

## Outputs and cache

Under the output directory:

- `report.json`: all aggregate metrics and denominator definitions.
- `cases.jsonl`: selected cases, original compact chains, exclusions, calls and
  per-case metrics. Includes cases with zero calls.
- `calls.jsonl`, `calls.csv`: per-call labels, node IDs and explanations.
- `intents.jsonl`: when enabled, merged within-case groups with member queries,
  summaries and group-level verdict consistency.
- `judge_cache.jsonl`: append-only judgments, raw output, usage and errors.
- `intent_group_cache.jsonl`: append-only semantic grouping responses.
- `run_config.json`: exact judge prompt, safe model identity, signature and an
  example judge input for review.

Cache keys include the full question/chain/query, system prompt, evaluator
version and judge identity/configuration signature. Same-payload calls share
one judge request without removing calls from metric denominators. Changes to
queries, chains, prompt, alias, served model or sampling configuration invalidate
their cache entries. Failed judge attempts are retried on resume. Keep one
output directory per judge configuration when using `--cache-only`, which reads
the last run's identity without initializing a model client.

The evaluator takes an output-directory lock to prevent concurrent writers.
Completed judgments are flushed individually, so interrupted evaluations can
resume. It does not modify source traces or their answer-evaluation results.

Exit status: 0 for a complete evaluation or successful dry-run, 1 for pending
or failed judge calls / unreadable source files, 2 for configuration errors or
no eligible cases. A completed judge returning `uncertain` is not an operational
failure. Never interpret a partial report as a complete model evaluation.

## Offline verification

```bash
python -m unittest opensearch_vl.tests.test_eval_t2i_alignment
```

The tests mock the judge and verify failed-case inclusion, Responses call
deduplication, zero-call denominators, bounds, malformed judge output, retries,
cache invalidation, resume and no-API dry-run behavior.
