# FVQA hard-case trajectory pipeline

This is a new pipeline, separate from synthesis/post_process/fix. It adds an
FVQA example only when:

1. the current target SFT model was independently judged wrong;
2. an expert using the production SFT tools solves it without seeing an answer
   or intermediate statements;
3. a separate answer judge marks the expert final answer correct;
4. a label-blind trajectory-quality judge finds the evidence sufficient and
   finds no internal-knowledge jump or material logical gap;
5. visual-comparison strength is measured, and strong visual traces also pass
   a blind image-only review; and
6. the raw trace passes label-safety, completion, tool-error, and leakage
   gates.

Every expensive stage is append-only. Every decision gets an audit row with
the exact rejection reasons and parsed tool provenance.

## Safety boundary

CanonicalCase keeps gold_answer only for base/expert evaluation. The writer
only receives a case ID, question, image path, split, and dataset name. The
collector preserves the existing synthesis.sft.api_tools system prompt and
tool runtime, but appends a binding FVQA override: no answer, hop chain, or
private fact exists in the writer session.

It never calls debug_vqa_batch._build_user_prompt_text, because that legacy
helper intentionally adds Answer: and private hop statements for synthetic
VQA generation.

The prepare command refuses source rows that look like fvqa_test when asked to
produce a train split. A filename alone is not evidence of a train split. In
particular, do not use data/benchmarks/fvqa/train.parquet as a training source
if its IDs/categories are test-labelled; use a confirmed FVQA source/train
split instead.

## Pilot workflow

Use fresh output paths. Let SOURCE_TRAIN be a verified source/train Parquet,
and first run the existing base-model inference and evaluator:

~~~bash
python opensearch_vl/run_infer.py \
  --data-path SOURCE_TRAIN --output-dir runs/fvqa_base_8b \
  --model qwen3-vl-8b --dataset train
python opensearch_vl/eval_infer_with_llm.py \
  --traj-dir runs/fvqa_base_8b --answer-file SOURCE_TRAIN \
  --output-path runs/fvqa_base_8b/llm_eval_report.json
~~~

Prepare cases and mine only judged-wrong base answers. Evaluation errors are
never treated as hard cases.

~~~bash
python -m synthesis.post_process.fix2.run prepare \
  --input-parquet SOURCE_TRAIN \
  --output-cases runs/fvqa_fix2/cases.jsonl \
  --image-dir runs/fvqa_fix2/images --split train

python -m synthesis.post_process.fix2.run mine-failures \
  --cases runs/fvqa_fix2/cases.jsonl \
  --base-eval-details runs/fvqa_base_8b/llm_eval_report_details.jsonl \
  --output-cases runs/fvqa_fix2/base_failures.jsonl \
  --output-manifest runs/fvqa_fix2/base_failure_manifest.jsonl --limit 100
~~~

Profile the legacy dataset, then make a human-sanitized copy of
style_examples.template.json. Do not put raw old trajectories, answers, source
titles, URLs, snippets, resource IDs, or image facts in that anchor file.

~~~bash
python -m synthesis.post_process.fix2.run profile-style \
  --input-sharegpt data/sharegpt_dataset_refined_gpt54_only_17k_final/trajectories_sharegpt.json \
  --output-profile runs/fvqa_fix2/legacy_style_profile.json --limit 500

python -m synthesis.post_process.fix2.run collect \
  --cases runs/fvqa_fix2/base_failures.jsonl \
  --output-jsonl runs/fvqa_fix2/expert_raw.jsonl \
  --model EXPERT_MODEL_ALIAS --workdir runs/fvqa_fix2/runtime \
  --style-examples-file runs/fvqa_fix2/style_examples.json \
  --max-turns 8 --resume
~~~

The answer judge sees the label only after the writer completes. The trajectory
quality judge does not see the label: it receives the original image, public
question, visible trace, and final answer, and rejects unsupported evidence,
internal-knowledge jumps, and logical gaps. Its default alias is
`gemini38flash_internal_azure`. The blind vision reviewer sees only original
pixels, opened candidate image pixels, and the public question: no title,
snippet, URL, trace, candidate name, or label.

~~~bash
python -m synthesis.post_process.fix2.run judge-expert \
  --input-jsonl runs/fvqa_fix2/expert_raw.jsonl \
  --private-cases runs/fvqa_fix2/cases.jsonl \
  --output-jsonl runs/fvqa_fix2/expert_answer_reviews.jsonl \
  --model-alias ANSWER_JUDGE_ALIAS --resume
python -m synthesis.post_process.fix2.run judge-trajectory-quality \
  --input-jsonl runs/fvqa_fix2/expert_raw.jsonl \
  --output-jsonl runs/fvqa_fix2/trajectory_quality_reviews.jsonl \
  --model-alias gemini38flash_internal_azure --resume
python -m synthesis.post_process.fix2.run blind-visual-review \
  --input-jsonl runs/fvqa_fix2/expert_raw.jsonl \
  --output-jsonl runs/fvqa_fix2/blind_visual_reviews.jsonl \
  --model-alias VISION_REVIEWER_ALIAS --resume
python -m synthesis.post_process.fix2.run filter \
  --input-jsonl runs/fvqa_fix2/expert_raw.jsonl \
  --private-cases runs/fvqa_fix2/cases.jsonl \
  --answer-reviews runs/fvqa_fix2/expert_answer_reviews.jsonl \
  --trajectory-quality-reviews runs/fvqa_fix2/trajectory_quality_reviews.jsonl \
  --blind-reviews runs/fvqa_fix2/blind_visual_reviews.jsonl \
  --accepted-output runs/fvqa_fix2/accepted_raw.jsonl \
  --rejected-output runs/fvqa_fix2/rejected_raw.jsonl \
  --audit-output runs/fvqa_fix2/filter_audit.jsonl
~~~

The accepted file is compatible with
scripts/convert_debug_vqa_jsonl_to_sharegpt.py. The filter injects the
independently correct answer_judge, so convert it normally, without
include-incorrect. Run the existing
filter_trajectory_information_leakage.py on exported ShareGPT before a
separate merge. Keep the FVQA export in a new directory until the pilot and
human sample review are approved.

## LiveVQA direct-teacher workflow

LiveVQA is intentionally separate from FVQA hard-case mining: it never runs
the target/base model. The source must be the **train** split of
`ONE-Lab/LiveVQA-new`; never use its benchmark test split for SFT. Build the
normalized source once when it is not already present:

~~~bash
python scripts/build_benchmark_dataset.py \
  --benchmark livevqa --split train \
  --cache-dir /tmp/$USER/livevqa_hf_cache \
  --output data/benchmarks/livevqa/train.parquet
~~~

The upstream LiveVQA repository stores its QA data in top-level Xet/LFS
`Dataset_part_*.parquet` files. Its dataset viewer currently cannot read those
files through the generic `datasets` path, so the builder downloads and reads
the shards directly through `huggingface_hub`; use a local `/tmp` cache rather
than HDFS. This metadata download excludes the source image archives.

The direct sampler uses only a fixed PRNG seed, the raw `level` field, and
downloaded image SHA256. It takes `N` random Level 1 rows and `3N` random
Level 2 rows; no source/topic/difficulty/model filter is applied. The Level 1
selection is materialized first, then Level 2 is sampled against the same
seen SHA256 set, so one exact image can appear in at most one teacher task.
Multiple LiveVQA references remain private as `acceptable_answers`; the
independent answer judge accepts any listed reference, while the teacher sees
only the question and local image.

~~~bash
# Default: 512 candidate cases = 128 L1 + 384 L2, teacher max_turns=30.
bash synthesis/post_process/fix2/launch_livevqa_pipeline.sh

# Example expansion; SAMPLE_COUNT must be divisible by four.
SAMPLE_COUNT=2048 SAMPLE_SEED=20260919 \
TEACHER_WORKERS=16 DOWNLOAD_WORKERS=16 \
bash synthesis/post_process/fix2/launch_livevqa_pipeline.sh
~~~

The launcher immediately enqueues the public manifest into a local SQLite
teacher queue; it starts no `run_infer`, base evaluator, or failure router.
LiveVQA source pixels are packaged upstream in `news.zip`,
`youtube_images.zip`, and `paper_images.zip`, rather than individual Hub image
URLs. The launcher stages only archives needed by selected samples under
`LOCAL_STATE_DIR/livevqa_archives` (local `/tmp`) and extracts only their
selected members. A random cross-source pilot may need all three archives
(about 5.2 GB total); retain this local cache for resume.
After the queue has no pending/leased tasks, use the same independent answer
judge, label-blind trajectory-quality judge, blind visual review, deterministic
filter, and ShareGPT leakage audit as FVQA:

~~~bash
QUEUE_NAME=teacher_collect_livevqa_responses \
bash synthesis/post_process/fix2/finalize_livevqa_pipeline.sh \
  RUN_DIR LOCAL_STATE_DIR
~~~

## Evaluation-only expert probe

To test whether expert success remains high on held-out FVQA failures, use
`launch_fvqa_test_expert_probe.sh`. It reads an already-completed base test
evaluation, reproducibly samples only independently judged base failures, and
materializes only the selected test images. It runs the same hidden-answer
expert, answer judge, trajectory-quality judge, and blind visual review. Its
output is an evaluation artifact, never an SFT source or ShareGPT export.

~~~bash
bash synthesis/post_process/fix2/launch_fvqa_test_expert_probe.sh
~~~

The default is 20 cases with seed `20260918`. Override `PROBE_SIZE=10` for a
cheaper smoke run, or set `SAMPLE_SEED` to obtain an independent replicate.
The selection report records the sampled test IDs and seed, while the public
writer manifest contains neither test labels nor base judge output.

After an asynchronous run reaches its quota, the same sequence is available as
one short command. It derives the coordinator-local SQLite path from the run
name (or accepts it as an optional second argument):

~~~bash
bash synthesis/post_process/fix2/finalize_fvqa_pipeline.sh \
  runs/fix2_fvqa_20260918_014403
~~~

It defaults to the mixed export: `strong` visual trajectories must pass blind
image review, while `weak`/`none` trajectories may pass only through the
label-blind trajectory-quality review. The final filter permits up to 30
assistant turns, matching the collector ceiling; independently wrong answers
and label-blind quality-review failures remain hard rejections. Set
`REQUIRE_VISUAL_COMPARE=1` before the command to export only the strong visual
branch. After a successful training-run finalization, it automatically writes
the final, leakage-filtered ShareGPT JSON to
`RUN_DIR/sharegpt_export/trajectories_sharegpt.json`, with materialized
images and `dataset_info.json`. The converted pre-leakage JSON and the leakage
audit are retained alongside it for auditability. The information-leakage
judge defaults to `gemini38flash_internal_azure`; override it only with
`LEAKAGE_JUDGE_MODEL`. Evaluation-only test probes set `EXPORT_SHAREGPT=0`
and can never produce a ShareGPT training export. The merged/review/filter
JSONL state files are deliberately kept in
`LOCAL_STATE_DIR/finalization` (default `/tmp/$USER/<run-name>/finalization`),
because they are append/resume state; the immutable per-case teacher records
remain in the shared `RUN_DIR`.

Inspect a completed finalization with one command; it prints answer-correctness
counts, quality verdicts, visual-strength distribution, final keep/filter
counts, rejection reasons, and a compact per-case audit:

~~~bash
bash synthesis/post_process/fix2/inspect_fvqa_finalization.sh \
  runs/fix2_fvqa_20260918_014403
~~~

## Asynchronous base-to-teacher queue

For the standard four-process launch, use
`launch_fvqa_pipeline.sh`. It prepares the manifests, profiles legacy surface
style, then backgrounds the base inference, base judge, difficulty router, and
teacher group with one log/PID file per process. It requires an explicit
coordinator-local `LOCAL_STATE_DIR` for SQLite and tool scratch; only results
and source manifests belong under the shared run directory. See `--help` for
the required endpoint and reviewed-style-anchor environment variables.

`run_eval` resumes by append-writing its detail JSONL. Therefore the launch
script also puts `llm_eval_report.json` and
`llm_eval_report_details.jsonl` in `LOCAL_STATE_DIR/base_eval`, rather than
under a HDFS `RUN_DIR`; the router reads that same local detail file.

To stop the background producer group, run
`bash synthesis/post_process/fix2/stop_fvqa_pipeline.sh`. It targets only the
most recently started fix2 run's recorded PIDs and verifies their command
line before sending `TERM`.

For throughput, run four independent processes on the coordinator host. The
base inference process is the existing run_infer worker group. The other three
processes can start before base inference has completed:

~~~text
run_infer worker group
    -> watch-base-eval worker group
    -> enqueue-failures difficulty router
    -> teacher-workers group
~~~

The router only enqueues an independently judged wrong base answer. Queue
tasks contain a PublicCase payload only: case ID, question, local image path,
split, and source dataset. They cannot contain answer, hop-chain, or judge
text. Each teacher worker leases one case, runs its multi-turn trajectory in
its own workdir, atomically writes one result file, acknowledges the lease,
then immediately leases another case.

~~~bash
# Terminal A: existing base workers (can use run_infer --parallel-workers).
python opensearch_vl/run_infer.py ... --output-dir runs/fvqa_base_8b

# Terminal B: independently evaluate newly arriving base trajectories.
python -m synthesis.post_process.fix2.run watch-base-eval \
  --trajectory-dir runs/fvqa_base_8b --answer-file SOURCE_TRAIN \
  --output-path runs/fvqa_base_8b/llm_eval_report.json \
  --judge-model-alias ANSWER_JUDGE_ALIAS --max-workers 8 --poll-s 10

# Terminal C: follow the append-only evaluator details file.  Route base
# failures, plus base-correct trajectories which took at least six tool calls.
# QUEUE_DB must be on coordinator-local disk, not HDFS/NFS.
python -m synthesis.post_process.fix2.run enqueue-failures \
  --cases runs/fvqa_fix2/cases.jsonl \
  --base-eval-details runs/fvqa_base_8b/llm_eval_report_details.jsonl \
  --base-trajectory-dir runs/fvqa_base_8b/trajectories \
  --min-correct-tool-calls 6 \
  --queue-db /path/on/coordinator-local-ssd/fix2/teacher_queue.sqlite \
  --max-ready-tasks 32 --follow --poll-s 10

# Terminal D: teacher workers. idle-exit-s=0 keeps them waiting for the
# producer indefinitely; workers each own one sample at a time.
python -m synthesis.post_process.fix2.run teacher-workers \
  --queue-db /path/on/coordinator-local-ssd/fix2/teacher_queue.sqlite \
  --results-dir runs/fvqa_fix2/teacher_results \
  --model EXPERT_MODEL_ALIAS --workdir runs/fvqa_fix2/runtime \
  --workers 8 --idle-exit-s 0 --lease-s 3600 \
  --style-examples-file runs/fvqa_fix2/style_examples.json

# Stop producers once the desired hard-case quota is reached, then merge the
# completed per-case records in deterministic case_id order.
python -m synthesis.post_process.fix2.run merge-teacher-results \
  --queue-db /path/on/coordinator-local-ssd/fix2/teacher_queue.sqlite \
  --output-jsonl runs/fvqa_fix2/expert_raw.jsonl
~~~

The initial durable backend is SQLite and is safe for many worker
threads/processes on one coordinator host. It must not be placed on the HDFS
mount used by this repository, nor on NFS. For cross-machine teacher workers,
keep the same lease/ack contract and use a central queue backend such as Redis
Streams consumer groups; do not share a SQLite file over a network filesystem.
Set max-ready-tasks to roughly 2–5 times the teacher worker count to apply
backpressure and avoid paying for an unbounded number of base failures.

### Correct-but-expensive base trajectories

The standard training launcher sets `MIN_CORRECT_TOOL_CALLS=6`.  A case enters
the teacher queue when the base answer is independently wrong **or** when it
is independently correct but its immutable `run_infer` trajectory reports at
least six tool calls.  The router incrementally indexes only newly written
trajectory files, so it remains safe in asynchronous follow mode.

The route reason, base answer, evaluator output, and tool-call count never
enter the `PublicCase` sent to the teacher.  They are mining signals only.
Set `MIN_CORRECT_TOOL_CALLS=0` to return to wrong-answer-only routing.

`HARD_CAP` applies to the combined queue.  Raise it (or set it to the desired
production quota) when using the additional cost bucket; retaining a smoke
test cap of 10 does not create more than 10 teacher tasks.

### Staged scale-up and resume

For a staged production collection, use the same `RUN_TAG` and local state
directory for every expansion. `launch_fvqa_scale.sh` fixes `BASE_LIMIT=0`
(the verified full FVQA train source) and `MIN_CORRECT_TOOL_CALLS=6`. On its
first run it extracts a bounded set of non-final assistant excerpts from the
previous ShareGPT SFT corpus into `style_examples.production.legacy.json`; the
excerpts preserve real writing/tool-use style but never include a legacy final
answer or tool observation. Start with 512 total queue tasks:

~~~bash
bash synthesis/post_process/fix2/launch_fvqa_scale.sh 512
~~~

After the queue reports 512 succeeded tasks and no pending/leased tasks, stop
only that run, then reuse its tag while raising the cumulative queue cap:

~~~bash
bash synthesis/post_process/fix2/stop_fvqa_pipeline.sh runs/fix2_fvqa_train_scale_v1
bash synthesis/post_process/fix2/launch_fvqa_scale.sh 2048
~~~

`run_infer` skips completed case IDs, the base evaluator resumes its local
detail JSONL, the router retains durable task IDs, and teacher workers retain
their atomic per-case results. Thus the second command adds only new work. Do
not run finalization between these two stages: its accepted/rejected exports
are intentionally immutable. Finalize and produce ShareGPT only after the
last desired expansion. `HARD_CAP` is an upper bound, not a guarantee that
the source contains that many eligible cases.

For a bounded source smoke test, override the number of FVQA rows while
retaining the same teacher-cap interface. For example, this evaluates exactly
the first 128 FVQA train rows and can enqueue at most 128 teacher tasks:

~~~bash
BASE_LIMIT=128 bash synthesis/post_process/fix2/launch_fvqa_scale.sh 128
~~~

## Quality and visual gates

Every accepted trace must pass the label-blind trajectory-quality review. It
may have `strong`, `weak`, or `none` visual verification; weak/non-comparison
cases remain eligible only when their visible image anchor and external
evidence chain are sufficient. A trace labelled `strong` still must pass the
causal visual and blind-image gates below. This preserves FVQA diversity
without lowering the bar for genuine visual comparisons or teaching unsupported
recognitions.

Use `filter --require-visual-compare` only to export the high-confidence
visual-comparison subset. That branch additionally requires this causal chain:

~~~text
i2i_search or t2i_search returns image IDs
    -> read_url opens at least one returned image ID
    -> their image attachments or runtime resources are recoverable
    -> assistant records a grounded visual verification
    -> blind VLM sees only original and candidate pixels
    -> independent answer judge passes
~~~

Search titles/snippets do not count as visual evidence. SAME_ENTITY and
CONSISTENT are the only blind-review keep verdicts. SIMILAR, DIFFERENT_PHOTO,
and UNRESOLVED filter the trace. Add require-external-fact-evidence for the
visual-to-text branch; it requires a successful non-image read_url observation.
When several candidates are genuinely plausible, the collector prompt requires
opening and comparing them; it never requires inventing a wrong candidate just
to satisfy a fixed count. Use --require-candidate-rejection only for a
separately labelled negative-candidate branch.

The audit row retains `trajectory_quality_visual_verification_level`, so the
final training mix can be stratified or weighted by visual strength without
discarding grounded nonvisual cases.

## Style preservation

Use the same runtime and tool schema first. Then append a compact style guide
and 2–4 curated/sanitized anchors; start at about 2k–3.5k tokens. Include a
visual comparison, visual-to-text handoff, and grounded recovery example.

For a 50–100 sample pilot, compare legacy and new surface profiles: role
alternation, assistant length, action/tool distribution, visual-search rate,
tool-error rate, and final-answer-tag rate. Few-shots guide prose and
decision style; provenance filters enforce evidence quality.
