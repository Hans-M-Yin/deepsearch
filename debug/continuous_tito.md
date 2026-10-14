# Opt-in cumulative TITO (Qwen + vLLM)

Enable `RLLM_MM_TITO=1` with `RLLM_MM_SEQUENCE_MODE=cumulative`.
The engine default is 0: all existing cumulative rendering remains available.
The full 4node rloo-baseline launcher now defaults RLLM_MM_TITO to 1.
The `RLLM_MM_` prefix is forwarded by `rllm/trainer/verl/ray_runtime_env.py`.

Minimal experiment:

```bash
bash RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-4node-vllm-rloo-baseline-minimal-tito.sh > /tmp/rloo_minimal_tito.log 2>&1
```

This inherits the smoke dataset (96 train / 32 test), mini batch 96 and TIS
from the existing minimal launcher. It uses a separate experiment/checkpoint
directory. Its inherited resume mode is auto; choose another EXPERIMENT_NAME
for a fresh rerun. It is NOT a one-step job: stop after the desired validation.
Disable with `RLLM_MM_TITO=0` or use the old launcher without setting the flag.

## Upstream comparison and version boundary

The design follows upstream verl's `ContinuousTokenBuilder`,
`QwenVLContinuousTokenBuilder` and AgentLoop, inspected 2026-09-21:

- https://github.com/verl-project/verl/blob/main/verl/utils/tokenizer/continuous_token.py
- https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py

This is a scoped adaptation, NOT a wholesale backport of current upstream.
Keep our SFT-compatible Qwen parser/XML tools. Encode only initial/new context;
retain sampled assistant IDs verbatim, with synthetic EOS/newline and all
observation tokens masked out. Do not render historical assistant messages.

Our vLLM 0.11 adapter accepts unexpanded placeholders and image objects. Keep
both raw request IDs and expanded training IDs. New images are processed once
locally; retain images for later inference and concatenate pixel/grid tensors
for training. The processor's new-message IDs must equal raw IDs with *only*
image placeholders expanded. No full-history processor output can overwrite
sampled IDs. Unlike upstream's final feature reconstruction, we reuse locally
cached features. Training position IDs still use the existing Qwen pipeline.

Every generation compares `RequestOutput.prompt_token_ids` returned by vLLM
against the expanded local prompt. If this installed vLLM reports unexpanded
IDs, or expands differently, the path errors rather than bypassing validation.
This proves IDs, not numerical equality of image embeddings / MRoPE / logits.
Real GPU validation of those remains necessary.

## Safety / unsupported cases

- vLLM + cumulative only; QwenChatTemplateParser only.
- Image-bearing observations must use our existing user-message `images` field.
- Only processor fields pixel_values and image_grid_thw are currently supported.
  Video/audio, generated image placeholders and unknown processor fields fail.
- Preparation is transactional; failed requests do not mutate the ledger.
- Changed history, missing image payloads, unexpected expansion fail explicitly.
- No separate 4096-token initial-prompt rejection in TITO. Enforce the model
  context and cumulative response budgets before generation. Training pads to
  actual prompt/response batch maxima without truncating image-bearing prompts.
  The old path keeps its configured fixed widths. Larger real prompts can still
  increase memory usage; this does not increase the model context limit.
- Existing workflow handling reports failed requests as error trajectories.
  Check the error/fatal rate: a run continuing alone does NOT prove correctness.
- Text messages may be stripped for tool parsing; they do not replace sampled
  tokens. Whole generated completions, including malformed tool text, are kept.

## Verification

CPU tests (no external requests / GPU required):

```bash
python -m unittest discover -s RL/rllm/tests -p 'test_continuous_tokens.py' -v
python -m unittest discover -s RL/rllm/tests -p 'test_rollout_consistency.py' -v
python -m unittest discover -s RL/rllm/tests -p 'test_token_tis.py' -v
```

Production cleanup: per-turn/batch/actor success logs and duplicate TOKEN_TIS
JSON dumps have been removed. Config dumps, extra forward probes and the
intentional debug stop before updates are no longer wired into training.
Normal TIS/consistency metrics still go through the training logger/W&B.
Check no TITO errors, no unexplained filtered/error groups, full active-token
TIS coverage, processed logprob differences, and one completed optimizer update.
Exercise initial image, new images on multiple turns, text-only tool results,
length-limited outputs, fatal masking, and partial-credit turn spans separately.
Do not equate CPU tests or TIS coverage with GPU end-to-end validation.

## Training-side sequence audit

TITO now always records each successful rollout prompt's SHA256 and length
(independent of CONSISTENCY_DEBUG). Since that prompt already matched the
vLLM-reported processed IDs, training can compare against the same sequence.
After padding/assembly, each active turn's entire unpadded prefix and exact
generated IDs are checked. Missing evidence or mismatch raises
`TITO_TRAIN_ALIGNMENT_ERROR`; success is silent. Fatal masked suffixes are
NOT counted as verified actions.

Full unpadded row fingerprints travel with DataProto through row reordering
and dispatch. Megatron actor RPC entry checks them before old-logprob compute
and before update, silently on success. Mismatch raises `TITO_ACTOR_ALIGNMENT_ERROR`.
These are CPU hashing checks, not extra forwards. They audit the input batch
at the actor boundary, not the subsequent packing/kernel internals, numerical
visual features, or equality of rollout/training MRoPE.

```bash
python -m unittest discover -s RL/rllm/tests -p test_tito_training_audit.py -v
```
