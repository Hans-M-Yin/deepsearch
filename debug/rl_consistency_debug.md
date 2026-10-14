# Forward-only consistency diagnosis

Run the existing RLOO baseline with `RLLM_MM_CONSISTENCY_DEBUG=1`.
The variable is forwarded via the existing `RLLM_MM_` Ray env prefix.
No experiment defaults, sampling temperature, rewards or kernels are fixed by
this diagnostic. Raw mode no longer triggers a startup failure.
The actor_raw/vs_rollout comparison uses a separate T=1 forward. Other
cross-engine probe comparisons still include logprob-definition differences.

This runs one rollout batch and three additional trainer forwards (repeat,
actor-update batching budget, and unfused if currently fused). It logs a
CONSISTENCY_DEBUG_REPORT and intentionally raises CONSISTENCY_DEBUG_COMPLETE
before optimizer update. Unfused may need substantially more GPU memory; use
a small diagnostic workload if it OOMs. Do not use this as a normal training run.

- repeat/vs_old: repeatability, stochastic forward or numerical instability.
- actor_batching/vs_old: batching/packing sensitivity, not an actual train-mode update comparison.
- unfused/vs_old: fused-path discrepancy; inspect fused_vocab_boundary for the
  local vocabulary tail issue discussed in verl PR 5349. No automatic fix.
- each variant/vs_rollout: cross-engine sampled-token differences, NOT KL.
- prompt_prefix_equal/different/unverified_tokens: expanded adapter prompt vs
  reconstructed training prefix, excluding padding, preserving positional turn mapping.
- mask_on_padding/response_input_mismatch/actor_nonfinite_tokens: input invariants.
- weight_sync_start/done: RPC lifecycle only, NOT proof of tensor equality.

Remaining unknowns: vLLM-internal image preprocessing, pixel values, mRoPE,
weight conversion correctness, backend/precision effects and cached states.
Config prints narrow these down but do not validate them. Never interpret
sampled-token coverage=1 as proof of those invariants.

Normal-mode code remains unchanged except retaining an all-unverified diagnostic
prefix tensor and computing an extra T=1 monitoring forward for raw vLLM
logprobs when the training temperature is not 1. No probe logprobs are used
for importance sampling or loss. Corrected W&B metrics are under
`training/raw_consistency/`; legacy comparisons are under
`training/cross_definition/`. Established `training/rollout_logprob_*`
names now report raw-to-raw values; do not compare their historical values
across this code change without accounting for the definition change.
