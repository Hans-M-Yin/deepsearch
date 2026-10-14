# RLOO baseline token TIS

`qwen3-vl-8b-4node-vllm-rloo-baseline.sh` defaults to `RLLM_MM_TIS=1`,
`RLLM_MM_TIS_THRESHOLD=2.0`. Other launchers remain off unless explicitly enabled.
Use `RLLM_MM_TIS=0` to return to the previous raw-monitoring baseline.
Keep `RLLM_MM_CONSISTENCY_DEBUG=0` for training (the separate diagnostic stops before update).

The shared launcher sets vLLM `logprobs_mode=processed_logprobs` only for TIS.
Runtime vLLM config is checked on each server. The trainer requires stochastic
sampling, top_p=1, top_k=-1, repetition_penalty=1, and vanilla PPO (no bypass).
These checks describe this initial integration, not all algorithms TIS can support.

After recomputing old logprobs at the training temperature, weights are computed
with verl's `compute_rollout_correction_weights`: token-level
`detach(min(exp(old - behavior), threshold))`, without batch normalization.
Weights enter the existing actor PPO loss. RLOO rewards, advantages, response
mask and current/old PPO clipping remain unchanged. No extra T=1 forward is
needed in processed mode. Sampling implementation may change with vLLM's
processed-logprob mode; do not assume bitwise reproducibility or identical speed.

All active loss tokens must have exactly aligned finite rollout logprobs.
Missing/truncated/re-tokenized records raise TOKEN_TIS_ALIGNMENT_ERROR;
we do not silently substitute a zero logprob, drop tokens or assign weight 1.
Use this error to fix alignment before training. Observations/padding are masked.

Monitor TOKEN_TIS log lines, `tis/coverage`, `tis/weight_mean`,
`tis/clipped_fraction`, verl weight statistics under `tis/`, and
`training/processed_consistency/rollout_logprob_abs_diff_mean` / `...p95`.
Existing unprefixed diff keys also refer to processed comparisons in this mode;
do not conflate them with historical raw or cross-definition comparisons.

TIS is a biased, truncated token-level correction, not a repair for incorrect
input contexts, weight synchronization bugs or missing exploration. It cannot
guarantee an accuracy improvement. Preserve a no-TIS control for comparison.
The existing experiment name and auto-resume behavior are NOT changed;
use a new EXPERIMENT_NAME / checkpoint directory for a fresh comparison.
