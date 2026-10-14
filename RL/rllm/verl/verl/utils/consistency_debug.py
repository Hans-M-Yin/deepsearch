"""Opt-in forward-only diagnostics; never replace the training log probabilities."""
import json
import os
import time

import torch


def enabled():
    return os.environ.get("RLLM_MM_CONSISTENCY_DEBUG", "0") == "1"


def emit(event, **fields):
    print("========== [CONSISTENCY_DEBUG] " + json.dumps(
        {"event": event, "pid": os.getpid(), **fields}, default=str
    ), flush=True)


def fields(obj, names):
    return {name: getattr(obj, name, None) for name in names.split()}


def raw_monitor_logprobs(actor, data, training_logprobs):
    """Extra T=1 forward for monitoring only; preserve training metadata/RNG."""
    temperature = data.meta_info["temperature"]
    if float(temperature) == 1.0:
        return training_logprobs
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        data.meta_info["temperature"] = 1.0
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            result, _ = actor.compute_log_prob(data=data, calculate_entropy=False)
        return result
    finally:
        data.meta_info["temperature"] = temperature


def forward_probes(actor, data, actor_config, calculate_entropy):
    """Same weights/temperature/input; independently vary batching and fusion.

    All distributed ranks must execute these in the same order. Do not swallow
    failures on a single rank: a failed collective cannot safely be skipped.
    """
    from verl.models.mcore.model_forward_fused import patch_fused_forward, unpatch_fused_forward

    original_meta = data.meta_info.copy()
    original_fused = actor.use_fused_kernels
    results = {}
    variants = ["repeat", "actor_batching"]
    if original_fused:
        variants.append("unfused")
    # Preserve RNG so diagnostics do not advance training's random stream.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    unpatched = []
    try:
        for variant in variants:
            data.meta_info.clear()
            data.meta_info.update(original_meta)
            if variant == "actor_batching":
                data.meta_info.update(
                    use_dynamic_bsz=actor_config.use_dynamic_bsz,
                    max_token_len=actor_config.ppo_max_token_len_per_gpu,
                    micro_batch_size=actor_config.ppo_micro_batch_size_per_gpu,
                )
            if variant == "unfused":
                for model in actor.actor_module:
                    unpatch_fused_forward(model)
                    unpatched.append(model)
                actor.use_fused_kernels = False
            emit("forward_start", variant=variant,
                 batching={k: data.meta_info.get(k) for k in (
                     "temperature", "micro_batch_size", "max_token_len", "use_dynamic_bsz")},
                 fused=actor.use_fused_kernels)
            start = time.monotonic()
            with torch.no_grad():
                value, _ = actor.compute_log_prob(data, calculate_entropy=calculate_entropy)
            results["diagnostic_" + variant + "_log_probs"] = value
            emit("forward_done", variant=variant, elapsed_s=time.monotonic() - start)
    finally:
        for model in unpatched:
            patch_fused_forward(model)
        actor.use_fused_kernels = original_fused
        data.meta_info.clear()
        data.meta_info.update(original_meta)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
    return results
