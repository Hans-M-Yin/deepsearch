import os
from pathlib import Path

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_USE_V1": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
    "worker_process_setup_hook": "rllm.patches.verl_patch_hook.setup",
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
    # OpenSearch-VL synthesis/SFT tool and reward configuration.
    "SERPER_",
    "SERPAPI_",
    "FIRECRAWL_",
    "ENHANCED_",
    "OSS_",
    "JUDGE_",
    "SFT_",
    "SYNTHESIS_",
    "WIKI_",
    "IMAGE_",
    "VISUAL_",
    "TEXT_",
    # Multimodal RL diagnostics (RLLM_MM_DEBUG, log path, rank filter, etc.).
    "RLLM_MM_",
]

# ``synthesis`` is a repository package rather than an installed wheel.  The
# driver launch script adds these directories to PYTHONPATH, but Ray's
# runtime_env does not inherit arbitrary environment variables unless they are
# explicitly forwarded below.
FORWARD_ENV_NAMES = {"PYTHONPATH"}


def _repository_pythonpath() -> str:
    """Return a deduplicated PYTHONPATH usable by Ray workers.

    The workflow imports both the repository-level ``synthesis`` package and
    the editable-installed ``rllm`` package.  Adding both roots makes the
    remote TaskRunner independent of the driver's current working directory.
    """

    # .../OpenSearch-VL/RL/rllm/rllm/trainer/verl/ray_runtime_env.py
    rllm_root = Path(__file__).resolve().parents[3]
    project_root = rllm_root.parents[1]

    # ``verl`` is a source checkout nested below ``RL/rllm/verl`` rather than
    # directly below ``RL/rllm``.  Put this path before any site-installed or
    # user-checkout version so Ray workers cannot mix verl modules from two
    # different installations.
    verl_root = rllm_root / "verl"
    entries = [str(project_root), str(rllm_root), str(verl_root)]
    entries.extend(item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            deduplicated.append(entry)
    return os.pathsep.join(deduplicated)


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    exclude_vars = set()
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            forward_prefix.remove(name.replace("*", "_"))
        else:
            exclude_vars.add(name)

    forwarded = {
        k: v
        for k, v in os.environ.items()
        if (k in FORWARD_ENV_NAMES or any(k.startswith(p) for p in forward_prefix))
        and k not in exclude_vars
    }
    # Always provide the repository roots.  This is intentionally done after
    # the generic forwarding logic so a stale/incomplete driver PYTHONPATH
    # cannot make ``synthesis`` disappear inside the Ray worker.
    if "PYTHONPATH" not in exclude_vars:
        forwarded["PYTHONPATH"] = _repository_pythonpath()
    return forwarded


def get_ppo_ray_runtime_env():
    env = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    env.update(_get_forwarded_env_vars())
    return {
        "env_vars": env,
        # "worker_process_setup_hook": PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"],
    }
