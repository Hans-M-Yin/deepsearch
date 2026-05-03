# OpenSearch-VL · Agentic SFT

This directory contains the supervised fine-tuning (SFT) pipeline for
**OpenSearch-VL**'s agentic cold-start. It is a trimmed fork of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — we keep only the
Ray-based full-parameter training path and the `agentic_full` configs used
to produce the paper's checkpoints.

## Repository layout

```
├── src/llamafactory/         # LLaMA-Factory source (trainer, data loader, CLI)
├── data/
│   └── dataset_info.json     # 7 agentic-SFT dataset entries (see §2)
├── examples/
│   ├── agentic_full/         # training YAMLs for Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL
│   └── deepspeed/            # ZeRO-2 / ZeRO-3 configs referenced by the YAMLs
├── requirements/             # pinned dependency groups
├── scripts/                  # helper scripts (tokenizer checks, sanity runs, etc.)
├── docker/                   # reference Dockerfiles
├── pyproject.toml
└── README.md
```

## Supported training configs

All YAMLs live in `examples/agentic_full/` and are wired to run with Ray on
`ray_num_workers = 256` (16 nodes × 8 GPU). Edit `ray_num_workers`,
`placement_strategy`, and `resources_per_worker` to match your cluster.

| YAML | Model | Template |
| ---- | ----- | -------- |
| `qwen3_vl_full_sft_8b_ray.yaml`      | `Qwen/Qwen3-VL-8B-Instruct`      | `qwen3_vl` |
| `qwen3_vl_full_sft_30_3b_ray.yaml`   | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `qwen3_vl` |
| `qwen3_vl_full_sft_32b_ray.yaml`     | `Qwen/Qwen3-VL-32B-Instruct`     | `qwen3_vl` |
| `qwen3_5vl_full_sft_27b_ray.yaml`    | `Qwen/Qwen3.5-VL-27B-Instruct`   | `qwen3_5`  |
| `qwen3_5vl_full_sft_35b_3b_ray.yaml` | `Qwen/Qwen3.5-VL-35B-A3B-Instruct` | `qwen3_5`|
| `qwen2_5_vl_full_sft_7b_ray.yaml`    | `Qwen/Qwen2.5-VL-7B-Instruct`    | `qwen2_vl` |
| `qwen2_5_vl_full_sft_32b_ray.yaml`   | `Qwen/Qwen2.5-VL-32B-Instruct`   | `qwen2_vl` |
| `qwen2_5_vl_full_sft_72b_ray.yaml`   | `Qwen/Qwen2.5-VL-72B-Instruct`   | `qwen2_vl` |

Shared hyper-parameters (8B example):

- `cutoff_len: 32000`, `bf16`, `lr = 2e-5`, `cosine` schedule, `warmup_ratio = 0.1`
- `num_train_epochs: 8`, `per_device_train_batch_size: 1`, `gradient_checkpointing: true`
- ZeRO-3: `deepspeed: examples/deepspeed/ds_z3_config.json`
- Full fine-tune of LLM + vision tower + projector (`freeze_vision_tower: false`, `freeze_multi_modal_projector: false`)

## 1. Install

Python ≥ 3.10, CUDA ≥ 12.1, PyTorch ≥ 2.4.

```bash
pip install -e ".[torch,metrics,deepspeed,ray]"
# Extra requirements for VL training:
pip install qwen-vl-utils pillow av decord torchvision flash-attn
```

Verify:

```bash
llamafactory-cli version
```

## 2. Prepare data

The training YAMLs point at the named datasets declared in
[`data/dataset_info.json`](data/dataset_info.json):

```
new_fvqa_agent_sft,palace_agent_sft,webqa_agent_sft,livevqa_agent_sft,
wikiart_agent_sft,wiki_zh_agent_sft,wiki_en_agent_sft
```

Each entry points at a **relative** path under `data/`. Download the
`OpenSearch-VL-SFT` release from HuggingFace (or build your own cold-start
data) and arrange it as below:

```
data/
├── dataset_info.json
├── new_fvqa/
│   └── fvqa_llama_factory_clean.json
├── palace/
│   └── palace_llama_factory_filtered.json
├── WebQA/
│   └── webqa_llama_factory_filtered.json
├── new_livevqa/
│   └── livevqa_llama_factory_filtered.json
├── wikiart/
│   └── wikiart_llama_factory_filtered.json
├── wiki_en/
│   └── wiki_en_llama_factory_filtered.json
└── wiki_zh/
    └── wiki_zh_llama_factory_filtered.json
```

Each JSON file is in ShareGPT format with the following columns:

- `conversations` (list of `{"from": "human|gpt|observation", "value": ...}`)
- `images` (list of image paths, relative to `data/` or absolute)
- `system` (string system prompt)
- `tools` (JSON-string tool schema, e.g. crop / search / visit)

Images may be stored next to the JSON or in a shared directory — both
absolute and relative (to `data/`) paths work.

## 3. Launch training (Ray, multi-node)

Use `USE_RAY=1` to route through the Ray launcher baked into the YAML
(`ray_run_name`, `ray_num_workers`, `placement_strategy`).

```bash
# On the Ray head node — start the cluster first (`ray start --head ...`),
# and `ray start --address=... ` on every worker node.

cd code/SFT

USE_RAY=1 \
llamafactory-cli train examples/agentic_full/qwen3_vl_full_sft_8b_ray.yaml
```

For quick single-node smoke tests:

```bash
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
llamafactory-cli train examples/agentic_full/qwen3_vl_full_sft_8b_ray.yaml
```

Checkpoints go to `saves/qwen3_vl_8b/full/sft_data_v1/` (controlled by the
`output_dir` / `ray_storage_path` fields of each YAML).

## 4. Cluster notes

The YAMLs pre-populate common NCCL / IB environment variables tuned for a
NIC-bonded (`bond1`) + `mlx5_bond*` fabric. Adjust these under
`ray_init_kwargs.runtime_env.env_vars` for your own cluster:

- `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `UCX_NET_DEVICES`
- `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_IB_SL`, `NCCL_IB_TC`,
  `NCCL_IB_QPS_PER_CONNECTION`

If you don't have RDMA, set `NCCL_IB_DISABLE: "1"` and remove the IB-specific
keys.

## Acknowledgements

This subtree derives from LLaMA-Factory(Apache-2.0). See `LICENSE` and `CITATION.cff` for attribution. Please cite
the upstream project when using this code.
