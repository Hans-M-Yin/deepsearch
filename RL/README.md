# OpenSearch-VL · RL Training

This directory contains the end-to-end reinforcement-learning (RL) pipeline
for **OpenSearch-VL** — a multimodal deep-research agent that reasons over
images and invokes a suite of visual / web tools. Training is built on
[rLLM](https://github.com/rllm-org/rllm) (the `AgentWorkflowEngine` flavor),
[verl](https://github.com/volcengine/verl) as the PPO/GRPO backend, and
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) + [mbridge](https://github.com/ISEEKYAN/mbridge)
for large-scale model-parallel training.

The main entry point is the Vision-DeepResearch async workflow:

```
rllm/vision_deepresearch_async_workflow/
├── deepresearch_agent.py              # ReAct-style agent loop (tool-call parsing, planning)
├── deepresearch_workflow.py           # rLLM Workflow that drives the agent and computes rewards
├── deepresearch_tools_async_executor.py
├── train_deepresearch_workflow_megatron.py  # main training script (Hydra)
├── tools/                             # async tool implementations
│   ├── crop_and_search_tool.py        # crop-then-image-search
│   ├── search_tool.py                 # web search (Serper / Jina / Polaris)
│   ├── visit_tool.py                  # visit URL
│   ├── visual_tools.py                # layout parsing, text/image search, super-res, ...
│   ├── python_interpreter_tool.py
│   └── shared.py                      # DeepResearchTool base class + async cache
├── utils/api_gateway_client.py        # optional LLM-judge OpenAI-compatible client
├── data_prepare/                      # parquet → jsonl → rLLM DatasetRegistry
│   ├── convert_parquet2jsonl.py / .sh
│   └── register_rl_dataset.py / .sh
└── run/                               # launch scripts
    ├── qwen3-vl-8b-multi-node.sh      # 8B dense, 8 nodes × 8 GPU  ← primary
    ├── qwen3-vl-8b-single-node.sh     # 8B dense, 1 node × 8 GPU  (smoke-test)
    ├── qwen3-vl-32b-multi-node.sh     # 32B dense, multi-node
    └── qwen3-vl-30b-3b-multi-node.sh  # 30B-A3B MoE, multi-node
```

## Repository layout (`code/RL/`)

```
├── rllm/           # rLLM + verl + the vision_deepresearch_async_workflow entry point
├── Megatron-LM/    # Megatron backend (pinned copy)
├── mbridge/        # bridge between HF checkpoints and Megatron parallelism
├── LICENSE
└── README.md       # (this file)
```

## 1. Install

We recommend a clean Python 3.10+ virtual environment with CUDA 12.x and
PyTorch ≥ 2.4. Inside that environment:

```bash
cd rllm
pip install -e .              # installs rllm + verl
cd ../Megatron-LM && pip install -e .
cd ../mbridge    && pip install -e .

# Required at runtime for async rollout + training
pip install "sglang[all]" transformer_engine flash-attn \
            ray==2.34.* hydra-core omegaconf wandb \
            pillow requests python-dotenv
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformer_engine.pytorch as te; print('TE ok')"
```

> **CUDA runtime conflicts.** If your host has a system CUDA (e.g.
> `/usr/local/cuda`) that clashes with the venv-bundled NVIDIA libraries,
> override `LD_LIBRARY_PATH` before launching — see the commented block at
> the top of the run scripts for a template.

## 2. Environment variables

Create `rllm/.env` (auto-loaded by the launch scripts) or export manually:

| Variable                 | Purpose                                              | Required |
| ------------------------ | ---------------------------------------------------- | -------- |
| `WANDB_API_KEY`          | Weights & Biases logging                             | optional |
| `WANDB_BASE_URL`         | Defaults to `https://api.wandb.ai`                   | optional |
| `SERP_API_KEY`           | Serper.dev (web search)                              | for search tools |
| `JINA_API_KEY`           | Jina AI reader (page visit / rerank)                 | for search tools |
| `ZHIPU_API_KEY`          | Zhipu (optional image search provider)               | optional |
| `API_GATEWAY_USER` / `API_GATEWAY_KEY` / `API_GATEWAY_HOST` | Opt-in gateway that proxies Serper/Jina through a single endpoint. Leave unset to call providers directly. | optional |
| `LAYOUT_PARSING_API_URL` / `LAYOUT_PARSING_TOKEN` | PP-StructureV3-compatible endpoint used by the `layout_parsing` tool | optional |
| `JUDGE_API_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | OpenAI-compatible judge used by the query-utility reward. Unset ⇒ reward defaults to 0.0 (training still runs). | optional |
| `COS_USERID` / `COS_UPLOAD_PATHS` | If you maintain an internal COS uploader (`upload.py`) for `image_search`, point the client at it. | optional |

## 3. Prepare data

We expect a JSONL where every line is
`{"id": ..., "question": ..., "answer": ..., "images": [...]}`.

If you start from a HuggingFace parquet shard (with embedded PNG bytes),
use the two helpers in `rllm/vision_deepresearch_async_workflow/data_prepare/`:

```bash
cd rllm/vision_deepresearch_async_workflow/data_prepare

# 1) Extract image bytes to files and produce a JSONL
DATA_ROOT=./data/Vision-DeepResearch-RL-Data bash convert_parquet2jsonl.sh

# 2) Register it with rLLM as "Vision-DeepResearch-QA" (90/10 train/test)
JSONL_PATH=./data/Vision-DeepResearch-RL-Data/vision-deepresearch_RL_Demo_1k.jsonl \
    bash register_rl_dataset.sh
```

## 4. Launch training

All run scripts `cd` to the `rllm/` root and launch
`python -m vision_deepresearch_async_workflow.train_deepresearch_workflow_megatron`.

Primary 8B multi-node run (8 nodes × 8 GPU, `NNODES=8`):

```bash
bash rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-multi-node.sh
```

Other presets (edit `NNODES`, batch sizes and parallelism inside each script
to match your cluster):

```bash
bash rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh    # smoke-test
bash rllm/vision_deepresearch_async_workflow/run/qwen3-vl-30b-3b-multi-node.sh # MoE
bash rllm/vision_deepresearch_async_workflow/run/qwen3-vl-32b-multi-node.sh    # 32B dense
```

Key knobs inside each script:

- `MODEL_PATH` — HuggingFace model id or local snapshot of `Qwen3-VL-*-Instruct`.
- `NNODES`, `trainer.n_gpus_per_node` — cluster shape.
- `train_tp` / `train_pp` / `train_cp` (and `train_ep` / `train_etp` for MoE) — Megatron parallelism.
- `gen_tp` — sglang rollout tensor-parallel size.
- `train_prompt_bsz`, `n_resp_per_prompt`, `train_prompt_mini_bsz` — RL batch.
- `max_prompt_length`, `max_response_length` — 4k prompt + 70k response by default.
- `adv_estimator` — `rloo` by default; set to `grpo` or `reinforce_plus_plus` as desired.

Checkpoints go to `checkpoints/${project_name}/${exp_name}/`.

## 5. Reproducing the paper

The reported numbers use:

| Variant     | Script                             | Cluster        |
| ----------- | ---------------------------------- | -------------- |
| Qwen3-VL-8B  | `qwen3-vl-8b-multi-node.sh`       | 8 × 8 H100/800 |
| Qwen3-VL-30B-A3B | `qwen3-vl-30b-3b-multi-node.sh` | 8 × 8 H100/800 |
| Qwen3-VL-32B | `qwen3-vl-32b-multi-node.sh`      | 16 × 8 H100/800 |

## License

This subtree bundles three open-source frameworks, each under its own
upstream license:

- `rllm/` — Apache-2.0 (see `rllm/LICENSE`)
- `Megatron-LM/` — see `Megatron-LM/LICENSE`
- `mbridge/` — Apache-2.0 (see `mbridge/LICENSE`)

Project-specific modifications (the `vision_deepresearch_async_workflow`
package and launch scripts) are released under the root `LICENSE`.
