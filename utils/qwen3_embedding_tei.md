# Qwen3 search-similarity embedding service

This service provides Qwen3-Embedding-0.6B or Qwen3-Embedding-4B through the
native TEI `POST /embed` endpoint. It is intentionally independent of the training
environment: RL workers only need HTTP access.

## Prerequisites

- Native TEI binary at `~/.cargo/bin/text-embeddings-router`, built with the
  `candle-cuda` feature. This is the only backend used by the scripts.
- NVIDIA driver/CUDA libraries and one GPU that is **not** assigned to actor or rollout.
- No Docker, host `torch`, `transformers`, or `sentence-transformers` installation
  is required for serving.

## Download and start

```bash
# 0.6B: default port 8086 and shared path
# /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-embedding-0p6b.
bash utils/download_qwen3_embedding.sh

# 4B: default port 8087 and shared path
# /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-embedding-4b.
EMBEDDING_MODEL_SIZE=4B bash utils/download_qwen3_embedding.sh

# Start native 0.6B TEI on a dedicated physical GPU. Default port: 8086.
EMBEDDING_GPU_DEVICE=0 bash utils/start_qwen3_embedding_tei.sh

# Start native 4B TEI. Default port: 8087.
EMBEDDING_MODEL_SIZE=4B EMBEDDING_GPU_DEVICE=0 bash utils/start_qwen3_embedding_tei.sh
```

The service uses port `8086` for 0.6B and `8087` for 4B, and binds to `0.0.0.0`.
Override with `EMBEDDING_PORT` and `EMBEDDING_BIND_HOST`; set
`EMBEDDING_MODEL_ROOT` or `EMBEDDING_MODEL_DIR` if the shared default model path
is not the desired storage. Use a routable internal host address when trainer and
service are on different nodes.

```bash
export SEARCH_SIMILARITY_EMBEDDING_URL='http://embedding-host:8086/embed'
bash utils/test_qwen3_embedding_tei.sh

# For 4B:
export SEARCH_SIMILARITY_EMBEDDING_URL='http://embedding-host:8087/embed'
EMBEDDING_MODEL_SIZE=4B bash utils/test_qwen3_embedding_tei.sh
```

Native TEI writes its service PID and log to `/tmp`. For 4B these default to
`/tmp/qwen3-embedding-4b.pid` and `/tmp/qwen3-embedding-4b.log`.

```bash
EMBEDDING_MODEL_SIZE=4B bash utils/stop_qwen3_embedding_tei.sh
```

## Reward-side usage

At trajectory completion, normalize and de-duplicate all search queries, send the
missing ones in one `inputs` list to `/embed`, then calculate cosine similarities
locally. Do not synchronously call the embedding service for every tool call.
Keep an LRU cache from normalized query text to vector; this saves both service
work and tail latency on repeated searches.

The start script defaults to bounded dynamic batching suitable for short search
queries: 128 concurrent requests, 64 batched requests/client inputs, and 8192
batch tokens. The router automatically truncates overlong inputs. Override the
corresponding `EMBEDDING_MAX_*` variables only after load testing.
