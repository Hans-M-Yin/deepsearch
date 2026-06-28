GRAPH_PATH=runs/multi_seed_visual_8192_1

SAMPLES=10
WORKERS=10
MIN_HOPS=2
MAX_HOPS=5

python -m synthesis.vqa.run_batch \
  --graph-dir $GRAPH_PATH \
  --output-dir $GRAPH_PATH/vqa \
  --samples $SAMPLES \
  --workers $WORKERS \
  --min-hops $MIN_HOPS \
  --max-hops $MAX_HOPS \
  --model-alias gemini25pro_internal_azure \
  --compress-hop-model-alias multimodal_process \
  --neighbor-selection-strategy llm_guided \
  --sampler-model-alias multimodal_process