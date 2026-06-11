GRAPH_PATH=runs/multi_seed_visual_smoke_6
python -m synthesis.vqa.run_batch \
  --graph-dir $GRAPH_PATH \
  --output-dir $GRAPH_PATH/vqa \
  --samples 10 \
  --workers 10 \
  --min-hops 1 \
  --max-hops 4 \
  --model-alias gpt54_internal_azure