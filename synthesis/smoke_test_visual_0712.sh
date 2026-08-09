mkdir -p runs/0712_multi_seed_visual_test_8192_6
echo "Start Generate."
source synthesis/.env
nohup python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test_8192_6 \
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 8 \
  --max-llm-neighbor-candidates 50 \
  --max-steps 12345678 \
  --max-nodes 25000 \
  --max-depth 128 \
  --parallel-workers 64 \
  --batch-size 120 \
  --max-inflight-text 32 \
  --image-backend serper \
  --image-entity-only \
  > synthesis/ignore/output.log 2>&1 &
  # --queue-pop-strategy random \