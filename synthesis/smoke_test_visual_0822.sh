mkdir -p runs/0712_multi_seed_visual_test_8192_8
echo "Start Generate."
source synthesis/.env
nohup python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds_v2.txt \
  --store-dir runs/0712_multi_seed_visual_test_8192_8 \
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 8 \
  --max-llm-neighbor-candidates 50 \
  --max-steps 12345678 \
  --max-nodes 10000 \
  --max-depth 128 \
  --parallel-workers 40 \
  --batch-size 120 \
  --max-inflight-text 20 \
  --image-backend serper \
  > /tmp/output_generate.log 2>&1 &
  # --queue-pop-strategy random \
