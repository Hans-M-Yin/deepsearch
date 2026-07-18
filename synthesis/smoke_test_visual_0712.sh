mkdir -p runs/0712_multi_seed_visual_test_8192_6
echo "Start Generate."
source synthesis/.env
nohup python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test_8192_6 \
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 10 \
  --max-llm-neighbor-candidates 80 \
  --max-steps 12345678 \
  --max-nodes 8192 \
  --max-depth 128 \
  --queue-pop-strategy random \
  --parallel-workers 120 \
  --batch-size 120 \
  --max-inflight-text 60 \
  --image-backend serper \
  --queue-pop-strategy random \
  > synthesis/ignore/output.log 2>&1 &
