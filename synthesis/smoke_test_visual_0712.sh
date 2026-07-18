mkdir -p runs/0712_multi_seed_visual_test_8192_4
echo "Start Generate."
source synthesis/.env
nohup python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test_8192_4 \
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 10 \
  --max-llm-neighbor-candidates 80 \
  --max-steps 12345678 \
  --max-nodes 8192 \
  --max-depth 128 \
  --parallel-workers 80 \
  --batch-size 80 \
  --image-backend serper \
  > synthesis/ignore/output.log 2>&1 &
