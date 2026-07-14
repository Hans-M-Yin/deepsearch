mkdir -p runs/0712_multi_seed_visual_test8_8192
echo "Start Generate."
source synthesis/.env
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test_8192\
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 8192 \
  --max-depth 128 \
  --parallel-workers 128 \
  --batch-size 128 \
  --image-backend serper 
