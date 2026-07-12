mkdir -p runs/0712_multi_seed_visual_test2
echo "Start Generate."
source synthesis/.env
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test2 \
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 64 \
  --max-depth 128 \
  --parallel-workers 32 \
  --batch-size 32 \
  --image-backend serper 
