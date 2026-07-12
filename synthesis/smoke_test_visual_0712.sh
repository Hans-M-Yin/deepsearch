mkdir -p runs/0712_multi_seed_visual_test4
echo "Start Generate."
source synthesis/.env
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test4 \
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 128 \
  --max-depth 128 \
  --parallel-workers 32 \
  --batch-size 32 \
  --image-backend serper 
