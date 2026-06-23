mkdir -p runs/multi_seed_visual_8192_1
echo "Start Generate."
source synthesis/.env
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/multi_seed_visual_8192_1 \
  --reader-base-url http://10.124.137.241:8004 \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 8192 \
  --max-depth 128 \
  --parallel-workers 128 \
  --batch-size 128 \
  --image-backend serper 
