mkdir -p runs/multi_seed_visual_smoke_6
echo "Start Generate."
source synthesis/.env
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/multi_seed_visual_smoke_6 \
  --reader-base-url http://10.124.136.252:8004 \
  --fresh \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 3072 \
  --max-depth 64 \
  --parallel-workers 32 \
  --batch-size 32 \
  --image-backend serper 
