mkdir -p runs/multi_seed_visual_smoke_4
echo "Start Generate."
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/multi_seed_visual_smoke_4 \
  --reader-base-url http://10.124.136.6:8004 \
  --fresh \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 512 \
  --max-depth 64 \
  --parallel-workers 48 \
  --batch-size 48 \
  --image-backend serper 
