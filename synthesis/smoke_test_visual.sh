mkdir -p runs/multi_seed_visual_smoke_2

python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/multi_seed_visual_smoke_2 \
  --reader-base-url http://10.124.136.6:8004 \
  --fresh \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 512 \
  --max-depth 64 \
  --parallel-workers 64 \
  --batch-size 64 \
  --image-backend serper \
  2>&1 | tee runs/multi_seed_visual_smoke_2/run.log
  
