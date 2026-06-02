python run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/multi_seed_visual_smoke \
  --reader-base-url http://10.124.136.6:8004 \
  --fresh \
  --skip-attributes \
  --max-neighbors 0 \
  --max-steps 32 \
  --max-nodes 4 \
  --max-depth 64 \
  --parallel-workers 1 \
  --batch-size 32 \
  --image-backend serper
  
