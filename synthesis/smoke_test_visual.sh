python run_min_graph.py \
  --seed-url https://en.wikipedia.org/wiki/Jay_Chou \
  --store-dir runs/kobe_with_images_serper_20_100_depth_33 \
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
  