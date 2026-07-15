mkdir -p runs/0712_multi_seed_visual_test_8192_2
echo "Start Generate."
source synthesis/.env
python synthesis/run_min_graph.py \
  --seed-urls-file synthesis/seeds.txt \
  --store-dir runs/0712_multi_seed_visual_test_8192_2\
  --reader-base-url http://localhost:8004 \
  --skip-attributes \
  --max-neighbors 4 \
  --max-steps 12345678 \
  --max-nodes 8192 \
  --max-depth 128 \
  --parallel-workers 100 \
  --batch-size 100 \
  --image-backend serper 
