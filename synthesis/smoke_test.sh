# Graph Construction without image fetch
python run_min_graph.py \
  --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
  --store-dir runs/kobe_text_only_20_100_depth_7 \
  --reader-base-url http://10.124.138.16:8004 \
  --fresh \
  --no-images \
  --skip-attributes \
  --max-steps 64 \
  --max-nodes 128 \
  --max-depth 64 \
  --parallel-workers 32 \
  --batch-size 32
# Graph Construction with image fetch
# python run_min_graph.py \
#   --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
#   --store-dir runs/kobe_min_graph \
#   --reader-base-url http://10.124.138.16:8004 \
#   --fresh \
#   --max-steps 5 \
#   --max-nodes 10