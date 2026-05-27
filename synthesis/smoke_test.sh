# Graph Construction without image fetch
python run_min_graph.py \
  --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
  --store-dir runs/kobe_text_only \
  --reader-base-url http://10.124.138.16:8004 \
  --fresh \
  --no-images \
  --skip-attributes \
  --max-steps 2 \
  --max-nodes 5

# Graph Construction with image fetch
# python run_min_graph.py \
#   --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
#   --store-dir runs/kobe_min_graph \
#   --reader-base-url http://10.124.138.16:8004 \
#   --fresh \
#   --max-steps 5 \
#   --max-nodes 10