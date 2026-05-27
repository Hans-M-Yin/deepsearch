# Graph Construction without image fetch
python synthesis/run_min_graph.py \
  --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
  --store-dir synthesis/runs/kobe_text_only \
  --fresh \
  --no-images \
  --skip-attributes \
  --max-steps 2 \
  --max-nodes 5

# Graph Construction with image fetch
python synthesis/run_min_graph.py \
  --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
  --store-dir synthesis/runs/kobe_min_graph \
  --fresh \
  --max-steps 5 \
  --max-nodes 10