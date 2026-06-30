python python_scripts/merge_signal_label.py \
  --prm_path      output/factuality/signal \
  --prm_model     qwen \
  --gen_model     Mistral-7B-Instruct-v0.3 \
  --critique_path output/factuality/critiques \
  --dataconfig   "all" \
  --output output/factuality