python python_scripts/annotate_reasoning.py \
    --generators qwen \
    --qwen_model Qwen/Qwen3-4B-Thinking-2507 \
    --format_mode steps \
    --dataset_name EleutherAI/hendrycks_math \
    --dataset_config algebra \
    --num_problems 1000 --sample_start 0 \
    --save_reasoning output/factuality/raw \
    --save_formatted output/factuality/formatted