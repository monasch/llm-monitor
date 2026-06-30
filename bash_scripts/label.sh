#python python_scripts/label_step.py \
#    --generators qwen \
#    --qwen_model Qwen/Qwen3-4B-Thinking-2507 \
#    --dataset_name EleutherAI/hendrycks_math \
#    --dataset_config algebra \
#    --dataset_split test \
#    --num_problems 900 \
#    --sample_start 100 \
#    --openai_api_key_file $HOME/openai_key.txt \
#    --critique_template templates/label.txt \
#    --save_formatted output/factuality/formatted \
#    --save_critiques output/factuality/critiques \
#    --critic_model o3-mini-2025-01-31 \
#    --batch \
#    --batch_dir output/factuality/critiques/batch_jobs


#python python_scripts/label_step.py \
#    --generators qwen \
#    --qwen_model Qwen/Qwen2.5-Math-7B-Instruct \
#    --dataset_name EleutherAI/hendrycks_math \
#    --dataset_config algebra \
#    --dataset_split test \
#    --num_problems 121 \
#    --sample_start 0 \
#    --openai_api_key_file $HOME/openai_key.txt \
#    --critique_template templates/label.txt \
#    --save_formatted output/factuality/formatted \
#    --save_critiques output/factuality/critiques \
#    --critic_model o3-mini-2025-01-31 \
#    --batch \
#    --batch_dir output/factuality/critiques/batch_jobs

subjects=(
#algebra
#counting_and_probability
#geometry
#intermediate_algebra
number_theory
#prealgebra
#precalculus
)

for subject in "${subjects[@]}"; do
    echo "Running subject: $subject"

    python python_scripts/label_step.py \
        --generators mistral \
        --qwen_model mistral/Mistral-7B-Instruct-v0.3 \
        --dataset_name EleutherAI/hendrycks_math \
        --dataset_config $subject \
        --dataset_split test \
        --num_problems 2500 \
        --sample_start 0 \
        --openai_api_key_file $HOME/openai_key.txt \
        --critique_template templates/label.txt \
        --save_formatted output/factuality/formatted \
        --save_critiques output/factuality/critiques \
        --critic_model o3-mini-2025-01-31 \
        --batch \
        --batch_dir output/factuality/critiques/batch_jobs
done