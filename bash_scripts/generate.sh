subjects=(
algebra
counting_and_probability
geometry
intermediate_algebra
number_theory
prealgebra
precalculus
)

for subject in "${subjects[@]}"; do
    echo "Running subject: $subject"

    python python_scripts/generate_reasoning.py \
        --generators mistral \
        --mistral_model mistralai/Mistral-7B-Instruct-v0.3 \
        --format_mode steps \
        --save_formatted output/factuality/formatted \
        --dataset_name EleutherAI/hendrycks_math \
        --dataset_config $subject \
        --num_problems 2500 \
        --sample_start 0

done