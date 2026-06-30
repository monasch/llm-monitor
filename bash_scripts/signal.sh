
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

    python python_scripts/get_signal_prm.py \
        --load_formatted output/factuality/formatted \
        --model_label Mistral-7B-Instruct-v0.3 \
        --dataset_config $subject \
        --num_problems 2500 \
        --sample_start 0 \
        --prm qwen \
        --save_path output/factuality/signal
done