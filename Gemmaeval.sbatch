#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
source ~/.bashrc
mamba activate py39
model_name="google/gemma-7b"

folder="gemma"

# select tasks
tasks_selected=(
    "bertaqa_eu"
    "bertaqa_en"
    #"bertaqa_en_mt"
)

num_fewshot=5

for group_name in "${tasks_selected[@]}"; do
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name \
        --tasks $group_name \
        --device cuda \
        --output_path /dss/dsshome1/07/ra47fey2/results/${folder}/${model}/${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
