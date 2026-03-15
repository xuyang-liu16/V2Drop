#!/bin/bash

model_id="/home/tata/G/CJJ/Qwen2-vl/Qwen2.5-VL-main/model/Qwen2-VL-7B-Instruct/"
model_name="Qwen2-VL-7B-Instruct"
output_path="./logs/${model_name}/${task}/"
mkdir -p "$output_path"

Sparse=$1
image_token_start_index=0
image_token_length=0

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 50008 \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=$model_id,device_map=cuda,use_flash_attention_2=True,Sparse=$Sparse,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --output_path "$output_path" \