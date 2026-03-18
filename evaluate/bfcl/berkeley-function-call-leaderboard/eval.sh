#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/myproject/LHTIR/bin/activate
conda activate BFCL
MODEL_PATH="/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b_ftrl_multiturn-MATCHTIR_KM-24"
MODEL_NAME="qwen3-4b-think-FC"
TEST_CATEGORY="multi_turn"
export CUDA_VISIBLE_DEVICES=6,7
bfcl generate\
    --model $MODEL_NAME\
    --local-model-path $MODEL_PATH\
    --test-category $TEST_CATEGORY\
    --backend sglang \
    --num-gpus 2 \
    --gpu-memory-utilization 0.9 \
    --output-file /share/home/sxjiang/myproject/LHTIR/bfcl/berkeley-function-call-leaderboard/ToolHop.jsonl