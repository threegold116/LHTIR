#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate 
conda activate verl_5_2
MODEL_PATH=/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b_ftrl_multiturn-MATCHTIR_KM-24
CUDA_VISIBLE_DEVICES=6,7 python -m sglang_router.launch_server\ 
    --model-path $MODEL_PATH \
    --served-model-name MatchTIR \
    --dp 2 \
    --host 0.0.0.0 \
    --port 7888 

curl http://localhost:7899/v1/models

curl http://localhost:7899/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "MatchTIR", "messages": [{"role": "user", "content": "Hello, how are you?"}]}'