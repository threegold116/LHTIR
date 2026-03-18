#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate 
conda activate verl_5_2
MODEL_PATH=/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b_ftrl_multiturn-MATCHTIR_KM-24
CUDA_VISIBLE_DEVICES=4,5 vllm serve $MODEL_PATH \
    --served-model-name Qwen/Qwen3-4B \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --uvicorn-log-level debug \
    --host 0.0.0.0 \
    --port 7899 

curl http://localhost:7899/v1/models

curl http://localhost:7899/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "MatchTIR", "messages": [{"role": "user", "content": "Hello, how are you?"}]}'