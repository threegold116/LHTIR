#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate vel_5

MODEL_PATH=/root/model/Qwen3-30B-A3B-Thinking-2507

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL_PATH \
    --served-model-name Qwen3-30B-A3B-Thinking-2507 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    --reasoning-parser deepseek_r1 \
    --trust-remote-code \
    --uvicorn-log-level debug \
    --host 0.0.0.0 \
    --port 7899 

curl http://61.161.168.20:7899/v1/models

curl http://localhost:7899/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "Qwen3-30B-A3B-Thinking-2507", "messages": [{"role": "user", "content": "Hello, who are you?"}]}'