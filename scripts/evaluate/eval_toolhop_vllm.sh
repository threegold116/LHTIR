#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate 
conda activate verl_5_2

export PROJECT_DIR="$(pwd)"
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/verl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

LOG_DIR="$PROJECT_DIR/logs/vllm_logs"
mkdir -p $LOG_DIR
CURRENT_TIME=$(date '+%m%d_%H%M')
MAIN_LOG="$LOG_DIR/eval_toolhop_vllm_${CURRENT_TIME}.log"

log() {
    local level="$1"
    local message="$2"
    # 同时输出到标准错误和主日志文件
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$MAIN_LOG"
}

vllm_kill(){
    echo "Shutting down vLLM..."
    # 找到占用 7899 端口的进程及其所有子进程
    PID=$(lsof -t -i:7899) || true
    if [ -z "$PID" ]; then
        echo "PID is empty"
        return 0
    fi
    echo "--------------------------------"
    echo "PID: $PID"
    killtree() {
        for child in $(pgrep -P $1); do
            killtree $child
        done
        kill -15 $1 2>/dev/null
    }
    killtree $PID
    echo "Killed vLLM"
}

vllm_run(){

    vllm_kill

    MODEL_PATH="$1"
    LOG_FILE="$LOG_DIR/vllm_server_${CURRENT_TIME}.log"
    log "INFO" "Starting vLLM server for model: $MODEL_PATH"
    
    # 根据 CUDA_VISIBLE_DEVICES 中 GPU 数量设置数据并行大小（如 "4,5" -> 2）
    DP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    vllm serve $MODEL_PATH \
        --served-model-name MatchTIR \
        --tensor-parallel-size 1 \
        --data-parallel-size $DP_SIZE \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --port 7899 > "$LOG_FILE" 2>&1 &
    
    # 轮询检查端口是否 Ready
    log "INFO" "Waiting for vLLM to initialize (logging to $LOG_FILE)..."
    while ! curl -s http://localhost:7899/v1/models > /dev/null; do
        sleep 5
        # 简单检查进程是否还在
        if ! kill -0 $! 2>/dev/null; then
            log "ERROR" "vLLM process died unexpectedly. Check $LOG_FILE"
            exit 1
        fi
    done
    log "SUCCESS" "vLLM server is READY."
}

# 调整 trap，只在异常退出时清理
cleanup() {
    trap - INT TERM EXIT
    echo "Interrupt received or error occurred."
    vllm_kill
}
trap cleanup INT TERM EXIT # 移除了 EXIT，因为正常结束我们手动杀



evaluation_file="evaluation_toolhop.py"
file="toolhop/ToolHop.jsonl"
model="/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b-2507_ftrl_multiturn-no_kl_no_ent-n_8-step_2048-gtpo_loss-clip_low3e-3_high4e-3_seq_mean_turn_mean_token_mean-process_reward-MATCHTIR_KM-24"

vllm_run $model
# for scenario in "Free" "Direct" "Mandatory"; do

for scenario in "Free"; do
    save_file="$PROJECT_DIR/results/ToolHop/Qwen3-4B/MatchTIR-KM-24-2507-no_kl_no_ent-n_8-step_2048-gtpo_loss-clip_low3e-3_high4e-3_seq_mean_turn_mean_token_mean-process_reward-seed1-2-${scenario}-vllm-4096.jsonl"
    LOG_FILE="$PROJECT_DIR/results/ToolHop/Qwen3-4B/eval_toolhop_vllm_${CURRENT_TIME}_${scenario}-MatchTIR-KM-24-2507-no_kl_no_ent-n_8-step_2048-gtpo_loss-clip_low3e-3_high4e-3_seq_mean_turn_mean_token_mean-process_reward-seed1-2.log"
    python3 $PROJECT_DIR/evaluate/${evaluation_file} \
        --scenario ${scenario} \
        --series qwen \
        --model_path ${model} \
        --input_file $PROJECT_DIR/data/${file} \
        --output_file ${save_file} \
        --enable_thinking \
        --base_url http://localhost:7899/v1 \
        --concurrency 128 \
        --batch_size 1024 \
        --max_tokens 4096 \
        --batch_mode async \
        --engine remote \
        --start_id 0 \
        --end_id -1  2>&1 | tee  $LOG_FILE 
done

vllm_kill

exit 0
