#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate


export LOCAL_SERVER_ENDPOINT=localhost        # 可省略，默认就是 localhost
export LOCAL_SERVER_PORT=7897                # 可省略，默认就是 1053
export REMOTE_OPENAI_BASE_URL=http://0.0.0.0:7897/v1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_DIR="/share/home/sxjiang/myproject/LHTIR"
LOG_DIR="$PROJECT_DIR/logs/vllm_logs"
mkdir -p $LOG_DIR
CURRENT_TIME=$(date '+%m%d_%H%M')
MAIN_LOG="$LOG_DIR/eval_bfcl_vllm_${CURRENT_TIME}.log"

log() {
    local level="$1"
    local message="$2"
    # 同时输出到标准错误和主日志文件
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$MAIN_LOG"
}

vllm_kill(){
    echo "Shutting down vLLM..."
    # 找到占用 7897 端口的进程及其所有子进程
    PID=$(lsof -t -i:7897) || true
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
    conda activate verl_5_2
    MODEL_PATH="$1"
    LOG_FILE="$LOG_DIR/vllm_server_${CURRENT_TIME}.log"
    log "INFO" "Starting vLLM server for model: $MODEL_PATH"
    
    # 根据 CUDA_VISIBLE_DEVICES 中 GPU 数量设置数据并行大小（如 "4,5" -> 2）
    DP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    vllm serve $MODEL_PATH \
        --served-model-name Qwen/Qwen3-4B \
        --tensor-parallel-size 1 \
        --data-parallel-size $DP_SIZE \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --port 7897 > "$LOG_FILE" 2>&1 &
    
    # 轮询检查端口是否 Ready
    log "INFO" "Waiting for vLLM to initialize (logging to $LOG_FILE)..."
    while ! curl -s http://localhost:7897/v1/models > /dev/null; do
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




# MODEL_PATH="/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b-instruct-2507_ftrl_multiturn-no_kl_no_ent-n_8-mask_func-MATCHTIR_KM-24"
MODEL_PATH="/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b-ftrl_multiturn-no_kl_no_ent-n_16-MATCHTIR_KM-24"
export REMOTE_OPENAI_TOKENIZER_PATH=$MODEL_PATH
MODEL_NAME="Qwen/Qwen3-4B-FC"
TEST_CATEGORY="multi_turn"
RESULT_DIR="$PROJECT_DIR/results/BFCL/Qwen3-4B/MatchTIR-KM-24-no_kl_no_ent-n_16-vllm_fc"
vllm_run $MODEL_PATH
conda activate BFCL


now() {
    date '+%Y-%m-%d_%H-%M-%S'
}
TIMESTAMP=$(now)
mkdir -p $RESULT_DIR
LOG_FILE="$RESULT_DIR/bfcl_eval_${TIMESTAMP}.log"
echo "MODEL_PATH: $MODEL_PATH" | tee $LOG_FILE
bfcl generate \
  --model $MODEL_NAME \
  --test-category $TEST_CATEGORY \
  --skip-server-setup \
  --num-threads 10\
  --result-dir $RESULT_DIR 2>&1 | tee -a $LOG_FILE

bfcl evaluate \
  --model $MODEL_NAME \
  --test-category $TEST_CATEGORY \
  --result-dir $RESULT_DIR \
  --score-dir $RESULT_DIR/scores 2>&1 | tee -a $LOG_FILE



vllm_kill

exit 0

