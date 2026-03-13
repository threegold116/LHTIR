#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate
conda activate verl_5_2

set -x

ulimit -n 65535

export WANDB_MODE=offline
export TRITON_CACHE_DIR="/tmp/triton_cache_$(whoami)"
export RAY_DEBUG_MODE="1"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/verl:$PYTHONPATH"
export LD_LIBRARY_PATH=/share/home/sxjiang/miniconda3/envs/envtuning/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export RAY_DEBUG_POST_MORTEM=1
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=9999999999
export CUDA_VISIBLE_DEVICES=4,5
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"

now() {
    date '+%Y-%m-%d_%H-%M-%S'
}
TIMESTAMP=$(now)
PROJECT_NAME="qwen2.5-3b_gsm8k_multiturn_grpo"
EXPERIMENT_NAME="qwen2.5-3b_gsm8k_multiturn_grpo"
ROLLOUT_DIR="$PROJECT_DIR/rollout/$PROJECT_NAME/$EXPERIMENT_NAME"
DEFAULT_LOCAL_DIR="$PROJECT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$DEFAULT_LOCAL_DIR"
LOG_FILE="$DEFAULT_LOCAL_DIR/$TIMESTAMP.log"
MODEL="/share/home/sxjiang/model/Qwen2.5-3B-Instruct"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    data.train_files=/share/home/sxjiang/myproject/LHTIR/data/gsm8k/train.parquet \
    data.val_files=/share/home/sxjiang/myproject/LHTIR/data/gsm8k/test.parquet \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR\
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/verl/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.total_epochs=15 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 2>&1 | tee $LOG_FILE

#gpu_memory_utilization