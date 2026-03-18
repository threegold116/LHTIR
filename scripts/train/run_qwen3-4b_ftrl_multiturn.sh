#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate
conda activate verl_5_2

set -x

ulimit -n 65535
export WANDB_MODE=offline
export TRITON_CACHE_DIR="/tmp/triton_cache_$(whoami)"
export RAY_DEBUG_MODE="0"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/verl:$PYTHONPATH"
export LD_LIBRARY_PATH=/share/home/sxjiang/miniconda3/envs/envtuning/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export RAY_DEBUG_POST_MORTEM=1
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=9999999999
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/prog_env/config"

now() {
    date '+%Y-%m-%d_%H-%M-%S'
}
TIMESTAMP=$(now)
PROJECT_NAME="qwen3-4b_ftrl_multiturn"
EXPERIMENT_NAME="qwen3-4b-instruct-2507_ftrl_multiturn"
ROLLOUT_DIR="$PROJECT_DIR/rollout/$PROJECT_NAME/$EXPERIMENT_NAME"
DEFAULT_LOCAL_DIR="$PROJECT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$DEFAULT_LOCAL_DIR"
LOG_FILE="$DEFAULT_LOCAL_DIR/$TIMESTAMP.log"
MODEL="/share/home/sxjiang/model/Qwen3-4B-Instruct-2507"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='ftrl_multiturn' \
    algorithm.adv_estimator=mathtir \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=7000 \
    data.max_response_length=23000 \
    data.prompt_key=messages\
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.custom_cls.path=$PROJECT_DIR/prog_env/dataset/rl_dataset.py \
    data.custom_cls.name=RLHFDataset \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=10 \
    +actor_rollout_ref.rollout.custom_cls.path=pkg://prog_env.agent_loop \
    +actor_rollout_ref.rollout.custom_cls.name=AgentLoopManager \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=4 \
    trainer.val_before_train=True \
    data.train_files=$PROJECT_DIR/data/ftrl/with_agent_name/train.parquet \
    data.val_files=$PROJECT_DIR/data/ftrl/with_agent_name/test.parquet \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR\
    trainer.total_epochs=3 \
    reward_model.reward_manager=matchtir \
    custom_reward_function.path=$PROJECT_DIR/prog_env/reward_score/matchtir.py \
    custom_reward_function.name=compute_process_KM \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 2>&1 | tee $LOG_FILE

#gpu_memory_utilization