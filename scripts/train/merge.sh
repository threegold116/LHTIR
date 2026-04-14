#!/bin/bash
source ~/.bashrc
source /share/home/sxjiang/miniconda3/bin/activate
conda activate verl_5_2

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/verl:$PYTHONPATH"
#------THREEGOLDCHANGE-------#
#ACTOR_DIR需是../actor
#------THREEGOLDCHANGE-------#
ACTOR_DIR="/share/home/sxjiang/myproject/LHTIR/checkpoints/qwen3-4b_ftrl_multiturn/qwen3-4b-2507_ftrl_multiturn-no_kl_no_ent-n_8-step_2048-answer_f1_recall/global_step_24/actor"
TARGET_DIR="/share/home/sxjiang/myproject/LHTIR/checkpoints/merged_checkpoints/qwen3-4b-2507_ftrl_multiturn-no_kl_no_ent-n_8-step_2048-answer_f1_recall-MATCHTIR_KM-24"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $ACTOR_DIR \
    --target_dir $TARGET_DIR