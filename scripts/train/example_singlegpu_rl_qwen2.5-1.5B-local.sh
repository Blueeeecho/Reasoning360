#!/bin/bash

# =================== User-Configurable Settings ===================
# --- Execution Environment ---
NUM_GPUS=1  # 单机单卡训练

# --- Resuming & Logging ---
RESUME_CKPT_DIR_NAME=""  # 可填入 W&B 实验名以便断点续训，否则留空
WANDB_PROJECT="Reasoning360" # 你的 wandb 项目名

# --- External Services ---
export STEM_LLM_JUDGE_URL="<STEM_LLM_JUDGE_URL>"  # 可选：填写 llm-as-judge 的 URL 以启用 STEM 域评估

# =================== Environment Setup ===================
export NCCL_DEBUG=info
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1 # 调试 CUDA 错误时可开启

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# =================== 设置所有数据的路径 ===================
SHARED_DATA_PATH=/root/autodl-tmp/data
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/   # 训练数据集
TEST_DATA_DIR=${SHARED_DATA_PATH}/online_eval/  # 测试数据集

# Math (train)
math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

# Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__livecodebench_440.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__taco_8.8k.parquet
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

# Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_1.2k.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k.parquet
# Logic (test)
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_100.parquet
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet
arcagi_test_path=${TEST_DATA_DIR}/logic__arcagi1_200.parquet

# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_3.7k.parquet
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_200.parquet

# Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_200.parquet

# Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k.parquet
# Stem (test)
supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_200.parquet

train_files="['${math_train_path}']"  # 以数学为示例，按需扩充
test_files="['${math_test_path}','${aime_test_path}']"  # 以数学为示例，按需扩充

# =================== Model ===================
# 适配本地模型路径：将 BASE_MODEL 改为你本地权重目录
# 例如：/root/autodl-tmp/models/Qwen2.5-7B 或已缓存的本地 HF 路径
BASE_MODEL="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"  # 请改为你的本地模型路径

# =================== Logging ===================
# 若断点续训，直接使用 RESUME_CKPT_DIR_NAME；否则生成唯一实验名
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME="$RESUME_CKPT_DIR_NAME"
else
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    WANDB_EXPERIMENT_NAME="single-gpu-${TIMESTAMP}-${BASE_MODEL##*/}"
fi

# =================== Local Mode ===================
echo "Single-GPU local training; skipping Ray start/stop."

# =================== RL Config ===================
# 继承原配置，调整为单卡+HF 本地 rollout

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512  # 按需调整，单卡下可能需减小
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=16
train_prompt_mini_bsz=64

# Algorithm
temperature=1.0
top_p=1.0
top_k=0  # HF rollout 使用 0；vLLM 使用 -1

# Training config
sp_size=1
gen_tp=1  # 单卡下张量并行为 1
gen_max_num_seqs=1024
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
offload=True

# =================== Start RL training ===================
echo "Starting single-GPU training with HF local rollout..."
python -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name="dapo_fsdp_config.yaml" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_seqs=${gen_max_num_seqs} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.mode="sync" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_multi_process \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.log_val_generations=50 \
    trainer.resume_mode=auto