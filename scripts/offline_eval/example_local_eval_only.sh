#!/usr/bin/env bash
set -euo pipefail

PY=python

# 与生成脚本保持一致的环境与参数（严格同源）
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/root/autodl-tmp/Reasoning360:${PYTHONPATH:-}"
# export DISABLE_RAY=1  # 评测不使用 Ray，但保持与示例脚本一致的注释
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export FLASH_ATTENTION_FORCE_DISABLED=1
export HF_USE_FLASH_ATTENTION_2=0

# 设备与路径（与生成脚本保持一致，评测仅用到保存目录）
n_nodes=${n_nodes:-1}
n_gpus_per_node=${n_gpus_per_node:-1}
gpu_ids=${gpu_ids:-"0"}
export CUDA_VISIBLE_DEVICES=${gpu_ids}

SHARED_DATA_PATH=/root/autodl-tmp/data
data_folder=${SHARED_DATA_PATH}/offline_eval  # 未使用，仅为保持一致
save_folder=./evaluation_results/test_offline_leaderboard_output
model_path=/root/autodl-tmp/Qwen2.5-1.5B-Instruct
model_name=$(basename "${model_path}")

# 日志目录
mkdir -p "${save_folder}"
logs_dir="${save_folder%/}/logs"
mkdir -p "${logs_dir}"

# 评测列表与域映射（与示例脚本一致）
leaderboard_list=(
  "aime"
  "math"
  # 可按需扩展，但此脚本严格对已生成的分片进行评测
)

declare -A domain_mappings
domain_mappings["aime"]="math"
domain_mappings["math"]="math"
domain_mappings["humaneval"]="codegen"
domain_mappings["livecodebench"]="codegen"
domain_mappings["mbpp"]="codegen"
domain_mappings["arcagi1"]="logic"
domain_mappings["zebra_puzzle_dataset"]="logic"
domain_mappings["finqa"]="table"
domain_mappings["hitab"]="table"
domain_mappings["multihier"]="table"
domain_mappings["codeio"]="simulation"
domain_mappings["cruxeval-i"]="simulation"
domain_mappings["cruxeval-o"]="simulation"
domain_mappings["gpqa_diamond"]="stem"
domain_mappings["supergpqa"]="stem"
domain_mappings["livebench_reasoning"]="ood"
domain_mappings["livebench_language"]="ood"
domain_mappings["livebench_data_analysis"]="ood"
domain_mappings["ifeval"]="ood"

# 生成参数（为严格保持一致而保留；评测不使用）
batch_size=${batch_size:-16}
temperature=${temperature:-1.0}
top_p=${top_p:-0.7}
use_vllm=${use_vllm:-0}

if [[ "${use_vllm}" -eq 1 ]]; then
  top_k=-1
  tensor_model_parallel_size=${tensor_model_parallel_size:-${n_gpus_per_node}}
  gpu_memory_utilization=${gpu_memory_utilization:-0.9}
  rollout_name=vllm
else
  top_k=0
  tensor_model_parallel_size=${tensor_model_parallel_size:-1}
  gpu_memory_utilization=${gpu_memory_utilization:-0.9}
  rollout_name=hf
fi

# 仅评测：遍历已生成的分片并调用 main_eval
for leaderboard in "${leaderboard_list[@]}"; do
  domain=${domain_mappings[$leaderboard]}

  # 文件匹配模式与示例脚本一致
  if [[ "${leaderboard}" == "aime" || "${leaderboard}" == "aime2025" ]]; then
    file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
  else
    file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
  fi

  eval_log_file="${logs_dir}/${model_name}_${leaderboard}_eval_only.log"

  # 在保存目录下查找已生成文件
  target_folder="${save_folder}/${model_name}"
  mapfile -t matched_files < <(find "${target_folder}" -type f -name "${file_pattern}" | sort)

  if [[ ${#matched_files[@]} -eq 0 ]]; then
    echo "No generated file found for pattern: ${file_pattern} in ${target_folder}. Skipping." | tee -a "${eval_log_file}"
    continue
  fi

  for gen_file in "${matched_files[@]}"; do
    echo "Starting evaluation for ${leaderboard}: ${gen_file}" | tee -a "${eval_log_file}"
    unset LD_LIBRARY_PATH
    {
      ${PY} -m verl.trainer.main_eval \
        data.path="${gen_file}" \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model
    } 2>&1 | tee -a "${eval_log_file}"
    echo "Completed evaluation for ${leaderboard}: ${gen_file}" | tee -a "${eval_log_file}"
  done

  echo "Completed processing ${leaderboard}. Evaluation log: ${eval_log_file}"
done

echo "All evaluations finished. Logs in: ${logs_dir}"