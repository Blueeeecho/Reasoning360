#!/usr/bin/env bash
set -euo pipefail

PY=python

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/root/autodl-tmp/Reasoning360:${PYTHONPATH:-}"
# export DISABLE_RAY=1  # 移除/注释掉该行以启用 Ray
export PYTHONUNBUFFERED=1  # 让 Python 实时输出，配合 tee 实时写日志
export HYDRA_FULL_ERROR=1  # 如遇错误，打印完整堆栈，便于定位

export FLASH_ATTENTION_FORCE_DISABLED=1
export HF_USE_FLASH_ATTENTION_2=0

# =================== 设备与路径（可按需修改） ===================
n_nodes=${n_nodes:-1}
n_gpus_per_node=${n_gpus_per_node:-1}   # 示例默认 2，可改为卡数（当前为 1 则设 1）
gpu_ids=${gpu_ids:-"0"}               # 逗号分隔的 GPU ID 列表；当前单卡可改为 "0"
export CUDA_VISIBLE_DEVICES=${gpu_ids}

SHARED_DATA_PATH=/root/autodl-tmp/data
data_folder=${SHARED_DATA_PATH}/test
save_folder=./evaluation_results/test_offline_leaderboard_output
model_path=/root/autodl-tmp/guru-7B
model_name=$(basename "${model_path}")

# 目录准备
mkdir -p "${save_folder}"
logs_dir="${save_folder%/}/logs"
mkdir -p "${logs_dir}"

# =================== 自动启动本地 Ray（可选） ===================
if command -v ray >/dev/null 2>&1; then
  if ! ray status >/dev/null 2>&1; then
    echo "Ray 未运行，尝试启动本地 head 节点..."
    # 启动本地 Ray；如果已运行则忽略错误
    ray start --head --num-cpus "$(nproc)" || true
  fi
else
  echo "未找到 ray 命令，请先安装 Ray（pip install ray）。" >&2
fi

# =================== (Offline) Leaderboard Eval Config ===================
leaderboard_list=(
  "aime"
  "math"
  # 如需更多任务，按需取消注释：
  # "mbpp" "humaneval" "livecodebench" "arcagi1" "zebra_puzzle_dataset"
  # "gpqa_diamond" "supergpqa" "finqa" "hitab" "multihier" "codeio"
  # "cruxeval-i" "cruxeval-o" "livebench_reasoning" "livebench_language"
  # "livebench_data_analysis" "ifeval"
)

# 域映射
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

# =================== 生成参数（Ray + vLLM 友好配置） ===================
batch_size=${batch_size:-16}
temperature=${temperature:-1.0}
top_p=${top_p:-0.7}
use_vllm=${use_vllm:-0}  # 1: 使用 vLLM（Ray友好，支持并行），0: 使用 HF（单机更保守）

if [[ "${use_vllm}" -eq 1 ]]; then
  top_k=-1                     # -1 表示 vLLM rollout
  tensor_model_parallel_size=${tensor_model_parallel_size:-${n_gpus_per_node}}
  gpu_memory_utilization=${gpu_memory_utilization:-0.9}
  rollout_name=vllm
else
  top_k=0                      # 0 表示 HF rollout
  tensor_model_parallel_size=${tensor_model_parallel_size:-1}
  gpu_memory_utilization=${gpu_memory_utilization:-0.9}
  rollout_name=hf
fi

for leaderboard in "${leaderboard_list[@]}"; do
  domain=${domain_mappings[$leaderboard]}

  # 采样倍数
  if [[ "${leaderboard}" == "aime" || "${leaderboard}" == "aime2025" ]]; then
    n_samples=4
  elif [[ "${leaderboard}" == "arcagi1" || "${leaderboard}" == "livecodebench" || "${leaderboard}" == "humaneval" || "${leaderboard}" == "zebra_puzzle_dataset" || "${leaderboard}" == "multihier" || "${leaderboard}" == "codeio" || "${leaderboard}" == "gpqa_diamond" ]]; then
    n_samples=4
  else
    n_samples=1
  fi

  # 长度
  if [[ "${leaderboard}" == "arcagi1" || "${leaderboard}" == "multihier" ]]; then
    prompt_length=4096
    response_length=4096
  else
    prompt_length=2048
    response_length=4096
  fi

gen_log_file="${logs_dir}/${model_name}_${leaderboard}_gen.log"
eval_log_file="${logs_dir}/${model_name}_${leaderboard}_eval.log"

  # 数据文件模式
  if [[ "${leaderboard}" == "aime" || "${leaderboard}" == "aime2025" ]]; then
    file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
  else
    file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
  fi

  # 遍历该任务下的全部分片（不再只取首个）
  mapfile -t matched_files < <(find "${data_folder}" -type f -name "${file_pattern}" | sort)
  if [[ ${#matched_files[@]} -eq 0 ]]; then
    echo "No file found matching pattern: ${file_pattern}. Skipping." | tee -a "${gen_log_file}"
    continue
  fi

  for data_file in "${matched_files[@]}"; do
    file_name=$(basename "${data_file}")
    save_path="${save_folder}/${model_name}/${file_name}"
    mkdir -p "$(dirname "${save_path}")"

    echo "Processing ${leaderboard}: ${data_file} -> ${save_path}" | tee -a "${gen_log_file}"

    # ===== Generate =====
    echo "Starting generation for ${leaderboard} at $(date)" | tee -a "${gen_log_file}"
    {
      ${PY} -m verl.trainer.main_generation \
        trainer.nnodes="${n_nodes}" \
        trainer.n_gpus_per_node="${n_gpus_per_node}" \
        model.path="${model_path}" \
        +model.trust_remote_code=True \
        +model.attn_implementation=eager \
        +model.use_flash_attention_2=false \
        data.path="${data_file}" \
        data.prompt_key=prompt \
        data.n_samples="${n_samples}" \
        data.batch_size="${batch_size}" \
        data.output_path="${save_path}" \
        rollout.name="${rollout_name}" \
        rollout.do_sample=True \
        rollout.temperature="${temperature}" \
        rollout.top_k="${top_k}" \
        rollout.top_p="${top_p}" \
        rollout.prompt_length="${prompt_length}" \
        rollout.response_length="${response_length}" \
        rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
        rollout.tensor_model_parallel_size="${tensor_model_parallel_size}"
    } 2>&1 | tee -a "${gen_log_file}"
    echo "Completed generation for ${leaderboard} at $(date)" | tee -a "${gen_log_file}"

    # ===== Evaluate =====
    echo "Starting evaluation for ${leaderboard} at $(date)" | tee -a "${eval_log_file}"
    unset LD_LIBRARY_PATH
    {
      ${PY} -m verl.trainer.main_eval \
        data.path="${save_path}" \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model
    } 2>&1 | tee -a "${eval_log_file}"
    echo "Completed evaluation for ${leaderboard} at $(date)" | tee -a "${eval_log_file}"
  done

  echo "Completed processing ${leaderboard}. Generation log: ${gen_log_file}, Evaluation log: ${eval_log_file}"
done

echo "All tasks finished. Logs in: ${logs_dir}"