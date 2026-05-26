#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:?usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]}"
OUTPUT_DIR="${2:?usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]}"
DATA_NAME="${3:?usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]}"
DATA_PATH="${4:?usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]}"
BASE_MODEL_PATH="${5:?usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]}"
RECALL_SERVER="${6:?usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]}"
EVAL_SCOPE="${7:-part}"
RECALL_TIMEOUT="${8:-180}"

if [[ "${EVAL_SCOPE}" != "part" && "${EVAL_SCOPE}" != "full" ]]; then
  echo "usage: bash script/eval_one_checkpoint_verl.sh <checkpoint_path> <output_dir> <data_name> <data_path> <base_model_path> <recall_server> [part|full] [recall_timeout]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p "${OUTPUT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

echo "[eval_one_checkpoint] CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "[eval_one_checkpoint] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[eval_one_checkpoint] TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}"
echo "[eval_one_checkpoint] NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
echo "[eval_one_checkpoint] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-}"
echo "[eval_one_checkpoint] NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-}"
echo "[eval_one_checkpoint] NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}"
echo "[eval_one_checkpoint] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-}"
echo "[eval_one_checkpoint] VLLM_USE_V1=${VLLM_USE_V1:-}"
echo "[eval_one_checkpoint] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}"

EXTRA_EVAL_ARGS=()
if [[ "${DISABLE_MULTI_TURN:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--disable_multi_turn)
fi
if [[ "${DISABLE_RECURRING_HISTORY:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--disable_recurring_history)
fi
if [[ "${DISABLE_GRAPH_CANDIDATES:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--disable_graph_candidates)
fi
if [[ "${DISABLE_GRAPH_PATHS:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--disable_graph_paths)
fi
if [[ "${DISABLE_GRAPH_REASONER_INTERACTION:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--disable_graph_reasoner_interaction)
fi
if [[ "${DISABLE_CONSISTENCY_GUIDANCE:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--disable_consistency_guidance)
fi

python -m evaluation.eval_pred_verl \
  --data_file "${DATA_PATH}" \
  --dataset "${DATA_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --recall_server "${RECALL_SERVER}" \
  --recall_timeout "${RECALL_TIMEOUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_scope "${EVAL_SCOPE}" \
  --chunk_size "${CHUNK_SIZE:-1000}" \
  --max_rounds "${MAX_ROUNDS:-3}" \
  --path_block_limit "${PATH_BLOCK_LIMIT:-10}" \
  --candidate_entity_limit "${CANDIDATE_ENTITY_LIMIT:-10}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-4}" \
  --gpu_memory_rate "${GPU_MEMORY_RATE:-0.55}" \
  --prompt_style "${PROMPT_STYLE:-fixed_multiturn}" \
  "${EXTRA_EVAL_ARGS[@]}"
