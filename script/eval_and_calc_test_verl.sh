#!/usr/bin/env bash
set -euo pipefail

# ice05: http://10.210.122.163:6001/tkgr_server
# ice18: http://10.211.247.214:6001/tkgr_server
# ice14: http://10.211.255.179:6001/tkgr_server
# gdelt: http://10.211.247.215:6001/tkgr_server
DATA_NAME="${1:-ICEWS14s_divide}"
RECALL_SERVER="${2:-http://10.211.255.179:6001/tkgr_server}"
CHECKPOINT_PATH="${3:-/root/work/externalstorage/gpfsprd/OpenSourceModels/Qwen2.5-32B-Instruct}"
# CHECKPOINT_PATH="${3:-/root/work/filestorage/GroupPostTrain/lifengzhi/TKGR-LLM/multiTurn-TKG-LLM/modelscope_cache/Qwen/Qwen2.5-3B-Instruct}"
# CHECKPOINT_PATH="${3:-/root/work/filestorage/GroupPostTrain/lifengzhi/TKGR-LLM/multiTurn-TKG-LLM/modelscope_cache/Llama/Llama-3.1-8B-Instruct}"
# CHECKPOINT_PATH="${3:-/root/work/filestorage/GroupPZzostTrain/lifengzhi/TKGR-LLM/multiTurn-TKG-LLM2/checkpoints/tkgr_verl/ice05_multiturn_PPO_full_pred_fixedRound3-4.24-noKL-step200-TBS512/global_step_100/actor/huggingface}"
EXPERIMENT_NAME="${4:-ice14s_qwen2.5-32b-noGuidance}"
TEST_DATA="${5:-test_recent_h10.jsonl}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-./modelscope_cache/Qwen/Qwen2.5-32B-Instruct}"
DATA_PATH="${DATA_PATH:-./data/dataset/${DATA_NAME}/${TEST_DATA}}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/test/${EXPERIMENT_NAME}}"
EVAL_SCOPE="full"
PATH_BLOCK_LIMIT="${PATH_BLOCK_LIMIT:-10}"
CANDIDATE_ENTITY_LIMIT="${CANDIDATE_ENTITY_LIMIT:-10}"
DISABLE_MULTI_TURN="${DISABLE_MULTI_TURN:-0}"
DISABLE_RECURRING_HISTORY="${DISABLE_RECURRING_HISTORY:-0}"
DISABLE_GRAPH_CANDIDATES="${DISABLE_GRAPH_CANDIDATES:-0}"
DISABLE_GRAPH_PATHS="${DISABLE_GRAPH_PATHS:-0}"
DISABLE_GRAPH_REASONER_INTERACTION="${DISABLE_GRAPH_REASONER_INTERACTION:-0}"
DISABLE_CONSISTENCY_GUIDANCE="${DISABLE_CONSISTENCY_GUIDANCE:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

export DISABLE_MULTI_TURN
export DISABLE_RECURRING_HISTORY
export DISABLE_GRAPH_CANDIDATES
export DISABLE_GRAPH_PATHS
export DISABLE_GRAPH_REASONER_INTERACTION
export DISABLE_CONSISTENCY_GUIDANCE
export PATH_BLOCK_LIMIT
export CANDIDATE_ENTITY_LIMIT
export VLLM_WORKER_MULTIPROC_METHOD
export VLLM_DISABLE_CUSTOM_ALL_REDUCE
export VLLM_USE_V1
export VLLM_ATTENTION_BACKEND

mkdir -p "${OUTPUT_DIR}"

OUTPUT_PARENT="$(cd "$(dirname "${OUTPUT_DIR}")" && pwd)"
OUTPUT_BASENAME="$(basename "${OUTPUT_DIR}")"

run_logged_step() {
  local step_name="$1"
  shift
  echo "[eval_and_calc_test_verl] START: ${step_name}"
  set +e
  "$@"
  local status=$?
  set -e
  echo "[eval_and_calc_test_verl] DONE: ${step_name} (exit=${status})"
  return $status
}

echo "[eval_and_calc_test_verl] DATA_NAME=${DATA_NAME}"
echo "[eval_and_calc_test_verl] DATA_PATH=${DATA_PATH}"
echo "[eval_and_calc_test_verl] CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "[eval_and_calc_test_verl] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[eval_and_calc_test_verl] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[eval_and_calc_test_verl] RECALL_SERVER=${RECALL_SERVER}"
echo "[eval_and_calc_test_verl] PATH_BLOCK_LIMIT=${PATH_BLOCK_LIMIT}"
echo "[eval_and_calc_test_verl] CANDIDATE_ENTITY_LIMIT=${CANDIDATE_ENTITY_LIMIT}"
echo "[eval_and_calc_test_verl] VLLM_USE_V1=${VLLM_USE_V1}"
echo "[eval_and_calc_test_verl] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}"

eval_status=0
if run_logged_step "evaluation" bash script/eval_one_checkpoint_verl.sh \
  "${CHECKPOINT_PATH}" \
  "${OUTPUT_DIR}" \
  "${DATA_NAME}" \
  "${DATA_PATH}" \
  "${BASE_MODEL_PATH}" \
  "${RECALL_SERVER}" \
  "${EVAL_SCOPE}"; then
  eval_status=0
else
  eval_status=$?
fi

if [[ $eval_status -ne 0 ]]; then
  if [[ "${ALLOW_METRIC_ON_EVAL_FAILURE:-0}" == "1" ]]; then
    echo "[eval_and_calc_test_verl] WARN: evaluation failed (exit=${eval_status}), ALLOW_METRIC_ON_EVAL_FAILURE=1 so metric calculation will continue." >&2
  else
    echo "[eval_and_calc_test_verl] ERROR: evaluation failed (exit=${eval_status}); skip metric calculation to avoid reporting partial/stale results." >&2
    exit "${eval_status}"
  fi
fi

if [[ "${EVAL_SCOPE}" == "full" && -f "${DATA_PATH}" ]]; then
  expected_result_count="$(wc -l < "${DATA_PATH}" | tr -d '[:space:]')"
  actual_result_count="0"
  if [[ -f "${OUTPUT_DIR}/test_text.jsonl" ]]; then
    actual_result_count="$(wc -l < "${OUTPUT_DIR}/test_text.jsonl" | tr -d '[:space:]')"
  fi
  echo "[eval_and_calc_test_verl] RESULT_COUNT expected=${expected_result_count} actual=${actual_result_count}"
  if [[ "${actual_result_count}" != "${expected_result_count}" ]]; then
    echo "[eval_and_calc_test_verl] ERROR: result row count mismatch; skip metric calculation to avoid invalid metrics." >&2
    exit 1
  fi
fi

metric_status=0
if run_logged_step "metric_calc_pred_verl" python -m evaluation.metric_calc_pred_verl \
  --text_results_dir "${OUTPUT_BASENAME}" \
  --eval_dir "${OUTPUT_PARENT}" \
  --dataset "${DATA_NAME}"; then
  metric_status=0
else
  metric_status=$?
fi

if [[ ${metric_status:-0} -ne 0 ]]; then
  exit "${metric_status}"
fi

case_study_status=0
if run_logged_step "case_study_verl" python -m evaluation.case_study_verl \
  --text_results_dir "${OUTPUT_BASENAME}" \
  --eval_dir "${OUTPUT_PARENT}" \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_PATH}"; then
  case_study_status=0
else
  case_study_status=$?
fi

if [[ ${case_study_status:-0} -ne 0 ]]; then
  exit "${case_study_status}"
fi

if [[ ${eval_status:-0} -ne 0 ]]; then
  exit "${eval_status}"
fi
