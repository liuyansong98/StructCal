# Structure Grounded Semantic Calibration with LLMs for Temporal Knowledge Graph Reasoning

## Abstract

Temporal Knowledge Graph Reasoning (TKGR) requires both structural reasoning over temporal graph patterns and semantic understanding beyond graph connectivity. Their relative importance varies across instances. Some queries are supported by reliable temporal structures, some require semantic disambiguation, and others benefit from both. However, existing methods often lack an explicit mechanism to determine when to trust structural predictions, when to revise them, and when to complement them with semantic knowledge. We propose \textbf{StructCal}, a structure-grounded semantic calibration framework with LLMs for TKGR. StructCal first employs a Temporal Subgraph Reasoner to produce structural predictions with path-level temporal evidence. It then verbalizes this evidence, together with structural candidates and recurring historical statistics, into prompts for evidence-conditioned semantic calibration. The resulting LLM feedback is used to calibrate the structural score distribution for final prediction. Experiments on benchmark TKG datasets show that StructCal improves predictive accuracy, provides interpretable structural evidence, and benefits consistently from stronger LLM backbones. Source code and data are available at \url{https://anonymous.4open.science/r/StructCal-5B4D}.

## Environment

```
python==3.10.12
numpy==1.26.4
torch==2.7.1
transformers==4.55.4
vllm==0.10.0
```

## Build GDELT26

```
py .\build_gdelt.py `
  --raw-csv '.\data\GDELT_raw\bq-results-2026.csv' `
  --cameo-csv '.\data\GDELT_raw\cameo.csv' `
  --output-dir .\data\GDELT26 `
  --start-date 20260101 `
  --end-date 20260105 `
  --date-filter-field BOTH `
  --time-field DATEADDED `
  --time-id-step 15 `
  --require-same-sqldate-and-dateadded-day
```

## Train TSR
```
cd TSR
./run_gdelt26.sh
./run_icews14.sh
./run_icews18.sh
./run_icews05-15.sh
```

## Start TSR Server
```
./script/tkgr_server_ice14.sh
./script/tkgr_server_ice18.sh
./script/tkgr_server_ice05.sh
./script/tkgr_server_gdelt.sh
```

## Data preprocessing
```
python process_data.py
```

## Evaluation

```
VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
DISABLE_MULTI_TURN=1 \
nohup bash ./script/eval_and_calc_test_verl.sh \
ICEWS14s_divide \
http://10.211.255.179:6001/tkgr_server \
/root/work/OpenSourceModels/Qwen2.5-14B-Instruct \
ice14_qwen-14b \
test_recent_h10.jsonl \
> res_ice14_qwen-14b.txt 2>&1 &

VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
DISABLE_MULTI_TURN=1 \
nohup bash ./script/eval_and_calc_test_verl.sh \
ICEWS18_divide \
http://10.211.247.214:6001/tkgr_server \
/root/work/OpenSourceModels/Qwen2.5-14B-Instruct \
ice18_qwen-14b \
test_recent_h10.jsonl \
> res_ice18_qwen-14b.txt 2>&1 &

VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
DISABLE_MULTI_TURN=1 \
nohup bash ./script/eval_and_calc_test_verl.sh \
ICEWS05_15_divide \
http://10.210.122.163:6001/tkgr_server \
/root/work/OpenSourceModels/Qwen2.5-14B-Instruct \
ice05_qwen-14b \
test_recent_h10.jsonl \
> res_ice05_qwen-14b.txt 2>&1 &

VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
DISABLE_MULTI_TURN=1 \
nohup bash ./script/eval_and_calc_test_verl.sh \
GDELT26 \
http://10.211.247.215:6001/tkgr_server \
/root/work/OpenSourceModels/Qwen2.5-14B-Instruct \
gdelt_qwen-14b \
test_recent_h10.jsonl \
> res_gdelt_qwen-14b 2>&1 &
```

