# GDELT26

This dataset was converted from GDELT raw events into the ICEWS-style temporal KG format used in this repository.

## Raw Source
- Raw CSV: D:\科研\知识图谱\时序知识图谱\code_LLM_TKGR\Build_dataset\data\GDELT_raw\bq-results-2026.csv
- CAMEO mapping: D:\科研\知识图谱\时序知识图谱\code_LLM_TKGR\Build_dataset\data\GDELT_raw\cameo.csv
- Date filter field: BOTH
- Date filter range: 20260101 to 20260105
- Require DATEADDED[:8] == SQLDATE: True
- head = Actor1Name, fallback to Actor1Code when Actor1Name is empty
- tail = Actor2Name, fallback to Actor2Code when Actor2Name is empty
- relation = CAMEO description (without the leading EventBaseCode)
- time = DATEADDED
- Split rule: chronological 80% train / 10% valid / 10% test
- ts2id step = 15

## Coverage
- First retained date: 2026-01-01 00:00:00
- Last retained date: 2026-01-05 23:45:00
- Dropped valid events with unseen train entities/relations: 0
- Dropped test events with unseen train entities/relations: 0
