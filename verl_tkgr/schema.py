"""Shared schemas and I/O helpers for the verl_tkgr migration layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TKGRBatchResult:
    idx_list: List[str]
    path_list: List[List[str]]
    rank_list: List[float]
    entity_num: int
    time_aware_score: List[List[float]]
    top10_entity_names: List[List[str]]

    def __len__(self) -> int:
        return len(self.idx_list)


@dataclass
class RolloutTrace:
    idx: str
    query: str
    target: Optional[str]
    history: str
    round_count: int
    recall_num: int
    finished: bool
    stop_reason: str
    final_text: str
    pred_block: str
    selected_path_history: List[str] = field(default_factory=list)
    initial_time_aware_score: Optional[List[float]] = None
    time_aware_score: Optional[List[float]] = None
    top10_entity_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RewardRecord:
    idx: str
    recall_num: int
    rewards: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingBatch:
    traces: List[RolloutTrace]
    reward_records: List[RewardRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces": [trace.to_dict() for trace in self.traces],
            "reward_records": [record.to_dict() for record in self.reward_records],
        }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
