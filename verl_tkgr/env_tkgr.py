"""TKGR environment wrapper for the future verl-based rollout pipeline."""

from __future__ import annotations
from typing import List, Optional
import requests
from verl_tkgr.schema import TKGRBatchResult


class TKGREnvironment:
    """Thin HTTP wrapper around the Graph Reasoner service."""

    def __init__(self, recall_server: str, timeout: float = 180.0):
        self.recall_server = recall_server.rstrip("/")
        self.timeout = timeout

    def recall(
        self,
        idx_list: List[str],
        selected_path_list: Optional[List[str]] = None,
    ) -> TKGRBatchResult:
        if selected_path_list is None:
            selected_path_list = [""] * len(idx_list)

        if len(idx_list) != len(selected_path_list):
            raise ValueError(
                f"idx_list and selected_path_list must have the same length, got "
                f"{len(idx_list)} and {len(selected_path_list)}."
            )

        response = requests.post(
            self.recall_server,
            json={
                "idx_list": idx_list,
                "selected_path_list": selected_path_list,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        result = TKGRBatchResult(
            idx_list=[str(idx) for idx in idx_list],
            path_list=payload["path_list"],
            rank_list=payload["rank_list"],
            entity_num=int(payload["entity_num"]),
            time_aware_score=payload["time_aware_score"],
            top10_entity_names=payload["top10_entity_names"],
        )
        self._validate_batch_result(result)
        return result

    @staticmethod
    def _validate_batch_result(result: TKGRBatchResult) -> None:
        batch_size = len(result.idx_list)
        fields = {
            "path_list": len(result.path_list),
            "rank_list": len(result.rank_list),
            "time_aware_score": len(result.time_aware_score),
            "top10_entity_names": len(result.top10_entity_names),
        }
        for field_name, field_len in fields.items():
            if field_len != batch_size:
                raise ValueError(
                    f"Graph Reasoner response field `{field_name}` has length {field_len}, "
                    f"expected {batch_size}."
                )
