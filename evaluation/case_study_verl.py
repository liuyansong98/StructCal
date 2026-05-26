import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


SETTINGS_PATH = os.path.dirname(__file__)
DATA_ROOT = os.path.join(SETTINGS_PATH, "../data/dataset")


@dataclass
class AnswerScore:
    answer: str
    score: float
    line_no: int


def _normalize_entity_for_match(name: str) -> str:
    return name.strip().strip("\"'`").strip("\u201c\u201d\u2018\u2019").lower()


def _build_normalized_entity2id(entity2id):
    normalized = {}
    for name, entity_id in entity2id.items():
        norm_name = _normalize_entity_for_match(str(name))
        if norm_name and norm_name not in normalized:
            normalized[norm_name] = int(entity_id)
    return normalized


def load_name_mapping(filepath, num_relations, is_rel=False):
    with open(filepath, "r", encoding="utf-8-sig") as file:
        name2id = json.load(file)
    if is_rel:
        inv_name2id = {}
        for key, value in name2id.items():
            inv_name2id["INV::" + key] = int(value) + num_relations
        name2id.update(inv_name2id)
    return name2id


def load_id_mapping(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as file:
        id2name = json.load(file)
    return {int(k): v for k, v in id2name.items()}


def read_jsonl(result_dir):
    rows = []
    for file in os.listdir(result_dir):
        if not file.endswith(".jsonl"):
            continue
        file_path = os.path.join(result_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def parse_top10(text: str) -> List[AnswerScore]:
    pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$")
    results: List[AnswerScore] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        results.append(
            AnswerScore(
                answer=match.group(2).strip(),
                score=float(match.group(3).strip()),
                line_no=int(match.group(1)),
            )
        )
    by_idx = {item.line_no: item for item in results}
    return [by_idx[i] for i in sorted(by_idx.keys())]


def rank_answers(items: List[AnswerScore]) -> List[AnswerScore]:
    return sorted(items, key=lambda x: (-x.score, x.line_no))


def has_recurring_history(history: str) -> bool:
    history = (history or "").strip()
    if not history:
        return False
    if "<recurring_entity_stats>" not in history:
        return False
    block_match = re.search(r"<recurring_entity_stats>\s*(.*?)\s*</recurring_entity_stats>", history, flags=re.DOTALL)
    if block_match is None:
        return False
    block = block_match.group(1).strip()
    if not block:
        return False
    lowered = block.lower()
    if lowered in {"none", "null", "n/a", "na"}:
        return False
    return bool(re.search(r"^\s*\d+\.", block, flags=re.MULTILINE))


def fused_top10_entities(sample, entity2id_norm, id2entity):
    score_vec = sample.get("time_aware_score")
    if score_vec is None:
        return []
    fused = np.asarray(score_vec, dtype=np.float32).copy()
    pred_text = sample.get("pred_ans") or ""
    for answer in rank_answers(parse_top10(pred_text)):
        entity_id = entity2id_norm.get(_normalize_entity_for_match(answer.answer))
        if entity_id is not None:
            fused[entity_id] += float(answer.score)

    top_k = min(10, fused.shape[0])
    if top_k <= 0:
        return []
    top_indices = np.argpartition(-fused, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-fused[top_indices], kind="stable")]
    return [id2entity.get(int(entity_id), str(int(entity_id))) for entity_id in top_indices]


def category_of(sample) -> Optional[str]:
    first_rank = float(sample.get("first_rank", -1))
    final_rank = float(sample.get("final_ranked", -1))
    structural_confidence_prob = sample.get("structural_confidence_prob")
    history_flag = has_recurring_history(sample.get("history", ""))

    if final_rank != 1:
        return None
    try:
        structural_confidence_prob = float(structural_confidence_prob) if structural_confidence_prob is not None else None
    except (TypeError, ValueError):
        structural_confidence_prob = None
    if structural_confidence_prob is not None and structural_confidence_prob < 0.5 and first_rank > 10:
        return "Case 5: LLM structural confidence<0.5, Graph Reasoner rank>10, final rank==1"
    if first_rank == 1 and not history_flag:
        return "Case 1: Graph Reasoner rank==1, 无历史信息, 最终 rank==1"
    if first_rank == 1 and history_flag:
        return "Case 2: Graph Reasoner rank==1, 有历史信息, 最终 rank==1"
    if first_rank > 3 and not history_flag:
        return "Case 3: Graph Reasoner rank>3, 无历史信息, 最终 rank==1"
    if first_rank > 3 and history_flag:
        return "Case 4: Graph Reasoner rank>3, 有历史信息, 最终 rank==1"
    return None


def group_cases(rows):
    buckets = {
        "Case 1: Graph Reasoner rank==1, 无历史信息, 最终 rank==1": [],
        "Case 2: Graph Reasoner rank==1, 有历史信息, 最终 rank==1": [],
        "Case 3: Graph Reasoner rank>3, 无历史信息, 最终 rank==1": [],
        "Case 4: Graph Reasoner rank>3, 有历史信息, 最终 rank==1": [],
        "Case 5: LLM structural confidence<0.5, Graph Reasoner rank>10, final rank==1": [],
    }
    for row in rows:
        category = category_of(row)
        if category is not None:
            buckets[category].append(row)
    for category, items in buckets.items():
        if category.startswith("Case 1") or category.startswith("Case 2"):
            buckets[category] = sorted(items, key=lambda x: (float(x.get("first_rank", 1e9)), str(x.get("idx", ""))))
        else:
            buckets[category] = sorted(items, key=lambda x: (-float(x.get("first_rank", -1)), str(x.get("idx", ""))))
    return buckets


def write_case_study(output_path, grouped_cases, entity2id_norm, id2entity):
    with open(output_path, "w", encoding="utf-8") as f:
        for category, samples in grouped_cases.items():
            f.write("=" * 100 + "\n")
            f.write(category + "\n")
            f.write("=" * 100 + "\n")
            f.write(f"Matched cases: {len(samples)}\n\n")
            if not samples:
                f.write("No matched case found.\n\n")
                continue

            for sample_idx, sample in enumerate(samples, start=1):
                fused_top10 = fused_top10_entities(sample, entity2id_norm, id2entity)
                f.write("-" * 100 + "\n")
                f.write(f"Sample #{sample_idx}\n")
                f.write("-" * 100 + "\n")
                f.write(f"case_type: {category}\n")
                f.write(f"idx: {sample.get('idx', '')}\n")
                f.write(f"query: {sample.get('query', '')}\n")
                f.write(f"target: {sample.get('target', '')}\n")
                f.write(f"graph_reasoner_rank: {sample.get('first_rank', '')}\n")
                f.write(f"final_rank: {sample.get('final_ranked', '')}\n")
                f.write(f"llm_structural_confidence_score: {sample.get('structural_confidence_score', '')}\n")
                f.write(f"llm_structural_confidence_prob: {sample.get('structural_confidence_prob', '')}\n")
                f.write(f"has_recurring_history: {has_recurring_history(sample.get('history', ''))}\n\n")

                f.write("Graph Reasoner Path list:\n")
                first_paths = sample.get("first_path_list") or []
                if first_paths:
                    for item in first_paths:
                        f.write(f"- {item}\n")
                else:
                    f.write("None\n")
                f.write("\n")

                f.write("Graph Reasoner prediction candidate:\n")
                candidates = sample.get("first_candidate_entity_list") or sample.get("top10_entity_names") or []
                if candidates:
                    for i, item in enumerate(candidates, start=1):
                        f.write(f"{i}. {item}\n")
                else:
                    f.write("None\n")
                f.write("\n")

                f.write("Query history:\n")
                f.write((sample.get("history") or "None").strip() + "\n\n")

                f.write("LLM predicted entity list:\n")
                f.write((sample.get("pred_ans") or "None").strip() + "\n\n")

                f.write("LLM+Graph Reasoner final top 10 entity:\n")
                if fused_top10:
                    for i, item in enumerate(fused_top10, start=1):
                        f.write(f"{i}. {item}\n")
                else:
                    f.write("None\n")
                f.write("\n\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_results_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = os.path.join(args.eval_dir, args.text_results_dir)
    rows = read_jsonl(result_dir)

    data_path = os.path.join(DATA_ROOT, args.dataset)
    entity2id = load_name_mapping(os.path.join(data_path, "entity2id.json"), 0)
    entity2id_norm = _build_normalized_entity2id(entity2id)
    id2entity = load_id_mapping(os.path.join(data_path, "id2entity.json"))

    grouped_cases = group_cases(rows)
    output_dir = os.path.dirname(os.path.abspath(args.data_file))
    output_path = os.path.join(output_dir, "case_study.txt")
    write_case_study(output_path, grouped_cases, entity2id_norm, id2entity)
    print(f"Case study saved to: {output_path}")


if __name__ == "__main__":
    main()
