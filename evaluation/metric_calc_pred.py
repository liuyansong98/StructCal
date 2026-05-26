
import argparse
import json
import math
import os
import pdb
import sys
import numpy as np
import re
import string
from collections import Counter, defaultdict, OrderedDict
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple
# import pandas as pd


def load_test_npy(filename):
    # 读取文件
    return np.load(filename, allow_pickle=True)

def read_jsonl(text_results_dir):
    preferred_file = os.path.join(text_results_dir, "test_text.jsonl")
    if os.path.isfile(preferred_file):
        files = ["test_text.jsonl"]
    else:
        files = sorted(os.listdir(text_results_dir))
        jsonl_files = [file for file in files if file.endswith(".jsonl")]
        if len(jsonl_files) > 1:
            print(
                "WARNING: test_text.jsonl not found; multiple jsonl files will be merged: "
                f"{jsonl_files}"
            )
        files = jsonl_files
    data = []
    for file in files:
        print(file)
        if file.endswith(".jsonl"):
            file_path = os.path.join(text_results_dir, file)
        else:
            continue
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


@dataclass
class AnswerScore:
    answer: str
    score: float
    line_no: int  # 原始行号(1-10)，用于同分稳定排序


def parse_top10(text: str) -> List[AnswerScore]:
    """
    解析形如：
    1. [answer entity 1]: [score (1-10)]
    ...
    返回 AnswerScore 列表（长度可能 <10，如果文本不完整）。
    """
    AnswerScore_PATTERN = re.compile(
        r'^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$'
    )
    results: List[AnswerScore] = []
    for i, line in enumerate(text.splitlines()):
        m = AnswerScore_PATTERN.match(line)
        if not m:
            continue
        i = int(m.group(1))
        ans = m.group(2).strip()
        score = float(m.group(3).strip())
        results.append(AnswerScore(answer=ans, score=score, line_no=i))

    # 可选：按 idx 去重（如果输入里重复了某一行号，保留最后一个）
    by_idx = {}
    for r in results:
        by_idx[r.line_no] = r
    results = [by_idx[i] for i in sorted(by_idx.keys())]

    return results

def rank_answers(items: List[AnswerScore]) -> List[AnswerScore]:
    """
    按 score 降序排序。
    同分时用 line_no 升序保证稳定（即原列表靠前的排名更高）。
    """
    return sorted(items, key=lambda x: (-x.score, x.line_no))


def _normalize_entity_for_match(name: str) -> str:
    return str(name).strip().strip("\"'`").strip("\u201c\u201d\u2018\u2019").lower()


def find_entity_rank(text: str, entity: str) -> Tuple[
    bool, Optional[int], List[AnswerScore]]:
    """
    返回：是否存在、排名(1-based, 按score)、排序后的完整榜单
    """
    items = parse_top10(text)
    ranked = rank_answers(items)
    entity_norm = _normalize_entity_for_match(entity)

    for i, r in enumerate(ranked, start=1):
        if _normalize_entity_for_match(r.answer) == entity_norm:
            return True, i, ranked

    return False, None, ranked

def load_name_mapping(filepath, num_relations, is_rel=False):
    with open(filepath, 'r') as file:
        name2id = json.load(file)
    if is_rel:
        inv_name2id = {}
        for key, value in name2id.items():
            inv_name2id["INV::" + key] = int(value) + num_relations
        name2id.update(inv_name2id)
    return name2id


def stat_ranks(total_rank, method):  # added  eval_paper_authors log for logging
    total_rank = np.array(total_rank).astype(float)
    valid_rank = total_rank[total_rank >= 1]
    if valid_rank.size == 0:
        print("MRR ({}): N/A".format(method))
        for hit in [1, 3, 10]:
            print("Hits ({}) @ {}: N/A".format(method, hit))
        return
    hits = [1, 3, 10]
    mrr = np.mean(1.0 / valid_rank)
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = np.mean((valid_rank <= hit).astype(float))
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))


def print_prediction_metrics(metric_list, reciprocal_rank_sum, total_data_num, error_count):
    if total_data_num == 0:
        print("MRR: N/A")
        for hit in [1, 3, 10]:
            print(f"hit@{hit}: N/A")
        return
    hits = [1, 3, 10]
    print(f"MRR: {reciprocal_rank_sum / total_data_num}")
    for i, hit in enumerate(hits):
        print(f"hit@{hit}: {metric_list[i]/total_data_num}")

    valid_pred_count = total_data_num - error_count
    if valid_pred_count != 0:
        print(f"(have pred block) MRR: {reciprocal_rank_sum / valid_pred_count}")
        for i, hit in enumerate(hits):
            print(f"(have pred block) hit@{hit}: {metric_list[i]/valid_pred_count}")


def print_result_integrity_diagnostics(data):
    total = len(data)
    required_keys = ["idx", "target", "pred_ans", "final_ranked", "first_rank", "last_rank"]
    missing_key_counts = Counter()
    idx_counter = Counter()
    invalid_idx_count = 0

    for row in data:
        for key in required_keys:
            if key not in row:
                missing_key_counts[key] += 1
        try:
            idx_counter[str(row.get("idx"))] += 1
        except Exception:
            invalid_idx_count += 1

    duplicate_idx_count = sum(count - 1 for count in idx_counter.values() if count > 1)
    print("Result integrity diagnostics:")
    print(f"  total_records: {total}")
    print(f"  unique_idx_count: {len(idx_counter)}")
    print(f"  duplicate_idx_extra_count: {duplicate_idx_count}")
    print(f"  invalid_idx_count: {invalid_idx_count}")
    print(f"  missing_required_key_counts: {dict(missing_key_counts)}")


def print_prediction_parse_diagnostics(data):
    total = len(data)
    if total == 0:
        print("Prediction parse diagnostics: N/A (no records)")
        return

    stop_reason_counter = Counter(str(d.get("stop_reason_final", "missing")) for d in data)
    parsed_line_counts = []
    missing_pred_block = 0
    empty_pred_ans = 0
    for d in data:
        pred_answer = str(d.get("pred_ans", "") or "")
        if pred_answer == "I don't know.":
            missing_pred_block += 1
        if not pred_answer.strip():
            empty_pred_ans += 1
        parsed_line_counts.append(len(parse_top10(pred_answer)))

    parsed_line_counts_np = np.asarray(parsed_line_counts, dtype=float)
    print("Prediction parse diagnostics:")
    print(f"  total_records: {total}")
    print(f"  stop_reason_final_counts: {dict(stop_reason_counter)}")
    print(f"  no_prediction_block_count: {missing_pred_block}")
    print(f"  empty_pred_ans_count: {empty_pred_ans}")
    print(f"  avg_parsed_prediction_lines: {float(np.mean(parsed_line_counts_np)):.6f}")
    print(f"  parsed_10_lines_count: {int(np.sum(parsed_line_counts_np >= 10))}")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_binary_auroc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(probs)
    rank_sum_pos = float(np.sum(ranks[pos_mask]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return auc


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Optional[float]:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return None
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    ece = 0.0
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(labels[mask]))
        bin_conf = float(np.mean(probs[mask]))
        ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return ece


def compute_brier_score(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return None
    return float(np.mean((probs - labels) ** 2))


def compute_mae(labels: np.ndarray, preds: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if len(labels) == 0:
        return None
    return float(np.mean(np.abs(preds - labels)))


def compute_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return None
    x_rank = _rankdata_average(x)
    y_rank = _rankdata_average(y)
    return compute_pearson(x_rank, y_rank)


def compute_soft_ece(targets: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Optional[float]:
    targets = np.asarray(targets, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(targets) == 0:
        return None
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(targets)
    ece = 0.0
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        bin_target = float(np.mean(targets[mask]))
        bin_conf = float(np.mean(probs[mask]))
        ece += (np.sum(mask) / total) * abs(bin_target - bin_conf)
    return ece


def _collect_structural_confidence_pairs(data, rank_threshold: int):
    probs = []
    labels = []
    for d in data:
        prob = d.get("structural_confidence_prob")
        rank = d.get("first_rank")
        if prob is None or rank is None:
            continue
        try:
            prob_f = float(prob)
            rank_f = float(rank)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= prob_f <= 1.0) or rank_f < 1:
            continue
        label_i = int(rank_f <= rank_threshold)
        probs.append(prob_f)
        labels.append(label_i)
    return probs, labels


def _collect_structural_confidence_soft_targets(data):
    probs = []
    targets = []
    for d in data:
        prob = d.get("structural_confidence_prob")
        rank = d.get("first_rank")
        if prob is None:
            continue
        try:
            prob_f = float(prob)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= prob_f <= 1.0):
            continue
        try:
            rank_i = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_i = None
        target = 0.0 if rank_i is None or rank_i <= 0 else 1.0 / float(rank_i)
        probs.append(prob_f)
        targets.append(target)
    return probs, targets


def _print_structural_confidence_metric_set(data, rank_threshold: int, label_name: str):
    probs, labels = _collect_structural_confidence_pairs(data, rank_threshold)
    if not probs:
        print(f"Structural confidence metrics ({label_name}): N/A (no valid Structural_Confidence records)")
        return

    probs_np = np.asarray(probs, dtype=float)
    labels_np = np.asarray(labels, dtype=int)
    auroc = compute_binary_auroc(labels_np, probs_np)
    ece = compute_ece(labels_np, probs_np, n_bins=10)
    brier = compute_brier_score(labels_np, probs_np)

    print(f"Structural confidence sample count ({label_name}): {len(probs_np)}")
    if auroc is None:
        print(f"Structural confidence AUROC ({label_name}): N/A")
    else:
        print(f"Structural confidence AUROC ({label_name}): {auroc:.6f}")
    if ece is None:
        print(f"Structural confidence ECE ({label_name}): N/A")
    else:
        print(f"Structural confidence ECE ({label_name}): {ece:.6f}")
    if brier is None:
        print(f"Structural confidence Brier score ({label_name}): N/A")
    else:
        print(f"Structural confidence Brier score ({label_name}): {brier:.6f}")


def print_structural_confidence_metrics(data):
    _print_structural_confidence_metric_set(data, 1, "top1")
    _print_structural_confidence_metric_set(data, 3, "top3")
    _print_structural_confidence_metric_set(data, 10, "top10")
    probs, soft_targets = _collect_structural_confidence_soft_targets(data)
    if not probs:
        print("Structural confidence soft metrics (target=1/rank): N/A (no valid Structural_Confidence records)")
        return

    probs_np = np.asarray(probs, dtype=float)
    soft_targets_np = np.asarray(soft_targets, dtype=float)
    soft_ece = compute_soft_ece(soft_targets_np, probs_np, n_bins=10)
    soft_brier = compute_brier_score(soft_targets_np, probs_np)
    mse = compute_brier_score(soft_targets_np, probs_np)
    mae = compute_mae(soft_targets_np, probs_np)
    pearson = compute_pearson(probs_np, soft_targets_np)
    spearman = compute_spearman(probs_np, soft_targets_np)

    print(f"Structural confidence soft sample count (target=1/rank): {len(probs_np)}")
    if soft_ece is None:
        print("Structural confidence soft-ECE (target=1/rank): N/A")
    else:
        print(f"Structural confidence soft-ECE (target=1/rank): {soft_ece:.6f}")
    if soft_brier is None:
        print("Structural confidence soft-Brier (target=1/rank): N/A")
    else:
        print(f"Structural confidence soft-Brier (target=1/rank): {soft_brier:.6f}")
    if mse is None:
        print("Structural confidence MSE (target=1/rank): N/A")
    else:
        print(f"Structural confidence MSE (target=1/rank): {mse:.6f}")
    if mae is None:
        print("Structural confidence MAE (target=1/rank): N/A")
    else:
        print(f"Structural confidence MAE (target=1/rank): {mae:.6f}")
    if pearson is None:
        print("Structural confidence Pearson (target=1/rank): N/A")
    else:
        print(f"Structural confidence Pearson (target=1/rank): {pearson:.6f}")
    if spearman is None:
        print("Structural confidence Spearman (target=1/rank): N/A")
    else:
        print(f"Structural confidence Spearman (target=1/rank): {spearman:.6f}")


def eval_(args):
    settings_path = os.path.dirname(__file__)
    DATA_ROOT = os.path.join(settings_path, '../data/dataset')
    DATA_PATH = os.path.join(DATA_ROOT, args.dataset)

    entity2id = load_name_mapping(os.path.join(DATA_PATH, "entity2id.json"), 0)
    test_result = load_test_npy(os.path.join(DATA_PATH, "test_result.npy"))

    print(f"eval_dir: {args.eval_dir}, text_results_dir: {args.text_results_dir}")
    result_dir = os.path.join(args.eval_dir, args.text_results_dir)
    print(f"result_dir: {result_dir}")
    data = read_jsonl(result_dir)
    total_data_num = len(data)
    print(f"Total data num: {total_data_num}")
    print(f"Eval {len(data)} from {args.text_results_dir}")
    print_result_integrity_diagnostics(data)
    print_prediction_parse_diagnostics(data)
    metric_list = [0, 0, 0]
    reciprocal_rank_sum = 0.0
    rank_out_10 = 0
    error_count = 0
    error_num = defaultdict(int)
    target_id_list = np.zeros(len(data)).astype(int)
    idx_list = np.zeros(len(data)).astype(int)
    for i, d in enumerate(data):
        idx = int(d["idx"])
        idx_list[i] = idx
    idx_min = np.min(idx_list)

    for d in data:
        target = d["target"]
        final_ranked = d["final_ranked"]
        pred_answer = d["pred_ans"]
        idx = int(d["idx"]) - idx_min
        target_id = entity2id.get(target)
        if idx < 10:
            print(target_id)
        if target_id is not None:
            target_id_list[idx] = target_id
        else:
            raise ValueError(f"未知target: {target}")
        if pred_answer == "I don't know.":
            error_num[d["stop_reason_final"]] += 1
            error_count += 1
            continue
        ok, rank, ranked_ans = find_entity_rank(pred_answer, target)
        for answer in ranked_ans:
            entityid = entity2id.get(answer.answer)
            if entityid is not None:
                test_result[idx][entityid] += answer.score * 0

        if ok:
            reciprocal_rank_sum += 1.0 / float(rank)
            if rank <=1 :
                metric_list[0] += 1
            if rank <= 3:
                metric_list[1] += 1
            if rank <= 10:
                metric_list[2] += 1
        else:
            rank_out_10 += 1

    assert len(data) == len(target_id_list), (len(data), len(target_id_list))
    time_aware_ranks = []
    for j in range(len(target_id_list)):
        tmp_score = test_result[j]
        pred_ground = tmp_score[target_id_list[j]]
        ob_pred_comp1 = (tmp_score > pred_ground)
        ob_pred_comp2 = (tmp_score == pred_ground)
        target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
        time_aware_ranks.append(target_rank_i)
    time_unaware_ranks = np.array(time_aware_ranks).astype(float)

    stat_ranks(time_unaware_ranks, "time aware filter")
    print(f"LLM Result: \n")
    print(f"Error type and num: {error_num}")
    print(f"rankd out 10 num{rank_out_10}")
    # 方式 B：显式使用 .keys() 方法
    print_prediction_metrics(metric_list, reciprocal_rank_sum, total_data_num, error_count)

    return metric_list


def eval(args):
    settings_path = os.path.dirname(__file__)

    print(f"eval_dir: {args.eval_dir}, text_results_dir: {args.text_results_dir}")
    result_dir = os.path.join(args.eval_dir, args.text_results_dir)
    print(f"result_dir: {result_dir}")
    data = read_jsonl(result_dir)
    total_data_num = len(data)
    print(f"Total data num: {total_data_num}")
    print(f"Eval {len(data)} from {args.text_results_dir}")
    print_result_integrity_diagnostics(data)
    print_prediction_parse_diagnostics(data)
    metric_list = [0, 0, 0]
    reciprocal_rank_sum = 0.0
    rank_out_10 = 0
    error_count = 0
    error_num = defaultdict(int)
    final_rank_list = []
    first_rank_list = []
    last_rank_list = []

    for d in data:
        target = d["target"]
        final_ranked = d["final_ranked"]
        first_ranked = d["first_rank"]
        last_ranked = d["last_rank"]
        pred_answer = d["pred_ans"]
        final_rank_list.append(final_ranked)
        first_rank_list.append(first_ranked)
        last_rank_list.append(last_ranked)

        if pred_answer == "I don't know.":
            error_num[d["stop_reason_final"]] += 1
            error_count += 1
            continue

        ok, rank, ranked_ans = find_entity_rank(pred_answer, target)
        if ok:
            reciprocal_rank_sum += 1.0 / float(rank)
            if rank <= 1:
                metric_list[0] += 1
            if rank <= 3:
                metric_list[1] += 1
            if rank <= 10:
                metric_list[2] += 1
        else:
            rank_out_10 += 1

    print(f"LLM Result: \n")
    print(f"Error type and num: {error_num}")
    print(f"rankd out 10 num{rank_out_10}")
    # 方式 B：显式使用 .keys() 方法
    print_prediction_metrics(metric_list, reciprocal_rank_sum, total_data_num, error_count)

    final_rank_list = np.array(final_rank_list).astype(float)
    first_rank_list = np.array(first_rank_list).astype(float)
    last_rank_list = np.array(last_rank_list).astype(float)
    stat_ranks(final_rank_list, "time aware filter final")
    stat_ranks(first_rank_list, "time aware filter first")
    stat_ranks(last_rank_list, "time aware filter last")
    print_structural_confidence_metrics(data)

    return metric_list


def calc_interaction_round(args):
    print(f"eval_dir: {args.eval_dir}, text_results_dir: {args.text_results_dir}")
    result_dir = os.path.join(args.eval_dir, args.text_results_dir)
    print(f"result_dir: {result_dir}")
    data = read_jsonl(result_dir)
    total_data_num = len(data)
    print(f"Total data num: {total_data_num}")
    print(f"Eval {len(data)} from {args.text_results_dir}")
    if total_data_num == 0:
        print("No samples found. Skip interaction_round statistics.")
        return {}

    round_counter = Counter()
    missing_count = 0
    invalid_count = 0

    for d in data:
        if "interaction_round" not in d:
            missing_count += 1
            continue
        try:
            interaction_round = int(d["interaction_round"])
            round_counter[interaction_round] += 1
        except (TypeError, ValueError):
            invalid_count += 1

    print("\nInteraction round statistics:")
    print("round\tcount\tratio")
    for r in sorted(round_counter.keys()):
        cnt = round_counter[r]
        ratio = cnt / total_data_num
        print(f"{r}\t{cnt}\t{ratio:.6f}")

    if missing_count > 0:
        print(f"missing_interaction_round_count: {missing_count} ({missing_count / total_data_num:.6f})")
    if invalid_count > 0:
        print(f"invalid_interaction_round_count: {invalid_count} ({invalid_count / total_data_num:.6f})")

    return {
        "total": total_data_num,
        "round_count": dict(sorted(round_counter.items())),
        "round_ratio": {k: v / total_data_num for k, v in sorted(round_counter.items())},
        "missing_count": missing_count,
        "invalid_count": invalid_count,
    }


def print_sample(args):
    print(f"eval_dir: {args.eval_dir}, text_results_dir: {args.text_results_dir}")
    result_dir = os.path.join(args.eval_dir, args.text_results_dir)
    print(f"result_dir: {result_dir}")
    data = read_jsonl(result_dir)
    total_data_num = len(data)
    print(f"Total data num: {total_data_num}")
    print(f"Eval {len(data)} from {args.text_results_dir}")
    if total_data_num == 0:
        print("No samples found. Skip interaction_round statistics.")
        return {}

    for d in data:
        print(f'stop_reason_store:{d["stop_reason_store"]},  stop_reason_final:{d["stop_reason_final"]},  pred_ans:{d["pred_ans"]}')

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_results_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, default="./results/eval/")
    parser.add_argument("--dataset", type=str, default="")
    # parser.add_argument("--result_dir_tkgr", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = eval(args)
    # calc_interaction_round(args)
