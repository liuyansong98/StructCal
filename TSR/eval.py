import copy
import json
import math
import csv
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
import os
import heapq
import random
import settings as settings
import json
import numpy as np
from utils import ARCCache1

TEST_BATCH_SIZE = 32

def save_rank_bucket_txt(records, filename, rank_key, lower=None, upper=None, include_lower=False, include_upper=True):
    selected = []
    for record in records:
        rank_value = record.get(rank_key)
        if rank_value is None:
            continue
        keep = True
        if lower is not None:
            keep = keep and (rank_value >= lower if include_lower else rank_value > lower)
        if upper is not None:
            keep = keep and (rank_value <= upper if include_upper else rank_value < upper)
        if keep:
            selected.append(record)

    with open(filename, "w", encoding="utf-8", newline="") as f:
        for record in selected:
            f.write(f"{int(record['src'])}\t{int(record['rel'])}\t{int(record['dst'])}\t{int(record['ts'])}\n")

    return len(selected)


def summarize_rank_list(ranks):
    if not ranks:
        return {
            "count": 0,
            "MRR": float("nan"),
            "Hit@1": float("nan"),
            "Hit@3": float("nan"),
            "Hit@10": float("nan"),
        }
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "count": int(len(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "Hit@1": float(np.mean(ranks <= 1)),
        "Hit@3": float(np.mean(ranks <= 3)),
        "Hit@10": float(np.mean(ranks <= 10)),
    }


def log_history_enhancement_subset(
        records, label, lower=None, upper=None, include_lower=False, include_upper=True,
        rank_key="raw_rank", enhanced_rank_key="raw_rank_his_enh", logger=None):
    subset = []
    for record in records:
        rank = record.get(rank_key)
        his_enh_rank = record.get(enhanced_rank_key)
        if rank is None or his_enh_rank is None:
            continue
        keep = True
        if lower is not None:
            keep = keep and (rank >= lower if include_lower else rank > lower)
        if upper is not None:
            keep = keep and (rank <= upper if include_upper else rank < upper)
        if keep:
            subset.append(record)

    before_metrics = summarize_rank_list([record[rank_key] for record in subset])
    after_metrics = summarize_rank_list([record[enhanced_rank_key] for record in subset])
    lines = [
        f"===========Time-aware Filter Rank {label} Sample Subset: before vs after history enhancement===========",
        (
            f"subset count={before_metrics['count']} | "
            f"before MRR={before_metrics['MRR']:.6f}, Hit@1={before_metrics['Hit@1']:.6f}, "
            f"Hit@3={before_metrics['Hit@3']:.6f}, Hit@10={before_metrics['Hit@10']:.6f}"
        ),
        (
            f"subset count={after_metrics['count']} | "
            f"after  MRR={after_metrics['MRR']:.6f}, Hit@1={after_metrics['Hit@1']:.6f}, "
            f"Hit@3={after_metrics['Hit@3']:.6f}, Hit@10={after_metrics['Hit@10']:.6f}"
        ),
    ]
    message = "\n".join(lines)
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    return {"before": before_metrics, "after": after_metrics}


def compute_attention_entropy(attn_weights, eps=1e-12):
    probs = np.asarray(attn_weights, dtype=np.float64)
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_rank_entropy_csv(records, filename):
    fieldnames = [
        "sample_id",
        "src",
        "rel",
        "dst",
        "ts",
        "raw_rank",
        "raw_rank_his_enh",
        "time_aware_rank",
        "time_aware_rank_his_enh",
        "time_unaware_rank",
        "attn_entropy",
        "path_num",
        "unique_tail_num",
        "path_unique_ratio",
        "path_tail_entropy",
        "most_common_tail_share",
        "top1_overlap_count",
        "top1_overlap_ratio",
        "top3_overlap_count",
        "top3_overlap_ratio",
        "top10_overlap_count",
        "top10_overlap_ratio",
        "top1_in_path",
        "any_top3_in_path",
        "any_top10_in_path",
        "target_in_path",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize_rank_entropy(records, logger=None, stage_name="test"):
    if not records:
        return {}

    raw_records = [record for record in records if record.get("raw_rank") is not None]
    raw_ranks = [record["raw_rank"] for record in raw_records]
    entropies = [record["attn_entropy"] for record in raw_records]
    summary = {
        "raw_rank_entropy_pearson": safe_corrcoef(raw_ranks, entropies),
        "raw_rank_entropy_spearman": float(pd.Series(raw_ranks).corr(pd.Series(entropies), method="spearman")),
    }

    time_aware_records = [record for record in records if record.get("time_aware_rank") is not None]
    if time_aware_records:
        time_aware_ranks = [record["time_aware_rank"] for record in time_aware_records]
        time_aware_entropies = [record["attn_entropy"] for record in time_aware_records]
        summary["time_aware_rank_entropy_pearson"] = safe_corrcoef(time_aware_ranks, time_aware_entropies)
        summary["time_aware_rank_entropy_spearman"] = float(
            pd.Series(time_aware_ranks).corr(pd.Series(time_aware_entropies), method="spearman")
        )

    time_unaware_records = [record for record in records if record.get("time_unaware_rank") is not None]
    if time_unaware_records:
        time_unaware_ranks = [record["time_unaware_rank"] for record in time_unaware_records]
        time_unaware_entropies = [record["attn_entropy"] for record in time_unaware_records]
        summary["time_unaware_rank_entropy_pearson"] = safe_corrcoef(time_unaware_ranks, time_unaware_entropies)
        summary["time_unaware_rank_entropy_spearman"] = float(
            pd.Series(time_unaware_ranks).corr(pd.Series(time_unaware_entropies), method="spearman")
        )

    message = [f"{stage_name} rank/entropy correlation summary:"]
    for key, value in summary.items():
        message.append(f"  {key}: {value:.6f}" if not np.isnan(value) else f"  {key}: nan")
    message = "\n".join(message)
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    return summary


def extract_path_tails_from_subgraph(subgraph_src, sample_idx):
    if subgraph_src is None:
        return []
    if not isinstance(subgraph_src, tuple):
        return []

    if len(subgraph_src) == 3 and hasattr(subgraph_src[0], "shape"):
        subgraphs = [subgraph_src]
    else:
        subgraphs = list(subgraph_src)

    tails = []
    for subgraph in subgraphs:
        if not isinstance(subgraph, tuple) or len(subgraph) != 3:
            continue
        node_records, _, _ = subgraph
        if sample_idx >= len(node_records):
            continue
        tails.extend([int(x) for x in node_records[sample_idx, :, -1].tolist()])
    return tails


def summarize_eval_metrics(records, logger=None, stage_name="test"):
    if not records:
        return {}

    df = pd.DataFrame(records)
    summary = {}
    metrics = [
        "attn_entropy",
        "path_unique_ratio",
        "path_tail_entropy",
        "most_common_tail_share",
        "top1_overlap_count",
        "top1_overlap_ratio",
        "top3_overlap_count",
        "top3_overlap_ratio",
        "top10_overlap_count",
        "top10_overlap_ratio",
        "top1_in_path",
        "any_top3_in_path",
        "any_top10_in_path",
        "target_in_path",
    ]
    rank_columns = ["raw_rank", "time_aware_rank", "time_unaware_rank"]
    for rank_col in rank_columns:
        rank_df = df[[rank_col] + metrics].dropna()
        if len(rank_df) < 2 or rank_df[rank_col].nunique() <= 1:
            continue
        for metric in metrics:
            summary[f"{rank_col}__{metric}__pearson"] = safe_corrcoef(rank_df[rank_col], rank_df[metric])
            summary[f"{rank_col}__{metric}__spearman"] = float(
                rank_df[rank_col].corr(rank_df[metric], method="spearman")
            )

    lines = [f"{stage_name} path-dispersion/top3-overlap correlation summary:"]
    for key, value in summary.items():
        lines.append(f"  {key}: {value:.6f}" if not np.isnan(value) else f"  {key}: nan")
    message = "\n".join(lines)
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    return summary

def save_test_npy(scores_np, filename):
    # 保存文件
    np.save(filename, scores_np)

def load_test_npy(filename):
    # 读取文件
    return np.load(filename)


def save_rank_json(sample_ids, ranks_final, filename="rank_results.json"):
    """
    根据排名筛选样本ID并保存为JSON。

    Args:
        sample_ids (list/np.array): 样本的ID列表
        ranks_final (list/np.array): 最终的排名列表 (0-based indexing assumed, e.g., 0 is Top1)
        filename (str): 输出文件名
    """

    # 辅助内部函数：根据阈值筛选ID
    def get_top_k_ids(ranks, ids, l, h):
        if ranks is None:
            return []
        # 确保是numpy array以便操作
        ranks_np = np.array(ranks)
        ids_np = np.array(ids)

        # 筛选 rank <= h 且 rank > l 的索引 (假设 rank 从 1 开始，即 0-100 为 top 100)
        mask = (ranks_np <= h) & (ranks_np > l)

        # 获取对应的 ID 并转换为 Python原生 int/str 类型 (JSON不支持numpy类型)
        selected_ids = ids_np[mask].tolist()

        # cold_train_count 是冷启动的训练样本数量，pred_train_count 是第二阶段训练的样本数量
        cold_train_count = 3000
        pred_train_count = 6000
        if len(selected_ids) < cold_train_count + pred_train_count:
            print(f"警告: 样本总数 ({len(selected_ids)}) 少于需要抽取的数量 ({cold_train_count + pred_train_count})。")
            # 根据需求处理：这里选择将全部作为 stage1，stage2 为空
            return selected_ids, [], []

        # 为了不破坏原始列表，先创建一个副本
        shuffled_ids = list(selected_ids)
        # 随机打乱列表顺序
        random.shuffle(shuffled_ids)
        # 利用切片进行分割
        selected_ids_stage1 = shuffled_ids[:cold_train_count]  # 随机7000个
        # selected_ids_stage2 = shuffled_ids[cold_train_count:cold_train_count+pred_train_count]  # 随机9000个
        selected_ids_stage2 = shuffled_ids[cold_train_count:]  # 剩余的

        return selected_ids_stage1, selected_ids_stage2, shuffled_ids

    # 准备输出字典
    output_data = {}

    # 定义阈值和对应的键名后缀
    thresholds_l = [0,   0,  0,  0,  10,  10, 3,  3,   3, 0]
    thresholds_h = [100, 50, 20, 10, 100, 50, 10, 100, 50, 1000000]

    for l,h in zip(thresholds_l, thresholds_h):
        key_suffix = f"top{l}_{h}"
        # 处理最终结果 (Final)
        output_data[f"{key_suffix}_stage1"], output_data[f"{key_suffix}_stage2"], output_data[f"{key_suffix}"] = get_top_k_ids(ranks_final, sample_ids, l, h)

    # 写入 JSON 文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"JSON result saved to {filename}")

def eval_one_epoch(model, all_nodes_l, src, dst, ts, e_idx_l, val_bs_idx, args, logger,
                   num_e, num_r, is_need_filter=False,
                   case_study=False, model_path="", stage='test'):

    with ((torch.no_grad())):
        model = model.eval()

        srt2o = defaultdict(list)
        sr2o = defaultdict(list)

        # 计算filter的过滤字典
        ts_discr = []
        if is_need_filter:
            for i in range(len(src)):
                sr2o[(src[i], e_idx_l[i])].append(dst[i])
                srt2o[(src[i], e_idx_l[i], ts[i])].append(dst[i])

        num_test_instance = len(src)
        # num_test_batch = math.ceil(num_test_instance / args.bs)
        num_test_batch = len(val_bs_idx) - 1

        train_node_degree = copy.deepcopy(model.ngh_finder.node_degree)
        t_results = {}
        t_results_time = {}
        t_results_static = {}
        hit_n_count_raw = np.zeros(15)
        hit_n_count_time = np.zeros(15)
        testSet_raw_rank = []
        testSet_time_rank = []
        mrr_raw_list = []
        mrr_time_list = []
        scores_np = None
        test_np_rows = []
        loss = 0
        rank_entropy_records = []

        num_batch = 0
        model.ngh_finder.node_degree = copy.deepcopy(train_node_degree)

        for k in tqdm(range(num_test_batch)):

            sample_start = val_bs_idx[k]
            sample_end = val_bs_idx[k + 1]
            if sample_start == sample_end:
                continue

            batch_rank_raw = []
            batch_rank_time = []

            n_batch = math.ceil((sample_end - sample_start) / args.bs)
            for x in range(n_batch):
                num_batch += 1
                s_idx = x * args.bs + sample_start
                e_idx = min(sample_end, s_idx + args.bs)

                if s_idx == e_idx:
                    continue
                src_l_cut, dst_l_cut = src[s_idx:e_idx], dst[s_idx:e_idx]
                ts_l_cut = ts[s_idx:e_idx]
                e_l_cut = e_idx_l[s_idx:e_idx]
                test_np_rows.append(
                    np.stack(
                        [
                            np.asarray(src_l_cut, dtype=np.int64),
                            np.asarray(e_l_cut, dtype=np.int64),
                            np.asarray(dst_l_cut, dtype=np.int64),
                            np.asarray(ts_l_cut, dtype=np.int64),
                        ],
                        axis=1,
                    )
                )


                batch_loss, score, attn_output_weights, subgraph_src = model.inference(src_l_cut, dst_l_cut, all_nodes_l, ts_l_cut, e_l_cut, stage=stage)

                b_range = torch.arange(score.shape[0], device=args.device)
                loss += batch_loss.item()
                ranks = []
                for i in range(len(src_l_cut)):
                    tmp_score = score[i]
                    pred_ground = tmp_score[dst_l_cut[i]]
                    ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
                    ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
                    target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                    ranks.append(target_rank_i)
                    attn_entropy_i = compute_attention_entropy(attn_output_weights[i].detach().cpu().numpy())
                    path_tail_ids = extract_path_tails_from_subgraph(subgraph_src, i)
                    path_tail_set = set(path_tail_ids)
                    top1_indices = torch.topk(
                        tmp_score, k=min(1, tmp_score.shape[0])
                    ).indices.detach().cpu().tolist()
                    top3_indices = torch.topk(
                        tmp_score, k=min(3, tmp_score.shape[0])
                    ).indices.detach().cpu().tolist()
                    top10_indices = torch.topk(
                        tmp_score, k=min(10, tmp_score.shape[0])
                    ).indices.detach().cpu().tolist()
                    top1_set = set(int(x) for x in top1_indices)
                    top3_set = set(int(x) for x in top3_indices)
                    top10_set = set(int(x) for x in top10_indices)
                    top1_overlap_count = len(path_tail_set & top1_set)
                    top3_overlap_count = len(path_tail_set & top3_set)
                    top10_overlap_count = len(path_tail_set & top10_set)
                    if path_tail_ids:
                        path_tail_counter = {}
                        for tail_id in path_tail_ids:
                            path_tail_counter[tail_id] = path_tail_counter.get(tail_id, 0) + 1
                        unique_tail_num = len(path_tail_set)
                        path_unique_ratio = unique_tail_num / len(path_tail_ids)
                        path_tail_entropy = compute_attention_entropy(
                            np.array(list(path_tail_counter.values()), dtype=np.float64)
                        )
                        most_common_tail_share = max(path_tail_counter.values()) / len(path_tail_ids)
                    else:
                        unique_tail_num = 0
                        path_unique_ratio = 0.0
                        path_tail_entropy = 0.0
                        most_common_tail_share = 0.0
                    rank_entropy_records.append({
                        "sample_id": int(s_idx + i),
                        "src": int(src_l_cut[i]),
                        "rel": int(e_l_cut[i]),
                        "dst": int(dst_l_cut[i]),
                        "ts": int(ts_l_cut[i]),
                        "raw_rank": float(target_rank_i),
                        "time_aware_rank": None,
                        "time_aware_rank_his_enh": None,
                        "time_unaware_rank": None,
                        "attn_entropy": float(attn_entropy_i),
                        "path_num": int(len(path_tail_ids)),
                        "unique_tail_num": int(unique_tail_num),
                        "path_unique_ratio": float(path_unique_ratio),
                        "path_tail_entropy": float(path_tail_entropy),
                        "most_common_tail_share": float(most_common_tail_share),
                        "top1_overlap_count": int(top1_overlap_count),
                        "top1_overlap_ratio": float(top1_overlap_count / 1.0),
                        "top3_overlap_count": int(top3_overlap_count),
                        "top3_overlap_ratio": float(top3_overlap_count / 3.0),
                        "top10_overlap_count": int(top10_overlap_count),
                        "top10_overlap_ratio": float(top10_overlap_count / 10.0),
                        "top1_in_path": int(bool(top1_indices) and int(top1_indices[0]) in path_tail_set),
                        "any_top3_in_path": int(top3_overlap_count > 0),
                        "any_top10_in_path": int(top10_overlap_count > 0),
                        "target_in_path": int(int(dst_l_cut[i]) in path_tail_set),
                    })

                testSet_raw_rank.extend(ranks)
                batch_rank_raw.extend(ranks)
                for j in range(len(ranks)):
                    if ranks[j] < 11:
                        hit_n_count_raw[int(ranks[j])] += 1
                    elif 10 < ranks[j] < 51:
                        hit_n_count_raw[11] += 1
                    elif 50 < ranks[j] < 101:
                        hit_n_count_raw[12] += 1
                    else:
                        hit_n_count_raw[13] += 1

                ranks = torch.tensor(ranks).float()
                t_results['count_raw'] = torch.numel(ranks) + t_results.get('count_raw', 0.0)
                t_results['mar_raw'] = torch.sum(ranks).item() + t_results.get('mar_raw', 0.0)
                t_results['mrr_raw'] = torch.sum(1.0 / ranks).item() + t_results.get('mrr_raw', 0.0)
                for j in range(10):
                    t_results['hits@{}_raw'.format(j + 1)] = torch.numel(ranks[ranks <= (j + 1)]) + t_results.get(
                        'hits@{}_raw'.format(j + 1), 0.0)

                # 计算time-aware filter 和 time-unaware filter
                if is_need_filter:

                    # 计算time-aware filter
                    time_aware_score = score.clone()
                    target_score = score[b_range, dst_l_cut]
                    for j in range(len(src_l_cut)):
                        if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
                            time_aware_score[j][srt2o[(src_l_cut[j], e_l_cut[j], ts_l_cut[j] // 3600)]] = -10000000
                        else:
                            time_aware_score[j][srt2o[(src_l_cut[j], e_l_cut[j], ts_l_cut[j])]] = -10000000
                    time_aware_score[b_range, dst_l_cut] = target_score

                    if scores_np is None:
                        scores_np = time_aware_score.clone().detach().cpu().numpy()
                    else:
                        scores_np = np.concatenate((scores_np, time_aware_score.clone().detach().cpu().numpy()), axis=0)

                    time_aware_ranks = []
                    for j in range(len(src_l_cut)):
                        tmp_score = time_aware_score[j]
                        pred_ground = tmp_score[dst_l_cut[j]]
                        ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
                        ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
                        target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                        time_aware_ranks.append(target_rank_i)
                        rank_entropy_records[-len(src_l_cut) + j]["time_aware_rank"] = float(target_rank_i)

                    testSet_time_rank.extend(time_aware_ranks)
                    batch_rank_time.extend(time_aware_ranks)
                    for j in range(len(time_aware_ranks)):
                        if time_aware_ranks[j] < 11:
                            hit_n_count_time[int(time_aware_ranks[j])] += 1
                        elif 10 < time_aware_ranks[j] < 51:
                            hit_n_count_time[11] += 1
                        elif 50 < time_aware_ranks[j] < 101:
                            hit_n_count_time[12] += 1
                        else:
                            hit_n_count_time[13] += 1

                    time_aware_ranks = torch.tensor(time_aware_ranks).float()
                    t_results_time['count_time_f'] = torch.numel(time_aware_ranks) + t_results_time.get('count_time_f', 0.0)
                    t_results_time['mar_time_f'] = torch.sum(time_aware_ranks).item() + t_results_time.get('mar_time_f', 0.0)
                    t_results_time['mrr_time_f'] = torch.sum(1.0 / time_aware_ranks).item() + t_results_time.get('mrr_time_f', 0.0)
                    for j in range(10):
                        t_results_time['hits@{}_time_f'.format(j + 1)] = torch.numel(time_aware_ranks[time_aware_ranks <= (j + 1)]) + t_results_time.get(
                            'hits@{}_time_f'.format(j + 1), 0.0)

                    # 计算time-unaware filter
                    time_unaware_score = score.clone()
                    for j in range(len(src_l_cut)):
                        time_unaware_score[j][sr2o[(src_l_cut[j], e_l_cut[j])]] = -10000000
                    time_unaware_score[b_range, dst_l_cut] = target_score

                    time_unaware_ranks = []
                    for j in range(len(src_l_cut)):
                        tmp_score = time_unaware_score[j]
                        pred_ground = tmp_score[dst_l_cut[j]]
                        ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
                        ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
                        target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                        time_unaware_ranks.append(target_rank_i)
                        rank_entropy_records[-len(src_l_cut) + j]["time_unaware_rank"] = float(target_rank_i)
                    time_unaware_ranks = torch.tensor(time_unaware_ranks).float()
                    t_results_static['count_time_uf'] = torch.numel(time_unaware_ranks) + t_results_static.get('count_time_uf', 0.0)
                    t_results_static['mar_time_uf'] = torch.sum(time_unaware_ranks).item() + t_results_static.get('mar_time_uf', 0.0)
                    t_results_static['mrr_time_uf'] = torch.sum(1.0 / time_unaware_ranks).item() + t_results_static.get('mrr_time_uf', 0.0)
                    for j in range(10):
                        t_results_static['hits@{}_time_uf'.format(j + 1)] = torch.numel(time_unaware_ranks[time_unaware_ranks <= (j + 1)]) + t_results_static.get(
                            'hits@{}_time_uf'.format(j + 1), 0.0)

                model.ngh_finder.update_node_degree(src_l_cut, dst_l_cut)
            mrr_raw_list.append(np.mean(1/np.array(batch_rank_raw)))
            if is_need_filter:
                mrr_time_list.append(np.mean(1/np.array(batch_rank_time)))

        loss = loss / num_batch
        t_results['mar_raw'] = round(t_results['mar_raw'] / t_results['count_raw'], 5)
        t_results['mrr_raw'] = round(t_results['mrr_raw'] / t_results['count_raw'], 5)
        for j in range(10):
            t_results['hits@{}_raw'.format(j + 1)] = round(
                t_results['hits@{}_raw'.format(j + 1)] / t_results['count_raw'], 5)

        if is_need_filter:
            t_results_time['mar_time_f'] = round(t_results_time['mar_time_f'] / t_results_time['count_time_f'], 5)
            t_results_time['mrr_time_f'] = round(t_results_time['mrr_time_f'] / t_results_time['count_time_f'], 5)
            for j in range(10):
                t_results_time['hits@{}_time_f'.format(j + 1)] = round(
                    t_results_time['hits@{}_time_f'.format(j + 1)] / t_results_time['count_time_f'], 5)

            t_results_static['mar_time_uf'] = round(t_results_static['mar_time_uf'] / t_results_static['count_time_uf'], 5)
            t_results_static['mrr_time_uf'] = round(t_results_static['mrr_time_uf'] / t_results_static['count_time_uf'], 5)
            for j in range(10):
                t_results_static['hits@{}_time_uf'.format(j + 1)] = round(
                    t_results_static['hits@{}_time_uf'.format(j + 1)] / t_results_static['count_time_uf'], 5)

        if stage == "gen_filter_idx":
            sample_ids = [i for i in range(len(testSet_time_rank))]
            save_rank_json(sample_ids, testSet_time_rank, f"./data/{args.data}/data_idx_two_stage.json")

        if stage == "test":
            if model_path != "":
                eval_output_dir = os.path.dirname(model_path)
            else:
                eval_output_dir = os.path.join(os.path.dirname(__file__), "result", args.data)
                os.makedirs(eval_output_dir, exist_ok=True)
            rank_entropy_csv = os.path.join(eval_output_dir, f"{stage}_rank_entropy.csv")
            save_rank_entropy_csv(rank_entropy_records, rank_entropy_csv)
            logger.info(f"Saved per-sample rank/entropy records to {rank_entropy_csv}")
            summarize_rank_entropy(rank_entropy_records, logger=logger, stage_name=stage)
            summarize_eval_metrics(rank_entropy_records, logger=logger, stage_name=stage)

            print("offline mrr raw list")
            for i in range(len(mrr_raw_list)):
                print(mrr_raw_list[i], flush=True)

            mrr_list_id = np.arange(len(mrr_raw_list))
            x_bar = np.mean(mrr_list_id)
            y_bar = np.mean(np.array(mrr_raw_list))
            decay = np.sum((np.array(mrr_raw_list) - y_bar) * (mrr_list_id - x_bar)) / np.sum(
                (mrr_list_id - x_bar) ** 2)
            print(f"\noffline raw mrr decay: {decay}\n", flush=True)

            if is_need_filter:
                print("offline mrr time list")
                for i in range(len(mrr_time_list)):
                    print(mrr_time_list[i], flush=True)
                x_bar = np.mean(mrr_list_id)
                y_bar = np.mean(np.array(mrr_time_list))
                decay = np.sum((np.array(mrr_time_list) - y_bar) * (mrr_list_id - x_bar)) / np.sum(
                    (mrr_list_id - x_bar) ** 2)
                print(f"offline time mrr decay: {decay}\n", flush=True)

            dataset_output_dir = os.path.join(os.path.dirname(__file__), "data", args.data)
            os.makedirs(dataset_output_dir, exist_ok=True)
            score_npy_path = os.path.join(dataset_output_dir, "score.npy")
            test_npy_path = os.path.join(dataset_output_dir, "test.npy")

            if scores_np is None:
                scores_to_save = np.empty((0, len(all_nodes_l)), dtype=np.float32)
            else:
                scores_to_save = scores_np

            if test_np_rows:
                test_np = np.concatenate(test_np_rows, axis=0)
            else:
                test_np = np.empty((0, 4), dtype=np.int64)

            save_test_npy(scores_to_save, score_npy_path)
            save_test_npy(test_np, test_npy_path)
            logger.info(f"Saved score.npy to {score_npy_path}")
            logger.info(f"Saved test.npy to {test_npy_path}")

            rank_bucket_key = "time_aware_rank" if is_need_filter else "raw_rank"
            rank_eq_1_path = os.path.join(dataset_output_dir, "test_rank_eq_1.txt")
            rank_1_3_path = os.path.join(dataset_output_dir, "test_rank_1_to_3.txt")
            rank_1_10_path = os.path.join(dataset_output_dir, "test_rank_1_to_10.txt")
            rank_3_100_path = os.path.join(dataset_output_dir, "test_rank_3_to_100.txt")
            rank_3_10_path = os.path.join(dataset_output_dir, "test_rank_3_to_10.txt")
            rank_10_100_path = os.path.join(dataset_output_dir, "test_rank_10_to_100.txt")
            rank_gt_10_path = os.path.join(dataset_output_dir, "test_rank_gt_10.txt")
            rank_gt_3_path = os.path.join(dataset_output_dir, "test_rank_gt_3.txt")

            rank_eq_1_count = save_rank_bucket_txt(
                rank_entropy_records, rank_eq_1_path, rank_bucket_key, lower=1, upper=1, include_lower=True, include_upper=True
            )
            rank_1_3_count = save_rank_bucket_txt(
                rank_entropy_records, rank_1_3_path, rank_bucket_key, lower=1, upper=3, include_lower=True, include_upper=True
            )
            rank_1_10_count = save_rank_bucket_txt(
                rank_entropy_records, rank_1_10_path, rank_bucket_key, lower=1, upper=10, include_lower=True, include_upper=True
            )
            rank_3_100_count = save_rank_bucket_txt(
                rank_entropy_records, rank_3_100_path, rank_bucket_key, lower=3, upper=100, include_lower=False, include_upper=True
            )
            rank_3_10_count = save_rank_bucket_txt(
                rank_entropy_records, rank_3_10_path, rank_bucket_key, lower=3, upper=10, include_lower=False, include_upper=True
            )
            rank_10_100_count = save_rank_bucket_txt(
                rank_entropy_records, rank_10_100_path, rank_bucket_key, lower=10, upper=100, include_lower=False, include_upper=True
            )
            rank_gt_10_count = save_rank_bucket_txt(
                rank_entropy_records, rank_gt_10_path, rank_bucket_key, lower=10, upper=None, include_lower=False
            )
            rank_gt_3_count = save_rank_bucket_txt(
                rank_entropy_records, rank_gt_3_path, rank_bucket_key, lower=3, upper=None, include_lower=False
            )
            logger.info(f"Saved rank bucket txts using `{rank_bucket_key}` to {dataset_output_dir}")
            logger.info(
                f"rank==1: {rank_eq_1_count}, 1<=rank<=3: {rank_1_3_count}, 1<=rank<=10: {rank_1_10_count}, 3<rank<=100: {rank_3_100_count}, "
                f"3<rank<=10: {rank_3_10_count}, 10<rank<=100: {rank_10_100_count}, rank>10: {rank_gt_10_count}, rank>3: {rank_gt_3_count}"
            )

            if is_need_filter:
                history_subset_kwargs = {
                    "rank_key": "time_aware_rank",
                    "enhanced_rank_key": "time_aware_rank_his_enh",
                    "logger": logger,
                }
                log_history_enhancement_subset(
                    rank_entropy_records, "1-3", lower=1, upper=3, include_lower=True, include_upper=True,
                    **history_subset_kwargs
                )
                log_history_enhancement_subset(
                    rank_entropy_records, "3<rank<=10", lower=3, upper=10, include_lower=False, include_upper=True,
                    **history_subset_kwargs
                )
                log_history_enhancement_subset(
                    rank_entropy_records, "1-10", lower=1, upper=10, include_lower=True, include_upper=True,
                    **history_subset_kwargs
                )
                log_history_enhancement_subset(
                    rank_entropy_records, "rank>3", lower=3, upper=None, include_lower=False,
                    **history_subset_kwargs
                )
                log_history_enhancement_subset(
                    rank_entropy_records, "rank>10", lower=10, upper=None, include_lower=False,
                    **history_subset_kwargs
                )
                log_history_enhancement_subset(
                    rank_entropy_records, "4-100", lower=4, upper=100, include_lower=True, include_upper=True,
                    **history_subset_kwargs
                )
                log_history_enhancement_subset(
                    rank_entropy_records, "11-100", lower=11, upper=100, include_lower=True, include_upper=True,
                    **history_subset_kwargs
                )
            else:
                logger.info("Skip time-aware filter history enhancement subset summary because is_need_filter=False")

        logger.info("===========evaluating or testing RAW===========")
        logger.info("HITS10 {}".format(t_results['hits@10_raw']))
        logger.info("HITS3 {}".format(t_results['hits@3_raw']))
        logger.info("HITS1 {}".format(t_results['hits@1_raw']))
        logger.info("MRR {}".format(t_results['mrr_raw']))
        logger.info("MAR {}".format(t_results['mar_raw']))

        if is_need_filter:
            logger.info("===========evaluating or testing time-aware filter===========")
            logger.info("HITS10 {}".format(t_results_time['hits@10_time_f']))
            logger.info("HITS3 {}".format(t_results_time['hits@3_time_f']))
            logger.info("HITS1 {}".format(t_results_time['hits@1_time_f']))
            logger.info("MRR {}".format(t_results_time['mrr_time_f']))
            logger.info("MAR {}".format(t_results_time['mar_time_f']))

            logger.info("===========evaluating or testing time-unaware filter===========")
            logger.info("HITS10 {}".format(t_results_static['hits@10_time_uf']))
            logger.info("HITS3 {}".format(t_results_static['hits@3_time_uf']))
            logger.info("HITS1 {}".format(t_results_static['hits@1_time_uf']))
            logger.info("MRR {}".format(t_results_static['mrr_time_uf']))
            logger.info("MAR {}".format(t_results_static['mar_time_uf']))

        logger.info("===========raw ranks distribution===========")
        for i in range(len(hit_n_count_raw)):
            logger.info(hit_n_count_raw[i])
        logger.info("===========time ranks distribution===========")
        for i in range(len(hit_n_count_time)):
            logger.info(hit_n_count_time[i])

    return t_results['mrr_raw']
