import numpy as np
import os
import pandas as pd
import torch
from server.models.module import myModel
from server.models.graph import NeighborFinder
from server.models.utils import *
import logging
from datetime import datetime
import heapq
from collections import OrderedDict
from typing import Dict
import re
import json
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

np.set_printoptions(threshold=1e6)


def is_start_with_number_dot(text):
    # ^ 表示开头
    # \d+ 表示一个或多个数字
    # \. 表示匹配原始的点字符
    pattern = r"^\d+\."

    if re.match(pattern, text):
        return True
    return False

def load_id_mapping(filepath, num_relations, is_rel=False):
    with open(filepath, 'r', encoding='utf-8-sig') as file:
        id2name = json.load(file)
    id2name = {int(k): v for k, v in id2name.items()}
    if is_rel:
        inv_id2name = {}
        for key, value in id2name.items():
            inv_id2name[int(key) + num_relations] = "INV::" + value
        id2name.update(inv_id2name)
    return id2name

def load_name_mapping(filepath, num_relations, is_rel=False):
    with open(filepath, 'r', encoding='utf-8-sig') as file:
        name2id = json.load(file)
    if is_rel:
        inv_name2id = {}
        for key, value in name2id.items():
            inv_name2id["INV::" + key] = int(value) + num_relations
        name2id.update(inv_name2id)
    return name2id

args, sys_argv = get_args()
print(args)
set_random_seed(args.seed)
device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
args.device = device

settings_path = os.path.dirname(__file__)
DATA_ROOT = os.path.join(settings_path, '../data/dataset')
MODEL_ROOT = os.path.join(settings_path, './model_ckpts')
DATA_PATH = os.path.join(DATA_ROOT, args.data)

heads, tails, rels, timestamps, edge_idxs, num_entities, num_relations = \
    load_temporal_knowledge_graph(DATA_ROOT, args.data)
logger = logging.getLogger()

srt2o = defaultdict(list)

# 计算filter的过滤字典
for i in range(len(heads)):
    srt2o[(heads[i], rels[i], timestamps[i])].append(tails[i])

relation2id = load_name_mapping(os.path.join(DATA_PATH, "relation2id.json"), num_relations, True)
entity2id = load_name_mapping(os.path.join(DATA_PATH, "entity2id.json"), num_relations)
id2relation = load_id_mapping(os.path.join(DATA_PATH, "id2relation.json"), num_relations, True)
id2entity = load_id_mapping(os.path.join(DATA_PATH, "id2entity.json"), num_relations)

full_adj_list = [[] for _ in range(num_entities)]
for src, dst, eidx, ts in zip(heads, tails, rels, timestamps):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx + num_relations, ts))

full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=args.temporal_bias,
                                 limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span,
                                 num_entity=num_entities, data_name=args.data)

logger.info('Sampling module - temporal bias: {}'.format(args.temporal_bias))
"""init dynamic entity embeddings"""
init_dynamic_entity_embeds = get_embedding(num_entities, args.embed_dim, zero_init=False)
"""init relation embeddings"""
init_dynamic_relation_embeds = get_embedding(num_relations * 2, args.embed_dim, zero_init=False)

model = myModel(n_feat=init_dynamic_entity_embeds, e_feat=init_dynamic_relation_embeds, device=args.device,
                pos_dim=args.pos_dim, num_neighbors=args.n_degree,
                n_head=args.n_head, path_encode=args.path_encode)

model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
model.update_ngh_finder(full_ngh_finder)
model.to(device)
model.eval()



def path_resolution(selected_paths_list, src_l_cut, e_l_cut, dst_l_cut, ts_l_cut):
    # 例：selected_paths_str 形如：
    # "PATH_1: s1 -> r1(t1) -> r2(t2) -> o1;
    #  PATH_2: s2 -> r3(t3) -> o2;
    #  PATH_3: s3 -> r4(t4) -> r5(t5) -> r6(t6) -> o3;"

    # 解析 "relation(t)" 片段的正则：
    # STEP_PATTERN = re.compile(
    #     r"(?P<rel>[^(>\n]+?)\s*\(\s*(?P<time>[^)]+)\s*\)"
    # )
    STEP_PATTERN = re.compile(r"^\s*(?P<rel>.*)\(\s*(?P<time>\d+)\s*\)\s*$")
    extr_subgraph1_list = []
    extr_subgraph2_list = []
    extr_subgraph3_list = []
    l_matrix = np.zeros((len(selected_paths_list), 3), dtype=int)
    for i, selected_paths_str in enumerate(selected_paths_list):
        if selected_paths_str is None:
            continue
        node_records = [None, None, None]
        rel_records = [None, None, None]
        t_records = [None, None, None]
        # 按行遍历
        for raw_line in selected_paths_str.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # if not line.startswith("PATH_"):
            #     continue

            # 是否以数字加'.'开头
            if is_start_with_number_dot(line):
                line = line.split(".")[1]

            # 去掉末尾分号
            if line.endswith(";"):
                line = line[:-1]
            if ";" in line:
                line = line.split(";")[0]

            # # 去掉 "PATH_x:" 前缀，保留真正的路径表达式 "s -> r1(t1) -> r2(t2) -> o"
            # try:
            #     _, path_expr = line.split(":", 1)
            # except ValueError:
            #     # 格式不对，跳过
            #     continue
            path_expr = line.strip()

            # 按 "->" 分割：['s', 'r1(t1)', 'r2(t2)', 'o']
            tokens = [t.strip() for t in path_expr.split("->")]
            if len(tokens) < 3:
                # 至少要 s -> r(t) -> o 才算合法路径
                continue

            head_name = tokens[0]
            tail_name = tokens[-1]
            middle_tokens = tokens[1:-1]  # 中间的 relation(t)

            L = len(middle_tokens)  # 路径长度 = 关系数
            if L == 0 or L > 3:
                # 只保留长度 1~3 的路径
                continue

            # 解析每个 middle token 里的 rel(t)
            rel_names = []
            time_int = []
            valid = True
            for step in middle_tokens:
                m = STEP_PATTERN.fullmatch(step)
                if not m:
                    # 有一个关系没匹配上，整条路径丢弃（也可以选择更宽松的策略）
                    valid = False
                    break
                rel_names.append(m.group("rel").strip())
                time_int.append(int(m.group("time").strip()))
            if not valid:
                continue

            # 映射实体/关系到 id
            try:
                head_id = entity2id[head_name]
                tail_id = entity2id[tail_name]
            except KeyError:
                # 实体不在字典中，跳过这条路径（你也可以选择 raise）
                continue

            try:
                rel_ids = [relation2id[r] for r in rel_names]
            except KeyError:
                # 关系不在字典中，跳过这条路径
                continue

            if src_l_cut[i] != head_id:
                continue
            if l_matrix[i][L - 1] == 0:
                node_records[L - 1] = np.zeros((1, L + 1))
                rel_records[L - 1] = np.zeros((1, L + 1))
                t_records[L - 1] = np.zeros((1, L + 1))
                node_records[L - 1][0][0] = head_id
                node_records[L - 1][0][-1] = tail_id
                rel_records[L - 1][0][-L:] = np.array(rel_ids)
                rel_records[L - 1][0][0] = e_l_cut[i]
                t_records[L - 1][0][-L:] = np.array(time_int)
                t_records[L - 1][0][0] = ts_l_cut[i]
            else:
                tmp_node_records = np.zeros((1, L + 1))
                tmp_rel_records = np.zeros((1, L + 1))
                tmp_t_records = np.zeros((1, L + 1))
                tmp_node_records[0][0] = head_id
                tmp_node_records[0][-1] = tail_id
                tmp_rel_records[0][-L:] = np.array(rel_ids)
                tmp_rel_records[0][0] = e_l_cut[i]
                tmp_t_records[0][-L:] = np.array(time_int)
                tmp_t_records[0][0] = ts_l_cut[i]
                node_records[L - 1] = np.concatenate((node_records[L - 1], tmp_node_records), axis=0)
                rel_records[L - 1] = np.concatenate((rel_records[L - 1], tmp_rel_records), axis=0)
                t_records[L - 1] = np.concatenate((t_records[L - 1], tmp_t_records), axis=0)

            l_matrix[i][L - 1] += 1

        if l_matrix[i][0] == 0:
            extr_subgraph1_list.append((None, None, None))
        else:
            assert len(node_records[0]) == len(rel_records[0]) == len(t_records[0]) == l_matrix[i][0]
            extr_subgraph1_list.append((node_records[0], rel_records[0], t_records[0]))
        if l_matrix[i][1] == 0:
            extr_subgraph2_list.append((None, None, None))
        else:
            assert len(node_records[1]) == len(rel_records[1]) == len(t_records[1]) == l_matrix[i][1]
            extr_subgraph2_list.append((node_records[1], rel_records[1], t_records[1]))
        if l_matrix[i][2] == 0:
            extr_subgraph3_list.append((None, None, None))
        else:
            assert len(node_records[2]) == len(rel_records[2]) == len(t_records[2]) == l_matrix[i][2]
            extr_subgraph3_list.append((node_records[2], rel_records[2], t_records[2]))

    # path_num_max = l_matrix.max(axis=0)
    # node_records1 = np.zeros((len(src_l_cut), path_num_max[0], 2))
    # rel_records1 = np.zeros((len(src_l_cut), path_num_max[0], 2))
    # t_records1 = np.zeros((len(src_l_cut), path_num_max[0], 2))
    # node_records2 = np.zeros((len(src_l_cut), path_num_max[1], 3))
    # rel_records2 = np.zeros((len(src_l_cut), path_num_max[1], 3))
    # t_records2 = np.zeros((len(src_l_cut), path_num_max[1], 3))
    # node_records3 = np.zeros((len(src_l_cut), path_num_max[2], 4))
    # rel_records3 = np.zeros((len(src_l_cut), path_num_max[2], 4))
    # t_records3 = np.zeros((len(src_l_cut), path_num_max[2], 4))
    # mask1 = np.zeros((len(src_l_cut), path_num_max[0]))
    # mask2 = np.zeros((len(src_l_cut), path_num_max[1]))
    # mask3 = np.zeros((len(src_l_cut), path_num_max[2]))
    # for j in range(len(src_l_cut)):
    #     if l_matrix[j][0] > 0:
    #         mask1[j][:l_matrix[j][0]] = 1
    #         node_records1[j][:l_matrix[j][0]], rel_records1[j][:l_matrix[j][0]], t_records1[j][:l_matrix[j][0]] = \
    #             extr_subgraph1_list[j]
    #     if l_matrix[j][1] > 0:
    #         mask2[j][:l_matrix[j][1]] = 1
    #         node_records2[j][:l_matrix[j][1]], rel_records2[j][:l_matrix[j][1]], t_records2[j][:l_matrix[j][1]] = \
    #             extr_subgraph2_list[j]
    #     if l_matrix[j][2] > 0:
    #         mask3[j][:l_matrix[j][2]] = 1
    #         node_records3[j][:l_matrix[j][2]], rel_records3[j][:l_matrix[j][2]], t_records3[j][:l_matrix[j][2]] = \
    #             extr_subgraph3_list[j]

    path_number = 30
    node_records1 = np.zeros((len(src_l_cut), path_number, 2))
    rel_records1 = np.zeros((len(src_l_cut), path_number, 2))
    t_records1 = np.zeros((len(src_l_cut), path_number, 2))
    node_records2 = np.zeros((len(src_l_cut), path_number, 3))
    rel_records2 = np.zeros((len(src_l_cut), path_number, 3))
    t_records2 = np.zeros((len(src_l_cut), path_number, 3))
    node_records3 = np.zeros((len(src_l_cut), path_number, 4))
    rel_records3 = np.zeros((len(src_l_cut), path_number, 4))
    t_records3 = np.zeros((len(src_l_cut), path_number, 4))
    mask1 = np.zeros((len(src_l_cut), path_number))
    mask2 = np.zeros((len(src_l_cut), path_number))
    mask3 = np.zeros((len(src_l_cut), path_number))
    for j in range(len(src_l_cut)):
        if l_matrix[j][0] > 0:
            k = path_number // l_matrix[j][0]
            for x in range(k):
                size = l_matrix[j][0]
                beg = x * size
                ed = (x + 1) * size
                mask1[j][beg:ed] = 1
                node_records1[j][beg:ed], rel_records1[j][beg:ed], t_records1[j][beg:ed] = \
                    extr_subgraph1_list[j]
        if l_matrix[j][1] > 0:
            k = path_number // l_matrix[j][1]
            for x in range(k):
                size = l_matrix[j][1]
                beg = x * size
                ed = (x + 1) * size
                mask2[j][beg:ed] = 1
                node_records2[j][beg:ed], rel_records2[j][beg:ed], t_records2[j][beg:ed] = \
                    extr_subgraph2_list[j]
        if l_matrix[j][2] > 0:
            k = path_number // l_matrix[j][2]
            for x in range(k):
                size = l_matrix[j][2]
                beg = x * size
                ed = (x + 1) * size
                mask3[j][beg:ed] = 1
                node_records3[j][beg:ed], rel_records3[j][beg:ed], t_records3[j][beg:ed] = \
                    extr_subgraph3_list[j]

    return (node_records1, rel_records1, t_records1), (node_records2, rel_records2, t_records2), (
        node_records3, rel_records3, t_records3), mask1, mask2, mask3


def deal_request(idx_list, selected_path_list, **kwargs):
    idx_list = [int(idx) for idx in idx_list]
    src_l_cut = heads[idx_list]
    e_l_cut = rels[idx_list]
    dst_l_cut = tails[idx_list]
    ts_l_cut = timestamps[idx_list]
    extr_subgraph_src1, extr_subgraph_src2, extr_subgraph_src3, mask1, mask2, mask3 = path_resolution(selected_path_list, src_l_cut, e_l_cut, dst_l_cut, ts_l_cut)
    batch_loss, score, attn_output_weights, subgraph_src = model.inference(src_l_cut, dst_l_cut, ts_l_cut,
                                                                           e_l_cut, extr_subgraph_src1, extr_subgraph_src2,
                                                                           extr_subgraph_src3, mask1, mask2, mask3)
    b_range = torch.arange(score.shape[0], device=args.device)
    ranks = []
    path_block_list = []
    for i in range(len(src_l_cut)):
        tmp_score = score[i]
        pred_ground = tmp_score[dst_l_cut[i]]
        ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
        ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
        target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
        ranks.append(target_rank_i)

        subgraph_src1, subgraph_src2, subgraph_src3 = subgraph_src
        node_records1, eidx_records1, t_records1 = subgraph_src1
        node_records2, eidx_records2, t_records2 = subgraph_src2
        node_records3, eidx_records3, t_records3 = subgraph_src3
        path_num1 = eidx_records1.shape[1]
        path_num2 = eidx_records2.shape[1]
        path_num3 = eidx_records3.shape[1]

        '''
        [PATH_1: subject -> relation(t1) -> relation(t2) -> entity1; 0.20]
        ...
        [PATH_n: ...]
        '''
        path_block = []
        attn_output_weights_i = attn_output_weights[i].tolist()
        # max_val_lis = heapq.nlargest(10, attn_output_weights_i)
        # print(f"set(attn_output_weights_i):{attn_output_weights_i}")
        max_val_lis = heapq.nlargest(20, set(attn_output_weights_i))
        max_idx_lis = []
        for j, item in enumerate(max_val_lis):
            idx = attn_output_weights_i.index(item)
            max_idx_lis.append(idx)
            path_score = str(round(float(attn_output_weights_i[idx]), 3))
            attn_output_weights_i[idx] = float('-inf')
            try:
                if idx < path_num1:
                    path_block.append(f"{id2entity[int(node_records1[i, idx, 0])]}->"
                                   f"{id2relation[int(eidx_records1[i, idx, 1])]}({int(t_records1[i, idx, 1])})->"
                                   f"{id2entity[int(node_records1[i, idx, 1])]};\n")
                elif path_num1 <= idx < path_num1 + path_num2:
                    idx = idx - path_num1
                    path_block.append(f"{id2entity[int(node_records2[i, idx, 0])]}->"
                                   f"{id2relation[int(eidx_records2[i, idx, 1])]}({int(t_records2[i, idx, 1])})->"
                                   f"{id2relation[int(eidx_records2[i, idx, 2])]}({int(t_records2[i, idx, 2])})->"
                                   f"{id2entity[int(node_records2[i, idx, 2])]};\n")

                else:
                    idx = idx - path_num1 - path_num2
                    path_block.append(f"{id2entity[int(node_records3[i, idx, 0])]}->"
                                   f"{id2relation[int(eidx_records3[i, idx, 1])]}({int(t_records3[i, idx, 1])})->"
                                   f"{id2relation[int(eidx_records3[i, idx, 2])]}({int(t_records3[i, idx, 2])})->"
                                   f"{id2relation[int(eidx_records3[i, idx, 3])]}({int(t_records3[i, idx, 3])})->"
                                   f"{id2entity[int(node_records3[i, idx, 3])]};\n")
            except KeyError:
                # 关系或实体不在字典中，跳过这条路径
                continue
        path_block_list.append(path_block)

    # 计算time-aware filter
    time_aware_score = score
    target_score = score[b_range, dst_l_cut]
    for j in range(len(src_l_cut)):
        time_aware_score[j][srt2o[(src_l_cut[j], e_l_cut[j], ts_l_cut[j])]] = -10000000
    time_aware_score[b_range, dst_l_cut] = target_score

    time_aware_ranks = []
    top10_entity_names = []
    for j in range(len(src_l_cut)):
        tmp_score = time_aware_score[j]
        pred_ground = tmp_score[dst_l_cut[j]]
        ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
        ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
        target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
        time_aware_ranks.append(target_rank_i)
        topk_indices = torch.topk(tmp_score, k=min(20, tmp_score.shape[0])).indices.detach().cpu().tolist()
        top10_entity_names.append([id2entity[int(entity_id)] for entity_id in topk_indices])

    return path_block_list, time_aware_ranks, time_aware_score.detach().cpu().tolist(), top10_entity_names

if __name__ == "__main__":
    app = FastAPI()

    @app.post("/tkgr_server")
    async def recall(request: Request):
        data = await request.json()
        path_list, rank_list, score, top10_entity_names = deal_request(**data)
        result = {
            "path_list": path_list,
            "rank_list": rank_list,
            "entity_num": num_entities,
            "time_aware_score": score,
            "top10_entity_names": top10_entity_names,
        }
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
