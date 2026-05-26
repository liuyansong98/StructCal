import numpy as np
import random
import torch
import math
from bisect import bisect_left
from sample import *

PRECISION = 5


class NeighborFinder:
    def __init__(self, adj_list, temporal_bias=0, ts_precision=PRECISION, use_cache=False,
                 limit_ngh_span=False, ngh_span=None, num_entity=0,
                 data_name="ICEWS14"):
        self.limit_ngh_span = limit_ngh_span
        self.ngh_span_list = ngh_span
        self.temporal_bias = temporal_bias
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        self.use_cache = use_cache
        self.cache = {}
        self.ts_precision = ts_precision
        self.ngh_lengths = []
        self.ngh_time_lengths = []
        self.num_entity = num_entity
        self.node_degree = np.zeros([num_entity])
        # 最近30个邻居的平均时间间隔 * 0.5 (训练集上统计的)
        self.dataname2num = {"ICEWS14": 17, "ICEWS18": 252, "ICEWS05_15":1670, "GDELT26":113}
        self.data_name = data_name

    def init_node_degree(self):
        self.node_degree = np.zeros([self.num_entity])

    def update_node_degree(self, src_list, dst_list):
        self.node_degree[src_list] += 1
        self.node_degree[dst_list] += 1
    def init_off_set(self, adj_list):
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            ts_l = [x[2] for x in curr]
            n_ts_l.extend(ts_l)
            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time, e_idx=None):
        if self.use_cache:
            result = self.check_cache(src_idx, cut_time)
            if result is not None:
                return result[0], result[1], result[2], result[3]

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        start = off_set_l[src_idx]
        end = off_set_l[src_idx + 1]
        neighbors_idx = node_idx_l[start: end]
        neighbors_ts = node_ts_l[start: end]
        neighbors_e_idx = edge_idx_l[start: end]

        assert (len(neighbors_idx) == len(neighbors_ts) and len(neighbors_idx) == len(
            neighbors_e_idx))

        cut_idx = bisect_left_adapt(neighbors_ts, cut_time)
        result = (neighbors_idx[:cut_idx], neighbors_e_idx[:cut_idx], neighbors_ts[:cut_idx], None)
        if self.use_cache:
            self.update_cache(src_idx, cut_time, result)

        return result

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbor=20, e_idx_l=None, hop_flag=False, hop=None):
        assert (len(src_idx_l) == len(cut_time_l))
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.float32)
        begin_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            if cut_time < 0:
                continue
            ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = self.find_before(src_idx, cut_time, e_idx=e_idx_l[
                i] if e_idx_l is not None else None)

            cut_time = cut_time + 1
            if self.limit_ngh_span:
                if hop_flag:
                    k = int(self.ngh_span_list[hop])
                else:
                    k = int(self.ngh_span_list[0])

                if len(ngh_idx) >= k:
                    delta_t = cut_time - ngh_ts
                    sel_idx = np.argsort(delta_t)[:k]
                    ngh_idx = ngh_idx[sel_idx]
                    ngh_eidx = ngh_eidx[sel_idx]
                    ngh_ts = ngh_ts[sel_idx]

            sampled_times = np.zeros(len(ngh_idx))
            if len(ngh_idx) == 0:
                continue
            if len(ngh_idx) >= 30:
                delta_t_ = cut_time - ngh_ts
                time_delta_avg = np.mean(delta_t_[np.argsort(delta_t_)[:30]])
            else:
                delta_t_ = cut_time - ngh_ts
                time_delta_avg = np.mean(delta_t_)

            self.ngh_lengths.append(len(ngh_ts))
            self.ngh_time_lengths.append(ngh_ts[-1] - ngh_ts[0])

            if ngh_binomial_prob is None:
                # uniform sampling
                if math.isclose(self.temporal_bias, 0):
                    sampled_idx = np.sort(np.random.randint(0, len(ngh_idx), num_neighbor))
                # temporal sampling
                else:
                    time_delta = cut_time - ngh_ts
                    # {"GDELT":175, "ICEWS14s":17, "ICEWS18":252, "ICEWS05_15":1670, "WIKI":5, "YAGO":4, "Social_TKG":154901}
                    temperal_sampling_weight = np.exp(
                        - self.temporal_bias * self.dataname2num[self.data_name] * time_delta / time_delta_avg)
                    # temperal_sampling_weight = np.exp(
                    #     - self.temporal_bias * time_delta)
                    # temperal_sampling_weight[temperal_sampling_weight > 0.5] = 0.5    # 过大截断
                    # float64最小可表示4.9e-324 为了防止除0错误
                    temperal_sampling_weight_sum = temperal_sampling_weight.sum()
                    if temperal_sampling_weight_sum!=0:
                        sampling_weight = temperal_sampling_weight / temperal_sampling_weight.sum()
                    else:
                        sampling_weight = 1/len(ngh_idx) * np.ones(len(ngh_idx))
                    sampled_idx = np.sort(
                        np.random.choice(np.arange(len(ngh_idx)), num_neighbor, replace=True, p=sampling_weight))
            else:
                sampled_idx = seq_binary_sample(ngh_binomial_prob, num_neighbor)

            out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
            out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
            begin_ngh_t_batch[i, :] = cut_time
            out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

        return out_ngh_node_batch, out_ngh_eidx_batch, begin_ngh_t_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors, e_idx_l=None):
        if k == 0:
            return ([], [], [])
        batch = len(src_idx_l)
        layer_i = 0
        x, y, begin_ngh_t_batch, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors[layer_i],
                                             e_idx_l=e_idx_l, hop_flag=False)

        node_records = [x]
        eidx_records = [y]
        t_sample_record = [begin_ngh_t_batch]
        t_records = [z]

        for layer_i in range(1, k):
            # ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_sample_record[-1]
            ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_records[-1]
            ngh_node_est = ngh_node_est.flatten()
            ngh_e_est = ngh_e_est.flatten()
            ngh_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, begin_ngh_t_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est,
                                                                                                 ngh_t_est,
                                                                                                 num_neighbors[layer_i],
                                                                                                 e_idx_l=None,
                                                                                                 hop_flag=True,
                                                                                                 hop=layer_i)

            out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1)
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(batch, -1)
            begin_ngh_t_batch = begin_ngh_t_batch.reshape(batch, -1)
            out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_sample_record.append(begin_ngh_t_batch)
            t_records.append(out_ngh_t_batch)

        return (node_records, eidx_records, t_records)

    def find_k_hop_walk(self, k, src_idx_l, cut_time_l, n_walk=100, e_idx_l=None, recent_bias=1.0):
        if len(src_idx_l) == 0:
            return None, None, None
        n_idx_batch, e_idx_batch, ts_batch = [], [], []
        for sample_idx, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            e_idx = None if e_idx_l is None else e_idx_l[sample_idx]
            walks_n_idx, walks_e_idx, walks_ts = self.get_random_walks(src_idx, cut_time, n_walk=n_walk, len_walk=k,
                                                                       e_idx=e_idx, recent_bias=recent_bias)
            n_idx_batch.append(walks_n_idx)
            e_idx_batch.append(walks_e_idx)
            ts_batch.append(walks_ts)
        n_idx_batch, e_idx_batch, ts_batch = np.stack(n_idx_batch), np.stack(e_idx_batch), np.stack(ts_batch)
        return n_idx_batch, e_idx_batch, ts_batch

    def get_random_walks(self, src_idx, cut_time, n_walk=100, len_walk=5, e_idx=None, recent_bias=1.0):
        walks_n_idx, walks_e_idx, walks_ts = [], [], []
        for _ in range(n_walk):
            walk_n_idx, walk_e_idx, walk_ts = self.get_random_walk(src_idx, cut_time, seed=-1,
                                                                   len_walk=len_walk, e_idx=e_idx,
                                                                   recent_bias=recent_bias, packed=False)
            walks_n_idx.append(walk_n_idx)
            walks_e_idx.append(walk_e_idx)
            walks_ts.append(walk_ts)
        walks_n_idx, walks_e_idx, walks_ts = np.stack(walks_n_idx), np.stack(walks_e_idx), np.stack(walks_ts)
        return walks_n_idx, walks_e_idx, walks_ts

    def get_random_walk(self, src_idx, cut_time, seed=0, len_walk=5, e_idx=None, packed=False, recent_bias=1.0):
        if seed >= 0:
            random.seed(seed)
        cur_n_idx, cur_time, cur_e_idx = src_idx, cut_time, e_idx
        if packed:
            random_walk = [(src_idx, cut_time)]
            for hop in range(len_walk):
                n_idx_l, e_idx_l, ts_l = self.find_before(cur_n_idx, cur_time, e_idx=cur_e_idx)
                cur_len = len(n_idx_l)
                if cur_len == 0:
                    random_walk += [[0, 0.0]] * (len_walk - hop)
                    return random_walk
                r = random.random()
                r = -(1 - r) ** recent_bias + 1
                idx_picked = int(r * cur_len)
                cur_n_idx, cur_time, cur_e_idx = n_idx_l[idx_picked], ts_l[idx_picked], e_idx_l[
                    idx_picked]
                random_walk.append((cur_n_idx, cur_e_idx))
                return random_walk
        else:
            walk_n_idx, walk_e_idx, walk_ts = [cur_n_idx], [e_idx if e_idx is not None else -1], [cur_time]
            for hop in range(len_walk):
                n_idx_l, e_idx_l, ts_l = self.find_before(cur_n_idx, cur_time, e_idx=cur_e_idx)
                cur_len = len(n_idx_l)
                if cur_len == 0:
                    walk_n_idx.extend([0] * (len_walk - hop))
                    walk_e_idx.extend([0] * (len_walk - hop))
                    walk_ts.extend([0.0] * (len_walk - hop))
                    break
                r = random.random()
                r = -(1 - r) ** recent_bias + 1
                idx_picked = int(r * cur_len)
                cur_n_idx, cur_time, cur_e_idx = n_idx_l[idx_picked], ts_l[idx_picked], e_idx_l[
                    idx_picked]
                walk_n_idx.append(cur_n_idx)
                walk_e_idx.append(cur_e_idx)
                walk_ts.append(cur_time)
            walks_n_idx, walks_e_idx, walks_ts = np.array(walk_n_idx, dtype=int), \
                                                 np.array(walk_e_idx, dtype=int), np.array(walk_ts, dtype=float)
            return walks_n_idx, walks_e_idx, walks_ts

    def update_cache(self, node, ts, results):
        ts_str = str(round(ts, PRECISION))
        key = (node, ts_str)
        if key not in self.cache:
            self.cache[key] = results

    def check_cache(self, node, ts):
        ts_str = str(round(ts, PRECISION))
        key = (node, ts_str)
        return self.cache.get(key)
