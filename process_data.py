import os
import json
import heapq
import numpy as np
import pandas as pd
from contextlib import ExitStack
from collections import defaultdict, OrderedDict, deque
from typing import Dict, Any

class LRUCache1:
    def __init__(self):
        self.cache = OrderedDict()

    def put(self, key: int, value: int) -> None:
        freq = 1
        if key in self.cache:
            # 如果 key 已经存在，更新其值并将其移到末尾
            _, freq = self.cache[key]
            freq += 1
            self.cache.move_to_end(key)
        # 更新频率
        self.cache[key] = (value, freq)

    def pop(self):
        return self.cache.popitem(last=False)

    def remove(self, key):
        del self.cache[key]


class LFUCache1:
    def __init__(self):
        self.cache = {}  # key -> (value, freq)
        self.freq = defaultdict(OrderedDict)  # freq -> OrderedDict of keys preserving LRU order
        self.min_freq = 0

    def update_fre(self, key: int) -> int:
        if key not in self.cache:
            return -1
        value, freq = self.cache[key]
        # 将该键从当前频率列表中移除
        del self.freq[freq][key]
        # 空频率列表且频率正好是当前最小频率时，min_freq 加 1
        if not self.freq[freq]:
            del self.freq[freq]
            if freq == self.min_freq:
                self.min_freq += 1
        # 更新频率并加入新的 freq 列表
        self.freq[freq + 1][key] = value
        self.cache[key] = (value, freq + 1)
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 更新已有条目内容并递增频率
            _, freq = self.cache[key]
            self.cache[key] = (value, freq)
            self.update_fre(key)  # 更新频率
            return

        # 插入新键，频率初始化为 1
        self.cache[key] = (value, 1)
        self.freq[1][key] = value
        self.min_freq = 1

    def pop(self):
        evict_key, _ = self.freq[self.min_freq].popitem(last=False)
        # 空频率列表且频率正好是当前最小频率时，min_freq 加 1
        if not self.freq[self.min_freq]:
            del self.freq[self.min_freq]
            if self.freq:
                self.min_freq = min(self.freq.keys())
            else:
                self.min_freq = 0
        del self.cache[evict_key]
        return evict_key


class ARCCache1:
    def __init__(self, capacity):
        self.capacity = capacity
        self.p = 0  # T1 的目标大小
        self.cache = {}  # key -> value
        self.t1 = LRUCache1()  # 最近访问但只访问一次的缓存页（T1）, LRU
        self.b1 = deque()  # T1 的 ghost 列表（只保存 key）
        self.t2 = LFUCache1()  # 访问频繁的缓存页（T2）, LFU
        self.b2 = deque()  # T2 的 ghost 列表

    def set_capccity(self, capacity):
        self.capacity = capacity

    def replace(self, key):
        # 按 ARC 论文策略决定从 T1 还是 T2 驱逐
        if self.t1.cache and ((key in self.b2 and len(self.t1.cache) == self.p) or (len(self.t1.cache) > self.p)):
            old, _ = self.t1.pop()
            self.b1.appendleft(old)
        else:
            old = self.t2.pop()
            self.b2.appendleft(old)
        del self.cache[old]

    def put(self, key, value):
        if key in self.cache:
            # 命中：若在 T1 移入 T2，否则保持在 T2
            if key in self.t1.cache:
                self.t1.remove(key)
                self.t2.put(key, value)
                freq = 2
            else:
                _, freq = self.cache[key]
                self.t2.put(key, value)
                freq += 1
            # 写入缓存
            self.cache[key] = (value, freq)
            # a = set(self.cache.keys())
            # b = set(self.t1.cache.keys()).union(set(self.t2.cache.keys()))
            # if a != b:
            #     print("cache集合不相等", )
            return

        # 未命中：先加载
        # 如果在 B1，说明曾经近期被驱逐
        if key in self.b1:
            self.p = min(self.capacity, self.p + max(len(self.b2) // len(self.b1), 1))
            self.replace(key)
            self.b1.remove(key)
            self.t2.put(key, value)
            freq = 2
        # 如果在 B2，说明曾经频繁访问过但被驱逐
        elif key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) // len(self.b2), 1))
            self.replace(key)
            self.b2.remove(key)
            self.t2.put(key, value)
            freq = 2
        elif len(self.t1.cache) + len(self.b1) == self.capacity:
            # 新 key
            if len(self.t1.cache) < self.capacity:
                self.b1.pop()
                self.replace(key)
            else:
                old, _ = self.t1.pop()
                del self.cache[old]
            self.t1.put(key, value)
            freq = 1
        else:
            total = len(self.t1.cache) + len(self.b1) + len(self.t2.cache) + len(self.b2)
            if total >= self.capacity:
                if total == 2 * self.capacity:
                    self.b2.pop()
                self.replace(key)
            self.t1.put(key, value)
            freq = 1

        # 最后写入缓存
        self.cache[key] = (value, freq)
        # a = set(self.cache.keys())
        # b = set(self.t1.cache.keys()).union(set(self.t2.cache.keys()))
        # if a != b:
        #     print("cache集合不相等", )
        return

# 配置路径
DATA_ROOT = './data/dataset'
DATA_DIR = ''
TRAIN_FILE = ''
VALID_FILE = ''
TEST_FILE = ''
TEST_FILE_1_to_3 = ''
TEST_FILE_3_to_10 = ''
TEST_FILE_eq_1 = ''
TEST_FILE_1_to_10 = ''
TEST_FILE_3_to_100 = ''
TEST_FILE_10_to_100 = ''
TEST_FILE_gt_10 = ''
TEST_FILE_gt_3 = ''
ENT2ID_FILE = ''
REL2ID_FILE = ''
STAT_FILE = ''


def configure_dataset_paths(dataset_name: str):
    global DATA_DIR
    global TRAIN_FILE
    global VALID_FILE
    global TEST_FILE
    global TEST_FILE_1_to_3
    global TEST_FILE_3_to_10
    global TEST_FILE_eq_1
    global TEST_FILE_1_to_10
    global TEST_FILE_3_to_100
    global TEST_FILE_10_to_100
    global TEST_FILE_gt_10
    global TEST_FILE_gt_3
    global ENT2ID_FILE
    global REL2ID_FILE
    global STAT_FILE

    DATA_DIR = os.path.join(DATA_ROOT, dataset_name)
    TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
    VALID_FILE = os.path.join(DATA_DIR, 'valid.txt')
    TEST_FILE = os.path.join(DATA_DIR, 'test.txt')
    TEST_FILE_1_to_3 = os.path.join(DATA_DIR, 'test_rank_1_to_3.txt')
    TEST_FILE_3_to_10 = os.path.join(DATA_DIR, 'test_rank_3_to_10.txt')
    TEST_FILE_eq_1 = os.path.join(DATA_DIR, 'test_rank_eq_1.txt')
    TEST_FILE_1_to_10 = os.path.join(DATA_DIR, 'test_rank_1_to_10.txt')
    TEST_FILE_3_to_100 = os.path.join(DATA_DIR, 'test_rank_3_to_100.txt')
    TEST_FILE_10_to_100 = os.path.join(DATA_DIR, 'test_rank_10_to_100.txt')
    TEST_FILE_gt_10 = os.path.join(DATA_DIR, 'test_rank_gt_10.txt')
    TEST_FILE_gt_3 = os.path.join(DATA_DIR, 'test_rank_gt_3.txt')
    ENT2ID_FILE = os.path.join(DATA_DIR, 'entity2id.txt')
    REL2ID_FILE = os.path.join(DATA_DIR, 'relation2id.txt')
    STAT_FILE = os.path.join(DATA_DIR, 'stat.txt')


configure_dataset_paths('ICEWS14s_divide')


def build_output_paths(data_dir, history_len, mode):
    suffix = f"{mode}_h{history_len}"
    return (
        os.path.join(data_dir, f"train_{suffix}.jsonl"),
        os.path.join(data_dir, f"valid_{suffix}.jsonl"),
        os.path.join(data_dir, f"test_{suffix}.jsonl"),
    )


def build_filtered_test_output_paths(data_dir, history_len, mode):
    suffix = f"{mode}_h{history_len}"
    return {
        "test_rank_1_to_3": os.path.join(data_dir, f"test_rank_1_to_3_{suffix}.jsonl"),
        "test_rank_3_to_10": os.path.join(data_dir, f"test_rank_3_to_10_{suffix}.jsonl"),
        "test_rank_eq_1": os.path.join(data_dir, f"test_rank_eq_1_{suffix}.jsonl"),
        "test_rank_1_to_10": os.path.join(data_dir, f"test_rank_1_to_10_{suffix}.jsonl"),
        "test_rank_3_to_100": os.path.join(data_dir, f"test_rank_3_to_100_{suffix}.jsonl"),
        "test_rank_10_to_100": os.path.join(data_dir, f"test_rank_10_to_100_{suffix}.jsonl"),
        "test_rank_gt_10": os.path.join(data_dir, f"test_rank_gt_10_{suffix}.jsonl"),
        "test_rank_gt_3": os.path.join(data_dir, f"test_rank_gt_3_{suffix}.jsonl"),
    }


def load_id_mapping(filepath):
    """加载 ID 到 Name 的映射字典，并将下划线替换为空格"""
    id2name = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                name, _id = parts
                # 【修改点】将下划线替换为空格，提升 LLM 理解能力
                name = name.replace('_', ' ')
                id2name[int(_id)] = name
    return id2name


def load_name_mapping(filepath):
    """加载 ID 到 Name 的映射字典，并将下划线替换为空格"""
    name2id = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                name, _id = parts
                # 【修改点】将下划线替换为空格，提升 LLM 理解能力
                name = name.replace('_', ' ')
                name2id[name] = int(_id)
    return name2id


def _build_recurring_history_text(h_name, r_name, current_time, interactions, top_k=None):
    """Build structured recurring history text for prompt consumption."""
    if top_k is not None:
        interactions = interactions[:top_k]

    lines = [
        f"Recurring historical statistics for ({h_name}, {r_name}, *, {current_time}):",
        "Use higher freq and lower recency as stronger recurring evidence.",
        "<recurring_entity_stats>",
    ]
    if not interactions:
        lines.append("None")
    else:
        for i, item in enumerate(interactions, 1):
            latest_time = int(item["latest_time"])
            recency = max(int(current_time) - latest_time, 0)
            lines.append(
                f"{i}. {item['name']} | freq={int(item['frequency'])} | recency={recency}"
            )

    lines.append("</recurring_entity_stats>")
    return "\n".join(lines)


def get_history_str_freq(head_id, h_name, r_id, r_name, current_time, history_data, id2entity, top_k=5):
    """Generate recurring history text sorted by frequency then recency."""
    if (head_id, r_id) not in history_data or not history_data[(head_id, r_id)]:
        return _build_recurring_history_text(h_name, r_name, current_time, interactions=[])

    interactions = []
    for tail_id, stats in history_data[(head_id, r_id)].items():
        interactions.append({
            'name': id2entity.get(tail_id, f"Entity_{tail_id}"),
            'frequency': int(stats['freq']),
            'latest_time': int(stats['last_time'])
        })

    interactions.sort(key=lambda x: (-x['frequency'], -x['latest_time']))
    return _build_recurring_history_text(
        h_name=h_name,
        r_name=r_name,
        current_time=current_time,
        interactions=interactions,
        top_k=top_k,
    )


def get_history_str_arc(head_id, h_name, r_id, r_name, current_time, history_entity_dic, id2entity):
    """Generate recurring history text from ARC cache state."""
    if len(history_entity_dic[(head_id, r_id)].cache) == 0:
        return _build_recurring_history_text(h_name, r_name, current_time, interactions=[])

    key_o = list(history_entity_dic[(head_id, r_id)].cache.keys())
    value_freq = list(history_entity_dic[(head_id, r_id)].cache.values())
    value_t, freq_t = zip(*value_freq)

    interactions = []
    for object_id, latest_time, freq in zip(key_o, value_t, freq_t):
        interactions.append({
            'name': id2entity[int(object_id)],
            'frequency': int(freq),
            'latest_time': int(latest_time),
        })

    interactions.sort(key=lambda x: (-x['frequency'], -x['latest_time']))
    return _build_recurring_history_text(
        h_name=h_name,
        r_name=r_name,
        current_time=current_time,
        interactions=interactions,
    )


def get_history_str_recent(head_id, h_name, r_id, r_name, current_time, history_data, id2entity, top_k=5):
    """Generate recurring history text sorted by recency then frequency."""
    if (head_id, r_id) not in history_data or not history_data[(head_id, r_id)]:
        return _build_recurring_history_text(h_name, r_name, current_time, interactions=[])

    interactions = []
    for tail_id, stats in history_data[(head_id, r_id)].items():
        interactions.append({
            'name': id2entity.get(tail_id, f"Entity_{tail_id}"),
            'frequency': int(stats['freq']),
            'latest_time': int(stats['last_time'])
        })

    interactions.sort(key=lambda x: (-x['latest_time'], -x['frequency']))
    return _build_recurring_history_text(
        h_name=h_name,
        r_name=r_name,
        current_time=current_time,
        interactions=interactions,
        top_k=top_k,
    )

def update_history(history_data, head_id, r_id, tail_id, timestamp):
    """更新历史记录"""
    if tail_id not in history_data[(head_id, r_id)]:
        history_data[(head_id, r_id)][tail_id] = {'freq': 0, 'last_time': -1}

    history_data[(head_id, r_id)][tail_id]['freq'] += 1
    history_data[(head_id, r_id)][tail_id]['last_time'] = timestamp


def process_file(input_path, output_path, id2entity, id2rel, history_data, history_entity_dic, history_len, mode, start_idx=0):
    print(f"Processing {input_path} -> {output_path} ...")

    # 假设无表头，列为: head, rel, tail, time, ignore
    df = pd.read_csv(input_path, sep='\t', header=None, names=['head', 'rel', 'tail', 'time', 'ignore'])

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, row in df.iterrows():
            h_id = int(row['head'])
            r_id = int(row['rel'])
            t_id = int(row['tail'])
            time_val = int(row['time'])

            h_name = id2entity.get(h_id, str(h_id))
            r_name = id2rel.get(r_id, str(r_id))
            t_name = id2entity.get(t_id, str(t_id))

            # 生成历史 (基于当前时刻之前)
            if mode == "arc":
                history_str = get_history_str_arc(h_id, h_name, r_id, r_name, time_val, history_entity_dic, id2entity)
            elif mode == "freq":
                history_str = get_history_str_freq(h_id, h_name, r_id, r_name, time_val, history_data, id2entity, top_k=history_len)
            elif mode == "recent":
                history_str = get_history_str_recent(h_id, h_name, r_id, r_name, time_val, history_data, id2entity, top_k=history_len)
            else:
                raise ValueError(f"Unsupported mode: {mode}. Expected one of ['arc', 'freq', 'recent'].")

            data_obj = {
                "idx": start_idx + i,
                "query": f"({h_name}, {r_name}, ?, {time_val})",
                "target": t_name,
                "history": history_str
            }

            f_out.write(json.dumps(data_obj, ensure_ascii=False) + '\n')

            # 更新历史
            update_history(history_data, h_id, r_id, t_id, time_val)
            if mode == "arc":
                history_entity_dic[(h_id, r_id)].put(t_id, time_val)

    print(f"data length: {len(df)}")
    return start_idx + len(df)


def process_file_filter(input_path, output_path, id2entity, id2rel, history_data, history_entity_dic, history_len, mode, start_idx=0):
    print(f"Processing filtered test buckets from {input_path} ...")

    df = pd.read_csv(input_path, sep='\t', header=None, names=['head', 'rel', 'tail', 'time', 'ignore'])
    filtered_output_paths = build_filtered_test_output_paths(DATA_DIR, history_len, mode)
    filter_key_sets = {
        "test_rank_1_to_3": load_filter_key_set(TEST_FILE_1_to_3),
        "test_rank_3_to_10": load_filter_key_set(TEST_FILE_3_to_10),
        "test_rank_eq_1": load_filter_key_set(TEST_FILE_eq_1),
        "test_rank_1_to_10": load_filter_key_set(TEST_FILE_1_to_10),
        "test_rank_3_to_100": load_filter_key_set(TEST_FILE_3_to_100),
        "test_rank_10_to_100": load_filter_key_set(TEST_FILE_10_to_100),
        "test_rank_gt_10": load_filter_key_set(TEST_FILE_gt_10),
        "test_rank_gt_3": load_filter_key_set(TEST_FILE_gt_3),
    }

    for path in filtered_output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    bucket_counts = defaultdict(int)
    with ExitStack() as stack:
        writers = {
            bucket_name: stack.enter_context(open(path, 'w', encoding='utf-8'))
            for bucket_name, path in filtered_output_paths.items()
        }

        for i, row in df.iterrows():
            h_id = int(row['head'])
            r_id = int(row['rel'])
            t_id = int(row['tail'])
            time_val = int(row['time'])

            h_name = id2entity.get(h_id, str(h_id))
            r_name = id2rel.get(r_id, str(r_id))
            t_name = id2entity.get(t_id, str(t_id))

            if mode == "arc":
                history_str = get_history_str_arc(h_id, h_name, r_id, r_name, time_val, history_entity_dic, id2entity)
            elif mode == "freq":
                history_str = get_history_str_freq(h_id, h_name, r_id, r_name, time_val, history_data, id2entity, top_k=history_len)
            elif mode == "recent":
                history_str = get_history_str_recent(h_id, h_name, r_id, r_name, time_val, history_data, id2entity, top_k=history_len)
            else:
                raise ValueError(f"Unsupported mode: {mode}. Expected one of ['arc', 'freq', 'recent'].")

            data_obj = {
                "idx": start_idx + i,
                "query": f"({h_name}, {r_name}, ?, {time_val})",
                "target": t_name,
                "history": history_str
            }
            key = (h_id, r_id, t_id, time_val)
            json_line = json.dumps(data_obj, ensure_ascii=False) + '\n'

            for bucket_name, key_set in filter_key_sets.items():
                if key in key_set:
                    writers[bucket_name].write(json_line)
                    bucket_counts[bucket_name] += 1

            update_history(history_data, h_id, r_id, t_id, time_val)
            if mode == "arc":
                history_entity_dic[(h_id, r_id)].put(t_id, time_val)

    for bucket_name, output_path_i in filtered_output_paths.items():
        print(f"{bucket_name} -> {output_path_i}, data length: {bucket_counts[bucket_name]}")

    print(f"all test data length: {len(df)}")
    return start_idx + len(df)

def update_cache(input_path, output_path, history_data, history_entity_dic, mode, start_idx=0):
    print(f"Updating cache ")

    # 假设无表头，列为: head, rel, tail, time, ignore
    df = pd.read_csv(input_path, sep='\t', header=None, names=['head', 'rel', 'tail', 'time', 'ignore'])
    for i, row in df.iterrows():
        h_id = int(row['head'])
        r_id = int(row['rel'])
        t_id = int(row['tail'])
        time_val = int(row['time'])
        # 更新历史
        update_history(history_data, h_id, r_id, t_id, time_val)
        if mode == "arc":
            history_entity_dic[(h_id, r_id)].put(t_id, time_val)

    print(f"data length: {len(df)}")
    return start_idx + len(df)

def load_stat(path):
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    parts = line.split("\t")
    if len(parts) < 2:
        raise ValueError(f"Invalid stat file format in {path}: expected two tab-separated integers, got: {line!r}")
    entity_num = int(parts[0])
    relation_num = int(parts[1])
    return entity_num, relation_num


def load_filter_key_set(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: filter file not found, skip this bucket: {filepath}")
        return set()

    df = pd.read_csv(filepath, sep='\t', header=None, names=['head', 'rel', 'tail', 'time', 'ignore'])
    key_set = set()
    for _, row in df.iterrows():
        key_set.add((
            int(row['head']),
            int(row['rel']),
            int(row['tail']),
            int(row['time']),
        ))
    return key_set

def get_data_jsonl(history_len: int, modes: list[str]):
    for mode in modes:
        mode = mode.lower()
        if mode not in {"arc", "freq", "recent"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of ['arc', 'freq', 'recent'].")

        output_train, output_valid, output_test = build_output_paths(DATA_DIR, history_len, mode)

        print("Loading mappings (converting underscores to spaces)...")
        if not os.path.exists(ENT2ID_FILE) or not os.path.exists(REL2ID_FILE):
            print("Error: Mapping files not found.")
            return

        id2entity = load_id_mapping(ENT2ID_FILE)
        id2rel = load_id_mapping(REL2ID_FILE)
        num_entity, num_rels = load_stat(STAT_FILE)
        print(f"Loaded {len(id2entity)} entities and {len(id2rel)} relations.")

        global_history = defaultdict(dict)
        current_idx = 0

        history_entity_dic: Dict[(int, int), ARCCache1] = {}
        for i in range(num_entity):
            for j in range(2 * num_rels):
                history_entity_dic[(i, j)] = ARCCache1(history_len)

        if os.path.exists(TRAIN_FILE):
            current_idx = process_file(
                TRAIN_FILE, output_train, id2entity, id2rel, global_history, history_entity_dic, history_len, mode, current_idx
            )

        if os.path.exists(VALID_FILE):
            current_idx = process_file(
                VALID_FILE, output_valid, id2entity, id2rel, global_history, history_entity_dic, history_len, mode, current_idx
            )

        if os.path.exists(TEST_FILE):
            current_idx = process_file(
                TEST_FILE, output_test, id2entity, id2rel, global_history, history_entity_dic, history_len, mode, current_idx
            )

    print("Done! JSONL files generated with cleaned names.")


def get_data_jsonl_filter_test(history_len: int, modes: list[str]):
    for mode in modes:
        mode = mode.lower()
        if mode not in {"arc", "freq", "recent"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of ['arc', 'freq', 'recent'].")

        output_train, output_valid, output_test = build_output_paths(DATA_DIR, history_len, mode)

        print("Loading mappings (converting underscores to spaces)...")
        if not os.path.exists(ENT2ID_FILE) or not os.path.exists(REL2ID_FILE):
            print("Error: Mapping files not found.")
            return

        id2entity = load_id_mapping(ENT2ID_FILE)
        id2rel = load_id_mapping(REL2ID_FILE)
        num_entity, num_rels = load_stat(STAT_FILE)
        print(f"Loaded {len(id2entity)} entities and {len(id2rel)} relations.")

        global_history = defaultdict(dict)
        current_idx = 0

        history_entity_dic: Dict[(int, int), ARCCache1] = {}
        for i in range(num_entity):
            for j in range(2 * num_rels):
                history_entity_dic[(i, j)] = ARCCache1(history_len)

        current_idx = update_cache(
            TRAIN_FILE, output_train, global_history, history_entity_dic, mode, current_idx
        )

        current_idx = update_cache(
            VALID_FILE, output_valid, global_history, history_entity_dic, mode, current_idx
        )

        if os.path.exists(TEST_FILE):
            current_idx = process_file_filter(
                TEST_FILE, output_test, id2entity, id2rel, global_history, history_entity_dic, history_len, mode, current_idx
            )



    print("Done! JSONL files generated with cleaned names.")

def deal_id_mapping():
    id2entity = load_id_mapping(ENT2ID_FILE)
    id2rel = load_id_mapping(REL2ID_FILE)
    entity2id = load_name_mapping(ENT2ID_FILE)
    rel2id = load_name_mapping(REL2ID_FILE)
    with open(os.path.join(DATA_DIR, 'id2entity.json'), "w", encoding="utf-8") as f:
        json.dump(id2entity, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, 'id2relation.json'), "w", encoding="utf-8") as f:
        json.dump(id2rel, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, 'entity2id.json'), "w", encoding="utf-8") as f:
        json.dump(entity2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, 'relation2id.json'), "w", encoding="utf-8") as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    dataset_names = [
        "ICEWS14",
        "GDELT26",
        "ICEWS18",
        "ICEWS05_15",
    ]
    # history_lens = [2, 6, 10, 14, 18]
    history_lens = [10]
    modes = ["freq", "recent"]

    for dataset_name in dataset_names:
        configure_dataset_paths(dataset_name)
        print(f"\n========== Dataset: {dataset_name} ==========")
        for history_len in history_lens:
            print(f"\n------ history_len={history_len} ------")
            get_data_jsonl(history_len, modes)
    # deal_id_mapping()
