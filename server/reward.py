import argparse
import json
import os
import numpy as np
import torch.nn
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.datasets.prompts_dataset import *
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer, LlamaModel
import torch.nn.functional as F
import pandas as pd
from server.utils import parse_command_line_args
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict

@dataclass
class AnswerScore:
    answer: str
    score: float
    line_no: int  # 原始行号(1-10)，用于同分稳定排序

logger = init_logger(__name__)

class RecRuleProxy:
    def __init__(self, config):

        self.config = config
        self.idx2label = dict()
        self.idx2query = dict()
        with open(config["data_file_path"], encoding='utf-8') as f:
            for line in f:
                data_dict = json.loads(line)
                # 样本id --> object
                self.idx2label[str(data_dict['idx'])] = data_dict['target'].strip()
                self.idx2query[str(data_dict['idx'])] = data_dict['query'].strip()

        self.data_root = config["data_root"]
        self.data_name = config["data_name"]
        self.relation2id = self.load_name_mapping(config["relation2id"], config["num_relations"], True)
        self.entity2id = self.load_name_mapping(config["entity2id"], config["num_relations"])
        self.entity2id_norm = self._build_normalized_entity2id(self.entity2id)
        self.id2relation = self.load_id_mapping(config["id2entity"], config["num_relations"], True)
        self.id2entity = self.load_id_mapping(config["id2entity"], config["num_relations"])

        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name_or_path"])
        self.eos_token = tokenizer.eos_token
        self.pad_token_escaped = re.escape(tokenizer.pad_token)
        self.eos_token_escaped = re.escape(tokenizer.eos_token)
        self.path_split_identifier = config["path_split_identifier"]
        self.log_file_path = os.path.join(config["log_dir"], config["log_file"])
        if self.log_file_path is not None:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        self.stage = config["stage"]
        self.target_interaction_round = int(config.get("interaction_round", 0))
        self.interaction_round_mismatch_penalty = float(config.get("interaction_round_mismatch_penalty", 0.0))
        self.format_error = config["format_error"]
        self.format_correct = config["format_correct"]
        self.invalid_entity_hard_threshold = int(config.get("invalid_entity_hard_threshold", 8))
        self.outcome_error = config["outcome_error"]
        self.outcome_correct = config["outcome_correct"]
        self.outcome_linear = config["outcome_linear"]
        # self.recall_num_base = config["recall_num_base"]
        # self.recall_num_max = config["recall_num_max"]
        self.tkgr_gain = config["tkgr_gain"]
        self.llm_gain = config["llm_gain"]
        # self.test_result = self.load_test_npy(config["test_result"])
        self.train_result = None
        self.start_idx = int(config.get("start_idx", 0))
        if config.get("train_result"):
            self.train_result = self.load_test_npy(config["train_result"])

        self.all_heads, self.all_tails, self.all_rels, self.all_timestamps = self.load_temporal_knowledge_graph(self.data_name)
        self.srt2o = defaultdict(list)
        # 计算filter的过滤字典
        for i in range(len(self.all_heads)):
            self.srt2o[(self.all_heads[i], self.all_rels[i], self.all_timestamps[i])].append(self.all_tails[i])

    def load_temporal_knowledge_graph(self, dataset_name):

        train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
        column_names = ['head', 'rel', 'tail', 'time', '_']
        train_data_table = self.load_data_table(dataset_name, train_file, column_names)
        val_data_table = self.load_data_table(dataset_name, val_file, column_names)
        test_data_table = self.load_data_table(dataset_name, test_file, column_names)
        all_data_table = pd.concat([train_data_table, val_data_table, test_data_table], ignore_index=True)
        eidx = np.arange(len(all_data_table), dtype=int)
        print(
            f"dataName:{dataset_name}\n train:{len(train_data_table)}, val:{len(val_data_table)}, test:{len(test_data_table)}, all data:{len(all_data_table)}")

        stat_table = self.load_data_table(dataset_name, "stat.txt", column_names=['num_entities', 'num_relations', '_'])
        num_entities, num_relations = stat_table['num_entities'].item(), stat_table['num_relations'].item()

        all_heads = all_data_table['head'].to_numpy()
        all_tails = all_data_table['tail'].to_numpy()
        all_rels = all_data_table['rel'].to_numpy()
        all_timestamps = all_data_table['time'].to_numpy()
        _ = all_data_table['_'].to_numpy()
        # all_timestamps = eidx
        edge_idxs = eidx

        # return all_heads, all_tails, all_rels, all_timestamps, _, edge_idxs, num_entities, num_relations
        return all_heads, all_tails, all_rels, all_timestamps

    def load_data_table(self, graph_name, file_name, column_names=None):
        data_fpath = os.path.join(self.data_root, graph_name, file_name)
        return pd.read_table(data_fpath, sep='\t', names=column_names)

    def load_test_npy(self, filename):
        # 读取文件
        return np.load(filename, allow_pickle=True)

    def load_id_mapping(self, filepath, num_relations, is_rel=False):
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            id2name = json.load(file)
        id2name = {int(k): v for k, v in id2name.items()}
        if is_rel:
            inv_id2name = {}
            for key, value in id2name.items():
                inv_id2name[int(key) + num_relations] = "INV::" + value
            id2name.update(inv_id2name)
        return id2name

    def load_name_mapping(self, filepath, num_relations, is_rel=False):
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            name2id = json.load(file)
        if is_rel:
            inv_name2id = {}
            for key, value in name2id.items():
                inv_name2id["INV::" + key] = int(value) + num_relations
            name2id.update(inv_name2id)
        return name2id

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        s = str(name).strip().lower()
        quote_chars = {'"', "'", '`', chr(0x2018), chr(0x2019), chr(0x201C), chr(0x201D)}
        while s and s[0] in quote_chars:
            s = s[1:].lstrip()
        while s and s[-1] in quote_chars:
            s = s[:-1].rstrip()
        return s

    def _build_normalized_entity2id(self, entity2id):
        normalized = {}
        for name, entity_id in entity2id.items():
            norm_name = self.normalize_entity_name(name)
            if norm_name and norm_name not in normalized:
                normalized[norm_name] = int(entity_id)
        return normalized

    def _entity_id_from_name(self, name: str):
        norm_name = self.normalize_entity_name(name)
        return self.entity2id_norm.get(norm_name)

    def _log_str(self, s):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(s + '\n')

    def _log_json(self, d):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    def is_start_with_number_dot(self, text):
        # ^ 表示开头
        # \d+ 表示一个或多个数字
        # \. 表示匹配原始的点字符
        pattern = r"^\d+\."

        if re.match(pattern, text):
            return True
        return False

    def _process_query(self, text):
        pattern = f"^({self.eos_token_escaped}|{self.pad_token_escaped})+"
        text = re.sub(pattern, "", text)

        pattern = f"({self.eos_token_escaped}|{self.pad_token_escaped})+$"
        text = re.sub(pattern, "", text)

        return text

    def _get_qa(self, text):
        remove_prefix = " ".join(text.split("**Input**")[1:])
        question = remove_prefix.split(f"<|im_start|>assistant")[0].strip()
        solution = text.split(f"<|im_start|>assistant")[-1].strip()
        return question, solution

    def _get_pred(self, text):
        return text.split(PRED_BEG)[-1].split(PRED_END)[0].strip()

    def parse_top(self, text: str) -> List[AnswerScore]:
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

    def rank_answers(self, items: List[AnswerScore]) -> List[AnswerScore]:
        """
        按 score 降序排序。
        同分时用 line_no 升序保证稳定（即原列表靠前的排名更高）。
        """
        return sorted(items, key=lambda x: (-x.score, x.line_no))

    def find_entity_rank(self, text: str, entity: str) -> Tuple[
        bool, Optional[int], List[AnswerScore]]:
        """
        返回：是否存在、排名(1-based, 按score)、排序后的完整榜单
        """
        items = self.parse_top(text)
        ranked = self.rank_answers(items)

        entity_norm = self.normalize_entity_name(entity)

        for i, r in enumerate(ranked, start=1):
            if self.normalize_entity_name(r.answer) == entity_norm:
                return True, i, ranked

        return False, None, ranked

    def calc_ranks(self, tmp_score, target_id, head_id, rel_id, timestamp):
        pred_ground = tmp_score[target_id]
        tmp_score[self.srt2o[head_id, rel_id, timestamp]] = -10000000
        tmp_score[target_id] = pred_ground
        ob_pred_comp1 = (tmp_score > pred_ground)
        ob_pred_comp2 = (tmp_score == pred_ground)
        rank = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1

        return rank

    def parse_query_str(self, query_str: str):
        pattern = re.compile(r"^\(\s*(.*)\s*,\s*(.*)\s*,\s*\?\s*,\s*([^)]+)\s*\)$")
        match = pattern.match(query_str.strip())
        if not match:
            raise ValueError(f"Invalid query format: {query_str}")

        head_str = match.group(1).strip()
        relation_str = match.group(2).strip()
        timestamp_str = match.group(3).strip()

        head_id = self._entity_id_from_name(head_str)
        if head_id is None:
            raise KeyError(f"Unknown head entity: {head_str}")
        rel_id = self.relation2id[relation_str]

        return head_str, relation_str, timestamp_str, head_id, rel_id, int(timestamp_str)


    def _get_outcome_reward(self, query, pred, label, idx, time_aware_score=None) -> float:
        pred_list = pred.split(self.path_split_identifier)
        pred_list = [pred.split('. ', 1)[-1].strip() for pred in pred_list]
        target_id = self._entity_id_from_name(label)
        if target_id is None:
            return 0
        query_str = self.idx2query[idx]
        print(idx, query_str)
        head_str, relation_str, timestamp_str, head_id, rel_id, timestamp = self.parse_query_str(query_str)

        res_np = np.asarray(time_aware_score, dtype=np.float32).copy()
        id = int(idx) - self.start_idx
        init_train_res_np = self.train_result[id].copy()


        # 计算传统TKGR预测指标提升的奖励

        first_rank = self.calc_ranks(init_train_res_np, target_id, head_id, rel_id, timestamp)
        last_rank = self.calc_ranks(res_np, target_id, head_id, rel_id, timestamp)
        path_denoise_reward =  max((1 / last_rank - 1 / first_rank) * self.tkgr_gain, 0)

        # LLM 预测的排名
        ok, llm_target_rank, ranked = self.find_entity_rank(pred, label)
        for answer in ranked:
            entityid = self._entity_id_from_name(answer.answer)
            if entityid is not None:
                res_np[entityid] += answer.score * 1

        target_rank = self.calc_ranks(res_np, target_id, head_id, rel_id, timestamp)

        llm_adv_reward = 0
        # LLM+Graph Reasoner预测指标的奖励函数
        pred_reward = 0
        pred_reward = self.outcome_linear * (1.0 / target_rank)
        # LLM优于传统TKGR模型的奖励函数
        # llm_adv_reward = max((1/llm_target_rank - 1/first_rank) * self.llm_gain, 0)

        print(f"path_denoise_reward: {path_denoise_reward}, llm_adv_reward: {llm_adv_reward}, pred_reward: {pred_reward}")
        return path_denoise_reward + llm_adv_reward + pred_reward


    def _get_recall_num_reward(self, recall_num, base=0.05, num_max=4):

        recall_num = min(recall_num, num_max)
        if recall_num <= 1:
            return 0

        reward = (recall_num - 1) * base

        return reward

    def path_format_reward(self, selected_paths_str, idx)->bool:
        # 例：selected_paths_str 形如：
        # "PATH_1: s1 -> r1(t1) -> r2(t2) -> o1;
        #  PATH_2: s2 -> r3(t3) -> o2;
        #  PATH_3: s3 -> r4(t4) -> r5(t5) -> r6(t6) -> o3;"

        # 解析 "relation(t)" 片段的正则：
        # STEP_PATTERN = re.compile(
        #     r"(?P<rel>[^(>\n]+?)\s*\(\s*(?P<time>[^)]+)\s*\)"
        # )
        STEP_PATTERN = re.compile(r"^\s*(?P<rel>.*)\(\s*(?P<time>\d+)\s*\)\s*$")

        # 不符合要求。没有选出合适的路径不能不输出
        if selected_paths_str is None:
            return True

        # 按行遍历
        line_count = 0
        error_path_count = 0
        for raw_line in selected_paths_str.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_count += 1

            # 是否以数字加'.'开头
            if self.is_start_with_number_dot(line):
                line = line.split(".")[1]

            # 去掉末尾分号
            if line.endswith(";"):
                line = line[:-1]
            if line.strip().endswith("]"):
                line = line[:-1]
            if line.strip().startswith("]"):
                line = line[1:]
            if ";" in line:
                line = line.split(";")[0]

            path_expr = line.strip()

            # 按 "->" 分割：['s', 'r1(t1)', 'r2(t2)', 'o']
            tokens = [t.strip() for t in path_expr.split("->")]
            if len(tokens) < 3:
                # 至少要 s -> r(t) -> o 才算合法路径
                error_path_count += 1
                continue

            head_name = tokens[0]
            tail_name = tokens[-1]
            middle_tokens = tokens[1:-1]  # 中间的 relation(t)

            L = len(middle_tokens)  # 路径长度 = 关系数
            if L == 0 or L > 3:
                # 只保留长度 1~3 的路径
                error_path_count += 1
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
                error_path_count += 1
                continue

            # 映射实体/关系到 id
            try:
                head_id = self._entity_id_from_name(head_name)
                tail_id = self._entity_id_from_name(tail_name)
                if head_id is None or tail_id is None:
                    raise KeyError
            except KeyError:
                # 实体不在字典中，跳过这条路径（你也可以选择 raise）
                error_path_count += 1
                continue

            try:
                rel_ids = [self.relation2id[r] for r in rel_names]
            except KeyError:
                # 关系不在字典中，跳过这条路径
                error_path_count += 1
                continue
        if error_path_count > line_count / 2:
            return True
        else:
            return False

    def extract_last_content(self, text, beg_token, end_token):
        # 1. 找到最后一个 end_token 的位置
        end_idx = text.rfind(end_token)

        if end_idx == -1:
            return -1, -1, ""

        # 2. 在最后一个 end_token 之前，找到最后一个 beg_token 的位置
        # rfind 的第二个和第三个参数定义了搜索范围 [0, end_idx]
        beg_idx = text.rfind(beg_token)

        if beg_idx == -1:
            return -1, -1, ""

        # 3. 计算截取的起始位置（开始标记的索引 + 开始标记自身的长度）
        start_pos = beg_idx + len(beg_token)
        if start_pos >= end_idx:
             return -1, -1, ""

        # 4. 返回截取内容
        return start_pos, end_idx, text[start_pos:end_idx]

    def _get_format_reward(self, text, recall_num, pred, idx):
        '''
        :param text: 生成的文本
        :param recall_num: 交互次数
        :param pred: 预测的entity
        :param idx: 样本id
        :return:
        '''

        error_reason = ""
        format_reward = 0
        hard_format_error = False
        # ------------ 1. THINK / PRED 标签计数与顺序 ------------
        # count_think_beg = text.count(THINK_BEG)
        # count_think_end = text.count(THINK_END)
        count_pred_beg = text.count(PRED_BEG)
        count_pred_end = text.count(PRED_END)

        # 要求：成对出现
        # if not (count_think_beg == count_think_end == 1):
        #     self._log_str(f"[FORMAT] THINK token count error: beg={count_think_beg}, end={count_think_end}")
        #     format_reward -= self.format_error
        #     error_reason += "THINK token count error; "

        # 要求：每种各出现一次
        if not (count_pred_beg == count_pred_end):
            self._log_str(f"[FORMAT] PRED token count error: beg={count_pred_beg}, end={count_pred_end}")
            format_reward -= self.format_error
            error_reason += "PRED token count error; "
            hard_format_error = True

        # 如果标签数量就不对，后面的位置信息就不要再强求了
        if count_pred_beg == count_pred_end:
            pred_beg_pos = text.find(PRED_BEG)
            pred_end_pos = text.find(PRED_END)

            # 要求顺序： THINK_BEG < THINK_END < PRED_BEG < PRED_END
            if not (0 <= pred_beg_pos < pred_end_pos):
                self._log_str(
                    f"[FORMAT] PRED order error: "
                    f"pred_beg={pred_beg_pos}, pred_end={pred_end_pos}"
                )
                format_reward -= 2 * self.format_error
                error_reason += "PRED order error; "
                hard_format_error = True
        # ------------ 2. PATH_SET 标签成对出现 ------------
        count_path_beg = text.count(PATH_LIST_BEG)
        count_path_end = text.count(PATH_LIST_END)
        if count_path_beg != count_path_end:
            self._log_str(
                f"[FORMAT] PATH_SET block count error: "
                f"beg={count_path_beg}, end={count_path_end}"
            )
            format_reward -= self.format_error
            error_reason += "PATH_SET block count error; "
            hard_format_error = True

        # ------------ 3. SELE_PATH 标签成对出现 + 数量与交互轮次匹配 ------------
        count_sele_beg = text.count(SELE_PATH_BEG)
        count_sele_end = text.count(SELE_PATH_END)
        try:
            expected_sele_count = max(int(recall_num) - 1, 0)
        except Exception:
            expected_sele_count = None
        if count_sele_beg != count_sele_end:
            self._log_str(
                f"[FORMAT] SELE_PATH block count error: "
                f"beg={count_sele_beg}, end={count_sele_end}"
            )
            format_reward -= self.format_error
            error_reason += "SELE_PATH block count error; "
            hard_format_error = True
        elif expected_sele_count is None:
            self._log_str(f"[FORMAT] interaction_round parse error: recall_num={recall_num}")
            format_reward -= self.format_error
            error_reason += "interaction_round parse error; "
            hard_format_error = True
        elif count_sele_beg != expected_sele_count:
            self._log_str(
                f"[FORMAT] SELE_PATH count mismatch with interaction_round: "
                f"got={count_sele_beg}, expected={expected_sele_count}, interaction_round={recall_num}"
            )
            format_reward -= self.format_error
            error_reason += "SELE_PATH count mismatch interaction_round; "
            hard_format_error = True
        else:
            if self.stage == "cold":
                sel_path_beg_pos, sel_path_end_pos, sel_path_block = self.extract_last_content(text, SELE_PATH_BEG, SELE_PATH_END)
                # 空路径或者路径解析失败返回True
                format_punish = self.path_format_reward(sel_path_block, idx)
                if format_punish:
                    format_reward -= self.format_error
                    error_reason += "PATH parse error; "

        # ------------ 4. PRED 段中 10 条候选实体行 ------------
        pred_beg_pos, pred_end_pos, pred_block = self.extract_last_content(text, PRED_BEG, PRED_END)
        if pred_beg_pos != -1 and pred_end_pos != -1 and pred_beg_pos < pred_end_pos:
            lines = [ln.strip() for ln in pred_block.split("\n") if ln.strip()]
            invalid_entity_count = 0

            # 选出以 "1.", "2.", ... 开头的行
            pred_lines = [ln for ln in lines if re.match(r"^\s*\d+\.", ln)]
            if len(pred_lines) < 10:
                self._log_str(
                    f"[FORMAT] prediction line count error: expected 10, got {len(pred_lines)}"
                )
                format_reward -= 2 * self.format_error
                error_reason += "prediction line count error; "
                hard_format_error = True
            else:
                # 遍历检查每行是否符合 "序号. 内容:分数" 的格式
                for idx, ln in enumerate(pred_lines, start=1):
                    # 修改正则表达式：匹配冒号后紧跟的数字，允许冒号前后有空格
                    # r":\s*(\d+)$" 表示匹配冒号，后面可能有空格，最后是数字并结束
                    m = re.match(r"^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+)\s*$", ln)

                    if not m:
                        self._log_str(f"[FORMAT] line {idx} has no ':score' pattern: '{ln}'")
                        format_reward -= 2 * self.format_error
                        error_reason += "prediction no ':score' error; "
                        hard_format_error = True
                        break
                    try:
                        entity_name = m.group(2).strip()
                        score = int(m.group(3))
                        if not (1 <= score <= 20):
                            self._log_str(f"[FORMAT] line {idx} score out of range 1-20: '{ln}'")
                            format_reward -= 2 * self.format_error
                            error_reason += "score out of range 1-20 error; "
                            hard_format_error = True
                            break
                        if self._entity_id_from_name(entity_name) is None:
                            invalid_entity_count += 1
                    except ValueError:
                        self._log_str(f"[FORMAT] line {idx} score parse error: '{ln}'")
                        format_reward -= 2 * self.format_error
                        error_reason += "prediction score parse error; "
                        hard_format_error = True
                        break
                if invalid_entity_count > 0:
                    invalid_entity_penalty = invalid_entity_count * abs(self.format_error)
                    self._log_str(
                        f"[FORMAT] invalid prediction entities: count={invalid_entity_count}, "
                        f"penalty={invalid_entity_penalty}"
                    )
                    format_reward -= invalid_entity_penalty
                    error_reason += f"invalid prediction entity count={invalid_entity_count}; "
                    if invalid_entity_count >= self.invalid_entity_hard_threshold:
                        self._log_str(
                            f"[FORMAT] too many invalid prediction entities, block outcome reward: count={invalid_entity_count}"
                        )
                        hard_format_error = True
        else:
            self._log_str("[FORMAT] PRED block not found or malformed")
            error_reason += "PRED block not found error; "
            format_reward -= 2 * self.format_error
            hard_format_error = True

        if inner_PATH_LIST_BEG not in pred and inner_PATH_LIST_END not in pred and inner_SELE_PATH_LIST_BEG not in pred and inner_SELE_PATH_LIST_END not in pred:
            pass
        else:
            self._log_str('illegal token in answer')
            error_reason += "illegal token error; "
            format_reward -= self.format_error
            hard_format_error = True

        have_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        if have_chinese is True:
            self._log_str('has chinese')
            error_reason += "has chinese; "
            format_reward -= self.format_error
            hard_format_error = True

        return format_reward, error_reason, hard_format_error


    def get_reward(
        self,
        query_list,
        idx_list,
        current_step,
        recall_num_list,
        time_aware_score_list=None,
        **kwargs,
    ):
        '''
        :param query_list: prompt+generate文本
        :param idx_list: 样本id
        :param current_step:
        :param recall_num_list:
        :param kwargs:
        :return:
        '''
        pred_list = []
        label_list = []

        question_list = []
        solution_list = []

        for idx, query in zip(idx_list, query_list):
            query = self._process_query(query)
            question, solution = self._get_qa(query)
            question_list.append(question)
            solution_list.append(solution)

            pred = self._get_pred(solution)
            label = self.idx2label[idx]

            pred_list.append(pred)
            label_list.append(label)

        score_list = []
        if time_aware_score_list is None:
            time_aware_score_list = [None] * len(idx_list)

        for idx, query, solution, pred, label, recall_num, time_aware_score in zip(
            idx_list,
            query_list,
            solution_list,
            pred_list,
            label_list,
            recall_num_list,
            time_aware_score_list,
        ):

            # Format Reward
            format_reward, error_reason, hard_format_error = self._get_format_reward(solution, recall_num, pred, idx)
            if hard_format_error:
                print("---------------------")
                print(f"error_reason:"
                      f"\n{error_reason}\n"
                      f"---------------------\n"
                      f"solution: \n{solution}")
            else:
                print("===================good sample===================")
                if format_reward >= 0:
                    format_reward = self.format_correct
            outcome_reward = 0
            if (not hard_format_error) and self.stage == "pred":
                # List-Wise Reward
                print(f"pred:\n{pred}, "
                      f"\n---------------------"
                      f"\nlabel:{label}\n")
                outcome_reward = self._get_outcome_reward(query, pred, label, idx, time_aware_score=time_aware_score)

            tool_call_reward = 0
            # if self.recall_num_base != 0 and self.stage == "cold":
            # if self.recall_num_base != 0:
                # Invocation Count Reward.
                # tool_call_reward = self._get_recall_num_reward(recall_num, base=self.recall_num_base, num_max=self.recall_num_max)
            interaction_round_penalty = 0.0
            if (
                self.target_interaction_round > 0
                and self.interaction_round_mismatch_penalty > 0
                and int(recall_num) != self.target_interaction_round
            ):
                interaction_round_penalty = -abs(self.interaction_round_mismatch_penalty)

            score_list.append(
                list(map(float, [outcome_reward, format_reward, tool_call_reward + interaction_round_penalty]))
            )
            self._log_json(dict(
                step=current_step, idx=idx, recall_num=recall_num, interaction_round=recall_num,
                target_interaction_round=self.target_interaction_round,
                interaction_round_penalty=interaction_round_penalty,
                outcome_reward=outcome_reward,
                format_reward=format_reward,
                tool_call_reward=tool_call_reward,
                query=query,
            ))

        return score_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="../server/config/Reward_ice14.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    # print(args)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    config.update(command_line_configs)

    print(config)
    # server
    reward_model = RecRuleProxy(config)
    app = FastAPI()

    @app.post("/reward")
    async def get_reward(request: Request):
        data = await request.json()
        rewards = reward_model.get_reward(**data)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
