import argparse
import json
import os
import re
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from verl_tkgr.env_tkgr import TKGREnvironment
from verl_tkgr.prompt_tkgr import (
    PATH_LIST_BEG,
    PATH_LIST_END,
    PRED_BEG,
    PRED_END,
    SELE_PATH_BEG,
    SELE_PATH_END,
    build_pred_add_str,
    build_training_messages,
    format_graph_candidate_entity_block,
    render_chat_messages,
)


settings_path = os.path.dirname(__file__)
DATA_ROOT = os.path.join(settings_path, "../data/dataset")


def load_name_mapping(filepath, num_relations, is_rel=False):
    with open(filepath, "r", encoding="utf-8") as file:
        name2id = json.load(file)
    if is_rel:
        inv_name2id = {}
        for key, value in name2id.items():
            inv_name2id["INV::" + key] = int(value) + num_relations
        name2id.update(inv_name2id)
    return name2id


def load_data_table(graph_name, file_name, column_names=None):
    data_fpath = os.path.join(DATA_ROOT, graph_name, file_name)
    return pd.read_table(data_fpath, sep="\t", names=column_names)


def load_temporal_knowledge_graph(dataset_name):
    train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
    column_names = ["head", "rel", "tail", "time", "_"]
    train_data_table = load_data_table(dataset_name, train_file, column_names)
    val_data_table = load_data_table(dataset_name, val_file, column_names)
    test_data_table = load_data_table(dataset_name, test_file, column_names)
    all_data_table = pd.concat([train_data_table, val_data_table, test_data_table], ignore_index=True)
    print(
        f"dataName:{dataset_name}\n train:{len(train_data_table)}, val:{len(val_data_table)}, "
        f"test:{len(test_data_table)}, all data:{len(all_data_table)}"
    )
    all_heads = all_data_table["head"].to_numpy()
    all_tails = all_data_table["tail"].to_numpy()
    all_rels = all_data_table["rel"].to_numpy()
    all_timestamps = all_data_table["time"].to_numpy()
    return all_heads, all_tails, all_rels, all_timestamps


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


def parse_top10(text: str) -> List[AnswerScore]:
    pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$")
    results: List[AnswerScore] = []
    for _, line in enumerate(text.splitlines()):
        match = pattern.match(line)
        if not match:
            continue
        line_no = int(match.group(1))
        ans = match.group(2).strip()
        score = float(match.group(3).strip())
        results.append(AnswerScore(answer=ans, score=score, line_no=line_no))

    by_idx = {}
    for r in results:
        by_idx[r.line_no] = r
    return [by_idx[i] for i in sorted(by_idx.keys())]


def rank_answers(items: List[AnswerScore]) -> List[AnswerScore]:
    return sorted(items, key=lambda x: (-x.score, x.line_no))


def find_entity_rank(text: str, entity: str) -> Tuple[bool, Optional[int], List[AnswerScore]]:
    items = parse_top10(text)
    ranked = rank_answers(items)
    entity_norm = _normalize_entity_for_match(entity)
    for i, r in enumerate(ranked, start=1):
        if _normalize_entity_for_match(r.answer) == entity_norm:
            return True, i, ranked
    return False, None, ranked


def get_final_ranked(pred, label, target_id, head_id, rel_id, timestamp, entity2id_norm, srt2o, train_res_np):
    train_res_np = np.asarray(train_res_np, dtype=np.float32).copy()
    if pred is not None:
        _, _, ranked = find_entity_rank(pred, label)
        for answer in ranked:
            entityid = entity2id_norm.get(_normalize_entity_for_match(answer.answer))
            if entityid is not None:
                train_res_np[entityid] += answer.score * 1

    tmp_score = train_res_np
    pred_ground = tmp_score[target_id]
    tmp_score[srt2o[head_id, rel_id, timestamp]] = -10000000
    tmp_score[target_id] = pred_ground
    ob_pred_comp1 = tmp_score > pred_ground
    ob_pred_comp2 = tmp_score == pred_ground
    target_rank = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
    return target_rank


def parse_query_str(query_str: str, entity2id, relation2id):
    pattern = re.compile(r"^\(\s*(.*)\s*,\s*(.*)\s*,\s*\?\s*,\s*([^)]+)\s*\)$")
    match = pattern.match(query_str.strip())
    if not match:
        raise ValueError(f"Invalid query format: {query_str}")

    head_str = match.group(1).strip()
    relation_str = match.group(2).strip()
    timestamp_str = match.group(3).strip()
    head_id = entity2id[head_str]
    rel_id = relation2id[relation_str]
    return head_str, relation_str, timestamp_str, head_id, rel_id, int(timestamp_str)


def load_raw_data(data_file: str, eval_scope: str = "full", eval_part_size: int = 3000, eval_seed: int = 42):
    data_raw_all = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data_raw_all.append(json.loads(line))
    total = len(data_raw_all)
    if eval_scope == "part" and total > eval_part_size:
        rng = random.Random(eval_seed)
        data_raw_all = rng.sample(data_raw_all, eval_part_size)
        print(
            f"All Data Length: {total}, sampled {len(data_raw_all)} records "
            f"(scope=part, seed={eval_seed})"
        )
    else:
        print("All Data Length:", total)
    return data_raw_all


def _checkpoint_actor_dir(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.name == "actor":
        return path
    if (path / "actor").is_dir():
        return path / "actor"
    return path


def _has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    hf_weight_patterns = [
        "pytorch_model.bin",
        "model.safetensors",
        "*.safetensors",
        "*.bin",
    ]
    for pattern in hf_weight_patterns:
        if any(model_dir.glob(pattern)):
            return True
    return False


def _detect_model_layout(actor_dir: Path, base_model_path: str):
    hf_dir = actor_dir / "huggingface"
    adapter_dir = actor_dir / "lora_adapter"

    if adapter_dir.is_dir():
        adapter_cfg = adapter_dir / "adapter_config.json"
        max_lora_rank = 64
        if adapter_cfg.is_file():
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            max_lora_rank = int(cfg.get("r", max_lora_rank))
        return "lora_adapter", base_model_path, str(adapter_dir), max_lora_rank

    if _has_hf_weights(hf_dir):
        return "huggingface", str(hf_dir), None, 0

    if _has_hf_weights(actor_dir):
        return "direct_huggingface", str(actor_dir), None, 0

    fsdp_config = actor_dir / "fsdp_config.json"
    has_fsdp_shards = any(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if fsdp_config.is_file() and has_fsdp_shards:
        raise FileNotFoundError(
            "Cannot find a directly loadable HF model or LoRA adapter under actor checkpoint: "
            f"{actor_dir}. This looks like a raw FSDP full-finetune checkpoint. "
            "Please merge the FSDP shards into HuggingFace format first, for example: "
            f"python -m verl.model_merger merge --backend fsdp --local_dir {actor_dir} --target_dir {hf_dir}"
        )

    raise FileNotFoundError(
        f"Cannot find a directly loadable HF model or LoRA adapter under actor checkpoint: {actor_dir}"
    )


def _build_prompts(samples, tokenizer, prompt_style: str):
    prompts = []
    for sample in samples:
        messages = build_training_messages(
            query=sample["query"],
            history=sample["history"],
            style=prompt_style,
        )
        prompt = render_chat_messages(messages, tokenizer, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def _resolve_prompt_style(args) -> str:
    if args.disable_graph_reasoner_interaction:
        return "fixed_no_graph_reasoner"
    if args.disable_graph_paths:
        if args.disable_graph_candidates:
            return "fixed_no_graph_reasoner"
        return "fixed_singleturn_no_graph_paths"
    if args.disable_multi_turn:
        if args.disable_graph_candidates:
            return "fixed_singleturn_no_graph_candidates"
        return "fixed_singleturn"
    if args.disable_graph_candidates:
        return "fixed_multiturn_no_graph_candidates"
    return args.prompt_style


def _render_prompt(messages, tokenizer):
    return render_chat_messages(messages, tokenizer, add_generation_prompt=True)


def _build_stop_tokens(tokenizer) -> list[str]:
    stop_tokens = [SELE_PATH_END, PRED_END]
    for token in ("<|im_end|>", "<|endoftext|>", getattr(tokenizer, "eos_token", None)):
        if token and token not in stop_tokens:
            stop_tokens.append(token)
    return stop_tokens


def _safe_avg(total: float, count: int) -> float:
    return float(total) / float(count) if count else 0.0


def _print_eval_runtime_stats(stats: dict):
    total_queries = int(stats.get("total_queries", 0))
    llm_request_count = int(stats.get("llm_request_count", 0))
    llm_final_request_count = int(stats.get("llm_final_request_count", 0))
    tsr_request_count = int(stats.get("tsr_request_count", 0))

    print("[eval_runtime_stats] total_queries:", total_queries)
    print("[eval_runtime_stats] llm_batch_count:", int(stats.get("llm_batch_count", 0)))
    print("[eval_runtime_stats] llm_request_count(sample-round):", llm_request_count)
    print(
        "[eval_runtime_stats] avg_llm_input_tokens_per_query(all_rounds): "
        f"{_safe_avg(stats.get('llm_prompt_tokens_total', 0.0), total_queries):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_output_tokens_per_query(all_rounds): "
        f"{_safe_avg(stats.get('llm_output_tokens_total', 0.0), total_queries):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_input_tokens_per_llm_request: "
        f"{_safe_avg(stats.get('llm_prompt_tokens_total', 0.0), llm_request_count):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_output_tokens_per_llm_request: "
        f"{_safe_avg(stats.get('llm_output_tokens_total', 0.0), llm_request_count):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_generation_seconds_per_query(all_rounds): "
        f"{_safe_avg(stats.get('llm_query_time_total', 0.0), total_queries):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_generation_seconds_per_llm_request: "
        f"{_safe_avg(stats.get('llm_query_time_total', 0.0), llm_request_count):.6f}"
    )
    print("[eval_runtime_stats] llm_final_request_count(sample):", llm_final_request_count)
    print(
        "[eval_runtime_stats] avg_llm_input_tokens_per_query(final_round): "
        f"{_safe_avg(stats.get('llm_final_prompt_tokens_total', 0.0), total_queries):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_output_tokens_per_query(final_round): "
        f"{_safe_avg(stats.get('llm_final_output_tokens_total', 0.0), total_queries):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_llm_generation_seconds_per_query(final_round): "
        f"{_safe_avg(stats.get('llm_final_query_time_total', 0.0), total_queries):.6f}"
    )
    print("[eval_runtime_stats] tsr_batch_count:", int(stats.get("tsr_batch_count", 0)))
    print("[eval_runtime_stats] tsr_request_count(sample-round):", tsr_request_count)
    print(
        "[eval_runtime_stats] avg_tsr_request_seconds_per_query(all_rounds): "
        f"{_safe_avg(stats.get('tsr_query_time_total', 0.0), total_queries):.6f}"
    )
    print(
        "[eval_runtime_stats] avg_tsr_request_seconds_per_tsr_request: "
        f"{_safe_avg(stats.get('tsr_query_time_total', 0.0), tsr_request_count):.6f}"
    )


def _truncate_items(items, limit: int):
    if items is None:
        return None
    limit = max(int(limit), 0)
    return list(items)[:limit]


def _assistant_text_from_output(generated_text: str, generated_text_exact: str, stop_reason: str) -> str:
    assistant_text = str(generated_text or "").strip()
    exact_clean = str(generated_text_exact or "")
    exact_clean = re.sub(r"^\s*<\|im_start\|>assistant\s*\n?", "", exact_clean)
    exact_clean = re.sub(r"<\|im_end\|>\s*$", "", exact_clean)
    exact_clean = re.sub(r"<\|endoftext\|>\s*$", "", exact_clean)
    exact_clean = exact_clean.strip()

    if not assistant_text and exact_clean:
        assistant_text = exact_clean
    if exact_clean and ((SELE_PATH_END in exact_clean and SELE_PATH_END not in assistant_text) or (PRED_END in exact_clean and PRED_END not in assistant_text)):
        assistant_text = exact_clean
    if stop_reason in {SELE_PATH_END, PRED_END} and not assistant_text.endswith(stop_reason):
        assistant_text = assistant_text + f"\n{stop_reason}" if assistant_text else stop_reason
    return assistant_text.strip()


def _extract_structural_confidence(text: str) -> tuple[Optional[int], Optional[float]]:
    match = re.search(
        r"<Structural_Confidence>\s*(\d{1,3})\s*</Structural_Confidence>",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    raw_score = int(match.group(1))
    clipped_score = min(100, max(1, raw_score))
    return clipped_score, clipped_score / 100.0


def _structural_candidate_rank_info(
    first_candidate_entity_list: Optional[list[str]],
    target: str,
) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
    if not first_candidate_entity_list:
        return None, None, None, None, None
    top1_entity = str(first_candidate_entity_list[0]).strip()
    if not top1_entity:
        return None, None, None, None, None
    normalized_target = _normalize_entity_for_match(target)
    candidate_rank = None
    for idx, entity in enumerate(first_candidate_entity_list, start=1):
        if _normalize_entity_for_match(str(entity)) == normalized_target:
            candidate_rank = idx
            break
    top1_correct = int(candidate_rank == 1) if candidate_rank is not None else 0
    top3_correct = int(candidate_rank is not None and candidate_rank <= 3)
    top10_correct = int(candidate_rank is not None and candidate_rank <= 10)
    return top1_entity, candidate_rank, top1_correct, top3_correct, top10_correct


def _graph_rank_or_default(
    sample,
    pred_text,
    target,
    target_id,
    query_str,
    entity2id,
    relation2id,
    entity2id_norm,
    srt2o,
    disable_graph_reasoner_interaction: bool,
):
    if disable_graph_reasoner_interaction:
        return -1.0
    _, _, _, head_id, rel_id, timestamp = parse_query_str(query_str, entity2id, relation2id)
    return get_final_ranked(
        pred_text,
        target,
        target_id,
        head_id,
        rel_id,
        timestamp,
        entity2id_norm,
        srt2o,
        sample.get("time_aware_score"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--prompt_style", type=str, default="fixed_multiturn")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--recall_server", type=str, default="http://XXXXX:6001/tkgr_server")
    parser.add_argument("--recall_timeout", type=float, default=180.0)
    parser.add_argument("--output_dir", type=str, default="./results/eval_verl/")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--eval_scope", type=str, choices=["full", "part"], default="full")
    parser.add_argument("--eval_part_size", type=int, default=3000)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--disable_multi_turn", action="store_true")
    parser.add_argument("--disable_recurring_history", action="store_true")
    parser.add_argument("--disable_graph_candidates", action="store_true")
    parser.add_argument("--disable_graph_paths", action="store_true")
    parser.add_argument("--disable_graph_reasoner_interaction", action="store_true")
    parser.add_argument("--disable_consistency_guidance", action="store_true")
    parser.add_argument("--path_block_limit", type=int, default=10)
    parser.add_argument("--candidate_entity_limit", type=int, default=10)
    return parser.parse_args()


def main():
    print("=Begin=" * 10)
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "test_text.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)

    prompt_style = _resolve_prompt_style(args)
    effective_max_rounds = 1 if (args.disable_multi_turn or args.disable_graph_reasoner_interaction or args.disable_graph_paths) else args.max_rounds
    pred_add_str = build_pred_add_str(
        include_paths=not args.disable_graph_paths,
        include_graph_candidates=not args.disable_graph_candidates,
        include_consistency_guidance=not args.disable_consistency_guidance,
    )

    data_path = os.path.join(DATA_ROOT, args.dataset)
    entity2id = load_name_mapping(os.path.join(data_path, "entity2id.json"), 0)
    entity2id_norm = _build_normalized_entity2id(entity2id)
    relation2id = load_name_mapping(os.path.join(data_path, "relation2id.json"), 0)
    all_heads, all_tails, all_rels, all_timestamps = load_temporal_knowledge_graph(args.dataset)
    srt2o = defaultdict(list)
    for i in range(len(all_heads)):
        srt2o[(all_heads[i], all_rels[i], all_timestamps[i])].append(all_tails[i])

    actor_dir = _checkpoint_actor_dir(args.checkpoint_path)
    checkpoint_mode, model_path, lora_adapter_path, max_lora_rank = _detect_model_layout(actor_dir, args.base_model_path)
    print(f"Resolved actor checkpoint: {actor_dir}")
    print(f"Resolved checkpoint mode: {checkpoint_mode}")
    print(f"Resolved model path: {model_path}")
    print(f"Resolved LoRA adapter path: {lora_adapter_path}")

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_rate,
        "trust_remote_code": True,
    }
    if lora_adapter_path is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_lora_rank"] = max_lora_rank

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"Tokenizer name_or_path: {getattr(tokenizer, 'name_or_path', '')}")
    print(f"Tokenizer has chat_template: {bool(getattr(tokenizer, 'chat_template', None))}")
    env = None if args.disable_graph_reasoner_interaction else TKGREnvironment(args.recall_server, timeout=args.recall_timeout)
    data_raw_all = load_raw_data(args.data_file, eval_scope=args.eval_scope, eval_part_size=args.eval_part_size, eval_seed=args.eval_seed)
    runtime_stats = defaultdict(float)
    runtime_stats["total_queries"] = len(data_raw_all)

    chunk_num = len(data_raw_all) // args.chunk_size
    if len(data_raw_all) % args.chunk_size != 0:
        chunk_num += 1

    stop_tokens = _build_stop_tokens(tokenizer)
    print(f"Stop tokens: {stop_tokens}")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
    )
    lora_requests = None
    if lora_adapter_path is not None:
        lora_requests = [
            LoRARequest(lora_name="eval_lora", lora_int_id=1, lora_path=lora_adapter_path)
        ]

    for chunk_i in range(chunk_num):
        print("==" * 80)
        print("Begin Chunk:", chunk_i, "All:", chunk_num)
        data = data_raw_all[chunk_i * args.chunk_size : (chunk_i + 1) * args.chunk_size]
        if not data:
            continue
        finished_all_list = []
        continued_data = []
        for raw in data:
            prompt_history = "" if args.disable_recurring_history else raw["history"]
            messages = build_training_messages(
                query=raw["query"],
                history=prompt_history,
                style=prompt_style,
                include_consistency_guidance=not args.disable_consistency_guidance,
            )
            continued_data.append(
                {
                    "idx": raw["idx"],
                    "messages": messages,
                    "target": raw["target"],
                    "history": prompt_history,
                    "query": raw["query"],
                    "interaction_round": 1,
                    "pending_selected_paths": "",
                    "gen_text_store": "",
                    "stop_reason_store": "",
                    "first_rank": "",
                    "last_rank": "",
                    "first_path_list": None,
                    "first_candidate_entity_list": None,
                }
            )

        max_total_tokens = 16384
        for step in range(effective_max_rounds):
            if not continued_data:
                break

            if not args.disable_graph_reasoner_interaction:
                recall_idx_list = [str(d["idx"]) for d in continued_data]
                selected_paths_list = [d["pending_selected_paths"] for d in continued_data]
                tsr_start = time.perf_counter()
                result = env.recall(recall_idx_list, selected_paths_list)
                tsr_elapsed = time.perf_counter() - tsr_start
                runtime_stats["tsr_batch_count"] += 1
                runtime_stats["tsr_request_count"] += len(continued_data)
                runtime_stats["tsr_query_time_total"] += tsr_elapsed * len(continued_data)

                for sample, paths_for_sample, score_vec, top10_names in zip(
                    continued_data, result.path_list, result.time_aware_score, result.top10_entity_names
                ):
                    limited_paths = _truncate_items(paths_for_sample, args.path_block_limit)
                    limited_top10_names = _truncate_items(top10_names, args.candidate_entity_limit)
                    observation_parts = []
                    if not args.disable_graph_paths:
                        path_block = "".join(f"{p}" for p in limited_paths) if limited_paths else "None"
                        observation_parts.append(f"{PATH_LIST_BEG}\n" + path_block + f"{PATH_LIST_END}\n")
                    if not args.disable_graph_candidates:
                        observation_parts.append(format_graph_candidate_entity_block(limited_top10_names))
                    observation_text = "".join(observation_parts)
                    if step == 0:
                        sample["messages"][-1]["content"] = str(sample["messages"][-1]["content"]) + "\n\n" + observation_text
                    else:
                        sample["messages"].append({"role": "user", "content": observation_text})
                    sample["time_aware_score"] = score_vec
                    sample["top10_entity_names"] = limited_top10_names
                    if step == 0:
                        sample["first_path_list"] = limited_paths
                        sample["first_candidate_entity_list"] = limited_top10_names
                        _, _, _, head_id, rel_id, timestamp = parse_query_str(
                            sample["query"], entity2id, relation2id
                        )
                        sample["first_rank"] = get_final_ranked(
                            None,
                            sample["target"],
                            entity2id_norm.get(_normalize_entity_for_match(sample["target"])),
                            head_id,
                            rel_id,
                            timestamp,
                            entity2id_norm,
                            srt2o,
                            score_vec,
                        )
            else:
                for sample in continued_data:
                    sample["time_aware_score"] = None
                    sample["top10_entity_names"] = None
                    if step == 0:
                        sample["first_rank"] = -1.0

            if step == effective_max_rounds - 1 and not args.disable_multi_turn and not args.disable_graph_reasoner_interaction:
                print("add pred prompt")
                for sample in continued_data:
                    sample["messages"][-1]["content"] = str(sample["messages"][-1]["content"]) + "\n" + pred_add_str

            batch_prompts = [_render_prompt(d["messages"], tokenizer) for d in continued_data]

            lora_batch = None
            if lora_requests is not None:
                lora_batch = lora_requests * len(batch_prompts)
            llm_start = time.perf_counter()
            outputs = llm.generate(
                batch_prompts,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_batch,
            )
            llm_elapsed = time.perf_counter() - llm_start
            runtime_stats["llm_batch_count"] += 1
            runtime_stats["llm_request_count"] += len(outputs)
            runtime_stats["llm_query_time_total"] += llm_elapsed * len(outputs)
            if step == effective_max_rounds - 1:
                runtime_stats["llm_final_request_count"] += len(outputs)
                runtime_stats["llm_final_query_time_total"] += llm_elapsed * len(outputs)

            finished_texts = []
            continued_texts = []
            for k, output in enumerate(outputs):
                prompt = output.prompt
                sample = continued_data[k]
                idx = sample["idx"]
                target = sample["target"]
                query_str = sample["query"]
                target_id = entity2id_norm.get(_normalize_entity_for_match(target))
                history = sample["history"]
                gen_text_store = sample.get("gen_text_store", "")
                stop_reason = output.outputs[0].stop_reason
                generated_text = output.outputs[0].text
                generated_token_ids = list(output.outputs[0].token_ids)
                prompt_token_ids = list(output.prompt_token_ids)
                runtime_stats["llm_prompt_tokens_total"] += len(prompt_token_ids)
                runtime_stats["llm_output_tokens_total"] += len(generated_token_ids)
                if step == effective_max_rounds - 1:
                    runtime_stats["llm_final_prompt_tokens_total"] += len(prompt_token_ids)
                    runtime_stats["llm_final_output_tokens_total"] += len(generated_token_ids)
                generated_text_exact = tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ) if generated_token_ids else ""
                interaction_round = sample["interaction_round"]
                stop_reason_store = sample.get("stop_reason_store", "")
                all_token_ids = prompt_token_ids + generated_token_ids
                assistant_text = _assistant_text_from_output(generated_text, generated_text_exact, stop_reason)
                structural_confidence_score, structural_confidence_prob = _extract_structural_confidence(assistant_text)
                (
                    structural_top1_entity,
                    structural_candidate_rank,
                    structural_top1_correct,
                    structural_top3_correct,
                    structural_top10_correct,
                ) = _structural_candidate_rank_info(
                    sample.get("first_candidate_entity_list"),
                    target,
                )

                if len(all_token_ids) > max_total_tokens:
                    final_ranked = _graph_rank_or_default(
                        sample, None, target, target_id, query_str, entity2id, relation2id,
                        entity2id_norm, srt2o, args.disable_graph_reasoner_interaction
                    )
                    finished_texts.append(
                        {
                            "idx": idx,
                            "history": history,
                            "target": target,
                            "gen_text_store": gen_text_store + assistant_text,
                            "generated_text": assistant_text,
                            "stop_reason_final": "many_recall or max_token",
                            "pred_ans": "I don't know.",
                            "interaction_round": interaction_round,
                            "time_aware_score": sample.get("time_aware_score"),
                            "top10_entity_names": sample.get("top10_entity_names"),
                            "first_path_list": sample.get("first_path_list"),
                            "first_candidate_entity_list": sample.get("first_candidate_entity_list"),
                            "structural_confidence_score": structural_confidence_score,
                            "structural_confidence_prob": structural_confidence_prob,
                            "structural_top1_entity": structural_top1_entity,
                            "structural_candidate_rank": structural_candidate_rank,
                            "structural_top1_correct": structural_top1_correct,
                            "structural_top3_correct": structural_top3_correct,
                            "structural_top10_correct": structural_top10_correct,
                            "query": query_str,
                            "final_ranked": final_ranked,
                            "last_rank": final_ranked,
                            "first_rank": sample.get("first_rank"),
                        }
                    )
                    continue

                if step != effective_max_rounds - 1:
                    sele_content = None
                    if SELE_PATH_BEG in assistant_text:
                        sele_content = assistant_text.split(SELE_PATH_BEG, 1)[-1].strip()
                    block_content = re.search(
                        rf"{re.escape(SELE_PATH_BEG)}\s*(.*?)\s*{re.escape(SELE_PATH_END)}",
                        assistant_text,
                        flags=re.DOTALL,
                    )
                    if block_content:
                        sele_content = block_content.group(1).strip()
                else:
                    sele_content = None

                if sele_content is not None and step != effective_max_rounds - 1:
                    next_messages = [dict(item) for item in sample["messages"]]
                    next_messages.append({"role": "assistant", "content": assistant_text})
                    continued_texts.append(
                        {
                            "idx": idx,
                            "messages": next_messages,
                            "history": history,
                            "target": target,
                            "stop_reason_store": stop_reason_store + f"\n{interaction_round}.\n{stop_reason}",
                            "gen_text_store": gen_text_store + assistant_text,
                            "interaction_round": interaction_round + 1,
                            "time_aware_score": sample.get("time_aware_score"),
                            "top10_entity_names": sample.get("top10_entity_names"),
                            "pending_selected_paths": sele_content,
                            "query": query_str,
                            "last_rank": sample.get("last_rank"),
                            "first_rank": sample.get("first_rank"),
                            "first_path_list": sample.get("first_path_list"),
                            "first_candidate_entity_list": sample.get("first_candidate_entity_list"),
                        }
                    )
                elif PRED_BEG in assistant_text and PRED_END in assistant_text:
                    last_rank = _graph_rank_or_default(
                        sample, None, target, target_id, query_str, entity2id, relation2id,
                        entity2id_norm, srt2o, args.disable_graph_reasoner_interaction
                    )
                    final_ranked = _graph_rank_or_default(
                        sample,
                        assistant_text.split(PRED_BEG, 1)[-1].replace(PRED_END, "").strip(),
                        target,
                        target_id,
                        query_str,
                        entity2id,
                        relation2id,
                        entity2id_norm,
                        srt2o,
                        args.disable_graph_reasoner_interaction,
                    )
                    finished_texts.append(
                        {
                            "idx": idx,
                            "history": history,
                            "target": target,
                            "prompt": prompt + assistant_text,
                            "gen_text_store": gen_text_store + assistant_text,
                            "stop_reason_store": stop_reason_store + f"\n{interaction_round}.\n{stop_reason}",
                            "stop_reason_final": "finished",
                            "pred_ans": assistant_text.split(PRED_BEG, 1)[-1].replace(PRED_END, "").strip(),
                            "interaction_round": interaction_round,
                            "time_aware_score": sample.get("time_aware_score"),
                            "top10_entity_names": sample.get("top10_entity_names"),
                            "first_path_list": sample.get("first_path_list"),
                            "first_candidate_entity_list": sample.get("first_candidate_entity_list"),
                            "structural_confidence_score": structural_confidence_score,
                            "structural_confidence_prob": structural_confidence_prob,
                            "structural_top1_entity": structural_top1_entity,
                            "structural_candidate_rank": structural_candidate_rank,
                            "structural_top1_correct": structural_top1_correct,
                            "structural_top3_correct": structural_top3_correct,
                            "structural_top10_correct": structural_top10_correct,
                            "query": query_str,
                            "final_ranked": final_ranked,
                            "last_rank": last_rank,
                            "first_rank": sample.get("first_rank"),
                        }
                    )
                else:
                    last_rank = _graph_rank_or_default(
                        sample, None, target, target_id, query_str, entity2id, relation2id,
                        entity2id_norm, srt2o, args.disable_graph_reasoner_interaction
                    )
                    final_ranked = _graph_rank_or_default(
                        sample, None, target, target_id, query_str, entity2id, relation2id,
                        entity2id_norm, srt2o, args.disable_graph_reasoner_interaction
                    )
                    finished_texts.append(
                        {
                            "idx": idx,
                            "history": history,
                            "target": target,
                            "prompt": prompt + assistant_text,
                            "gen_text_store": gen_text_store + assistant_text,
                            "stop_reason_store": stop_reason_store + f"\n{interaction_round}.\n{stop_reason}",
                            "stop_reason_final": "shot_down",
                            "pred_ans": "I don't know.",
                            "interaction_round": interaction_round,
                            "time_aware_score": sample.get("time_aware_score"),
                            "top10_entity_names": sample.get("top10_entity_names"),
                            "first_path_list": sample.get("first_path_list"),
                            "first_candidate_entity_list": sample.get("first_candidate_entity_list"),
                            "structural_confidence_score": structural_confidence_score,
                            "structural_confidence_prob": structural_confidence_prob,
                            "structural_top1_entity": structural_top1_entity,
                            "structural_candidate_rank": structural_candidate_rank,
                            "structural_top1_correct": structural_top1_correct,
                            "structural_top3_correct": structural_top3_correct,
                            "structural_top10_correct": structural_top10_correct,
                            "query": query_str,
                            "final_ranked": final_ranked,
                            "last_rank": last_rank,
                            "first_rank": sample.get("first_rank"),
                        }
                    )

            finished_all_list.extend(finished_texts)
            print("==" * 80)
            print(
                "Step:",
                step,
                "New_Finished:",
                len(finished_texts),
                "All_Finished",
                len(finished_all_list),
                "Continued:",
                len(continued_texts),
            )

            if finished_texts:
                with open(output_file, "a", encoding="utf-8") as f:
                    for text in finished_texts:
                        f.write(json.dumps(text, ensure_ascii=False) + "\n")

            if not continued_texts:
                break
            continued_data = continued_texts

    _print_eval_runtime_stats(runtime_stats)


if __name__ == "__main__":
    main()
