from __future__ import annotations

import re
from typing import Optional

PRED_BEG = "<prediction_list>"
PRED_END = "</prediction_list>"
PATH_LIST_BEG = "<path_list>"
PATH_LIST_END = "</path_list>"
GRAPH_ENTITY_LIST_BEG = "<graph_candidate_entity_list>"
GRAPH_ENTITY_LIST_END = "</graph_candidate_entity_list>"
SELE_PATH_BEG = "<selected_path_list>"
SELE_PATH_END = "</selected_path_list>"
MAX_SELECTED_PATH_LINES = 8

_CONSISTENCY_GUIDANCE_LINE_PATTERNS = (
    "Decision guidance for the final round:",
    "Decision guidance:",
    "Final-round decision guidance:",
    "If graph-reasoner paths are mutually consistent",
    "If recurring historical evidence is rich, frequent, and temporally recent",
    "If recurring and graph evidence both support a candidate",
    "If recurring and graph evidence conflict",
    "If graph paths are inconsistent and recurring historical evidence is weak",
    "If graph-reasoner paths are mutually consistent or repeatedly point to the same entities",
    "If graph candidate entities repeatedly point to the same or similar entities",
)


def _without_consistency_guidance(text: str) -> str:
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip().lstrip("-").strip()
        if any(pattern in stripped for pattern in _CONSISTENCY_GUIDANCE_LINE_PATTERNS):
            continue
        kept_lines.append(raw_line)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def build_pred_add_str(
    *,
    include_paths: bool = True,
    include_graph_candidates: bool = True,
    include_consistency_guidance: bool = True,
) -> str:
    sources: list[str] = [
        "<recurring_entity_stats>...</recurring_entity_stats> (if provided)",
    ]
    if include_paths:
        sources.append(f"tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}")
    if include_graph_candidates:
        sources.append(f"{GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}")

    if len(sources) == 1:
        source_text = sources[0]
    elif len(sources) == 2:
        source_text = f"{sources[0]} and {sources[1]}"
    else:
        source_text = f"{', '.join(sources[:-1])}, and {sources[-1]}"

    guidance_lines = [
        "Final-round decision guidance:",
        f"- You may copy entity names from exactly these sources: {source_text}.",
        "- Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.",
    ]
    if include_consistency_guidance:
        guidance_lines.extend(
            [
                "- If graph-reasoner paths are mutually consistent, repeatedly point to the same or similar entities, or remain stable across rounds, give graph evidence more weight.",
                "- If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.",
                "- If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.",
                "- If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.",
                "- If graph paths are inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.",
            ]
        )

    return (
        "Maximum number of interaction rounds has been reached. Based on your background knowledge, "
        "the recurring historical information provided at the beginning, and the interaction information above, "
        f'predict the tail entity of the "Query". Wrap the results in {PRED_BEG}{PRED_END}. '
        + "\n".join(guidance_lines)
        + "\n"
    )

PRED_ADD_STR = build_pred_add_str(
    include_paths=True,
    include_graph_candidates=True,
    include_consistency_guidance=True,
)
'''
PRED_ADD_STR = (
    "Maximum number of interaction rounds has been reached. Based on your background knowledge, "
    "the recurring historical information provided at the beginning, and the interaction information above, "
    f'predict the tail entity of the "Query". Wrap the results in {PRED_BEG}{PRED_END}. '
    "Final-round decision guidance:\n"
    "- Valid prediction entity names must be copied from existing interaction evidence, not invented.\n"
    f"- You may copy entity names from exactly these three sources: <recurring_entity_stats>...</recurring_entity_stats> (if provided), tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}, and {GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}.\n"
    "- Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.\n"
    "- If graph-reasoner paths are mutually consistent, repeatedly point to the same or similar entities, or remain stable across rounds, give graph evidence more weight.\n"
    "- If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.\n"
    "- If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.\n"
    "- If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.\n"
    "- If evidence is sparse, still output 10 entities and prioritize canonical names from the graph candidate block.\n"
)
'''
'''
PRED_ADD_STR = (
    "Maximum number of interaction rounds has been reached. Based on your background knowledge, "
    "the recurring historical information provided at the beginning, and the interaction information above, "
    f'predict the tail entity of the "Query".'
    "Format guidance:\n"
    '- Wrap the results inside:\n'
    f'{PRED_BEG}\n'
    f'1. <entity_name>:<score>\n'
    f'2. <entity_name>:<score>\n'
    f'...\n'
    f'10. <entity_name>:<score>\n'
    f'{PRED_END}.\n'
    f'- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.'
    f'- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.'
    "- Do not repeat the same entity more than once in the final prediction list.\n"
    "Final-round prediction decision guidance:\n"
    "- Valid prediction entity names must be copied from existing interaction evidence, not invented.\n"
    f"- You may copy entity names from exactly these three sources: <recurring_entity_stats>...</recurring_entity_stats> (if provided), tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}, and {GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}.\n"
    "- Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.\n"
    "- If graph-reasoner paths are mutually consistent, repeatedly point to the same or similar entities, or remain stable across rounds, give graph evidence more weight.\n"
    "- If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.\n"
    "- If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.\n"
    "- If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.\n"
    "- If evidence is sparse, still output 10 entities and prioritize canonical names from the graph candidate block.\n"
)
'''

FIXED_MULTITURN_SYSTEM_PROMPT = f"""
You are a Temporal Knowledge Graph Reasoning (TKGR) expert. You collaborate with an environment over multiple rounds to predict the tail entity.

**Interaction Protocol**
- The Environment provides candidate paths and their confidence scores strictly inside:
  {PATH_LIST_BEG} ... {PATH_LIST_END}
- Each candidate path is a variable-length temporal multi-hop chain in the form entity -> relation(timestamp) -> ... -> relation(timestamp) -> entity, where the number of hops varies across paths.
- The Environment may also provide graph-reasoner top candidate entity names strictly inside:
  {GRAPH_ENTITY_LIST_BEG} ... {GRAPH_ENTITY_LIST_END}
- These names are canonical entity strings and may be copied directly when useful.

- In each non-final round, you MUST output a selected-path request strictly inside:
{SELE_PATH_BEG}
<path>
<path>
...
{SELE_PATH_END}
- Select as many paths as you consider semantically plausible, rather than choosing only the single best one.
- Each line contains exactly ONE path copied from the current path list.
- Each selected path must be unique. Do not repeat any path in the same selected-path block.
- Output at most {MAX_SELECTED_PATH_LINES} selected paths in each round.

- In the final round, you MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}
- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the interaction history and the recurring historical.
- Decision guidance for the final round:
  - You may copy entity names from exactly these three sources: <recurring_entity_stats>...</recurring_entity_stats> (if provided), tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}, and {GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}.
  - Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.
  - If graph-reasoner paths are mutually consistent, repeatedly point to the same or similar entities, or remain stable across rounds, give graph evidence more weight.
  - If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.
  - If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.
  - If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.
  - If graph paths are inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.
""".strip()

FIXED_SINGLETURN_SYSTEM_PROMPT = f"""
You are a Temporal Knowledge Graph Reasoning (TKGR) expert. Predict the tail entity from the provided evidence in a single interaction round.

**Available Evidence**
- The Environment provides candidate paths and their confidence scores strictly inside:
  {PATH_LIST_BEG} ... {PATH_LIST_END}
- Each candidate path is a variable-length temporal multi-hop chain in the form entity -> relation(timestamp) -> ... -> relation(timestamp) -> entity, where the number of hops varies across paths.
- The Environment may also provide Temporal Subgraph Reasoner top candidate entity names strictly inside:
  {GRAPH_ENTITY_LIST_BEG} ... {GRAPH_ENTITY_LIST_END}
- These names are canonical entity strings and may be copied directly when useful.

- You MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}
- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the recurring historical information and the recalled graph evidence.
- Decision guidance:
  - You may copy entity names from exactly these three sources: <recurring_entity_stats>...</recurring_entity_stats> (if provided), tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}, and {GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}.
  - Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.
  - If Temporal Subgraph Reasoner paths are mutually consistent or repeatedly point to the same entities, give graph evidence more weight.
  - If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.
  - If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.
  - If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.
  - If graph paths are inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.
""".strip()


'''
- Before the final prediction, output the inference confidence score of the Temporal Subgraph Reasoner in:
<Structural_Confidence> ... </Structural_Confidence>.
- The score ranges from 1 to 100.
- A higher score indicates that the answer inferred by the Temporal Subgraph Reasoner from structural reasoning is more accurate."



Before the final prediction, output a structural confidence score in:
<Structural_Confidence> ... </Structural_Confidence>

Definition:
- The score must be an integer from 1 to 100.
- It indicates whether the path list supports the prediction candidate.
- 1 means the path evidence provides almost no support for the inferred answer.
- 100 means the path evidence provides overwhelming and internally consistent support for the inferred answer.

Scoring Principles:
- Use only the provided paths and candidate evidence.
- Do not use external world knowledge.
- Consider the relevance of the evidence, multi-path consistency, and specificity.
- If the evidence is weak, indirect, contradictory, noisy, or overly general, assign a lower score.
- If multiple paths strongly and consistently support the inferred answer, assign a higher score.
- Use the full 1–100 range when appropriate.



- Decision guidance:
  - You may copy entity names from exactly these three sources: <recurring_entity_stats>...</recurring_entity_stats> (if provided), tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}, and {GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}.
  - Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.
  - If Temporal Subgraph Reasoner paths are mutually consistent or repeatedly point to the same entities, give graph evidence more weight.
  - If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.
  - If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.
  - If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.
  - If graph paths are inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.
'''


FIXED_MULTITURN_NO_GRAPH_CANDIDATES_SYSTEM_PROMPT = f"""
You are a Temporal Knowledge Graph Reasoning (TKGR) expert. You collaborate with an environment over multiple rounds to predict the tail entity.

**Interaction Protocol**
- The Environment provides candidate paths and their confidence scores strictly inside:
  {PATH_LIST_BEG} ... {PATH_LIST_END}
- Each candidate path is a variable-length temporal multi-hop chain in the form entity -> relation(timestamp) -> ... -> relation(timestamp) -> entity, where the number of hops varies across paths.

- In each non-final round, you MUST output a selected-path request strictly inside:
{SELE_PATH_BEG}
<path>
<path>
...
{SELE_PATH_END}
- Select as many paths as you consider semantically plausible, rather than choosing only the single best one.
- Each line contains exactly ONE path copied from the current path list.
- Each selected path must be unique. Do not repeat any path in the same selected-path block.
- Output at most {MAX_SELECTED_PATH_LINES} selected paths in each round.

- In the final round, you MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}
- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the recurring historical information and the recalled graph evidence.
- If graph evidence is inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.
""".strip()

FIXED_SINGLETURN_NO_GRAPH_CANDIDATES_SYSTEM_PROMPT = f"""
You are a Temporal Knowledge Graph Reasoning (TKGR) expert. Predict the tail entity from the provided evidence in a single interaction round.

**Available Evidence**
- The Environment provides candidate paths and their confidence scores strictly inside:
  {PATH_LIST_BEG} ... {PATH_LIST_END}
- Each candidate path is a variable-length temporal multi-hop chain in the form entity -> relation(timestamp) -> ... -> relation(timestamp) -> entity, where the number of hops varies across paths.

- You MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}
- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the recurring historical information and the recalled graph evidence.
- If graph evidence is inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.
""".strip()

FIXED_SINGLETURN_NO_GRAPH_PATHS_SYSTEM_PROMPT = f"""
You are a Temporal Knowledge Graph Reasoning (TKGR) expert. Predict the tail entity from the provided evidence in a single interaction round.

**Available Evidence**
- The Environment may provide graph-reasoner top candidate entity names strictly inside:
  {GRAPH_ENTITY_LIST_BEG} ... {GRAPH_ENTITY_LIST_END}
- These names are canonical entity strings and may be copied directly when useful.

- You MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}
- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the recurring historical information and the recalled graph evidence.
- If graph evidence is inconsistent and recurring historical evidence is weak, use semantic plausibility and contextual reasoning to select more likely candidate entities.
""".strip()

FIXED_NO_GRAPH_REASONER_SYSTEM_PROMPT = f"""
You are a Temporal Knowledge Graph Reasoning (TKGR) expert. Predict the tail entity without using any graph-reasoner interaction results.

**Available Evidence**
- You will be given the query and the recurring historical statistics for the same head-relation-time pattern.
- No recalled graph paths or graph candidate entities are available in this setting.

- You MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}
- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the query semantics and the recurring historical information.
- If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.
""".strip()

'''
- In the final round, you MUST output final predictions strictly inside:
{PRED_BEG}
1. <entity_name>:<score>
2. <entity_name>:<score>
...
10. <entity_name>:<score>
{PRED_END}

- Provide exactly 10 candidate entities with confidence scores as integers from 1 to 20.
- Never output fewer than 10 entities. If uncertain, still output 10 by filling the remaining slots with low-confidence candidates.
- Do not repeat the same entity more than once in the final prediction list.
- When predicting, give full consideration to the interaction history and the recurring historical.
- Decision guidance for the final round:
  - Valid prediction entity names must be copied from existing interaction evidence, not invented.
  - You may copy entity names from exactly these three sources: <recurring_entity_stats> (if provided), tail entities in {PATH_LIST_BEG}...{PATH_LIST_END}, and {GRAPH_ENTITY_LIST_BEG}...{GRAPH_ENTITY_LIST_END}.
  - Copy entity names exactly as shown in the source text, preserving spaces, parentheses, and punctuation.
  - If graph-reasoner paths are mutually consistent, repeatedly point to the same or similar entities, or remain stable across rounds, give graph evidence more weight.
  - If recurring historical evidence is rich, frequent, and temporally recent, give it more weight.
  - If recurring and graph evidence both support a candidate, rank it above candidates supported by only one source.
  - If recurring and graph evidence conflict, prefer candidates with stronger recency+frequency from recurring and stronger path-consistency from graph.
  - If evidence is sparse, still output 10 entities and prioritize canonical names from the graph candidate block.
'''

FIXED_MULTITURN_USER_PROMPT = """
**Input**
Query:
{query}

{history}

""".strip()

# Start interaction:

def build_training_messages(
    query: str,
    history: str,
    style: str = "fixed_multiturn",
    tool_name: str = "tkgr_recall",
    include_consistency_guidance: bool = True,
) -> list[dict[str, str]]:
    if style == "fixed_multiturn":
        system_prompt = FIXED_MULTITURN_SYSTEM_PROMPT
    elif style == "fixed_multiturn_no_graph_candidates":
        system_prompt = FIXED_MULTITURN_NO_GRAPH_CANDIDATES_SYSTEM_PROMPT
    elif style == "fixed_singleturn":
        system_prompt = FIXED_SINGLETURN_SYSTEM_PROMPT
    elif style == "fixed_singleturn_no_graph_candidates":
        system_prompt = FIXED_SINGLETURN_NO_GRAPH_CANDIDATES_SYSTEM_PROMPT
    elif style == "fixed_singleturn_no_graph_paths":
        system_prompt = FIXED_SINGLETURN_NO_GRAPH_PATHS_SYSTEM_PROMPT
    elif style == "fixed_no_graph_reasoner":
        system_prompt = FIXED_NO_GRAPH_REASONER_SYSTEM_PROMPT
    else:
        raise ValueError(
            f"Unsupported TKGR prompt style: {style}. "
            "Supported styles: 'fixed_multiturn', 'fixed_multiturn_no_graph_candidates', "
            "'fixed_singleturn', 'fixed_singleturn_no_graph_candidates', 'fixed_singleturn_no_graph_paths', "
            "'fixed_no_graph_reasoner'."
        )
    if not include_consistency_guidance:
        system_prompt = _without_consistency_guidance(system_prompt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": FIXED_MULTITURN_USER_PROMPT.format(query=query, history=history)},
    ]


def build_training_prompt(
    query: str,
    history: str,
    style: str = "fixed_multiturn",
    tool_name: str = "tkgr_recall",
    include_consistency_guidance: bool = True,
) -> str:
    messages = build_training_messages(
        query=query,
        history=history,
        style=style,
        tool_name=tool_name,
        include_consistency_guidance=include_consistency_guidance,
    )
    return "\n\n".join(message["content"] for message in messages)


def render_chat_messages(
    messages: list[dict[str, str]],
    tokenizer,
    *,
    add_generation_prompt: bool = True,
    **apply_chat_template_kwargs,
) -> str:
    """Render chat messages with a safe fallback for instruction-only tokenizers."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **apply_chat_template_kwargs,
        )

    tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    if "mpt" in tokenizer_name:
        instruction_parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                instruction_parts.append(content)
            elif role == "user":
                instruction_parts.append(f"User:\n{content}")
            elif role == "assistant":
                instruction_parts.append(f"Assistant:\n{content}")
            else:
                instruction_parts.append(f"{role.title() if role else 'Message'}:\n{content}")

        rendered = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            + "\n\n".join(instruction_parts).rstrip()
        )
        if add_generation_prompt:
            rendered = f"{rendered}\n\n### Response:"
        return f"{rendered}\n"

    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    rendered_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        label = role_labels.get(role, role.title() if role else "Message")
        content = str(message.get("content", "")).strip()
        rendered_parts.append(f"{label}:\n{content}")

    rendered = "\n\n".join(rendered_parts).rstrip()
    if add_generation_prompt:
        rendered = f"{rendered}\n\nAssistant:"
    return f"{rendered}\n"


def extract_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), flags=re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def extract_last_block(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    end_idx = text.rfind(end_tag)
    if end_idx == -1:
        return None
    start_idx = text.rfind(start_tag, 0, end_idx)
    if start_idx == -1:
        return None
    content_start = start_idx + len(start_tag)
    if content_start > end_idx:
        return None
    return text[content_start:end_idx].strip()


def append_path_block(prompt: str, paths_for_sample: list[str]) -> str:
    path_block = "".join(f"{p}" for p in paths_for_sample) if paths_for_sample else "None"
    return prompt + "\n\n" + f"{PATH_LIST_BEG}\n" + path_block + f"{PATH_LIST_END}\n"


def format_graph_candidate_entity_block(entity_names: list[str] | None) -> str:
    if not entity_names:
        entity_lines = ["None"]
    else:
        entity_lines = [f"{i}. {name}" for i, name in enumerate(entity_names, start=1)]
    return (
        f"{GRAPH_ENTITY_LIST_BEG}\n"
        + "\n".join(entity_lines)
        + f"\n{GRAPH_ENTITY_LIST_END}\n"
    )
