# GSC

## Overview

Temporal Knowledge Graph Reasoning (TKGR) requires not only structural reasoning over temporal graph patterns, but also semantic understanding beyond graph connectivity. The relative importance of these two signals varies across instances: some queries are better resolved by structural evidence, some by semantic reasoning, and others by their synergy. Despite this need, existing methods often lack explicit evidence-aware interaction between structural and semantic reasoning. We propose \textbf{G}raph-Grounded LLM \textbf{S}emantic \textbf{C}alibration (GSC), an explainable framework that uses interpretable structural evidence to guide LLM-based semantic calibration. GSC first employs a Temporal Subgraph Reasoner to produce structural predictions and path-level evidence. These outputs, together with recurring historical statistics, are verbalized into prompts for evidence-conditioned semantic re-prediction. The LLM-generated scores are then used to augment structural scores for final prediction. Experiments on benchmark TKG datasets show that GSC improves predictive accuracy while providing interpretable reasoning evidence.



## Environment

```
pip install torch==2.5.1
pip install transformers==4.46.3

```

## Train TSR



## Evaluation

```shell

```

