#!/bin/bash
# Order in which the results of the paper were produced. Run from this directory. Steps 2-3 need a GPU and vLLM;
# every other step runs on a CPU. Outputs go to $ZD_OUT, else ./outputs.
set -euo pipefail

# 1. Transcript table from the public datasets (needs `hf auth login` and accepting the SLEIGHT-Bench terms).
python -m reproduce.monitors.data_prep --sources apps basharena sleight --n-apps 1500

# 2. Score matrix: seventeen monitor configurations. Each call writes one column; --think turns on reasoning
#    (Qwen3 thinking mode; for gpt-oss-120b, medium instead of low reasoning effort).
S="python -m reproduce.monitors.score_matrix"
TH="--think --max-new-tokens 3072 --n-samples 2"
for m in Qwen/Qwen3-8B Qwen/Qwen3-32B-AWQ Qwen/Qwen3-32B openai/gpt-oss-120b; do $S --monitor $m; $S --monitor $m $TH; done
$S --monitor Qwen/Qwen3-14B --max-model-len 12288; $S --monitor Qwen/Qwen3-14B --max-model-len 12288 $TH
$S --monitor Qwen/Qwen3-8B --prompt monitor_minimal
$S --monitor Qwen/Qwen3-32B --prompt monitor_minimal
$S --monitor Qwen/Qwen3-32B --prompt monitor_checklist --max-new-tokens 1024
$S --monitor openai/gpt-oss-120b --prompt monitor_minimal $TH
$S --monitor openai/gpt-oss-120b --prompt monitor_checklist $TH
$S --monitor openai/gpt-oss-120b --view action_only $TH
$S --monitor openai/gpt-oss-120b --view task_action $TH

# 3. Selectors for the tail stress test (attack transcripts only).
$S --monitor Qwen/Qwen3-8B --prompt selector_default --attacks-only
$S --monitor Qwen/Qwen3-32B --prompt selector_default --attacks-only
$S --monitor openai/gpt-oss-120b --prompt selector_default --attacks-only $TH

# 4. Re-parse stored outputs (a score is never read from a reasoning trace cut off by the token limit), then analyse
#    at every operating point: FPR 1, 2, 5, 10% with surface or pooled thresholds. The main one is 5% / surface.
python -m reproduce.monitors.reparse_scores
for f in 0.01 0.02 0.05 0.10; do for c in surface pooled; do
  ZD_FPR=$f ZD_CALIB=$c python -m reproduce.monitors.analyze build
  ZD_FPR=$f ZD_CALIB=$c python -m reproduce.monitors.analyze exp01
  ZD_FPR=$f ZD_CALIB=$c python -m reproduce.monitors.analyze exp06
  ZD_FPR=$f ZD_CALIB=$c python -m reproduce.monitors.analyze exp03
done; done

# 5. Simulations and exact computations (CPU). The design, hazard and audit steps read the measured recall and FPR of step 4.
python -m reproduce.simulations.run_all
python -m reproduce.simulations.exp05_11_design_cost; ZD_EXCLUDE=sleight python -m reproduce.simulations.exp05_11_design_cost
python -m reproduce.simulations.exp05b_true_rL
python -m reproduce.simulations.exp10_hazard
python - <<'PY'
import pandas as pd
M = pd.read_parquet("outputs/results/fpr05_surface/M_long.parquet"); main = "Qwen3-32B__monitor_default__full__think__n2"
M[(M.config_id == main) & (M.dataset == "apps")][["score", "is_attack"]].dropna().to_parquet("outputs/results/stage1_scores.parquet")
PY
python -m reproduce.simulations.exp07_audit --scores outputs/results/stage1_scores.parquet --tag real --fpr 0.05

# 6. Numbers of the theory sections, then every figure and LaTeX number macro used in the paper.
python -m reproduce.theory_examples.certificate_examples; python -m reproduce.theory_examples.design_examples
python -m reproduce.paper.paper_figs; python -m reproduce.paper.paper_facts
