# Reproducing the paper

Run everything from the repository root after `pip install -e ".[reproduce]"`. Outputs go to `$ZD_OUT`, else `./outputs`.
`pipeline.sh` lists every step in the order used for the paper.

## From shipped results (CPU only)

The score matrix and all downstream result files are in [`../results`](../results), so the figures and tables below can be
checked without a GPU.

| Paper | What | Command | Result file |
|---|---|---|---|
| Table 1, Figure 9, Sections 3 and 7 | Report-shaped tiers, minimax design, campaign bound, false certification | `python -m reproduce.theory_examples.certificate_examples`; `python -m reproduce.theory_examples.design_examples` | `results/latex_macros/numbers_certificate_examples.tex`, `numbers_design_examples.tex` |
| Figure 2 | Recall by stratum; certificate under uniform routing | `python -m reproduce.monitors.analyze exp01` | `results/monitors/fpr05_surface/exp01_*.csv` |
| Figure 3(a), Figure 13 | Same AUC, different certificate; deployment scale | `python -m reproduce.simulations.exp02a_08_tail` | `results/simulations/exp02a_same_auc.csv`, `exp08_scale.csv` |
| Figure 3(b) | Tail stress test on a real monitor | `python -m reproduce.monitors.analyze exp03` | `results/monitors/fpr05_surface/exp03_selectors.csv` |
| Figure 4(a), Table 5 | Exact validity of the certificate; baselines under concentration | `python -m reproduce.simulations.exp04_coverage` | `results/simulations/exp04a_*.csv`, `exp04b_baselines.csv` |
| Figure 4(b), Table 7 (left) | Five designs on measured recall bounds | `ZD_EXCLUDE=sleight python -m reproduce.simulations.exp05_11_design_cost` | `results/simulations/exp05_designs_nosleight.csv` |
| Table 7 (right) | Designs with the SLEIGHT-Bench strata, unfloored recall bounds | `python -m reproduce.simulations.exp05b_true_rL` | `results/simulations/exp05_designs_trueRL.csv` |
| Table 4, Appendix H.1 | Operating points and threshold bootstrap | `ZD_FPR=0.01 python -m reproduce.monitors.analyze exp01` (and 0.02, 0.05, 0.10) | `results/monitors/fpr*/exp01_threshold_ci.csv` |
| Table 6 | Recall audits and verification bias | `python -m reproduce.simulations.exp07_audit --scores results/monitors/stage1_scores.parquet --tag real --fpr 0.05` | `results/simulations/exp07_audit_real.csv` |
| Table 8 | Randomized monitor assignment, held out | `python -m reproduce.monitors.analyze exp06` | `results/monitors/fpr*/exp06_*.csv` |
| Figure 12 | Recall audit size | `python -m reproduce.simulations.exp12_recall_n` | `results/simulations/exp12_recall_n.csv` |
| Appendix H.3 | Hazard-scale design | `python -m reproduce.simulations.exp10_hazard` | `results/simulations/exp10_hazard.csv` |
| All figures | Paper figures and LaTeX number macros | `python -m reproduce.paper.paper_figs`; `python -m reproduce.paper.paper_facts` | `outputs/paper/` |

The monitor analyses read the score matrix from `outputs/results/<operating point>/M_long.parquet`. To start from the shipped
matrix, copy it there first: `mkdir -p outputs/results && cp -r results/monitors/* outputs/results/`. `exp01` and `exp06` then run
directly. `exp03` also trains a bag-of-words probe on the transcript text, which is not shipped; build the transcript table with
step 3 below first (no GPU needed).

## From scratch (GPU)

1. `pip install -e ".[monitors]"` (vLLM 0.29.0; full pinned environment in [`requirements-full.txt`](requirements-full.txt)).
2. `hf auth login`, then accept the terms of the SLEIGHT-Bench dataset on its Hugging Face page.
3. `python -m reproduce.monitors.data_prep --sources apps basharena sleight --n-apps 1500` builds the table of 4,058 transcripts.
4. Score each monitor configuration with `python -m reproduce.monitors.score_matrix` (the 17 monitor and 3 selector configurations
   are listed in `pipeline.sh`). One configuration takes between 12 minutes and two hours on one GPU; the Qwen3-8B and 4-bit
   configurations fit on a 40 GB A100, gpt-oss-120b and Qwen3-32B in bf16 need a larger GPU. Scoring is resumable, and
   `--shard k/K` splits one configuration across K GPUs.
5. `python -m reproduce.monitors.reparse_scores`, then `python -m reproduce.monitors.analyze build|exp01|exp06|exp03` at each operating
   point (`ZD_FPR` in 0.01, 0.02, 0.05, 0.10 and `ZD_CALIB` in `surface`, `pooled`).

## Result files

- `results/monitors/fpr{01,02,05,10}_{surface,pooled}/`: one directory per operating point; `fpr05_surface` is the main one.
  - `M_long.parquet` is the score matrix: `row_id`, `config_id`, `dataset`, `is_attack`, `attack_type`, `score` (mean over
    samples), `unparsed`, `p_detect` (flag at the threshold) and `n_chars`.
  - The CSV files hold recall by stratum, thresholds, bootstrap intervals, selector results and the randomized-assignment study.
- `results/monitors/measured_rf.json`: recall lower bounds and false-positive rates of the main monitor at the main operating
  point. The two SLEIGHT-Bench bounds are stored with a floor of 0.001 so that the design and hazard simulations can cover every
  stratum; Table 7 uses the unfloored bounds.
- `results/monitors/stage1_scores.parquet`: the main monitor's APPS scores and labels.
- `results/simulations/`: simulation outputs. The per-configuration validity table `exp04a_coverage.csv` (510,390 rows, 45 MB) is
  not shipped; `exp04_coverage` regenerates it in a few minutes.
- `results/latex_macros/`: the number macros and small tables that the manuscript reads.
