"""Plug in measured recall: Qwen3-32B (reasoning) as monitor on 16 attack strata from APPS, BashArena and SLEIGHT-Bench,
5% false-positive rate, uniform routing pi = 0.25 and equal traffic shares. Needs pandas (pip install -e ".[reproduce]")."""
from pathlib import Path
import pandas as pd
import gammacert as gc

strata = pd.read_csv(Path(__file__).parents[1] / "results/monitors/fpr05_surface/exp01_strata_main.csv")
print(strata[["dataset", "attack_type", "n", "r", "rL"]].round(3).to_string(index=False), "\n")
for gamma in [1, 2, 10, 16]:
    point = gc.certificate(0.25, strata.r, gamma=gamma)
    lower = gc.certificate(0.25, strata.rL, gamma=gamma)
    print(f"Gamma {gamma:>3}: certificate {point:12,.0f} from measured recall, {lower:14,.0f} from recall lower bounds")
