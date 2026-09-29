"""Table 7 (designs on measured recall, WITH the SLEIGHT-Bench strata) recomputed from the actual simultaneous recall lower
bounds, without the 0.001 floor that measured_rf.json applies. A stratum with r^L = 0 has zero certified coverage under every
routing, so equal effective coverage is only attainable at level 0, and the floor of the floor+overlay design is 0.
Writes outputs/results/exp05_designs_trueRL.csv and prints the floored values next to it as a check of the old table."""
import json, os, numpy as np, pandas as pd
from gammacert.core import *

RES = out_dir("results"); d = json.load(open(os.path.join(RES, "measured_rf.json")))
st = pd.read_csv(os.path.join(RES, "fpr05_surface", "exp01_strata_main.csv")); st["name"] = st.dataset + ":" + st.attack_type
assert list(st.name) == d["strata"], "stratum order differs"
w = np.array(d["w"], float); w = w / w.sum(); pi_rep = np.array(d["pi"], float); pibar = (w * pi_rep).sum()


def eqcov(r):
    if r.min() <= 0: return np.zeros_like(r)           # the only equal effective coverage reachable is 0
    return equal_coverage(r, w, pibar)


def flov(r):
    if r.min() <= 0: return np.clip(design(r, w, 1.0, pibar)[1], 0, 1)   # floor c0 = 0.5*pibar*Hbar = 0: overlay only
    return floor_overlay(r, w, pibar, 0.5)[0]


rows = []
for lab, r in (("floored_0.001", np.array(d["r"], float)), ("true_rL", st.rL.values.astype(float))):
    des = {"report-shaped tiers": scaled_report(pi_rep, w, pibar), "risk-targeted": design(r, w, 1.0, pibar)[1],
           "equal effective coverage": eqcov(r), "uniform hidden floor": np.full(len(w), pibar), "floor + risk overlay": flov(r)}
    for name, p in des.items():
        rows.append(dict(rL=lab, design=name, budget_used=(w * p).sum(), **{f"U_G{g}": ucb0(p * r, w, g) for g in (1, 2, 10, 16, np.inf)}))
out = pd.DataFrame(rows); out[out.rL == "true_rL"].to_csv(os.path.join(RES, "exp05_designs_trueRL.csv"), index=False)
out[out.rL != "true_rL"].to_csv(os.path.join(RES, "exp05_designs_floored_check.csv"), index=False)   # reproduces the old table
print("strata rL:"); print(st[["name", "n", "r", "rL"]].to_string())
pd.set_option("display.width", 200); print(out.round(2).to_string())
