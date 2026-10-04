"""Experiments 5 + 11: monitoring designs at a fixed monitored fraction, and what robustness costs.

Uses measured recall / FPR (outputs/results/measured_rf.json, written by monitors/analyze.py exp01) when present,
otherwise the paper's illustrative tiers with a placeholder FPR of 1% per stratum.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
from gammacert.core import *

EXCL = os.environ.get("ZD_EXCLUDE") or None; SUF = f"_no{EXCL}" if EXCL else ""      # ZD_EXCLUDE=sleight: design comparison without the recall blind spot
emit = Emitter(f"numbers_exp05{SUF}.tex"); RES, FIG = out_dir("results"), out_dir("figures")
_m = emit.__call__; emit = (lambda n, v, fmt="{:.2f}": _m(n + ("NoBlind" if EXCL else ""), v, fmt))
w, pi_rep, r, f, measured = load_instance(EXCL); w = w / w.sum(); L = np.log(20)
pibar = (w * pi_rep).sum(); Hb = 1 / (w / r).sum(); print("measured inputs:", measured, "| excluded:", EXCL, "| H =", len(w), "| Thm 4 assumption pibar*Hbar <= r_min:", bool(pibar * Hb <= r.min()), "| pibar", pibar, "| Hbar", Hb)
N_WEEK = 1e6        # harm-capable units per week, used only to express false positives as a review load


def designs(pibar):
    d = {"report-shaped tiers": scaled_report(pi_rep, w, pibar), "risk-targeted ($\\Gamma{=}1$ optimum)": design(r, w, 1.0, pibar)[1],
         "equal effective coverage": equal_coverage(r, w, pibar), "uniform hidden floor": np.full(len(w), pibar)}
    d["floor + risk overlay (50/50)"], _ = floor_overlay(r, w, pibar, 0.5)
    return d


# ---- main figure: certificate against Gamma, same budget
Gs = np.logspace(0, 3, 80); rows = []
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for name, p in designs(pibar).items():
    U = [ucb0(p * r, w, G) for G in Gs]; ax.loglog(Gs, U, lw=2, label=name)
    rows.append(dict(design=name, budget_used=(w * p).sum(), min_coverage=(p * r).min(), mean_coverage=(w * p * r).sum(),
                     fp_per_week=N_WEEK * (w * p * f).sum(), **{f"U_G{g}": ucb0(p * r, w, g) for g in (1, 1.5, 2, 5, 10, 100, np.inf)}))
ax.axvline(Hb / r.min(), color="grey", lw=0.8, ls=":"); ax.set_xlabel(r"$\Gamma$"); ax.set_ylabel(r"$U_{0.05}(0;\Gamma)$")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp05_designs{SUF}.pdf")
tab = pd.DataFrame(rows); tab.to_csv(f"{RES}/exp05_designs{SUF}.csv", index=False); print(tab.round(3).T)
for name, tag in [("report-shaped tiers", "Rep"), ("equal effective coverage", "Eq"), ("floor + risk overlay (50/50)", "Hyb"), ("uniform hidden floor", "Unif")]:
    row = tab[tab.design == name].iloc[0]
    for g, gt in [(1, "One"), (2, "Two"), (10, "Ten"), (np.inf, "Inf")]: emit(f"des{tag}{gt}", row[f"U_G{g}"], "{:.1f}")

# ---- cost frontier: certificate strength against budget and against false-positive review load
budgets = np.logspace(-2.3, np.log10(0.8), 40); rows = []
for b in budgets:
    for name, p in designs(b).items():
        for G in (1, 2, np.inf):
            rows.append(dict(budget=b, design=name, G=G, U=ucb0(p * r, w, G), fp_per_week=N_WEEK * (w * p * f).sum()))
fr = pd.DataFrame(rows); fr.to_csv(f"{RES}/exp11_frontier{SUF}.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for (name, s), col in zip(fr[fr.G == 2].groupby("design", sort=False), plt.cm.tab10.colors):
    axes[0].loglog(s.budget, s.U, lw=1.8, color=col, label=name); axes[1].loglog(s.fp_per_week, s.U, lw=1.8, color=col)
axes[0].set_xlabel("monitored fraction (compute cost)"); axes[1].set_xlabel(f"false positives per {N_WEEK:.0e} units (review load)")
axes[0].set_ylabel(r"$U_{0.05}(0;\Gamma=2)$"); axes[0].legend(frameon=False, fontsize=6.5); fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp11_frontier{SUF}.pdf")

# ---- Theorem 4 under a false-positive budget (the 'design under operational constraints' limitation): still an LP
f_het = f if measured else np.array([0.005, 0.01, 0.02, 0.03, 0.05])     # placeholder heterogeneity until measured
rows = []
for G in (1, 1.5, 2, 5, 100):
    v0, p0 = design(r, w, G, pibar); fp0 = (w * p0 * f_het).sum()
    for share in (1.0, 0.75, 0.5, 0.25):
        v, p = design_fp(r, w, G, pibar, f_het, share * fp0)
        rows.append(dict(G=G, fp_budget_share=share, value=v, U=L / kappa(v) if v > 0 else np.inf, pi=np.round(p, 4).tolist()))
pd.DataFrame(rows).to_csv(f"{RES}/exp11_fp_constrained{SUF}.csv", index=False); print(pd.DataFrame(rows).drop(columns="pi").round(3))
print("exp05/11 done")
