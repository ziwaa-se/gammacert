"""Experiment 12: is the certificate limited by the monitoring design or by the size of the recall audit?

T attack types with true recalls r_t; n red-team attacks per type. Compare plug-in, marginal Clopper-Pearson,
simultaneous Clopper-Pearson (Proposition 1) and report (i) the resulting certificate, (ii) whether the recall
input is valid (r_input <= true worst-type recall), which is what Proposition 1 needs.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gammacert.core import *

emit = Emitter("numbers_exp12.tex"); RES, FIG = out_dir("results"), out_dir("figures"); rng = np.random.default_rng(12)
L = np.log(20); A2 = 0.025; PI0 = 0.25; REPS = 4000
NS = [5, 10, 20, 40, 80, 163, 500, 2000, 10000]
PROFILES = {"3 types (0.94, 0.58, 0.44)": np.array([0.94, 0.58, 0.44]),
            "11 types, SLEIGHT-like spread": np.linspace(0.14, 0.94, 11), "40 types, SLEIGHT-like spread": np.linspace(0.14, 0.94, 40)}
rows = []
for pname, rt in PROFILES.items():
    T = len(rt); U_true = L / kappa(PI0 * rt.min())
    for n in NS:
        X = rng.binomial(n, rt, size=(REPS, T))
        est = {"plug-in": (X / n).min(1), "marginal CP": np.array([[cp_lower(x, n, A2) for x in row] for row in X[:400]]).min(1),
               "simultaneous CP": np.array([[cp_lower(x, n, A2 / T) for x in row] for row in X[:400]]).min(1)}
        for m, rin in est.items():
            with np.errstate(divide="ignore"): U = L / kappa(PI0 * rin)               # a zero recall bound certifies nothing: U = inf
            rows.append(dict(profile=pname, n=n, method=m, U_median=np.median(U), U_q90=np.quantile(U, 0.9), U_true=U_true,
                             ratio_median=np.median(U) / U_true, recall_input_valid=(rin <= rt.min() + 1e-12).mean()))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/exp12_recall_n.csv", index=False); print(df.round(3).to_string())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for ax, (pname, s) in zip(axes, df.groupby("profile", sort=False)):
    for m, ss in s.groupby("method", sort=False): ax.loglog(ss.n, ss.ratio_median, "o-", label=f"{m} (valid {ss.recall_input_valid.min():.0%}+)")
    ax.axhline(1, color="k", lw=0.6); ax.set_title(pname, fontsize=8); ax.set_xlabel("red-team attacks per type $n$"); ax.legend(frameon=False, fontsize=6.5)
axes[0].set_ylabel(r"$U$ with estimated recall / $U$ with true recall"); fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp12_recall_n.pdf")
s = df[(df.method == "simultaneous CP") & df.profile.str.startswith("11")]
for n, tag in [(5, "Five"), (40, "Forty"), (163, "Metr"), (2000, "TwoK")]:
    v = s[s.n == n].ratio_median.iloc[0]; emit(f"recallInflation{tag}", v) if np.isfinite(v) else emit.f.write(f"\\newcommand{{\\recallInflation{tag}}}{{\\ensuremath{{\\infty}}}}\n")
emit("pluginValidForty", df[(df.method == "plug-in") & df.profile.str.startswith("11") & (df.n == 40)].recall_input_valid.iloc[0])
print("exp12 done")
