"""Experiments 2A and 8: same bulk metric, different certificate; rare blind spot x deployment scale.

Information structures: binormal signal (any AUC) plus a recognisable atom of mass eta on low-coverage
contexts, with the TOTAL AUC held exactly fixed (Theorem 3(b)-(c)).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gammacert.core import *

emit = Emitter("numbers_exp02a_08.tex"); RES, FIG = out_dir("results"), out_dir("figures")
wL, cH, cL, L = 0.1, 0.2, 1.6e-4, np.log(20)

# ------------------------------------------------------------------ 2A
N = 1e8; etaN = L / (cL * wL * N); BETA0 = 1e-4; rows = []
for auc in np.round(np.arange(0.52, 0.96, 0.01), 2):
    for mult in [0, 0.03, 0.1, 0.3, 1, 3]:
        s = TwoStrata(wL, cH, cL, auc, mult * etaN)
        rows.append(dict(auc=auc, eta=mult * etaN, eta_over_etaN=mult, U=s.cert(N), lcvar=s.lcvar(BETA0), acc=s.bayes_acc(), brier=s.brier()))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/exp02a_same_auc.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for (m, s), col in zip(df.groupby("eta_over_etaN"), plt.cm.viridis(np.linspace(0, 0.95, 6))):
    axes[0].semilogy(s.auc, s.U, "o-", ms=2.5, lw=1, color=col, label=fr"$\eta={m:g}\,\eta_N$")
    axes[1].semilogy(s.brier, s.U, "o", ms=2.5, color=col); axes[2].loglog(s.lcvar, s.U, "o", ms=2.5, color=col)
axes[0].set_xlabel("AUC of the system's signal"); axes[1].set_xlabel("Brier score of the Bayes posterior"); axes[2].set_xlabel(fr"$\mathrm{{LCVaR}}_{{{BETA0:g}}}(d)$")
axes[0].set_ylabel(r"certified count $U_{0.05}$"); axes[0].legend(frameon=False, fontsize=6.5); fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp02a_same_auc.pdf")
for auc, tag in [(0.55, "FiftyFive"), (0.8, "Eighty")]:
    s = df[np.isclose(df.auc, auc)]; emit(f"sameAucSpread{tag}", s.U.max() / s.U.min(), "{:.0f}")
    emit(f"sameAucAccGap{tag}", s.acc.max() - s.acc.min(), "{:.1e}"); emit(f"sameAucBrierGap{tag}", s.brier.max() - s.brier.min(), "{:.1e}")
lu = df[np.isfinite(df.U)]; emit("rankCorrAucU", lu[["auc", "U"]].corr("spearman").iloc[0, 1]); emit("rankCorrLcvarU", lu[["lcvar", "U"]].corr("spearman").iloc[0, 1])

# ------------------------------------------------------------------ 8
Ns = 10.0 ** np.arange(4, 8.01, 0.25); etas = 10.0 ** np.arange(-7, -0.99, 0.25); AUC = 0.55
Z = np.array([[TwoStrata(wL, cH, cL, AUC, e).cert(n) if e <= 2 * AUC - 1 + 1e-12 else np.nan for n in Ns] for e in etas])
pd.DataFrame(Z, index=etas, columns=Ns).to_csv(f"{RES}/exp08_scale.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.8))
im = ax.pcolormesh(Ns, etas, np.log10(Z), shading="nearest", cmap="magma"); ax.set_xscale("log"); ax.set_yscale("log")
ax.plot(Ns, L / (cL * wL * Ns), "w--", lw=1.5, label=r"$\eta_N=\log(1/\alpha)/(c_L w N)$"); ax.set_ylim(etas.min(), etas.max())
ax.set_xlabel("deployment scale $N$"); ax.set_ylabel(r"recognisable blind-spot mass $\eta$ (AUC fixed at 0.55)")
fig.colorbar(im, label=r"$\log_{10} U_{0.05}$"); ax.legend(frameon=False, fontsize=7, labelcolor="w", loc="lower left"); fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp08_scale.pdf")
for n, tag in [(1e4, "TenK"), (1e6, "OneM"), (1e8, "HundredM")]:
    emit(f"scaleU{tag}", TwoStrata(wL, cH, cL, AUC, 1e-4).cert(n), "{:.0f}"); emit(f"scaleOpp{tag}", n * wL * 1e-4, "{:.0f}")
print("exp02a/08 done")
