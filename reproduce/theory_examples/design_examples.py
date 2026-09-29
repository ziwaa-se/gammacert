"""Illustrative numbers and figures for the design results: hidden floors, campaign interruption, near-uniform coverage
(OpenAI illustration), design curves, floor-plus-overlay budget, randomized monitor assignment."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from gammacert.certlib import kappa, kstar, ucb0
alpha = 0.05; L = np.log(1 / alpha)
import os; os.makedirs("outputs/theory", exist_ok=True)
out = open("outputs/theory/numbers_design_examples.tex", "w")
def emit(name, val, fmt="{:.2f}"):
    txt = fmt.format(val)
    if "e" in txt and "{:d}" != fmt:
        m_, e_ = txt.split("e"); txt = f"{m_}\\times10^{{{int(e_)}}}"
    out.write(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{txt}}}}}\n"); print(name, txt)
floor = lambda K, r: (1 - alpha ** (1 / K)) / r
for K, kt in [(10, "Ten"), (30, "Thirty"), (100, "Hundred")]:
    emit(f"floorLow{kt}", floor(K, 0.44), "{:.2g}"); emit(f"floorHigh{kt}", floor(K, 0.94), "{:.2g}")
emit("floorApproxThirty", L / (30 * 0.44))
# report-style tiers (Section 7)
pi = np.array([1.0, 0.25, 0.05, 0.002, 0.0002]); w = np.array([0.40, 0.10, 0.20, 0.21, 0.09]); r = np.array([0.94, 0.94, 0.58, 0.58, 0.44])
pibar = (w * pi).sum(); Hbar = 1 / (w / r).sum()
# campaign interruption: k actions, P(no timely detection) <= exp(-k kappa*(Gamma))
k = 24
for G, tag in [(1, "One"), (1.5, "OnePointFive"), (2, "Two"), (5, "Five")]:
    emit(f"campaign{tag}", np.exp(-k * kstar(pi * r, w, G)), "{:.2g}")
# with timeliness x intervention factor 0.5 on every stratum
for G, tag in [(1, "One"), (2, "Two")]:
    emit(f"campaignHalf{tag}", np.exp(-k * kstar(pi * r * 0.5, w, G)), "{:.2g}")
emit("campaignK", k, "{:d}")
# OpenAI-style: near-uniform 99.9% coverage
for rl, tag in [(0.5, "Half"), (0.284, "Low")]:
    emit(f"oaiU{tag}", L / kappa(0.999 * rl), "{:.1f}")
emit("oaiGammaInf", 1 / 0.001, "{:.0f}")
# design curves: required monitored fraction to certify U_alpha(0;Gamma) <= K
Ks = np.logspace(np.log10(6), 3, 60)
def budget_scaled(K, G):
    # report-shaped design pi(s) = min(1, s*pi_report); minimal budget s.t. U <= K
    f = lambda s: L / max(kstar(np.minimum(1, s * pi) * r, w, G), 1e-300) - K
    if f(1e6) > 0: return np.nan
    s = brentq(f, 1e-9, 1e6); return (w * np.minimum(1, s * pi)).sum()
def budget_equal(K):
    c = 1 - alpha ** (1 / K)
    return c / Hbar if c <= r.min() else np.nan
def budget_floor(K):
    c = 1 - alpha ** (1 / K)
    return c / r.min() if c <= r.min() else np.nan
fig, ax = plt.subplots(figsize=(5.4, 3.5))
for G, ls in [(1, "-"), (1.5, "-"), (2, "-"), (5, "-")]:
    ax.loglog(Ks, [budget_scaled(K, G) for K in Ks], lw=1.8, label=fr"report-shaped tiers, $\Gamma={G}$")
ax.loglog(Ks, [budget_equal(K) for K in Ks], "k--", lw=2, label=r"equal coverage (any $\Gamma\geq\Gamma^\dagger$)")
ax.loglog(Ks, [budget_floor(K) for K in Ks], "k:", lw=2, label=r"uniform hidden floor (any $\Gamma$)")
ax.axhline(pibar, color="grey", lw=0.8); ax.text(250, pibar * 0.8, "illustrative current budget", color="grey", fontsize=7, ha="center", va="top")
ax.set_xlabel("target: certify at most $K$ harmful actions (95%, zero detections)")
ax.set_ylabel("required monitored fraction")
ax.set_ylim(1e-3, 1.2); ax.legend(frameon=False, fontsize=7, loc="lower left"); fig.tight_layout()
fig.savefig("outputs/theory/fig_design_curve.pdf")
for K, kt in [(10, "Ten"), (30, "Thirty"), (100, "Hundred")]:
    emit(f"budEq{kt}", budget_equal(K), "{:.3f}")
    for G, gt in [(1, "One"), (2, "Two")]:
        b = budget_scaled(K, G); emit(f"budRep{gt}{kt}", b if np.isfinite(b) else float('nan'), "{:.3f}")

# ---------- architecture section ----------
for K, kt in [(100, "Hundred"), (50, "Fifty"), (20, "Twenty"), (10, "Ten")]:
    ex = (1 - alpha ** (1 / K)) / 0.60; ap = L / (K * 0.60)
    emit(f"floorSixtyExact{kt}", ex, "{:.3f}"); emit(f"floorSixtyApprox{kt}", ap, "{:.3f}")
# floor-plus-target budget decomposition (report-style illustration)
c0 = 1 - alpha ** (1 / 30)
emit("hybridCzero", c0, "{:.4f}"); emit("hybridRobustBudget", c0 / Hbar, "{:.3f}")
emit("hybridResidual", pibar - c0 / Hbar, "{:.3f}"); emit("hybridResidualShare", (pibar - c0 / Hbar) / pibar * 100, "{:.0f}")
emit("hybridMaxCzero", pibar * Hbar, "{:.3f}"); emit("hybridMinK", L / kappa(pibar * Hbar), "{:.1f}")
# randomized monitor assignment: matrix game value (rows attack types, cols monitors)
from scipy.optimize import linprog
R = np.array([[0.9, 0.3], [0.2, 0.8]])
T, J = R.shape
res = linprog(np.r_[-1, np.zeros(J)], A_ub=np.c_[np.ones(T), -R], b_ub=np.zeros(T),
              A_eq=np.r_[0, np.ones(J)][None, :], b_eq=[1], bounds=[(None, None)] + [(0, 1)] * J)
emit("gameValue", -res.fun, "{:.2f}"); emit("gameP", res.x[1], "{:.2f}")
emit("gameBestSingle", R.min(axis=0).max(), "{:.2f}"); emit("gameEnsembleLB", R.max(axis=1).min(), "{:.2f}")
out.close()
