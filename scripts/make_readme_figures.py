"""Regenerate the README figures (assets/figures/*.png) from the library and the shipped result files.

    python scripts/make_readme_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gammacert as gc
from gammacert.core import design, equal_coverage, floor_overlay, scaled_report, ucb0

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical order (light surface) and recessive ink.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#4a3aa7", "#e34948")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
                     "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "lines.linewidth": 2.2})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.15); plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- 1. the headline: concentration breaks the certificate
PI = np.array([1, 0.25, 0.05, 0.002, 0.0002]); W = np.array([0.40, 0.10, 0.20, 0.21, 0.09])
R = np.array([0.94, 0.94, 0.58, 0.58, 0.44]); budget = W @ PI
gam = np.logspace(0, 2, 200)
as_designed = gc.certificate_curve(PI, R, W, gam)
equal = gc.certificate_curve(gc.equal_coverage_design(R, W, budget), R, W, gam)

strata = pd.read_csv(RES / "monitors/fpr05_surface/exp01_strata_main.csv")


def short(t):
    t = t.replace("control_tax:", "").replace("apps:unknown", "AI-Control set")
    for p in ("anthropic/", "openai/", "google/"): t = t.replace(p, "")
    t = t.replace("-latest", "").replace("claude-3-7-sonnet-thinking", "claude-3.7-sonnet (think)")
    t = t.replace("claude-3-7-sonnet", "claude-3.7-sonnet").replace("claude-3-5-haiku", "claude-3.5-haiku")
    return t.replace("other (small types pooled)", "10 small categories").replace("_", " ")


fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
a = ax[0]
a.loglog(gam, as_designed, color=BLUE, label="monitoring as designed")
a.loglog(gam, equal, color=ORANGE, label="same budget, equal coverage")
for g, u in [(1, as_designed[0]), (2, gc.certificate(PI, R, W, 2))]:
    a.plot(g, u, "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
a.annotate("2.6 harmful actions\n(proportional allocation)", (1, as_designed[0]), xytext=(1.25, 0.9), fontsize=10, color=INK2)
a.annotate("244 when harm may be\n2x over-represented", (2, gc.certificate(PI, R, W, 2)), xytext=(2.6, 60), fontsize=10,
           color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.annotate("8.3 for every $\\Gamma$", (40, 8.3), xytext=(12, 2.2), fontsize=10, color=INK2,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.set_xlabel(r"strategic concentration $\Gamma$"); a.set_ylabel("95% bound on harmful actions (0 detections)")
a.set_title("A silent record rules out less when harm can concentrate"); a.legend(loc="upper left")
a.set_ylim(0.7, 1e5)

b = ax[1]
order = strata.assign(label=strata.attack_type.map(short)).reset_index(drop=True)
colors = {"apps": BLUE, "basharena": ORANGE, "sleight": AQUA}
names = {"apps": "APPS backdoors", "basharena": "BashArena", "sleight": "SLEIGHT-Bench"}
y = np.arange(len(order))[::-1]
b.barh(y, order.r, height=0.72, color=[colors[d] for d in order.dataset], edgecolor=SURFACE, linewidth=2)
b.scatter(order.rL, y, marker="|", s=140, color=INK, linewidths=2, zorder=3, label="97.5% simultaneous lower bound")
b.set_yticks(y); b.set_yticklabels(order.label, fontsize=9); b.set_xlim(0, 1); b.grid(axis="y", visible=False)
from matplotlib.patches import Patch
handles = [Patch(color=colors[d], label=names[d]) for d in colors] + [b.collections[-1]]
b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=2, fontsize=9)
b.set_xlabel("recall at 5% false-positive rate"); b.set_title("One monitor, 16 attack strata: recall from 0 to 0.92")
fig.tight_layout(w_pad=3); save(fig, "hero_certificate_and_recall.png")

# ---------------------------------------------------------------- 2. tail stress test on a real monitor
sel = pd.read_csv(RES / "monitors/fpr05_surface/exp03_selectors.csv")
sel = sel[sel.surface == "all"]
proxies = sel[sel.selector.str.startswith("proxy")]
best_proxy = proxies[proxies.beta == 0.01].sort_values("mean_coverage_selected").selector.iloc[0]
lines = [("random", "random order", INK2, "--"), ("supervised probe (TF-IDF, cross-fitted)", "supervised probe", AQUA, "-"),
         ("LLM selector: gpt-oss-120b", "gpt-oss-120b selector", BLUE, "-"),
         (best_proxy, "another monitor, best of 16", ORANGE, "-"), ("oracle", "oracle", INK, ":")]
fig, a = plt.subplots(figsize=(7.6, 5.0))
for key, lab, col, ls in lines:
    s = sel[sel.selector == key].sort_values("beta")
    a.semilogx(s.beta, s.mean_coverage_selected, ls, color=col, marker="o", ms=5, label=f"{lab} (AUC {s.auc.iloc[0]:.2f})")
a.axvline(0.01, color=GRID, lw=1.2, zorder=0)
a.set_xlabel(r"fraction $\beta$ of 2,372 attacks selected"); a.set_ylabel("monitor recall on the selection")
a.set_title("A selector with AUC 0.58 finds a 1% tail with recall 0.125"); a.set_ylim(-0.03, 1.02)
a.annotate("0.125", (0.01, 0.125), xytext=(0.0042, 0.16), fontsize=10, color=BLUE, fontweight="bold")
a.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2); fig.tight_layout(); save(fig, "tail_stress_test.png")

# ---------------------------------------------------------------- 3. validity: the exact Poisson constant uses the full alpha
cov = pd.read_csv(RES / "latex_macros/coverage_by_d.csv")
fig, a = plt.subplots(figsize=(7.6, 4.2))
x = cov.d.values
a.bar(x - 0.19, cov.chernoff, 0.36, color=ORANGE, label="Chernoff constant", edgecolor=SURFACE, linewidth=2)
a.bar(x + 0.19, cov.garwood, 0.36, color=BLUE, label="exact Poisson constant (ours)", edgecolor=SURFACE, linewidth=2)
a.axhline(1, color=INK, lw=1, ls="--"); a.text(8.6, 1.02, r"nominal $\alpha$", ha="right", va="bottom", fontsize=9, color=INK2)
a.set_xticks(x); a.set_xlabel("number of detections d"); a.set_ylabel(r"worst exact miscoverage / $\alpha$")
a.set_title("510,390 exact checks: never above α, and tight"); a.set_ylim(0, 1.18); a.grid(axis="x", visible=False)
a.legend(loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.98)); fig.tight_layout(); save(fig, "validity.png")

# ---------------------------------------------------------------- 4. monitoring designs on measured recall bounds
m = json.load(open(RES / "monitors/measured_rf.json"))
keep = np.array([not s.startswith("sleight") for s in m["strata"]])
w = np.array(m["w"])[keep]; w = w / w.sum(); pi_rep = np.array(m["pi"])[keep]; r = np.array(m["r"])[keep]
pibar = w @ pi_rep
designs = [("risk-targeted", design(r, w, 1.0, pibar)[1], RED), ("report-shaped tiers", scaled_report(pi_rep, w, pibar), YELLOW),
           ("floor + overlay", floor_overlay(r, w, pibar, 0.5)[0], VIOLET), ("uniform floor", np.full(len(w), pibar), AQUA),
           ("equal coverage, capped", equal_coverage(r, w, pibar), BLUE)]
g = np.logspace(0, 2, 150)
fig, a = plt.subplots(figsize=(7.6, 4.6))
for lab, p, col in designs:
    u = np.array([ucb0(p * r, w, x) for x in g]); u[~np.isfinite(u)] = np.nan
    a.loglog(g, u, color=col, label=lab)
a.set_xlabel(r"strategic concentration $\Gamma$"); a.set_ylabel("95% bound on harmful actions (0 detections)")
a.set_title("Same budget, five designs: only equalizing stays flat"); a.set_ylim(2, 300)
a.text(4.6, 230, r"$\to\infty$", color=RED, fontsize=12, va="center")
a.legend(fontsize=9, loc="upper left"); fig.tight_layout(); save(fig, "designs.png")
print("designs at Gamma=1 and inf:", {lab: (round(ucb0(p * r, w, 1), 1), round(ucb0(p * r, w, np.inf), 1)) for lab, p, _ in designs})
