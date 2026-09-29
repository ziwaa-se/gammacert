"""Figures for the ICLR manuscript, drawn at the 5.5in text width with 7-8pt type. Reads only result files (CPU, < 2 min).

    python -m reproduce.paper.paper_figs --dest outputs/paper/figures
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gammacert.core import *

ap = argparse.ArgumentParser(); ap.add_argument("--dest", default="outputs/paper/figures"); ap.add_argument("--tag", default="fpr05_surface"); a = ap.parse_args()
os.makedirs(a.dest, exist_ok=True); R = out_dir("results"); RT = os.path.join(R, a.tag); W = 5.5
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "pdf.fonttype": 42, "font.family": "serif"})
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "black": "#000000"}   # Okabe-Ito
DS = {"apps": OK["blue"], "basharena": OK["orange"], "sleight": OK["red"]}; DSN = {"apps": "APPS", "basharena": "BashArena", "sleight": "SLEIGHT"}
save = lambda fig, name: (fig.savefig(os.path.join(a.dest, name), bbox_inches="tight", pad_inches=0.02), plt.close(fig), print("wrote", name))


def short(t):
    t = t.replace("control_tax:", "").replace("apps:unknown", "AI-Control set").replace("anthropic/", "").replace("openai/", "").replace("google/", "").replace("-latest", "")
    return t.replace("claude-3-7-sonnet-thinking", "claude-3.7-sonnet (think)").replace("claude-3-7-sonnet", "claude-3.7-sonnet").replace("claude-3-5-haiku", "claude-3.5-haiku").replace("other (small types pooled)", "10 small categories").replace("_", " ")


# ------------------------------------------------------------------ Fig. 1: measured heterogeneity and what it does to the certificate
tab = pd.read_csv(f"{RT}/exp01_strata_main.csv"); tab["o"] = tab.dataset.map({"apps": 0, "basharena": 1, "sleight": 2}); tab = tab.sort_values(["o", "r"], ascending=[True, False]).reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.62)); y = np.arange(len(tab))[::-1]
ax[0].barh(y, tab.r, color=[DS[d] for d in tab.dataset], height=0.72); ax[0].plot(tab.rL, y, "|", color="k", ms=6, mew=1.1)
ax[0].set_yticks(y); ax[0].set_yticklabels([short(t) for t in tab.attack_type], fontsize=5.8); ax[0].set_xlim(0, 1); ax[0].set_xlabel("recall at 5% FPR (bar); simultaneous lower bound (tick)")
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=DS[d], label=DSN[d]) for d in DS], frameon=False, loc="lower right", handlelength=1); ax[0].set_title("(a) one monitor, 16 attack strata", loc="left")
H = len(tab); w0 = np.full(H, 1 / H); Gs = np.logspace(0, 2, 60); PIB = 0.25; rng = np.random.default_rng(1)
for col, lab, rr in ((OK["blue"], "measured recall $r_h$", tab.r.values), (OK["red"], "lower bound $r_h^L$", tab.rL.values)):
    U = np.array([[ucb0(PIB * rr, w, G) for G in Gs] for w in [w0] + [rng.dirichlet(np.ones(H)) for _ in range(300)]]); U[~np.isfinite(U)] = 1e7
    ax[1].fill_between(Gs, np.percentile(U, 10, 0), np.percentile(U, 90, 0), color=col, alpha=0.18, lw=0); ax[1].loglog(Gs, U[0], color=col, label=lab)
ax[1].axhline(np.log(20) / kappa(PIB * np.average(tab.r, weights=tab.n)), color="k", ls=":", lw=1, label="average recall only")
ax[1].set_ylim(8, 3e5); ax[1].set_xlabel(r"strategic concentration $\Gamma$"); ax[1].set_ylabel(r"certified count $U_{0.05}(0;\Gamma)$"); ax[1].legend(frameon=False, loc="upper left"); ax[1].set_title(r"(b) uniform routing, $\pi=0.25$", loc="left")
save(fig, "fig_main_heterogeneity.pdf")

# ------------------------------------------------------------------ Fig. 2: bulk discrimination versus the tail
fig, ax = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw=dict(wspace=0.36)); df = pd.read_csv(f"{R}/exp02a_same_auc.csv")
for (m, s), col in zip(df.groupby("eta_over_etaN"), plt.cm.viridis(np.linspace(0, 0.92, df.eta_over_etaN.nunique()))):
    ax[0].semilogy(s.auc, s.U, color=col, label=(r"no blind spot" if m == 0 else fr"$\eta={m:g}\,\eta_N$"))
ax[0].set_xlabel("AUC of the system's signal"); ax[0].set_ylabel(r"bound $U_{0.05}$ on expected count $\mu$"); ax[0].legend(frameon=False, ncol=2, loc="lower right", columnspacing=0.8, handlelength=1.2); ax[0].set_title("(a) same AUC, different certificate", loc="left")
s3 = pd.read_csv(f"{RT}/exp03_selectors.csv"); s3 = s3[s3.surface == "all"]
pick = [("random", "random", OK["grey"], ":"), ("supervised probe (TF-IDF, cross-fitted)", "supervised probe", OK["green"], "-"), ("LLM selector: gpt-oss-120b", "gpt-oss-120b selector", OK["blue"], "-"),
        ("LLM selector: Qwen3-32B", "Qwen3-32B selector", OK["sky"], "-"), ("proxy monitor: gpt-oss-120b default full", "proxy: another open monitor", OK["red"], "-"), ("oracle", "oracle", OK["black"], "--")]
for name, lab, col, ls in pick:
    s = s3[(s3.selector == name) & (s3.beta >= 0.003)].sort_values("beta")
    if len(s): ax[1].semilogx(s.beta, s.mean_coverage_selected, ls, color=col, marker="o", ms=2.3, label=f"{lab} (AUC {s.auc.iloc[0]:.2f})")
ax[1].set_xlabel(r"selected fraction $\beta$ of 2372 attacks"); ax[1].set_ylabel("monitor recall on the selection"); ax[1].set_ylim(-0.03, 1.62); ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax[1].legend(frameon=False, loc="upper left", fontsize=5.4, ncol=1, borderaxespad=0.1, labelspacing=0.25); ax[1].set_title("(b) tail stress test on a real monitor", loc="left")
save(fig, "fig_main_tail.pdf")

# ------------------------------------------------------------------ Fig. 3: validity of the certificate; designs on measured recall
cov = pd.read_csv(f"{R}/exp04a_coverage.csv", usecols=["method", "alpha", "d", "miscov", "in_Q"]); cov = cov[cov.in_Q & (cov.alpha == 0.05)]
t = cov.groupby(["method", "d"]).miscov.max().unstack("method") / 0.05; t.to_csv(os.path.join(a.dest, "..", "numbers", "coverage_by_d.csv"))
fig, ax = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw=dict(wspace=0.36)); x = np.arange(len(t))
ax[0].bar(x - 0.2, t.chernoff, 0.4, color=OK["sky"], label="Chernoff constant"); ax[0].bar(x + 0.2, t.garwood, 0.4, color=OK["blue"], label="exact Poisson constant"); ax[0].axhline(1, color="k", lw=0.8, ls=":")
ax[0].set_xticks(x); ax[0].set_xticklabels(t.index); ax[0].set_xlabel("detections $d$"); ax[0].set_ylabel(r"max. exact miscoverage / $\alpha$"); ax[0].set_ylim(0, 1.25); ax[0].legend(frameon=False, loc="upper right"); ax[0].set_title("(a) 510,390 configurations", loc="left")
w, pi_rep, r, f, _ = load_instance("sleight"); pib = (w * pi_rep).sum(); Gs = np.logspace(0, 2, 70)
des = [("risk-targeted", design(r, w, 1.0, pib)[1], OK["red"]), ("report-shaped tiers", scaled_report(pi_rep, w, pib), OK["orange"]), ("floor + overlay", floor_overlay(r, w, pib, 0.5)[0], OK["purple"]),
       ("uniform floor", np.full(len(w), pib), OK["green"]), ("equal coverage, capped", equal_coverage(r, w, pib), OK["blue"])]
for lab, p, col in des:
    U = np.array([ucb0(p * r, w, G) for G in Gs]); U[~np.isfinite(U)] = 1e6; ax[1].loglog(Gs, U, color=col, label=lab)
ax[1].set_xlabel(r"strategic concentration $\Gamma$"); ax[1].set_ylabel(r"certified count $U_{0.05}(0;\Gamma)$"); ax[1].legend(frameon=False, loc="lower right", ncol=1, labelspacing=0.2); ax[1].set_ylim(1.2, 300); ax[1].set_title("(b) five designs, measured recall", loc="left")
save(fig, "fig_main_validity_design.pdf")

# ------------------------------------------------------------------ Appendix: recall of every monitor on every stratum; size ladder
M = pd.read_parquet(f"{RT}/M_long.parquet"); M = M[M.p_detect.notna() & M.is_attack]; base = sorted(c for c in M.config_id.unique() if "__monitor_default__full" in c)
pv = M[M.config_id.isin(base)].groupby(["dataset", "attack_type", "config_id"]).p_detect.mean().unstack("config_id")[base]; pv = pv.loc[[i for d in ("apps", "basharena", "sleight") for i in pv.index if i[0] == d]]
nm = lambda c: c.replace("__monitor_default__full", "").replace("__n2", "").replace("__think", " +R").replace("Qwen3-", "Q3-").replace("gpt-oss-120b", "oss-120b")
fig, axh = plt.subplots(figsize=(W, 0.17 * len(pv) + 0.9)); im = axh.imshow(pv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
axh.set_xticks(range(len(base))); axh.set_xticklabels([nm(c) for c in base], rotation=40, ha="right", fontsize=6); axh.set_yticks(range(len(pv))); axh.set_yticklabels([f"{DSN[d][:4]}: {short(t)}" for d, t in pv.index], fontsize=5.6)
for sp in axh.spines.values(): sp.set_visible(False)
fig.colorbar(im, fraction=0.035, pad=0.02, label="recall at 5% FPR"); save(fig, "fig_app_heatmap.pdf")
lad = pd.read_csv(f"{RT}/exp01_ladder.csv").pivot(index="config_id", columns="dataset", values="recall").loc[base]; lad.to_csv(os.path.join(a.dest, "..", "numbers", "ladder_fpr05.csv"))
fig, axl = plt.subplots(figsize=(W * 0.8, 2.2)); x = np.arange(len(lad))
for i, d in enumerate(("apps", "basharena", "sleight")): axl.bar(x + (i - 1) * 0.27, lad[d], 0.27, color=DS[d], label=DSN[d])
axl.set_xticks(x); axl.set_xticklabels([nm(c) for c in base], rotation=40, ha="right", fontsize=6); axl.set_ylabel("recall at 5% FPR"); axl.set_ylim(0, 1); axl.legend(frameon=False, ncol=3, loc="upper left"); save(fig, "fig_app_ladder.pdf")

# ------------------------------------------------------------------ Appendix: recall audit size; conservatism decomposition; operating points
d12 = pd.read_csv(f"{R}/exp12_recall_n.csv"); d12 = d12[d12.profile.str.startswith("11")]; fig, ax12 = plt.subplots(figsize=(W * 0.62, 2.0))
for (m, s), col in zip(d12.groupby("method", sort=False), (OK["orange"], OK["sky"], OK["blue"])): ax12.loglog(s.n, s.ratio_median, "o-", ms=2.5, color=col, label=f"{m} (valid in {s.recall_input_valid.min():.0%}+ of audits)")
ax12.axhline(1, color="k", lw=0.6); ax12.set_xlabel("red-team attacks per type"); ax12.set_ylabel("certificate / certificate with true recall"); ax12.legend(frameon=False, fontsize=5.6); save(fig, "fig_app_recall_n.pdf")
rows = []
for fpr in ("01", "02", "05", "10"):
    for cal in ("surface", "pooled"):
        p = f"{R}/fpr{fpr}_{cal}/exp01_ladder.csv"
        if os.path.exists(p):
            L_ = pd.read_csv(p); s = L_[L_.config_id == "Qwen3-32B__monitor_default__full__think__n2"].set_index("dataset"); rows.append(dict(fpr=int(fpr), calib=cal, **{f"r_{d}": s.recall.get(d, np.nan) for d in DS}, **{f"f_{d}": s.fpr.get(d, np.nan) for d in DS}))
pd.DataFrame(rows).to_csv(os.path.join(a.dest, "..", "numbers", "operating_points_main_monitor.csv"), index=False)
ci = pd.concat([pd.read_csv(f"{R}/fpr{f}_surface/exp01_threshold_ci.csv").assign(fpr=int(f)) for f in ("01", "02", "05", "10")]); ci.to_csv(os.path.join(a.dest, "..", "numbers", "threshold_ci.csv"), index=False)
g6 = pd.concat([pd.read_csv(f"{R}/fpr{f}_surface/exp06_game_splits.csv").mean().to_frame().T.assign(fpr=int(f), win=(lambda z: (z.optimized > z.best_single).mean())(pd.read_csv(f"{R}/fpr{f}_surface/exp06_game_splits.csv"))) for f in ("01", "02", "05", "10")])
g6.to_csv(os.path.join(a.dest, "..", "numbers", "game_by_fpr.csv"), index=False); print(g6.round(3).to_string()); print(pd.DataFrame(rows).round(3).to_string()); print(ci.round(3).to_string()); print(t.round(4).to_string())
for fn in ("exp05_designs_trueRL.csv", "exp05_designs_nosleight.csv", "exp07_audit_real.csv", "exp07_audit_sim.csv", "exp04b_baselines.csv", "exp10_hazard.csv"): pd.read_csv(f"{R}/{fn}").to_csv(os.path.join(a.dest, "..", "numbers", fn), index=False)
print("paper figures done")
