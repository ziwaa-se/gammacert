"""Experiment 10: Theorem 4 optimises on the coverage scale; how much is lost against the exact hazard scale?

Exact problem: max_pi LCVaR_{1/Gamma}(-log(1 - pi_h r_h)) s.t. sum w pi = pibar, 0 <= pi <= 1  (non-concave).
Epigraph form with LCVaR_b(k) = max_t { t - (1/b) sum_h w_h (t - k_h)_+ }; multi-start SLSQP, warm-started at the LP solution.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gammacert.core import *

emit = Emitter("numbers_exp10.tex"); RES, FIG = out_dir("results"), out_dir("figures"); rng = np.random.default_rng(10)
w0, _, r0, _, _ = load_instance(); w0 = w0 / w0.sum(); L = np.log(20)


def hazard_design(r, w, G, pibar, n_start=24):
    H = len(r); _, p_lp = design(r, w, G, pibar); p_lp = np.clip(p_lp, 0, 1)
    val = lambda p: kstar(np.clip(p, 0, 1 - 1e-9) * r, w, G)
    kap = lambda p: -np.log1p(-np.clip(p * r, 0, 1 - 1e-9))
    obj = lambda x: -(x[H] - G * (w * x[H + 1:]).sum())                       # x = (pi, t, b)
    cons = [{"type": "eq", "fun": lambda x: (w * x[:H]).sum() - pibar},
            {"type": "ineq", "fun": lambda x: x[H + 1:] + kap(x[:H]) - x[H]}]
    best_p, best_v = p_lp, val(p_lp)
    for s in range(n_start):
        p = p_lp if s == 0 else np.clip(p_lp + rng.normal(0, 0.15, H), 0.001, 0.999)
        p = np.clip(p * pibar / (w * p).sum(), 0, 1); k = kap(p); t = np.quantile(k, min(1, 1 / G))
        x0 = np.r_[p, t, np.maximum(t - k, 0)]
        res = minimize(obj, x0, constraints=cons, bounds=[(0, 1)] * H + [(None, None)] + [(0, None)] * H, method="SLSQP", options=dict(maxiter=300, ftol=1e-12))
        if res.success and abs((w * res.x[:H]).sum() - pibar) < 1e-6 and val(res.x[:H]) > best_v: best_p, best_v = res.x[:H], val(res.x[:H])
    return best_v, best_p, val(p_lp), p_lp


rows = []
for pibar in [0.05, 0.1, 0.2, 0.435, 0.6, 0.8]:
    for G in [1, 1.25, 1.5, 2, 5]:
        for rname, r in [("measured/report", r0), ("high recall", np.minimum(0.99, r0 + 0.3)), ("low recall", r0 * 0.5)]:
            v_h, p_h, v_lp, p_lp = hazard_design(r, w0, G, pibar)
            rows.append(dict(pibar=pibar, G=G, recall=rname, kappa_lp=v_lp, kappa_hazard=v_h, U_lp=L / v_lp, U_hazard=L / v_h,
                             rel_gain=v_h / v_lp - 1, max_abs_dpi=np.abs(p_h - p_lp).max(), thm4_assumption=bool(pibar / (w0 / r).sum() <= r.min())))
df = pd.DataFrame(rows); df.to_csv(f"{RES}/exp10_hazard.csv", index=False); print(df.round(4).to_string())
ok = df[(df.G >= 2) & df.thm4_assumption]; bad = df[(df.G >= 2) & ~df.thm4_assumption]
emit("hazardMaxGainRobust", 100 * ok.rel_gain.max() if len(ok) else float("nan"), "{:.2f}")            # Gamma >= 2 AND pibar*Hbar <= r_min
emit("hazardMaxGainRobustAssumptionFails", 100 * bad.rel_gain.max() if len(bad) else 0.0, "{:.2f}"); emit("hazardShareAssumptionHolds", df.thm4_assumption.mean())      # Gamma >= Gamma-dagger: is equal coverage still optimal on the hazard scale?
emit("hazardMaxGainGammaOne", 100 * df[(df.G == 1) & df.thm4_assumption].rel_gain.max(), "{:.0f}")
emit("hazardMaxGain", 100 * df.rel_gain.max(), "{:.1f}"); emit("hazardMedianGain", 100 * df.rel_gain.median(), "{:.2f}")
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for (rn, s) in df[df.G == 1.5].groupby("recall"): ax.plot(s.pibar, 100 * s.rel_gain, "o-", label=rn)
ax.set_xlabel(r"monitored fraction $\bar\pi$"); ax.set_ylabel(r"gain in $\kappa^*$ over the LP design (%), $\Gamma=1.5$"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp10_hazard.pdf"); print("exp10 done")
