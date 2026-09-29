"""Experiment 4: certificate validity, tightness, and where the conservatism comes from.

Miscoverage P(A > U(D)) is computed EXACTLY (D is a sum of independent binomials given the harmful
configuration), so there is no Monte Carlo error. Any dependent / randomised adversary is a mixture over
fixed configurations, hence its miscoverage is a mixture of the numbers below; a Monte Carlo cross-check
with an all-or-nothing adversary is included.

  4a  coverage grid: paper bound (Chernoff) over instances x Gamma x allocations x A
  4e  same grid with the Garwood (exact Poisson) limit  -> is max miscoverage still <= alpha ?
  4b  validity of naive baselines under strategic allocation
  4c  decomposition of log(U / A) into Chernoff / strategic / recall-uncertainty factors
  4d  Theorem 3 attainability: the adversary that makes 18,723 real
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gammacert.core import *

rng = np.random.default_rng(4); emit = Emitter("numbers_exp04.tex"); RES, FIG = out_dir("results"), out_dir("figures")
GAMMAS = [1, 1.5, 2, 5, 10, np.inf]; ALPHAS = [0.01, 0.05, 0.1]; DS = range(0, 9)


def instances(n_random=200):
    yield "report", W_REP, PI_REP * R_REP
    for i in range(n_random):
        H = int(rng.choice([2, 3, 5, 10])); w = rng.dirichlet(np.ones(H))
        c = np.clip(rng.uniform(0, 1, H) ** rng.choice([1, 3]) * rng.uniform(0.05, 1), 1e-4, 0.98)
        yield f"rand{i}", w, c


def allocations(c, w, G, n_rand=6):
    qw = worst_q(kappa(c), w, G); yield "worst", qw
    yield "proportional", w / w.sum()
    for lam in (0.25, 0.5, 0.75): yield f"mix{lam}", lam * qw + (1 - lam) * w / w.sum()
    for k in range(n_rand): yield f"random{k}", random_q(w, G, rng)


# ------------------------------------------------------------------ 4a + 4e
rows = []
for name, w, c in instances():
    for G in GAMMAS:
        for alpha in ALPHAS:
            for method in ("chernoff", "garwood"):
                Ud = [U_cert(d, c, w, G, alpha, method) for d in range(81)]
                Uf = lambda d, Ud=Ud: Ud[d]
                for d in DS:                                   # the binding cases: A just above U(d)
                    if not np.isfinite(Ud[d]) or Ud[d] > 5e4: continue
                    A = int(np.floor(Ud[d])) + 1
                    for qn, q in allocations(c, w, G):
                        Ah = int_worst(A, kappa(c), w, G) if qn == "worst" else int_alloc(A, q)
                        ok = Ah is not None and feasible(Ah, w, G)           # the theorem is about configurations IN Q_Gamma; rounding can leave it
                        if Ah is None: Ah = int_alloc(A, q)
                        rows.append(dict(inst=name, H=len(w), G=G, alpha=alpha, method=method, d=d, A=A, alloc=qn, in_Q=ok,
                                         miscov=miscoverage(Ah, c, Uf)))
full = pd.DataFrame(rows); full.to_csv(f"{RES}/exp04a_coverage.csv", index=False); df = full[full.in_Q]
out = full[~full.in_Q]; print(f"[4a] {len(out)} of {len(full)} rounded configurations fall outside Q_Gamma (small A); among them max miscoverage/alpha = {(out.miscov / out.alpha).max():.2f}")
emit("covOutsideQShare", len(out) / len(full)); emit("covOutsideQMaxRatio", (out.miscov / out.alpha).max())
summ = df.groupby(["method", "alpha"]).miscov.agg(["max", "mean", lambda x: (x > 0).mean()]).rename(columns={"<lambda_0>": "frac_pos"})
summ["ratio_max_to_alpha"] = summ["max"] / summ.index.get_level_values("alpha"); print(summ); summ.to_csv(f"{RES}/exp04a_summary.csv")
for method in ("chernoff", "garwood"):
    for alpha, tag in zip(ALPHAS, ("One", "Five", "Ten")):
        emit(f"cov{method.capitalize()}Max{tag}", summ.loc[(method, alpha), "max"], "{:.4f}")
# rounding A q to integers can push a configuration marginally outside Q_Gamma; report how often that matters
viol = df[(df.miscov > df.alpha * 1.0001)]; print(f"[4a/4e] configurations with miscoverage > alpha: {len(viol)} of {len(df)}")
viol.to_csv(f"{RES}/exp04a_violations.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, method in zip(axes, ("chernoff", "garwood")):
    s = df[(df.method == method) & (df.alpha == 0.05) & (df.alloc == "worst")]
    piv = s.groupby(["G", "d"]).miscov.max().unstack("d")
    im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=0.05, cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([r"$\infty$" if not np.isfinite(g) else g for g in piv.index])
    ax.set_xlabel("detections $d$ (with $A=\\lfloor U_\\alpha(d)\\rfloor+1$)"); ax.set_title({"chernoff": "Lemma 1(c): Chernoff", "garwood": "exact Poisson limit"}[method], fontsize=9)
axes[0].set_ylabel(r"$\Gamma$"); fig.colorbar(im, ax=axes, label="max exact miscoverage (nominal 0.05)")
fig.savefig(f"{FIG}/fig_exp04_coverage.pdf", bbox_inches="tight")

# Monte Carlo cross-check of the exact computation (all-or-nothing adversary)
c = np.array([0.8, 0.05, 0.002]); w = np.array([0.5, 0.3, 0.2]); G = 2.5
A = int(np.floor(ucb0(c, w, G))) + 1; Ah = int_alloc(A, worst_q(kappa(c), w, G)); R = 400000
D = sum(rng.binomial(Ah[h], c[h], size=R) for h in range(3)); act = rng.random(R) < 0.5
emit("mcCheckSim", ((D == 0) & act).mean(), "{:.4f}"); emit("mcCheckExact", 0.5 * pmf_D(Ah, c, 5)[0], "{:.4f}")

# ------------------------------------------------------------------ 4b baselines under strategic allocation
w, c = W_REP, PI_REP * R_REP; L = np.log(20); rows = []
for Gt in [1, 1.5, 2, 5, 10]:
    bounds = {"paper, correct Gamma": ucb0(c, w, Gt), "paper, Gamma=infinity": ucb0(c, w, np.inf),
              "misspecified Gamma=1": ucb0(c, w, 1.0), "naive average coverage": L / (w * c).sum(),
              "naive average sampling rate (recall ignored)": L / (w * PI_REP).sum()}
    for bname, U in bounds.items():
        if not np.isfinite(U): continue
        A = int(np.floor(U)) + 1; Ah = int_worst(A, kappa(c), w, Gt)       # None: no integer configuration with this A lies in Q_Gamma
        rows.append(dict(true_Gamma=Gt, bound=bname, U=U, P_zero_detections_at_A=np.nan if Ah is None else pmf_D(Ah, c, 1)[0]))
pd.DataFrame(rows).to_csv(f"{RES}/exp04b_baselines.csv", index=False); print(pd.DataFrame(rows).pivot(index="bound", columns="true_Gamma", values="P_zero_detections_at_A").round(3))

# ------------------------------------------------------------------ 4c decomposition of the conservatism
rows = []
for Gt, Ga in [(1, 1), (1, 2), (1.5, 2), (2, 2), (2, 10)]:          # (true concentration, Gamma the analyst allows)
    for n in [20, 163, 2000]:                                       # red-team sample size per stratum
        rl = np.array([np.mean([cp_lower(x, n, 0.025 / 5) for x in rng.binomial(n, r, 200)]) for r in R_REP])
        for d in [0, 1, 3, 10]:
            qtrue = worst_q(kappa(c), w, Gt)
            lam_g, lam_c = (lam_garwood(d), lam_chernoff(d))
            U_oracle = lam_g / (qtrue * (kappa(c) if d == 0 else c)).sum()            # knows allocation and recall, exact tail
            U_strat = lam_g / (kstar(c, w, Ga) if d == 0 else rstar(c, w, Ga))        # + allocation unknown within Q_Gamma
            U_recall = lam_g / (kstar(PI_REP * rl, w, Ga) if d == 0 else rstar(PI_REP * rl, w, Ga))   # + recall only lower-bounded
            U_paper = U_recall * lam_c / lam_g                                         # + Chernoff relaxation (d >= 1 only)
            rows.append(dict(true_Gamma=Gt, analyst_Gamma=Ga, n_redteam=n, d=d, U_oracle=U_oracle, f_strategic=U_strat / U_oracle,
                             f_recall=U_recall / U_strat, f_chernoff=lam_c / lam_g, U_paper=U_paper))
dec = pd.DataFrame(rows); dec.to_csv(f"{RES}/exp04c_decomposition.csv", index=False)
s = dec[(dec.n_redteam == 163)]; lab = [f"$\\Gamma$={a}/{b}, d={d}" for a, b, d in zip(s.true_Gamma, s.analyst_Gamma, s.d)]
fig, ax = plt.subplots(figsize=(9, 3.2)); bot = np.zeros(len(s))
for col, name in [("f_strategic", "strategic uncertainty ($Q_\\Gamma$ vs true allocation)"), ("f_recall", "recall lower bound (n=163 per stratum)"), ("f_chernoff", "Chernoff vs exact Poisson")]:
    v = np.log10(s[col].values); ax.bar(range(len(s)), v, bottom=bot, label=name); bot += v
ax.set_xticks(range(len(s))); ax.set_xticklabels(lab, rotation=70, fontsize=6); ax.set_ylabel(r"$\log_{10}(U_{\rm paper}/U_{\rm oracle})$")
ax.legend(frameon=False, fontsize=7); fig.savefig(f"{FIG}/fig_exp04_decomposition.pdf", bbox_inches="tight")
for d, tag in [(1, "One"), (3, "Three"), (10, "Ten")]: emit("chernoffFactor" + tag, lam_chernoff(d) / lam_garwood(d))

# ------------------------------------------------------------------ 4d Theorem 3 is attained, not loose
N, wL, cH, cL = 1e8, 0.1, 0.2, 1.6e-4; etaN = L / (cL * wL * N)
base, pert = TwoStrata(wL, cH, cL, 0.55), TwoStrata(wL, cH, cL, 0.55, etaN)
U_base, U_vis = base.cert(N), pert.cert(N); n_star = int(round(N * wL * etaN))      # number of star-contexts
A = int(np.floor(U_vis)); sims = rng.binomial(A, cL, size=2_000_000)                 # adversary acts on star-contexts only
emit("attainUbase", U_base, "{:.0f}"); emit("attainUvis", U_vis, "{:.0f}"); emit("attainStarContexts", n_star, "{:d}")
emit("attainPzeroExact", (1 - cL) ** A, "{:.4f}"); emit("attainPzeroSim", (sims == 0).mean(), "{:.4f}")
# coverage of the AUC-based certificate (U_base at D=0, Chernoff on the average coverage otherwise) against that adversary
cbar = (1 - wL) * cH + wL * cL; Unaive = lambda d: U_base if d == 0 else lam_chernoff(d) / cbar
emit("attainNaiveCoverage", 1 - miscoverage([A], [cL], Unaive, dmax=60), "{:.4f}")
emit("attainAucGap", abs(pert.auc_total - base.auc_total), "{:.1e}"); emit("attainAccGap", abs(pert.bayes_acc() - base.bayes_acc()), "{:.1e}")
print("exp04 done")

# ------------------------------------------------------------------ 4f the two classical inequalities behind the Garwood version of Lemma 1(c)
# For a Poisson-binomial S (n trials, mean lam) and d <= lam - 1:   P(S<=d) <= P(Bin(n, lam/n)<=d) <= P(Poisson(lam)<=d).
from scipy.stats import poisson
worst_h, worst_as, n_checked, rev = -np.inf, -np.inf, 0, 0
for _ in range(20000):
    n = int(rng.integers(2, 250)); p = rng.beta(*rng.choice([0.3, 1.0, 3.0], 2), size=n) * rng.uniform(0.02, 1.0); lam = p.sum()
    pm = np.zeros(n + 1); pm[0] = 1.0
    for pi_ in p: pm[1:] = pm[1:] * (1 - pi_) + pm[:-1] * pi_; pm[0] *= (1 - pi_)
    cdf = np.cumsum(pm); ds = np.arange(0, n + 1); cb, cp = binom.cdf(ds, n, lam / n), poisson.cdf(ds, lam)
    ok = ds <= lam - 1
    if ok.any(): worst_h = max(worst_h, (cdf[ok] - cb[ok]).max()); worst_as = max(worst_as, (cb[ok] - cp[ok]).max()); n_checked += int(ok.sum())
    rev += int(((cdf - cp)[ds >= np.ceil(lam)] > 1e-12).any())                    # outside the range the ordering flips: the condition matters
print(f"[4f] {n_checked} (instance, d) pairs with d <= lam-1: max[P(S<=d)-P(Bin<=d)] = {worst_h:.2e}, max[P(Bin<=d)-P(Poi<=d)] = {worst_as:.2e}; ordering reversed for some d >= lam in {rev} of 20000 instances")
emit("ineqPairsChecked", n_checked, "{:d}"); emit("ineqHoeffdingWorst", worst_h, "{:.1e}"); emit("ineqAndersonSamuelsWorst", worst_as, "{:.1e}")
for a_ in (0.01, 0.05, 0.1, 0.3): assert all(lam_garwood(d, a_) >= d + 1 for d in range(0, 2000)), a_      # lambda_alpha(d) >= d+1 whenever alpha <= e^{-1}
print("exp04f done")
