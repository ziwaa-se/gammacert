"""Illustrative numbers and figures for the certificate results: report-shaped tiers, false certification under
concentration, the two-stratum information example (AUC versus the tail), and recall confidence bounds."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
from gammacert.certlib import *
rng = np.random.default_rng(1); alpha = 0.05; L = np.log(20)
import os; os.makedirs("outputs/theory", exist_ok=True)
out = open("outputs/theory/numbers_certificate_examples.tex", "w")
def emit(name, val, fmt="{:.1f}"):
    out.write(f"\\newcommand{{\\{name}}}{{{fmt.format(val)}}}\n"); print(name, fmt.format(val))

# checks
e = max(abs(kstar(rng.uniform(0, 1, H) ** 3, rng.dirichlet(np.ones(H)), G) - kstar_lp(c, w, G))
        for H in [5] for _ in range(1) for c, w, G in [(rng.uniform(0, 1, 5) ** 3, rng.dirichlet(np.ones(5)), 7.0)])
errs = []
for _ in range(2000):
    H = rng.integers(2, 9); c = rng.uniform(0, 0.99, H) ** 3; w = rng.dirichlet(np.ones(H)); G = np.exp(rng.uniform(0, 7))
    errs.append(abs(kstar(c, w, G) - kstar_lp(c, w, G)))
emit("chkLP", max(errs), "{:.0e}")
# Monte Carlo: realized-count bound, feasible allocation in Q_Gamma, dependent (all-or-nothing) actions
c = np.array([0.8, 0.05, 0.002]); w = np.array([0.5, 0.3, 0.2]); G = 2.5
k = kstar(c, w, G); U = L / k; A = int(2 * np.ceil(U / 2)); Ah = np.array([0, A // 2, A // 2])
assert np.all(Ah / A <= G * w + 1e-12)
R = 400000; act = rng.random(R) < 0.5                     # adversary acts (A actions) w.p. 1/2, else none
D = sum(rng.binomial(Ah[h], c[h], size=R) for h in range(3)) * act
emit("mcU", U); emit("mcA", A, "{:d}")
emit("mcPzero", ((D == 0) & act).mean(), "{:.4f}"); emit("mcBound", 0.5 * np.exp(-(Ah * kappa(c)).sum()), "{:.4f}")
# report-modelled tiers (shares assumed)
pi = np.array([1.0, 0.25, 0.05, 0.002, 0.0002]); w = np.array([0.40, 0.10, 0.20, 0.21, 0.09])
r = np.array([0.94, 0.94, 0.58, 0.58, 0.44]); pibar = (w * pi).sum()
emit("pibar", pibar, "{:.3f}")
Hbar = 1 / (w / r).sum(); emit("Hbar", Hbar, "{:.3f}"); emit("Gdag", Hbar / r.min(), "{:.2f}")
Gs = np.logspace(0, 4, 60)
rep = [ucb0(pi * r, w, G) for G in Gs]
rob = []
for G in Gs:
    v, p = design(r, w, G, pibar); rob.append(L / (-np.log1p(-v)))   # coverage-scale design, kappa-scale report
# exact kappa-scale UCB for the robust design too
rob_k = []
for G in Gs:
    v, p = design(r, w, G, pibar); rob_k.append(ucb0(p * r, w, G))
for G in [1, 10, 100]:
    emit(f"rep{['One','Ten','Hun'][[1,10,100].index(G)]}", ucb0(pi * r, w, G))
    v, p = design(r, w, G, pibar); emit(f"rob{['One','Ten','Hun'][[1,10,100].index(G)]}", ucb0(p * r, w, G))
for G in [1.5, 2, 5]:
    tag = {1.5: "OnePointFive", 2: "Two", 5: "Five"}[G]
    emit(f"rep{tag}", ucb0(pi * r, w, G)); v, p = design(r, w, G, pibar); emit(f"rob{tag}", ucb0(p * r, w, G))
# design theorem check: LP value = pibar*Hbar iff Gamma >= Gdag
for G in [1.0, 1.3, Hbar / r.min(), 2.0, 10.0, 1e3]:
    v, p = design(r, w, G, pibar); print(f"  design check Gamma={G:.3f}: LP value {v:.5f}  pibar*Hbar {pibar*Hbar:.5f}")
emit("robCoverage", pibar * Hbar, "{:.3f}")
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.loglog(Gs, rep, lw=2, label="report-style tiers")
ax.loglog(Gs, rob_k, lw=2, ls="--", label=r"$\Gamma$-robust design, same budget")
ax.axvline(Hbar / r.min(), color="grey", lw=0.8, ls=":"); ax.text(Hbar / r.min() * 1.1, 3e3, r"$\Gamma^\dagger$", color="grey")
ax.set_xlabel(r"$\Gamma$ (max over-representation of harmful actions in a stratum)")
ax.set_ylabel("95% bound on # harmful actions\n(zero detections)"); ax.legend(frameon=False); fig.tight_layout()
fig.savefig("outputs/theory/fig_ucb_gamma.pdf")

# false certification: P(D=0 | A = U_1) under true Gamma, over random traffic shares
Gt = np.logspace(0, 1, 40); curves = []
for _ in range(500):
    ww = rng.dirichlet(np.ones(5))
    k1 = kstar(pi * r, ww, 1.0)
    curves.append([alpha ** (kstar(pi * r, ww, g) / k1) for g in Gt])
curves = np.array(curves)
w0 = w.copy(); k1 = kstar(pi * r, w0, 1.0)
emit("fcOnePointFive", alpha ** (kstar(pi * r, w0, 1.5) / k1), "{:.2f}"); emit("fcTwo", alpha ** (kstar(pi * r, w0, 2) / k1), "{:.2f}"); emit("fcFive", alpha ** (kstar(pi * r, w0, 5) / k1), "{:.2f}")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.fill_between(Gt, np.percentile(curves, 10, 0), np.percentile(curves, 90, 0), alpha=0.25, label="10–90% over random traffic shares")
ax.semilogx(Gt, np.median(curves, 0), lw=2, label="median")
ax.axhline(alpha, color="k", lw=0.8, ls=":"); ax.text(1.1, alpha + 0.02, "nominal 0.05")
ax.set_xlabel(r"true $\Gamma$"); ax.set_ylabel(r"$P(D=0 \mid A = U_{\Gamma=1})$"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("outputs/theory/fig_false_cert.pdf")

# information version: AUC is the wrong metric
N, wL, cH, cL = 1e8, 0.1, 0.25 * 0.8, 0.0002 * 0.8
emit("infoNaive", L / ((1 - wL) * cH + wL * cL)); emit("infoWorst", L / cL, "{:.0f}")
aucs = np.linspace(0.5, 0.97, 48); ub_norm, ub_bdd = [], []
for a in aucs:
    dp = np.sqrt(2) * norm.ppf(a)
    ub_norm.append(info_ucb(N, wL, cH, cL, lambda t: norm.sf(t), lambda t: norm.sf(t - dp)))
    # bounded-LR signal on [0,1]: f_H = 1 - e(2s-1), f_L = 1 + e(2s-1); AUC = 1/2 + e/3 (capped)
    e = min(3 * (a - 0.5), 0.999)
    sfH = lambda t, e=e: np.clip(1 - (np.clip(t, 0, 1) - e * (np.clip(t, 0, 1) ** 2 - np.clip(t, 0, 1))), 0, 1)
    sfL = lambda t, e=e: np.clip(1 - (np.clip(t, 0, 1) + e * (np.clip(t, 0, 1) ** 2 - np.clip(t, 0, 1))), 0, 1)
    ub_bdd.append(info_ucb(N, wL, cH, cL, sfH, sfL) if a <= 0.5 + 0.999 / 3 else np.nan)
for a in [0.8]:
    i = np.argmin(abs(aucs - a)); emit(f"infoNormEighty", ub_norm[i], "{:.0f}"); emit("infoBddEighty", ub_bdd[i], "{:.0f}")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.semilogy(aucs, ub_norm, lw=2, label="Gaussian signal (unbounded LR)")
ax.semilogy(aucs, ub_bdd, lw=2, ls="--", label=r"bounded-LR signal, LR $\leq (1+\eta)/(1-\eta)$")
ax.axhline(L / cL, color="grey", lw=0.8, ls=":"); ax.text(0.505, L / cL * 0.5, "worst case: visible strata, or any AUC with 0.19% of\nlow-coverage contexts revealed (Theorem 3)", color="grey", fontsize=7, va="top"); ax.set_ylim(10, 4e4)
ax.set_xlabel("AUC of the model's discrimination between strata"); ax.set_ylabel("95% bound on E[# harmful actions]")
ax.legend(frameon=False, loc="center left"); fig.tight_layout(); fig.savefig("outputs/theory/fig_info_auc.pdf")

# recall uncertainty: METR 153/163 (94%), internal 4/5
emit("cpMETR", cp_lower(153, 163, 0.025 / 1), "{:.3f}"); emit("cpInternal", cp_lower(4, 5, 0.025), "{:.3f}")
i9 = np.argmin(abs(aucs - 0.9)); emit("infoNormNinety", ub_norm[i9], "{:.0f}")
out.close()

# ---------- Theorem (no bulk discrimination summary bounds the certificate) ----------
# Two strata; base binormal structure with AUC A0; add an atom of mass eta on L-contexts that reveals L.
out = open("outputs/theory/numbers_certificate_examples.tex", "a")
L = np.log(20); N, wL, cH, cL = 1e8, 0.1, 0.25 * 0.8, 0.0002 * 0.8
etaN = L / (cL * wL * N); m_, e_ = f"{etaN:.2e}".split("e"); out.write(f"\\newcommand{{\\etaN}}{{{m_}\\times10^{{{int(e_)}}}}}\n"); print("etaN", etaN)
Uvis = L / cL
def cert_with_atom(A_target, eta):
    A0 = (A_target - eta) / (1 - eta); dp = np.sqrt(2) * norm.ppf(A0)
    # adversary: first the atom (mass wL*eta, coverage cL), then top binormal signals
    atom_mass, atom_det = wL * eta, wL * eta * cL
    def det(t):   # expected detections per context when acting on atom + {S > t}
        return atom_det + (1 - wL) * cH * norm.sf(t) + wL * (1 - eta) * cL * norm.sf(t - dp)
    def mass(t):
        return atom_mass + (1 - wL) * norm.sf(t) + wL * (1 - eta) * norm.sf(t - dp)
    if N * atom_det >= L:                      # certificate reached inside the atom
        return L / cL, A0
    t = brentq(lambda t: N * det(t) - L, -50, 50)
    return N * mass(t), A0
for A in [0.55, 0.8, 0.95]:
    U_base = info_ucb(N, wL, cH, cL, lambda t: norm.sf(t), lambda t, dp=np.sqrt(2) * norm.ppf(A): norm.sf(t - dp))
    U_pert, A0 = cert_with_atom(A, etaN)
    print(f"  AUC {A}: base binormal cert {U_base:.0f}; perturbed (eta_N) cert {U_pert:.0f}; Uvis {Uvis:.0f}; base AUC used {A0:.5f}")
    tag = {0.55: "FiftyFive", 0.8: "Eighty", 0.95: "NinetyFive"}[A]
    emit(f"aucBase{tag}", U_base, "{:.0f}"); emit(f"aucPert{tag}", U_pert, "{:.0f}")
emit("Uvis", Uvis, "{:.0f}")
out.close()
