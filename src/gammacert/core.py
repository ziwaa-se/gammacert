"""Numerical core used by the experiments: exact Poisson and Chernoff constants, exact miscoverage, worst integer
allocations, monitoring designs, randomized monitor assignment and the two-stratum information model.

Notation follows the paper: c_h = pi_h r_h, kappa_h = -log(1-c_h), Q_Gamma = {q <= Gamma w},
kappa*(Gamma) = LCVaR_{1/Gamma}(kappa), rho*(Gamma) = LCVaR_{1/Gamma}(c).
"""
import json, os
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.stats import binom, chi2, norm
from .certlib import lcvar, kappa, kstar, ucb0, design, cp_lower  # noqa: F401  (re-exported)

ALPHA = 0.05
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz
# Illustrative report-shaped tiers of Section 7 of the paper (shares and recall mapping are assumed there).
PI_REP = np.array([1.0, 0.25, 0.05, 0.002, 0.0002])
W_REP = np.array([0.40, 0.10, 0.20, 0.21, 0.09])
R_REP = np.array([0.94, 0.94, 0.58, 0.58, 0.44])


# ---------------------------------------------------------------- output helpers
def out_dir(sub=""):
    """outputs/ is an optional symlink to large-file storage; otherwise fall back to ./results_local."""
    root = os.environ.get("ZD_OUT") or ("outputs" if os.path.isdir("outputs") else "results_local")
    p = os.path.join(root, sub); os.makedirs(p, exist_ok=True); return p


def transcripts_path():
    """The canonical transcript table: <out>/data/transcripts.parquet if it exists, else the copy shipped in data_canonical/."""
    p = os.path.join(out_dir("data"), "transcripts.parquet")
    return p if os.path.exists(p) else os.path.join("data_canonical", "transcripts.parquet")


class Emitter:
    """Same convention as theory_examples/: every number used in the paper becomes a LaTeX macro."""
    def __init__(self, fname):
        self.f = open(os.path.join(out_dir("numbers"), fname), "w")
    def __call__(self, name, val, fmt="{:.2f}"):
        txt = fmt.format(val)
        if "e" in txt and fmt != "{:d}":
            m_, e_ = txt.split("e"); txt = f"{m_}\\times10^{{{int(e_)}}}"
        self.f.write(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{txt}}}}}\n"); self.f.flush(); print(name, txt)


def load_instance(exclude=None):
    """(w, pi, r, f, measured): measured recall / FPR from the monitor experiments if <out>/results/measured_rf.json exists, else the paper's illustrative tiers.
    `exclude`: drop strata whose name starts with this dataset prefix (e.g. "sleight") and renormalise the shares."""
    p = os.path.join(out_dir("results"), "measured_rf.json")
    if os.path.exists(p):
        d = json.load(open(p)); keep = np.array([not (exclude and n.startswith(exclude)) for n in d["strata"]])
        w, pi, r, f = (np.array(d[k], float)[keep] for k in ("w", "pi", "r", "f")); return w / w.sum(), pi, r, f, True
    return W_REP, PI_REP, R_REP, np.full(5, 0.01), False


# ---------------------------------------------------------------- certificates for d >= 0
@lru_cache(maxsize=None)
def lam_chernoff(d, alpha=ALPHA):
    """Lemma 1(c): lambda > d solving exp(-lam) (e lam / d)^d = alpha."""
    if d == 0: return np.log(1 / alpha)
    g = lambda lam: -lam + d * (1 + np.log(lam / d)) - np.log(alpha)
    return brentq(g, d, d + 50 * (d + 10))


@lru_cache(maxsize=None)
def lam_garwood(d, alpha=ALPHA):
    """Exact Poisson upper limit: P(Poisson(lam) <= d) = alpha.  Candidate replacement for lam_chernoff."""
    return 0.5 * chi2.ppf(1 - alpha, 2 * (d + 1))


def rstar(c, w, G):
    c = np.asarray(c, float)
    return lcvar(c, w, 1.0 / G) if np.isfinite(G) else c[np.asarray(w) > 0].min()


def U_cert(d, c, w, G, alpha=ALPHA, method="chernoff"):
    """U_alpha(d; Gamma). d = 0 uses the exact hazard scale; d >= 1 uses lambda_alpha(d) / rho*."""
    if d == 0: return ucb0(c, w, G, alpha)
    lam = lam_chernoff(d, alpha) if method == "chernoff" else lam_garwood(d, alpha)
    rho = rstar(c, w, G); return np.inf if rho <= 0 else lam / rho


# ---------------------------------------------------------------- allocations
def worst_q(v, w, G):
    """Minimiser of sum q_h v_h over Q_Gamma (fractional knapsack: fill strata in increasing v)."""
    v, w = np.asarray(v, float), np.asarray(w, float); w = w / w.sum()
    if not np.isfinite(G):
        q = np.zeros_like(w); q[np.where(w > 0, v, np.inf).argmin()] = 1.0; return q
    q, rem = np.zeros_like(w), 1.0
    for h in np.argsort(v):
        q[h] = min(G * w[h], rem); rem -= q[h]
        if rem <= 1e-15: break
    return q


def random_q(w, G, rng):
    """A random point of Q_Gamma: random priority order, random fill levels."""
    w = np.asarray(w, float) / np.sum(w); q, rem = np.zeros_like(w), 1.0
    order = rng.permutation(len(w))
    for i, h in enumerate(order):
        cap = min(G * w[h], rem) if np.isfinite(G) else rem
        q[h] = cap if i == len(order) - 1 else cap * rng.uniform(0.3, 1.0); rem -= q[h]
    if rem > 1e-12:                                   # top up within caps
        for h in order:
            add = min((G * w[h] if np.isfinite(G) else 1.0) - q[h], rem); q[h] += add; rem -= add
    return q / q.sum()


def int_alloc(A, q):
    """Largest-remainder rounding of A q to integers (the paper's 'up to rounding')."""
    x = A * np.asarray(q, float); a = np.floor(x).astype(int)
    for h in np.argsort(-(x - a))[: int(A - a.sum())]: a[h] += 1
    return a


def int_worst(A, v, w, G):
    """Worst INTEGER configuration inside Q_Gamma: fill strata in increasing v under caps floor(Gamma w_h A).
    Returns None when no integer configuration with this A lies in Q_Gamma (possible for small A)."""
    w = np.asarray(w, float) / np.sum(w); cap = np.full(len(w), A) if not np.isfinite(G) else np.floor(G * w * A + 1e-9).astype(int)
    if cap.sum() < A: return None
    a, rem = np.zeros(len(w), int), A
    for h in np.argsort(v):
        a[h] = min(cap[h], rem); rem -= a[h]
    return a


def feasible(Ah, w, G):
    w = np.asarray(w, float) / np.sum(w); return bool(np.all(Ah <= G * w * np.sum(Ah) + 1e-9)) if np.isfinite(G) else True


# ---------------------------------------------------------------- exact law of D given the harmful configuration
def pmf_D(Ah, c, dmax):
    """P(D = 0..dmax) for D = sum_h Binomial(A_h, c_h): exact convolution, truncated at dmax."""
    p = np.zeros(dmax + 1); p[0] = 1.0
    for a, ch in zip(Ah, c):
        if a == 0 or ch <= 0: continue
        p = np.convolve(p, binom.pmf(np.arange(dmax + 1), a, min(ch, 1.0)))[: dmax + 1]
    return p


def miscoverage(Ah, c, U_of_d, dmax=80):
    """Exact P(A > U(D)) for a fixed configuration: sum of P(D=d) over d with U(d) < A."""
    A = int(np.sum(Ah)); p = pmf_D(Ah, c, dmax)
    bad = np.array([U_of_d(d) < A for d in range(dmax + 1)])
    return float((p * bad).sum())


# ---------------------------------------------------------------- designs at a fixed monitored fraction
def scaled_report(pi_shape, w, pibar):
    f = lambda s: (w * np.minimum(1, s * pi_shape)).sum() - pibar
    return np.minimum(1, brentq(f, 1e-12, 1e9) * pi_shape)


def equal_coverage(r, w, pibar):
    Hb = 1 / (w / r).sum(); return np.minimum(1, pibar * Hb / r)


def floor_overlay(r, w, pibar, share=0.5):
    """Floor on the effective scale with `share` of the budget; the rest goes where recall is highest (Gamma=1 objective).
    Returns total inclusion probability 1-(1-floor)(1-risk)."""
    Hb = 1 / (w / r).sum(); c0 = share * pibar * Hb; fl = np.minimum(1, c0 / r)
    _, risk = design(r, w, 1.0, pibar - (w * fl).sum())
    return 1 - (1 - fl) * (1 - np.clip(risk, 0, 1)), c0


def design_fp(r, w, G, pibar, f, B):
    """Theorem 4 LP with an extra false-positive / human-review budget: sum_h w_h pi_h f_h <= B."""
    r, w, f = (np.asarray(x, float) for x in (r, w, f)); w = w / w.sum(); H = len(r); n = 1 + 2 * H
    obj = np.zeros(n); obj[0] = -1; obj[1:1 + H] = G * w
    A = np.zeros((H + 1, n)); A[:H, 0] = 1; A[np.arange(H), 1 + np.arange(H)] = -1; A[np.arange(H), 1 + H + np.arange(H)] = -r
    A[H, 1 + H:] = w * f; b = np.zeros(H + 1); b[H] = B
    Aeq = np.zeros((1, n)); Aeq[0, 1 + H:] = w
    res = linprog(obj, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[pibar], bounds=[(None, None)] + [(0, None)] * H + [(0, 1)] * H)
    return (-res.fun, res.x[1 + H:]) if res.success else (np.nan, np.full(H, np.nan))


def game_value(R):
    """Proposition 3: max_p min_t sum_j p_j R[t, j].  Returns (value, p)."""
    T, J = R.shape
    res = linprog(np.r_[-1, np.zeros(J)], A_ub=np.c_[np.ones(T), -R], b_ub=np.zeros(T),
                  A_eq=np.r_[0, np.ones(J)][None, :], b_eq=[1], bounds=[(None, None)] + [(0, 1)] * J)
    return -res.fun, res.x[1:]


# ---------------------------------------------------------------- partial visibility (Theorems 2-3)
class TwoStrata:
    """Strata L, H; binormal signal with separation dp plus an atom of mass eta on L that is never emitted under H."""
    def __init__(self, wL, cH, cL, auc_total, eta=0.0):
        self.wL, self.cH, self.cL, self.eta = wL, cH, cL, eta
        self.auc_total = auc_total; self.auc_base = (auc_total - eta) / (1 - eta)
        self.dp = np.sqrt(2) * norm.ppf(self.auc_base)
    def mass(self, t):   # traffic mass of {atom} U {signal > t}
        return self.wL * self.eta + (1 - self.wL) * norm.sf(t) + self.wL * (1 - self.eta) * norm.sf(t - self.dp)
    def det(self, t):    # expected detection probability on that set
        return self.wL * self.eta * self.cL + (1 - self.wL) * self.cH * norm.sf(t) + self.wL * (1 - self.eta) * self.cL * norm.sf(t - self.dp)
    def lcvar(self, beta):
        a = self.wL * self.eta
        if beta <= a: return self.cL
        t = brentq(lambda t: self.mass(t) - beta, -60, 60); return self.det(t) / beta
    def cert(self, N, alpha=ALPHA):
        L = np.log(1 / alpha)
        if N * self.wL * self.eta * self.cL >= L: return L / self.cL
        if N * self.det(-60) < L: return np.inf
        t = brentq(lambda t: N * self.det(t) - L, -60, 60); return N * self.mass(t)
    def bayes_acc(self):
        s = np.linspace(-12, 12 + self.dp, 40001); w = self.wL
        return w * self.eta + _trapz(np.maximum(w * (1 - self.eta) * norm.pdf(s - self.dp), (1 - w) * norm.pdf(s)), s)
    def brier(self):
        """Brier score of the Bayes posterior P(L | signal): E[p(1-p)] off the atom, 0 on it."""
        s = np.linspace(-12, 12 + self.dp, 40001); w = self.wL
        fL, fH = w * (1 - self.eta) * norm.pdf(s - self.dp), (1 - w) * norm.pdf(s); m = fL + fH
        p = np.divide(fL, m, out=np.zeros_like(m), where=m > 0); return _trapz(p * (1 - p) * m, s)
