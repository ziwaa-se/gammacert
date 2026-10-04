"""Certificate library: zero-detection bounds under strategic allocation (lower-tail CVaR), design linear program.

Allocation class Q_Gamma = {q in simplex : q_h <= Gamma * w_h}: harmful actions are at most
Gamma times over-represented in any stratum relative to its traffic share.
kappa_h = -log(1 - c_h), c_h = pi_h * r_h.  kappa*(Gamma) = LCVaR_{1/Gamma}(kappa) under w.
Zero detections => U = log(1/alpha) / kappa*(Gamma) (design-based, realized count).
"""
import numpy as np
from scipy.optimize import linprog, brentq
from scipy.stats import norm, beta as betad

def lcvar(vals, w, level):
    """Lower CVaR at `level` of a discrete distribution (vals, w): mean of the lowest `level` mass."""
    vals, w = np.asarray(vals, float), np.asarray(w, float); w = w / w.sum()
    if level >= 1: return (w * vals).sum()
    o = np.argsort(vals); v, p = vals[o], w[o]
    rem, tot = level, 0.0
    for vi, pi in zip(v, p):
        take = min(pi, rem); tot += take * vi; rem -= take
        if rem <= 1e-15: break
    return tot / level

def kappa(c):
    c = np.asarray(c, float)
    with np.errstate(divide="ignore"):
        return -np.log1p(-np.minimum(c, 1.0))

def kstar(c, w, G):
    return lcvar(kappa(c), w, 1.0 / G) if np.isfinite(G) else kappa(c)[np.asarray(w) > 0].min()

def ucb0(c, w, G, alpha=0.05):
    k = kstar(c, w, G)
    return np.inf if k <= 0 else np.log(1 / alpha) / k

def kstar_lp(c, w, G):
    k = kappa(c); w = np.asarray(w, float) / np.sum(w); H = len(k)
    res = linprog(k, A_eq=np.ones((1, H)), b_eq=[1], bounds=[(0, min(1, G * wh)) for wh in w])
    return res.fun

def design(r, w, G, pibar):
    """max_pi min_{q<=G w} sum q_h pi_h r_h  s.t. sum w pi = pibar, 0<=pi<=1  (coverage-scale LP).
    Dual of inner: max t - G sum w_h b_h  s.t. t - b_h <= pi_h r_h, b>=0."""
    r, w = np.asarray(r, float), np.asarray(w, float); w = w / w.sum(); H = len(r)
    n = 1 + 2 * H                                   # t, b(H), pi(H)
    obj = np.zeros(n); obj[0] = -1; obj[1:1 + H] = G * w
    A = np.zeros((H, n)); A[:, 0] = 1; A[np.arange(H), 1 + np.arange(H)] = -1; A[np.arange(H), 1 + H + np.arange(H)] = -r
    Aeq = np.zeros((1, n)); Aeq[0, 1 + H:] = w
    res = linprog(obj, A_ub=A, b_ub=np.zeros(H), A_eq=Aeq, b_eq=[pibar],
                  bounds=[(None, None)] + [(0, None)] * H + [(0, 1)] * H)
    return -res.fun, res.x[1 + H:]

def cp_lower(x, n, a):
    return 0.0 if x == 0 else betad.ppf(a, x, n - x + 1)

# ---- information version: two strata, model sees a scalar signal ----
def info_ucb(N, wL, cH, cL, sfH, sfL, alpha=0.05):
    """Adversary acts on contexts with the largest signal (lowest posterior coverage, monotone LR).
    sfH(t), sfL(t): survival functions of the signal in H and L. Returns certified mu."""
    L = np.log(1 / alpha)
    g = lambda t: N * ((1 - wL) * cH * sfH(t) + wL * cL * sfL(t)) - L   # expected detections - L
    lo, hi = -50.0, 50.0
    if g(lo) < 0: return np.inf
    t = brentq(g, lo, hi) if g(hi) < 0 else hi
    return N * ((1 - wL) * sfH(t) + wL * sfL(t))
