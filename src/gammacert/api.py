"""Public API of gammacert.

A harm-capable unit in monitoring stratum h is routed to a monitor with probability pi[h] and, once routed, flagged with
probability recall[h]. Its detection probability ("coverage") is c[h] = pi[h] * recall[h]. A strategic system can
over-represent its harmful units in weakly covered strata by at most a factor gamma relative to the traffic shares.

Every function takes plain numbers or array-likes, one entry per stratum.
"""
import numpy as np

from .certlib import cp_lower, design, kappa, lcvar
from .core import equal_coverage as _equal_coverage
from .core import game_value, lam_chernoff, lam_garwood, rstar, ucb0

__all__ = ["coverage", "certificate", "certificate_curve", "lower_tail", "hidden_floor", "floor_certificate",
           "single_action_certified", "campaign_escape", "minimax_design", "equal_coverage_design",
           "recall_lower_bounds", "randomized_monitor", "selection_coverage"]


def _arrays(pi, recall, shares=None):
    pi, recall = np.broadcast_arrays(np.asarray(pi, float), np.asarray(recall, float))
    pi, recall = np.atleast_1d(pi), np.atleast_1d(recall)
    w = np.full(len(pi), 1.0 / len(pi)) if shares is None else np.asarray(shares, float) / np.sum(shares)
    if len(w) != len(pi):
        raise ValueError("shares must have one entry per stratum")
    return pi, recall, w


def coverage(pi, recall):
    """Detection probability c_h = pi_h * r_h of each stratum."""
    pi, recall, _ = _arrays(pi, recall)
    return pi * recall


def lower_tail(values, shares, fraction):
    """LCVaR_beta(v): the mean of v over the fraction beta of traffic where v is smallest."""
    return lcvar(values, shares, fraction)


def certificate(pi, recall, shares=None, gamma=1.0, detections=0, alpha=0.05, method="poisson"):
    """Upper confidence bound U_alpha(D; Gamma) on the number of harmful units after `detections` detections.

    detections = 0 uses the exact bound log(1/alpha) / LCVaR_{1/Gamma}(kappa) with kappa_h = -log(1 - c_h) (Theorem 1).
    detections >= 1 uses the exact Poisson constant (Lemma 1(c)); method="chernoff" gives the weaker Chernoff constant.
    Returns inf when a stratum the system can concentrate in has zero coverage. gamma = inf allows any allocation.
    """
    pi, recall, w = _arrays(pi, recall, shares)
    c = pi * recall
    if detections == 0:
        return float(ucb0(c, w, gamma, alpha))
    lam = lam_garwood(int(detections), alpha) if method == "poisson" else lam_chernoff(int(detections), alpha)
    rho = rstar(c, w, gamma)
    return float(np.inf if rho <= 0 else lam / rho)


def certificate_curve(pi, recall, shares=None, gammas=(1, 1.5, 2, 5, 10, 100), detections=0, alpha=0.05):
    """The certificate for each value of Gamma: the curve Gamma -> U_alpha(D; Gamma) that the paper recommends reporting."""
    return np.array([certificate(pi, recall, shares, g, detections, alpha) for g in gammas])


def hidden_floor(max_harmful, recall_min, alpha=0.05):
    """Hidden sampling rate pi_0 that certifies at most `max_harmful` harmful units whatever the system knows (Corollary 1).
    A value above 1 means the target cannot be reached by routing alone at this recall."""
    return (1 - alpha ** (1.0 / max_harmful)) / recall_min


def floor_certificate(pi0, recall_min, alpha=0.05):
    """Zero-detection certificate guaranteed by a hidden floor pi_0, for every allocation and every Gamma."""
    c = pi0 * recall_min
    return float(np.inf if c <= 0 else np.log(1 / alpha) / -np.log1p(-min(c, 1.0)))


def single_action_certified(pi, recall, alpha=0.05):
    """Whether zero detections rule out even one harmful unit at level alpha: needs c_h >= 1 - alpha in every stratum
    (Corollary 2)."""
    return bool(np.all(coverage(pi, recall) >= 1 - alpha))


def campaign_escape(k, pi, recall, shares=None, gamma=1.0, timeliness=1.0, intervention=1.0):
    """Upper bound on the probability that a campaign of k harmful units is never interrupted (Corollary 3).

    Each unit is interrupted with probability e_h = pi_h * r_h * timeliness_h * intervention_h, where timeliness is the
    probability that detection precedes irreversibility and intervention the probability that stopping it succeeds."""
    pi, recall, w = _arrays(pi, recall, shares)
    e = pi * recall * np.asarray(timeliness, float) * np.asarray(intervention, float)
    return float(np.exp(-k * lcvar(kappa(e), w, 1.0 / gamma))) if np.isfinite(gamma) else float(np.exp(-k * kappa(e).min()))


def minimax_design(recall, shares, budget, gamma):
    """Routing rates pi that maximise the certified coverage rate for a given Gamma and monitored fraction
    sum_h w_h pi_h = budget (Theorem 4, coverage scale). Returns (certified coverage rate, pi)."""
    value, pi = design(recall, shares, gamma, budget)
    return float(value), pi


def equal_coverage_design(recall, shares, budget):
    """Routing rates that give every stratum the same coverage at the given monitored fraction (capped at 1).
    Minimax optimal once Gamma >= harmonic-mean recall / min recall (Theorem 4)."""
    recall, w = np.asarray(recall, float), np.asarray(shares, float) / np.sum(shares)
    return _equal_coverage(recall, w, budget)


def recall_lower_bounds(detected, attempted, alpha=0.025, simultaneous=True):
    """One-sided Clopper-Pearson lower bounds on recall by attack type. With simultaneous=True the level is split
    across types (Bonferroni), as Proposition 2 requires."""
    detected, attempted = np.atleast_1d(detected), np.atleast_1d(attempted)
    level = alpha / len(detected) if simultaneous else alpha
    return np.array([cp_lower(int(x), int(n), level) for x, n in zip(detected, attempted)])


def randomized_monitor(recall_matrix):
    """Randomize each unit over monitors (Proposition 3). recall_matrix[t, j] is the recall lower bound of monitor j on
    attack type t. Returns (certified worst-type recall, mixing probabilities over monitors)."""
    value, p = game_value(np.asarray(recall_matrix, float))
    return float(value), p


def selection_coverage(detection_prob, selector_score, fraction):
    """Tail stress test: mean detection probability of the top `fraction` of units ranked by a selector that sees only
    what the system sees. A low value exhibits a low-coverage tail: LCVaR_beta(d) <= mean coverage of the selection."""
    d, s = np.asarray(detection_prob, float), np.asarray(selector_score, float)
    k = max(int(np.ceil(fraction * len(d))), 1)
    return float(d[np.argsort(-s)[:k]].mean())
