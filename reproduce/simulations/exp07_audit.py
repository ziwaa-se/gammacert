"""Experiment 7: verification bias in the recall audit, and what it does to the certificate.

Stage 1 scores every unit; a reference review (ground truth) is applied to an audited subset. Five audit
policies. Simulated scores by default; with --scores <parquet from monitors/> the real Stage-1 scores of one
monitor are resampled instead (columns: score, is_attack).

Reported per policy: the recall estimate, the recall lower bound fed to the certificate, the certificate, and
the ACTUAL probability of zero detections when the true count is just above that certificate (nominal 0.05).
"""
import argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
from gammacert.core import *

ap = argparse.ArgumentParser(); ap.add_argument("--scores", default=None); ap.add_argument("--tag", default="sim"); ap.add_argument("--fpr", type=float, default=0.01); args = ap.parse_args()
emit = Emitter(f"numbers_exp07_{args.tag}.tex"); RES, FIG = out_dir("results"), out_dir("figures"); rng = np.random.default_rng(7)
L = np.log(20); PI0 = 0.25; N = 200_000; PREV = 2e-3; AUDIT_BUDGET = 4000; N_INJ = 300; REPS = 300

if args.scores:
    sc = pd.read_parquet(args.scores); pos_pool, neg_pool = sc[sc.is_attack].score.values.astype(float), sc[~sc.is_attack].score.values.astype(float)
    draw = lambda y: np.where(y, rng.choice(pos_pool, y.size), rng.choice(neg_pool, y.size)) + rng.uniform(-0.5, 0.5, y.size)   # jitter breaks integer ties
    # threshold and true recall are defined on the same jittered scores the simulation draws (otherwise ties at integer scores bias r_true)
    thr = np.quantile(draw(np.zeros(400_000, bool)), 1 - args.fpr); r_true = (draw(np.ones(400_000, bool)) > thr).mean()
else:
    DELTA = 2.4; draw = lambda y: rng.normal(DELTA * y, 1.0); thr = norm.ppf(1 - args.fpr); r_true = norm.sf(thr - DELTA)


def cp_weighted(flag, wts, a=0.025):
    """Recall estimate and a conservative lower bound from inverse-probability-weighted audited positives (Kish effective n)."""
    if wts.sum() == 0: return np.nan, 0.0
    est = (wts * flag).sum() / wts.sum(); neff = wts.sum() ** 2 / (wts ** 2).sum()
    return est, cp_lower(int(np.floor(est * neff)), max(int(np.floor(neff)), 1), a)


rows = []
for rep in range(REPS):
    y = rng.random(N) < PREV; s = draw(y); flag = s > thr; rk = s.argsort().argsort() / N      # score rank in [0,1)
    pol = {"audit Stage-1 positives only": flag.astype(float),
           "audit prob. increasing in score, unweighted": np.clip(AUDIT_BUDGET / N * 6 * rk ** 5, 0, 1),
           "uniform random audit": np.full(N, AUDIT_BUDGET / N),
           "known a(s)>0 + inverse-probability weights": np.clip(0.5 * AUDIT_BUDGET / N + 0.5 * AUDIT_BUDGET / N * 6 * rk ** 5, 0, 1)}
    for name, a in pol.items():
        aud = (rng.random(N) < a) & y                                                    # audited units that are truly harmful
        wts = (1 / a[aud]) if "inverse" in name else np.ones(aud.sum())
        est, low = cp_weighted(flag[aud].astype(float), wts); rows.append(dict(rep=rep, policy=name, n_pos=int(aud.sum()), r_hat=est, r_low=low))
    inj = draw(np.ones(N_INJ, bool)) > thr                                                # hidden injected attacks, all known
    rows.append(dict(rep=rep, policy=f"hidden injected attacks (n={N_INJ})", n_pos=N_INJ, r_hat=inj.mean(), r_low=cp_lower(int(inj.sum()), N_INJ, 0.025)))
df = pd.DataFrame(rows); kt = kappa(PI0 * r_true)
df["U"] = L / kappa(np.maximum(PI0 * df.r_low, 1e-12)); df["p_zero_at_U"] = np.exp(-(np.floor(df.U) + 1) * kt)   # nominal <= 0.05 if r_low <= r_true
summ = df.groupby("policy", sort=False).agg(n_pos=("n_pos", "mean"), r_hat=("r_hat", "mean"), r_hat_sd=("r_hat", "std"), r_low=("r_low", "mean"),
        U=("U", "median"), p_zero=("p_zero_at_U", "mean"), invalid=("r_low", lambda x: (x > r_true).mean()))
summ["r_true"] = r_true; summ["U_true"] = L / kt; summ.to_csv(f"{RES}/exp07_audit_{args.tag}.csv"); print(summ.round(4).to_string())
fig, ax = plt.subplots(figsize=(6.4, 3.2)); names = list(summ.index)
ax.barh(range(len(names)), summ.p_zero, color=["C3" if v > 0.05 else "C0" for v in summ.p_zero]); ax.axvline(0.05, color="k", ls=":")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.set_xlabel("P(zero detections | true count just above the certificate); nominal 0.05")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp07_audit_{args.tag}.pdf")
emit("auditTrueRecall" + args.tag.capitalize(), r_true); emit("auditPosOnlyPzero" + args.tag.capitalize(), summ.p_zero.iloc[0]); emit("auditIpwPzero" + args.tag.capitalize(), summ.p_zero.iloc[3])
print("exp07 done")
