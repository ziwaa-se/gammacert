"""Regression test: the library reproduces the closed-form values printed in the paper (Sections 3-5 and 7).
Run with `pytest` or `python -m tests.test_regression`."""
import numpy as np
from gammacert.core import *


def test_paper_values():
    c = PI_REP * R_REP; ok = True
    def check(name, got, want, tol):
        nonlocal ok; good = abs(got - want) <= tol; ok &= good; print(f"{'PASS' if good else 'FAIL'}  {name}: got {got:.6g}, paper has {want}")
    for G, want in [(1, 2.6), (1.5, 9.1), (2, 244.1), (5, 4418.7), (10, 15341.4), (100, 34040.9)]: check(f"U(0;Gamma={G}) report tiers", ucb0(c, W_REP, G), want, 0.06)
    pibar = (W_REP * PI_REP).sum(); Hb = 1 / (W_REP / R_REP).sum()
    check("pibar", pibar, 0.435, 6e-4); check("Hbar", Hb, 0.693, 6e-4); check("equal-coverage U", ucb0(equal_coverage(R_REP, W_REP, pibar) * R_REP, W_REP, 50), 8.3, 0.06)
    check("design LP = pibar*Hbar at Gamma=2", design(R_REP, W_REP, 2.0, pibar)[0], pibar * Hb, 1e-6)
    N, wL, cH, cL = 1e8, 0.1, 0.2, 1.6e-4; etaN = np.log(20) / (cL * wL * N)
    check("eta_N", etaN, 0.00187233, 1e-7); check("Thm3 base cert AUC .55", TwoStrata(wL, cH, cL, 0.55).cert(N), 19, 1.0)
    check("Thm3 perturbed cert", TwoStrata(wL, cH, cL, 0.55, etaN).cert(N), 18723, 1.0); check("binormal AUC .8", TwoStrata(wL, cH, cL, 0.8).cert(N), 461, 2.0)
    check("game value", game_value(np.array([[0.9, 0.3], [0.2, 0.8]]))[0], 0.55, 1e-6)
    check("CP lower 153/163", cp_lower(153, 163, 0.025), 0.890, 6e-4); check("Garwood d=0", lam_garwood(0), np.log(20), 1e-9); check("Garwood d=1", lam_garwood(1), 4.7439, 1e-3)
    check("worst_q matches LCVaR", (worst_q(kappa(c), W_REP, 2.0) * kappa(c)).sum(), kstar(c, W_REP, 2.0), 1e-12)
    assert ok, "regression against the paper values failed"; print("all regression checks passed")


if __name__ == "__main__":
    test_paper_values()
