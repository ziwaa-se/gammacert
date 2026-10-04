"""Theorem 3: two information structures with the same AUC can certify very different numbers of harmful actions."""
import numpy as np
from gammacert.core import TwoStrata

N, w_low, c_high, c_low = 1e8, 0.1, 0.2, 1.6e-4        # deployment size, low-coverage share, coverages
eta_N = np.log(20) / (c_low * w_low * N)               # smallest recognizable blind spot that matters
smooth = TwoStrata(w_low, c_high, c_low, 0.55)         # binormal signal, AUC 0.55
blind = TwoStrata(w_low, c_high, c_low, 0.55, eta_N)   # same AUC, plus a blind spot of mass eta_N
print(f"blind-spot mass {eta_N:.2e} of low-coverage traffic")
print(f"95% bound on expected harmful actions: {smooth.cert(N):,.0f} (binormal) vs {blind.cert(N):,.0f} (same AUC, blind spot)")
print(f"Bayes accuracy {smooth.bayes_acc():.4f} vs {blind.bayes_acc():.4f}")
