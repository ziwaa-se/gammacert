"""Same monitoring budget, different allocation: equal effective coverage makes the certificate independent of Gamma."""
import numpy as np
import gammacert as gc

pi = np.array([1.0, 0.25, 0.05, 0.002, 0.0002])
recall = np.array([0.94, 0.94, 0.58, 0.58, 0.44])
shares = np.array([0.40, 0.10, 0.20, 0.21, 0.09])
budget = shares @ pi                                    # monitored fraction, about 0.435

equal = gc.equal_coverage_design(recall, shares, budget)
gammas = [1, 1.5, 2, 5, 10, 100]
print(f"monitored fraction {budget:.3f}")
print("Gamma             " + "".join(f"{g:>9}" for g in gammas))
print("as designed       " + "".join(f"{u:9.1f}" for u in gc.certificate_curve(pi, recall, shares, gammas)))
print("equal coverage    " + "".join(f"{u:9.1f}" for u in gc.certificate_curve(equal, recall, shares, gammas)))
print("minimax (per G)   " + "".join(f"{gc.certificate(gc.minimax_design(recall, shares, budget, g)[1], recall, shares, g):9.1f}"
                                     for g in gammas))
print("\nP(24-step campaign never interrupted), Gamma = 2:", round(gc.campaign_escape(24, pi, recall, shares, gamma=2), 2))
