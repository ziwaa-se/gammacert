"""Quickstart: how many harmful actions does a silent monitoring record rule out, and how does the answer change when the system can see the design?"""
import gammacert as gc

# Five monitoring strata: routing probability, recall once routed, and share of harm-capable traffic.
pi = [1.0, 0.25, 0.05, 0.002, 0.0002]
recall = [0.94, 0.94, 0.58, 0.58, 0.44]
shares = [0.40, 0.10, 0.20, 0.21, 0.09]

print("95% upper bound on the number of harmful actions after zero detections")
for gamma in [1, 1.5, 2, 5, 10]:
    print(f"  harm up to {gamma:>4}x over-represented in weak strata: {gc.certificate(pi, recall, shares, gamma=gamma):10.1f}")

print("\nAfter 3 detections, Gamma = 1:", round(gc.certificate(pi, recall, shares, detections=3), 1))
