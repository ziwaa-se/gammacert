"""A hidden randomized floor certifies a bound whatever the system knows (Corollary 1)."""
import gammacert as gc

for r in [0.44, 0.94]:
    rates = ", ".join(f"K={k}: {gc.hidden_floor(k, r):.3f}" for k in [100, 30, 10])
    print(f"recall {r}: hidden sampling rate needed to certify at most K harmful units -> {rates}")

print("Can zero detections rule out a single catastrophic action with 99.9% routing and recall 0.9?",
      gc.single_action_certified([0.999], [0.9]))
