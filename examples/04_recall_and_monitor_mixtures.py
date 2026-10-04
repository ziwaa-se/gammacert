"""Recall enters as a simultaneous lower bound by attack type; randomizing over monitors can raise the worst case."""
import gammacert as gc

detected, attempted = [153, 4, 40], [163, 5, 60]
print("simultaneous 97.5% lower bounds:", gc.recall_lower_bounds(detected, attempted).round(3))

# rows: attack types, columns: monitors
value, p = gc.randomized_monitor([[0.9, 0.3], [0.2, 0.8]])
print(f"best single monitor guarantees {max(min(0.9, 0.2), min(0.3, 0.8)):.2f}; randomizing with p = {p.round(3)} "
      f"guarantees {value:.2f}")
