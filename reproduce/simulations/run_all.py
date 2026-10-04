"""Run every simulation (no GPU). Pure numpy/scipy/pandas/matplotlib: works on a laptop (about 10-20 minutes).
    python -m reproduce.simulations.run_all
"""
import runpy, sys, time, traceback
MODS = ["tests.test_regression", "reproduce.simulations.exp04_coverage", "reproduce.simulations.exp05_11_design_cost", "reproduce.simulations.exp02a_08_tail",
        "reproduce.simulations.exp10_hazard", "reproduce.simulations.exp12_recall_n", "reproduce.simulations.exp07_audit"]
fails = []
for m in MODS:
    t = time.time(); print(f"\n{'=' * 30} {m}")
    try: sys.argv = [m]; runpy.run_module(m, run_name="__main__")
    except Exception: traceback.print_exc(); fails.append(m)
    print(f"[{m}] {time.time() - t:.0f}s")
print("\nFAILED:" if fails else "\nALL OK", fails); sys.exit(1 if fails else 0)
