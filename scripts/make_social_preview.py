"""Social preview card (1280 x 640) for the repository settings: assets/social_preview.png

    python scripts/make_social_preview.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_logo import NAVY, NAVY2, TEAL, CYAN, AMBER, WHITE, gradient, radar

OUT = Path(__file__).resolve().parents[1] / "assets" / "social_preview.png"
fig = plt.figure(figsize=(6.4, 3.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.8); ax.set_ylim(0, 6.4); ax.axis("off")
gradient(ax, FancyBboxPatch((0, 0), 12.8, 6.4, boxstyle="square,pad=0", zorder=0), "#08142a", NAVY2, angle=20, extent=(0, 12.8, 0, 6.4))
radar(ax, cx=2.45, cy=3.2, s=2.05, fs=0.19, lw=0.85)
t0 = ax.text(5.05, 4.6, "gamma", fontsize=34, fontweight="bold", color=WHITE, va="center", family="DejaVu Sans")
fig.canvas.draw(); bb = t0.get_window_extent().transformed(ax.transData.inverted())
ax.text(bb.x1 + 0.03, 4.6, "cert", fontsize=34, fontweight="bold", color=CYAN, va="center", family="DejaVu Sans")
ax.text(5.1, 3.55, "How much can a silent monitor rule out,\nwhen the AI can see the monitor?", fontsize=11.5, color="#cbd5e1",
        va="center", family="DejaVu Sans", linespacing=1.35)
for k, (v, lab) in enumerate([("2.6 → 244", "bound when harm\ncan concentrate 2×"), ("8.3", "same budget,\nequal coverage"),
                              ("19 vs 18,723", "same AUC,\ndifferent tail")]):
    x = 5.1 + k * 2.5
    ax.add_patch(FancyBboxPatch((x, 1.35), 2.3, 1.3, boxstyle="round,pad=0,rounding_size=0.18", fc="#ffffff10", ec="#5eead455", lw=1))
    ax.text(x + 1.15, 2.25, v, ha="center", va="center", fontsize=10.5, fontweight="bold", color=AMBER if k == 0 else WHITE, family="DejaVu Sans")
    ax.text(x + 1.15, 1.72, lab, ha="center", va="center", fontsize=7.2, color="#cbd5e1", family="DejaVu Sans", linespacing=1.2)
ax.text(5.1, 0.62, "Zihua She · Xiao Wang      github.com/ziwaa-se/gammacert", fontsize=8.5, color="#94a3b8", family="DejaVu Sans")
fig.savefig(OUT, dpi=200); print("wrote", OUT, "size", OUT.stat().st_size // 1024, "KB")
