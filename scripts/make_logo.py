"""Draw the repository logo (three concepts) and the README banner into assets/logo/.

    python scripts/make_logo.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, Wedge, PathPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parents[1] / "assets" / "logo"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, NAVY2, TEAL, CYAN, AMBER, CORAL, WHITE = "#0b1b33", "#13294b", "#14b8a6", "#5eead4", "#f59e0b", "#fb7185", "#f8fafc"


def canvas(size=6):
    fig = plt.figure(figsize=(size, size)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal"); ax.axis("off"); fig.patch.set_alpha(0); return fig, ax


def gradient(ax, patch, c0, c1, angle=90, extent=(-1.1, 1.1, -1.1, 1.1)):
    """Fill `patch` with a linear gradient from c0 to c1."""
    t = np.deg2rad(angle); x, y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
    z = x * np.cos(t) + y * np.sin(t)
    im = ax.imshow(z, cmap=LinearSegmentedColormap.from_list("g", [c0, c1]), extent=extent, origin="lower", zorder=patch.get_zorder())
    ax.add_patch(patch); patch.set_facecolor("none"); patch.set_edgecolor("none"); im.set_clip_path(patch)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, transparent=True)
    plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------ A. blind-spot radar
def radar(ax, cx=0.0, cy=0.0, s=1.0, fs=1.0, lw=None):
    lw = s if lw is None else lw
    gradient(ax, Circle((cx, cy), 1.0 * s, zorder=1), NAVY, NAVY2, angle=60, extent=(cx - 1.1 * s, cx + 1.1 * s, cy - 1.1 * s, cy + 1.1 * s))
    for i, (r, a) in enumerate(zip(np.linspace(0.93, 0.38, 5), [0.95, 0.75, 0.55, 0.38, 0.22])):
        ax.add_patch(Circle((cx, cy), r * s, fill=False, lw=2.2 * lw, ec=TEAL, alpha=a, zorder=3))
    for k in range(12):                                                    # tick marks
        th = np.deg2rad(k * 30); r0, r1 = 0.95 * s, 1.0 * s
        ax.plot([cx + r0 * np.cos(th), cx + r1 * np.cos(th)], [cy + r0 * np.sin(th), cy + r1 * np.sin(th)], color=CYAN, lw=1.5 * lw, alpha=0.7, zorder=3)
    for j, a in enumerate(np.linspace(0.34, 0.0, 26)):                     # radar sweep with a fading trail
        ax.add_patch(Wedge((cx, cy), 0.93 * s, 25 + 3 * j, 28 + 3 * j, color=CYAN, alpha=a, lw=0, zorder=2))
    ax.plot([cx, cx + 0.93 * s * np.cos(np.deg2rad(28))], [cy, cy + 0.93 * s * np.sin(np.deg2rad(28))], color=CYAN, lw=2.6 * lw, zorder=4)
    ax.add_patch(Wedge((cx, cy), 0.93 * s, 232, 262, width=0.55 * s, color=AMBER, alpha=0.22, lw=0, zorder=2))   # the blind spot
    ax.add_patch(Wedge((cx, cy), 0.93 * s, 232, 262, width=0.55 * s, fill=False, ec=AMBER, lw=2.2 * lw, zorder=4))
    for th, rr in [(240, 0.70), (248, 0.55), (255, 0.80), (245, 0.85)]:    # undetected units hiding in it
        t = np.deg2rad(th); ax.add_patch(Circle((cx + rr * s * np.cos(t), cy + rr * s * np.sin(t)), 0.035 * s, color=AMBER, zorder=5))
    for th, rr in [(40, 0.62), (120, 0.78), (175, 0.5), (320, 0.7), (80, 0.45)]:   # detected units: rings
        t = np.deg2rad(th); ax.add_patch(Circle((cx + rr * s * np.cos(t), cy + rr * s * np.sin(t)), 0.03 * s, fill=False, ec=CYAN, lw=1.6 * lw, zorder=5))
    ax.text(cx, cy - 0.02 * s, "0", ha="center", va="center", fontsize=118 * s * fs, fontweight="bold", color=WHITE, family="DejaVu Sans", zorder=6)


def main():
    fig, ax = canvas(); radar(ax); save(fig, "logo_A_radar")

    # ------------------------------------------------------------------ B. certified shield with coverage tiers
    fig, ax = canvas()
    shield = MPath([(0, 1.0), (0.82, 0.72), (0.84, -0.20), (0, -1.02), (-0.84, -0.20), (-0.82, 0.72), (0, 1.0)],
                   [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.CLOSEPOLY])
    gradient(ax, PathPatch(shield, zorder=1), NAVY, NAVY2, angle=90)
    for i, (y0, a) in enumerate(zip(np.linspace(0.62, -0.58, 5), [0.85, 0.62, 0.42, 0.25, 0.10])):
        band = FancyBboxPatch((-0.9, y0 - 0.09), 1.8, 0.18, boxstyle="round,pad=0,rounding_size=0.04", color=TEAL, alpha=a, lw=0, zorder=2)
        ax.add_patch(band); band.set_clip_path(PathPatch(shield, transform=ax.transData))
    ax.add_patch(PathPatch(shield, fill=False, ec=CYAN, lw=4, zorder=4))
    check = [(-0.42, 0.08), (-0.10, -0.62), (0.52, 0.52)]
    ax.plot(*zip(*check), color=WHITE, lw=15, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    ax.plot(*zip(*check[:2]), color=AMBER, lw=15, solid_capstyle="round", zorder=6)      # the tail of the check sits in the weakest tier
    save(fig, "logo_B_shield")

    # ------------------------------------------------------------------ C. a zero made of monitored units, with a lens
    fig, ax = canvas()
    gradient(ax, Circle((0, 0), 1.0, zorder=1), NAVY, NAVY2, angle=45)
    rng = np.random.default_rng(3)
    for ring, (rx, ry) in enumerate([(0.52, 0.72), (0.43, 0.63), (0.34, 0.54)]):
        n = 44 - 6 * ring
        for k in range(n):
            th = 2 * np.pi * k / n + ring * 0.07; x, y = rx * np.cos(th), ry * np.sin(th)
            hidden = -2.35 < np.arctan2(y, x) < -1.55                           # an arc of unmonitored units
            ax.add_patch(Circle((x, y), 0.028, color=AMBER if hidden else TEAL, alpha=1.0 if hidden else 0.55 + 0.45 * rng.random(), zorder=3))
    ax.add_patch(Circle((0.42, 0.40), 0.30, fill=False, ec=WHITE, lw=6, zorder=5))           # lens
    ax.plot([0.63, 0.86], [0.19, -0.05], color=WHITE, lw=12, solid_capstyle="round", zorder=5)
    gradient(ax, Circle((0.42, 0.40), 0.27, zorder=4), "#ffffff22", "#5eead455", angle=135)
    save(fig, "logo_C_units")

    # ------------------------------------------------------------------ README banner: mark + wordmark (concept A)
    fig = plt.figure(figsize=(12, 3.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    fig.patch.set_alpha(0)
    radar(ax, cx=1.6, cy=1.6, s=1.38, fs=0.36)
    t0 = ax.text(3.45, 1.95, "gamma", fontsize=62, fontweight="bold", color=NAVY, va="center", family="DejaVu Sans")
    fig.canvas.draw(); bb = t0.get_window_extent().transformed(ax.transData.inverted())
    ax.text(bb.x1 + 0.02, 1.95, "cert", fontsize=62, fontweight="bold", color=TEAL, va="center", family="DejaVu Sans")
    ax.text(3.5, 0.95, "How much can a silent monitor rule out when the AI can see it?", fontsize=16.5, color="#475569",
            va="center", family="DejaVu Sans")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"banner.{ext}", dpi=200, transparent=True)
    print("wrote banner")


if __name__ == "__main__":
    main()
