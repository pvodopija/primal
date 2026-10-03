"""
The delta over one lap, as a driver would see it: the true delta, and PRIMAL's.

Reads runs/RUN/consistency_series.npz (experiments/consistency.py) and plots one lap:
by default the unseen-track lap at a human-like pace whose share of ticks over 100 ms
is the median of all of them. Writes runs/demos/delta_<RUN>.png.

Usage: PYTHONPATH=. python experiments/show_delta_plot.py [RUN] [KEY]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRUE, OURS = "#52514e", "#2a78d6"
HZ, SKIP_S = 15.0, 2.0

run = sys.argv[1] if len(sys.argv) > 1 else "v3_mobilenet_lr1_s0"
series = np.load(f"runs/{run}/consistency_series.npz")
keys = [k for k in series.files if k.startswith("imperfect/G2/")]
share = {k: float(np.mean(np.abs(series[k][0]) > 100)) for k in keys}
key = sys.argv[2] if len(sys.argv) > 2 else min(keys, key=lambda k: abs(share[k] - np.median(list(share.values()))))
d, true, _ = series[key]
t = SKIP_S + np.arange(d.size) / HZ
true_s = (true - true[0]) / 1000.0
ours_s = true_s + d / 1000.0
track = key.split("/")[-1].removeprefix("ks_").removeprefix("rt_").replace("__", " ").replace("_", " ").title()

plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.family": "sans-serif"})
fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, sharey=True, facecolor=SURFACE)
lo, hi = min(true_s.min(), ours_s.min()) - 0.05, max(true_s.max(), ours_s.max()) + 0.05
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylim(lo, hi)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("delta (s)")
axes[0].plot(t, true_s, color=TRUE, linewidth=2, solid_capstyle="round")
axes[0].set_title("True delta: what a perfect timer would show", loc="left", color=INK, fontsize=13)
axes[1].plot(t, true_s, color=TRUE, linewidth=1, alpha=0.45, label="true delta")
axes[1].plot(t, ours_s, color=OURS, linewidth=2, solid_capstyle="round", label="PRIMAL's delta (camera only)")
axes[1].set_title("PRIMAL's delta, with the true one faint behind it", loc="left", color=INK, fontsize=13)
axes[1].legend(loc="upper left", frameon=False, labelcolor=MUTED)
axes[1].set_xlabel("time into the lap (s)")
over = 100 * share[key]
fig.suptitle(f"{track} (never trained on): one lap at a human-like pace against a reference lap from another session.\n"
             f"Above zero = slower than the reference. PRIMAL is off by more than 0.1 s for {over:.0f}% of the lap "
             f"(median error {np.median(np.abs(d)) / 1000:.3f} s): a typical lap of the 81 tested.",
             x=0.01, ha="left", color=MUTED, fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = Path(f"runs/demos/delta_{run}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out, key)
