import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import erf

N, C, G = 6, 16, 4   # frames in a batch, channels, channels per group (drawing sizes)
fig = plt.figure(figsize=(13, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.55, wspace=0.28)
panels = [
    ("BatchNorm", "one channel, across ALL frames in the batch",
     lambda n, c: c == 5),
    ("LayerNorm", "one frame, all channels",
     lambda n, c: n == 2),
    ("GroupNorm (ours)", "one frame, one group of channels",
     lambda n, c: n == 2 and 4 <= c < 8),
]
for k, (title, sub, shaded) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    for n in range(N):
        for c in range(C):
            on = shaded(n, c)
            ax.add_patch(Rectangle((n, c), 0.92, 0.88, facecolor="#e4572e" if on else "#dfe4ea",
                                   edgecolor="white", linewidth=0.6))
    ax.set_xlim(-0.2, N + 0.1); ax.set_ylim(C + 0.2, -0.3)
    ax.set_xticks([n + 0.46 for n in range(N)]); ax.set_xticklabels([f"{n + 1}" for n in range(N)], fontsize=8)
    ax.set_yticks([]); ax.set_xlabel("frame in the batch", fontsize=9)
    ax.set_ylabel("channels (each box = a whole H x W map)" if k == 0 else "", fontsize=9)
    ax.set_title(f"{title}\n", fontsize=13, fontweight="bold", loc="left")
    ax.text(0, -1.25, sub, fontsize=9.5, color="#444")
    for s in ax.spines.values(): s.set_visible(False)
fig.text(0.07, 0.462, "Red = the numbers averaged together to get ONE mean and ONE spread. They are then shifted to mean 0, spread 1, and a learned scale and shift is applied.",
         fontsize=10, color="#222")

ax = fig.add_subplot(gs[1, 0:2])
x = np.linspace(-4, 3, 600)
relu = np.maximum(0, x)
gelu = x * 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))
ax.plot(x, relu, color="#2e86ab", lw=2.5, label="ReLU:  max(0, x)")
ax.plot(x, gelu, color="#e4572e", lw=2.5, label="GELU:  x · P(a normal variable < x)")
# 8-bit lookup table: 256 possible inputs -> 256 stored outputs
xs = np.linspace(-4, 3, 32)
gx = xs * 0.5 * (1 + np.vectorize(erf)(xs / np.sqrt(2)))
ax.scatter(xs, gx, s=14, color="#8b1e3f", zorder=5, label="on the NPU: a lookup table (256 entries for 8-bit input; 32 shown)")
ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
ax.set_title("Activations: what each neuron does to its number", fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9.5, loc="upper left"); ax.set_xlabel("input", fontsize=9); ax.set_ylabel("output", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ax.set_title("On Halo's NPU (Vela, per frame)", fontsize=13, fontweight="bold", loc="left")
rows = [("encoder with GroupNorm", "9.0 ms on the NPU\n+ 40 steps on the CPU\n265 separate pieces", "#e4572e"),
        ("encoder without it (GELU)", "2.5 ms, all on the NPU\n6 pieces", "#2e86ab"),
        ("encoder without it (ReLU)", "2.4 ms, all on the NPU", "#2e86ab")]
for i, (name, what, col) in enumerate(rows):
    y = 0.86 - i * 0.31
    ax.add_patch(Rectangle((0, y - 0.2), 0.035, 0.22, color=col, transform=ax.transAxes))
    ax.text(0.07, y, name, fontsize=11, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.07, y - 0.07, what, fontsize=9.5, color="#333", transform=ax.transAxes, va="top")
out = "/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/norms_and_activations.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white"); print(out)
