"""
Where along the lap PRIMAL's night error sits, and whether the picture there explains it: every
lap against the fastest (as gps_compare.py), each tick placed on the reference lap by the GPS,
binned by the reference's seconds. Per bin: the median |PRIMAL - GPS| over laps, how often the
delta jumps (more than JUMP_S within a second, which no driving does), and what the model sees
there on the reference lap (its 148x80 frames): brightness (mean grey) and detail (variance of
the Laplacian). Errors that come back at the same places lap after lap and follow the dark or
blurred stretches are the picture's; errors that wander are the tracker's or the GPS's.

Usage: REAL_FOOTAGE=night1 PYTHONPATH=. python experiments/night_signal.py RUN
       (writes ROOT/night_signal_<run>.json and .png)
"""
from __future__ import annotations

import json
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.gps_compare import gps_delta, gps_xy
from experiments.real_demo import lap_series, setup
from experiments.real_footage import ROOT, SUFFIX, VIDEO, frames

JUMP_S = 0.3


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def main() -> None:
    run = sys.argv[1]
    S = setup(run)
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    t, xy = gps_xy()
    ra, rb = S.laps[S.r]
    n_bins = int(np.floor(rb - ra))
    err = [[] for _ in range(n_bins)]
    jumps = [[] for _ in range(n_bins)]
    per_lap = []
    for lap_no in S.live:
        L = lap_series(S, lap_no, fps)
        g = gps_delta(t, xy, (ra, rb), (L.a, L.b), L.ticks)
        where = (L.ticks - L.a) - g  # seconds into the reference lap, by GPS
        e = L.delta - g
        step = int(round(1 / np.median(np.diff(L.ticks))))  # ticks in a second
        jump = np.zeros(len(e), bool)
        jump[step:] = np.abs(L.delta[step:] - L.delta[:-step]) > JUMP_S
        b = np.floor(where).astype(int)
        ok = (b >= 0) & (b < n_bins)
        lap_err = np.full(n_bins, np.nan)
        for k in range(n_bins):
            sel = ok & (b == k)
            err[k] += list(np.abs(e[sel]))
            jumps[k] += list(jump[sel])
            if sel.any():
                lap_err[k] = np.median(np.abs(e[sel]))
        per_lap.append(lap_err)
    f, ft = frames()
    grey = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    bright, detail, thumbs = [], [], []
    for k in range(n_bins):
        idx = np.flatnonzero((ft >= ra + k) & (ft < ra + k + 1))
        g_ = [grey(f[i])[:64] for i in idx]  # the picture above the masked band
        bright.append(float(np.mean([x.mean() for x in g_])))
        detail.append(float(np.median([cv2.Laplacian(x, cv2.CV_32F).var() for x in g_])))
        thumbs.append(f[idx[len(idx) // 2]])
    med = np.array([np.median(x) if x else np.nan for x in err])
    jump_rate = np.array([np.mean(x) if x else np.nan for x in jumps])
    bright, detail = np.array(bright), np.array(detail)
    P = np.array(per_lap)
    # do the bad places repeat? each lap's per-bin error against the other laps' median
    repeat = np.nanmedian([spearman(np.nan_to_num(P[i]), np.nan_to_num(np.nanmedian(np.delete(P, i, 0), 0)))
                           for i in range(len(P))])
    report = {"bins_s": n_bins, "median_error_ms": (1000 * med).round(0).tolist(), "jump_rate": jump_rate.round(3).tolist(),
              "brightness": np.round(bright, 1).tolist(), "detail": np.round(detail, 1).tolist(),
              "spearman_error_brightness": spearman(med, bright), "spearman_error_detail": spearman(med, detail),
              "spearman_jumps_brightness": spearman(jump_rate, bright), "spearman_jumps_detail": spearman(jump_rate, detail),
              "repeat_across_laps": float(repeat)}
    worst, best = np.argsort(-med)[:4], np.argsort(med)[:4]
    print(f"{run}{SUFFIX}: {n_bins} one-second bins of the reference lap")
    print(f"  error by bin: median of bins {1000 * np.median(med):.0f} ms, worst four {np.round(1000 * med[worst]).astype(int)} "
          f"at {worst.tolist()} s, best four {np.round(1000 * med[best]).astype(int)} at {best.tolist()} s")
    print(f"  the same places bad lap after lap: rank correlation {repeat:+.2f} (0: errors wander, 1: always the same places)")
    print(f"  error vs brightness {report['spearman_error_brightness']:+.2f}, vs detail {report['spearman_error_detail']:+.2f}; "
          f"jumps vs brightness {report['spearman_jumps_brightness']:+.2f}, vs detail {report['spearman_jumps_detail']:+.2f}")
    print(f"  brightness (0-255) worst bins {np.round(bright[worst]).astype(int)}, best bins {np.round(bright[best]).astype(int)}; "
          f"detail worst {np.round(detail[worst]).astype(int)}, best {np.round(detail[best]).astype(int)}")
    (ROOT / f"night_signal_{run}{SUFFIX}.json").write_text(json.dumps(report, indent=2))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.07, 0.55, 0.86, 0.4])
    x = np.arange(n_bins) + 0.5
    ax.bar(x, 1000 * med, width=0.85, color="#d9534f", alpha=0.8, label="median |PRIMAL - GPS| over laps (ms)")
    ax.set_ylabel("error (ms)")
    ax.set_xlabel("seconds into the reference lap")
    ax2 = ax.twinx()
    ax2.plot(x, bright, color="#222222", lw=1.6, label="brightness the model sees")
    ax2.plot(x, detail / detail.max() * bright.max(), color="#2c7fb8", lw=1.2, ls="--", label="detail (scaled)")
    ax2.set_ylabel("brightness (0-255)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title(f"{run}: where the night error sits (repeat across laps {repeat:+.2f}; "
                 f"error vs brightness {report['spearman_error_brightness']:+.2f}, vs detail {report['spearman_error_detail']:+.2f})", fontsize=10)
    for j, (k, label) in enumerate([(k, "worst") for k in worst] + [(k, "best") for k in best]):
        a = fig.add_axes([0.03 + 0.24 * (j % 4), 0.27 - 0.24 * (j // 4), 0.22, 0.2])
        a.imshow(cv2.cvtColor(thumbs[k], cv2.COLOR_BGR2RGB))
        a.set_title(f"{label}: {k} s, {1000 * med[k]:.0f} ms", fontsize=8)
        a.axis("off")
    fig.savefig(ROOT / f"night_signal_{run}{SUFFIX}.png", dpi=110)


if __name__ == "__main__":
    main()
