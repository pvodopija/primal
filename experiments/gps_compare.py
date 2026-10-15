"""
PRIMAL's delta along whole laps against the camera's own GPS (a GoPro's GPS5,
experiments/gpmf_read.py), for footage whose video truth fails (night). Every lap against the
session's fastest, as session_bars.py's review: the GPS delta at each of PRIMAL's ticks is the
lap's time so far less the time the reference lap took to the nearest point of its own GPS
path (searched near the last match, the path at 200 Hz between fixes). A constant GPS lag
cancels: both laps' fixes carry it, and the line crossings are the video's.

Pavle's night session (HERO7, 17.8 fixes a second, 3D fix, DOP 1.2-1.7): GPS lap times from
a line through the fixes at the transponder's crossings match the transponder's to 22 ms
median (max 67), the positions at the 13 crossings within 0.7 m of each other.

Usage: REAL_FOOTAGE=night1 [REAL_STREAM=name] PYTHONPATH=. python experiments/gps_compare.py RUN
       (writes ROOT/gps_compare_<run><suffix>.json and .png)
"""
from __future__ import annotations

import json
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.real_demo import CONF_MIN, lap_series, setup
from experiments.real_footage import ROOT, SUFFIX, VIDEO

DENSE_HZ = 200.0
SEARCH_S = 4.0  # the reference's match searched this far either side of the last one


def gps_xy() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(ROOT / "gpmf.npz")
    g, t = z["GPS5"], z["GPS5_t"]
    lat0, lon0 = g[0, :2]
    return t, np.stack([(g[:, 1] - lon0) * 111320 * np.cos(np.radians(lat0)), (g[:, 0] - lat0) * 110540], 1)


def gps_delta(t: np.ndarray, xy: np.ndarray, ref: tuple[float, float], lap: tuple[float, float], ticks: np.ndarray) -> np.ndarray:
    ra, rb = ref
    rt = np.arange(ra - 1.0, rb + 1.0, 1 / DENSE_HZ)
    rp = np.stack([np.interp(rt, t, xy[:, k]) for k in range(2)], 1)
    p = np.stack([np.interp(ticks, t, xy[:, k]) for k in range(2)], 1)
    out, last = np.full(len(ticks), np.nan), ra + (ticks[0] - lap[0])
    for i, (tick, q) in enumerate(zip(ticks, p)):
        w = np.flatnonzero(np.abs(rt - last) <= SEARCH_S)
        k = w[np.argmin(np.sum((rp[w] - q) ** 2, 1))]
        last = rt[k]
        out[i] = (tick - lap[0]) - (rt[k] - ra)
    return out


def main() -> None:
    run = sys.argv[1]
    S = setup(run)
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    t, xy = gps_xy()
    report, errs, errs_shown, plots = {"laps": []}, [], [], []
    for lap_no in S.live:
        L = lap_series(S, lap_no, fps)
        g = gps_delta(t, xy, S.laps[S.r], (L.a, L.b), L.ticks)
        inside = (L.ticks - L.a > 1.0) & (L.b - L.ticks > 1.0)  # a second from each line: the search's wrap
        e = (L.delta - g)[inside]
        shown = (L.conf >= CONF_MIN)[inside]
        errs.append(e)
        errs_shown.append(e[shown])
        plots.append((lap_no, L.ticks - L.a, g, L.delta, L.conf >= CONF_MIN))
        report["laps"].append({"lap": lap_no, "median_ms": float(1000 * np.median(np.abs(e))),
                               "over_100ms": float(np.mean(np.abs(e) > 0.1)),
                               "shown_median_ms": float(1000 * np.median(np.abs(e[shown]))) if shown.any() else None,
                               "shown_share": float(shown.mean())})
        print(f"  lap {lap_no:2d} against lap {S.r + 1}: PRIMAL vs GPS median {1000 * np.median(np.abs(e)):4.0f} ms, "
              f"{100 * np.mean(np.abs(e) > 0.1):4.1f}% over 100 ms; where shown ({100 * shown.mean():.0f}%) "
              f"{1000 * np.median(np.abs(e[shown])) if shown.any() else float('nan'):4.0f} ms", flush=True)
    a, s = np.abs(np.concatenate(errs)), np.abs(np.concatenate(errs_shown))
    report.update({"all_median_ms": float(1000 * np.median(a)), "all_p90_ms": float(1000 * np.percentile(a, 90)),
                   "all_over_100ms": float(np.mean(a > 0.1)), "shown_median_ms": float(1000 * np.median(s)),
                   "shown_p90_ms": float(1000 * np.percentile(s, 90)), "shown_over_100ms": float(np.mean(s > 0.1)),
                   "shown_share": float(len(s) / len(a))})
    print(f"{run}{SUFFIX}: PRIMAL vs GPS over every lap, median {report['all_median_ms']:.0f} ms, p90 {report['all_p90_ms']:.0f}, "
          f"{100 * report['all_over_100ms']:.1f}% over 100 ms; where shown ({100 * report['shown_share']:.0f}%) "
          f"median {report['shown_median_ms']:.0f}, p90 {report['shown_p90_ms']:.0f}, {100 * report['shown_over_100ms']:.1f}% over 100 ms")
    (ROOT / f"gps_compare_{run}{SUFFIX}.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, (lap_no, x, g, d, sh) in zip(axes, plots[:: max(1, len(plots) // 4)][:4]):
        ax.plot(x, g, color="#222222", lw=2, label="GoPro GPS (18 Hz)")
        ax.plot(x, np.where(sh, d, np.nan), color="#2c7fb8", lw=1.4, label="PRIMAL (camera), shown")
        ax.plot(x, np.where(sh, np.nan, d), color="#2c7fb8", lw=0.8, alpha=0.35, label="PRIMAL, hidden")
        ax.set_ylabel(f"lap {lap_no} (s)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"{run}{SUFFIX}: delta against the fastest lap, PRIMAL and the camera's GPS")
    axes[-1].set_xlabel("time into the lap (s)")
    fig.tight_layout()
    fig.savefig(ROOT / f"gps_compare_{run}{SUFFIX}.png", dpi=110)


if __name__ == "__main__":
    main()
