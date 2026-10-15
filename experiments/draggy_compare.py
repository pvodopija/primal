"""
PRIMAL against Dragy (25 Hz GPS, its delta read off the video by experiments/draggy_read.py) on
lap 2 against lap 1, both against the ORB truth, under Dragy's own line crossings. Writes
ROOT/draggy_compare.json and .png.

Usage: REAL_FOOTAGE=draggy PYTHONPATH=. python experiments/draggy_compare.py RUN
"""
from __future__ import annotations

import json
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.real_demo import lap_series, setup
from experiments.real_footage import ROOT, VIDEO


def main() -> None:
    run = sys.argv[1]
    S = setup(run)
    L = lap_series(S, 2, cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS))
    ocr = pd.read_csv(ROOT / "ocr.csv").dropna(subset=["delta_s"])
    ocr = ocr[(ocr.t > L.a + 2.0) & (ocr.t < L.b)]  # just past the line Dragy still shows the last lap
    med = ocr.delta_s.rolling(9, center=True, min_periods=1).median()
    ocr = ocr[(ocr.delta_s - med).abs() < 0.15]  # misreads out
    keep = (L.ticks >= ocr.t.min()) & (L.ticks <= ocr.t.max()) & np.isfinite(L.true)
    t, primal, true = L.ticks[keep], L.delta[keep], L.true[keep]
    dragy = np.interp(t, ocr.t, ocr.delta_s)
    report = {}
    for name, (x, y) in {"PRIMAL vs truth": (primal, true), "Dragy vs truth": (dragy, true), "PRIMAL vs Dragy": (primal, dragy)}.items():
        e = np.abs(x - y) * 1000
        report[name] = {"median_ms": float(np.median(e)), "p90_ms": float(np.percentile(e, 90)), "over_100ms": float(np.mean(e > 100))}
        print(f"  {name:<16} median {report[name]['median_ms']:4.0f} ms, p90 {report[name]['p90_ms']:4.0f} ms, over 100 ms {100 * report[name]['over_100ms']:4.1f}%")
    print(f"  at the line: PRIMAL {np.median(L.delta[-8:]):+.2f} s, Dragy's lap times {L.b - L.a - S.T:+.2f} s, last Dragy reading {ocr.delta_s.iloc[-1]:+.2f} s")
    (ROOT / "draggy_compare.json").write_text(json.dumps({"run": run, "report": report}, indent=2))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t - L.a, true, color="#222222", lw=2, label="truth (video features)")
    ax.plot(t - L.a, dragy, color="#d9534f", lw=1.4, label="Dragy (25 Hz GPS)")
    ax.plot(t - L.a, primal, color="#2c7fb8", lw=1.4, label="PRIMAL (camera)")
    ax.axhline(0, color="#999999", lw=0.8); ax.set_xlabel("lap time, s"); ax.set_ylabel("delta, s")
    ax.set_title(f"lap 2 against lap 1: {run}"); ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(ROOT / "draggy_compare.png", dpi=110)


if __name__ == "__main__":
    main()
