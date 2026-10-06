"""
PRIMAL against RaceChrono (GPS) on RaceChrono's own video, and both against the ORB truth.

RaceChrono shows each lap's delta against its comparison lap R1 (1:26.5, not in the
video), one decimal, read off the overlay by experiments/racechrono_read.py. Its R1 time
at the car's place is the lap clock minus that delta, so two laps are at the same place
where those agree: that gives RaceChrono's own delta of a test lap against our reference
lap from its GPS alone, independent of the video. PRIMAL places each live clip on the
reference lap from the camera (the overlays greyed out), the ORB truth by geometric
feature matching (experiments/real_truth_orb.py). All three are expressed as the
reference lap's video moment matched to each live moment, under the same lap starts, so
they differ only in where they place the car.

Scored at RaceChrono's 10 Hz readings, from 5 s into each lap (before that RaceChrono
shows the previous lap) to the line. Writes ROOT/racechrono_compare.json and .png.

Usage: REAL_FOOTAGE=race-chrono PYTHONPATH=. python experiments/racechrono_compare.py RUN
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
from experiments.real_footage import CROSSINGS_S, ROOT, VIDEO
from experiments.real_signs import cells


def r1_time(ocr: pd.DataFrame, lap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RaceChrono's lap clock and R1 time at the car's place (clock - delta), monotone; and its lap start."""
    x = ocr[(ocr.lap == lap)].dropna()
    start = float((x.t - x.clock_s).median())
    x = x[np.abs((x.t - x.clock_s) - start) < 0.3]  # frozen or misread clocks out
    tau = np.maximum.accumulate(pd.Series((x.clock_s - x.delta_s).to_numpy()).rolling(5, center=True, min_periods=1).median().to_numpy())
    return x.t.to_numpy(), x.clock_s.to_numpy(), tau, start


def main() -> None:
    run = sys.argv[1]
    S = setup(run)
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    ocr = pd.read_csv(ROOT / "ocr.csv")
    r = S.r
    rc_ref_lap = r + 2  # our laps start at RaceChrono's lap 2
    _, ref_clock, ref_tau, ref_start_rc = r1_time(ocr, rc_ref_lap)
    a_r = S.laps[r][0]
    report, fig = {}, plt.figure(figsize=(11, 3.4 * (len(S.laps) - 1)))
    for n, lap_no in enumerate([k for k in range(1, len(S.laps) + 1) if k != r + 1]):
        L = lap_series(S, lap_no, fps)
        a, b = L.a, L.b
        t, clock, tau, _ = r1_time(ocr, lap_no + 1)
        keep = t < b
        t, clock, tau = t[keep], clock[keep], tau[keep]
        # each system's reference-lap video moment for the live moment t, as a delta under our lap starts
        rc_matched = ref_start_rc + np.interp(tau, ref_tau, ref_clock)
        rc = (t - a) - (rc_matched - a_r)
        primal = np.interp(t, L.ticks, L.delta)
        true = np.interp(t, L.ticks, L.true)
        ok = np.isfinite(true)
        t, rc, primal, true = t[ok], rc[ok], primal[ok], true[ok]
        # the ORB truth's own glitches: a delta cannot jump seconds within a second
        sane = np.abs(true - pd.Series(true).rolling(31, center=True, min_periods=1).median().to_numpy()) <= 0.25
        pairs = {"PRIMAL vs truth": (primal, true, sane), "RaceChrono vs truth": (rc, true, sane),
                 "PRIMAL vs RaceChrono": (primal, rc, np.ones_like(sane))}
        row = {"lap_s": b - a, "truth_rejected": float(1 - sane.mean()),
               "final": {"PRIMAL": float(primal[-1]), "RaceChrono": float(rc[-1]), "truth": float(true[-1])}}
        for name, (x, y, use) in pairs.items():
            x, y = np.where(use, x, np.nan), np.where(use, y, np.nan)
            e = np.abs(x - y)[use] * 1000
            row[name] = {"median_ms": float(np.median(e)), "p90_ms": float(np.percentile(e, 90)), "over_100ms": float(np.mean(e > 100))}
            for w in (2, 5):
                k = int(w * 10)
                dx, dy = x[k:] - x[:-k], y[k:] - y[:-k]
                c = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dy) >= 0.05)
                row[name][f"gain_lose_{w}s"] = cells(dx[c] < 0, dy[c] < 0)["balanced"]
        report[f"lap {lap_no}"] = row
        print(f"\nlap {lap_no} ({b - a:.2f} s) against lap {r + 1} ({S.T:.2f} s); at the line PRIMAL {primal[-1]:+.2f}, "
              f"RaceChrono {rc[-1]:+.2f}, truth {true[-1]:+.2f} s; truth glitches left out: {100 * (1 - sane.mean()):.1f}%")
        for name in pairs:
            x = row[name]
            print(f"  {name:<22} median {x['median_ms']:4.0f} ms, p90 {x['p90_ms']:4.0f}, >100 ms {100 * x['over_100ms']:5.1f}%; "
                  f"gaining/losing agree (balanced) 2 s {100 * x['gain_lose_2s']:4.1f}%, 5 s {100 * x['gain_lose_5s']:4.1f}%")
        ax = fig.add_subplot(len(S.laps) - 1, 1, n + 1)
        ax.plot(t - a, np.where(sane, true, np.nan), color="#222222", lw=2.0, label="truth (video features; its glitches left out)")
        ax.plot(t - a, rc, color="#d9534f", lw=1.4, label="RaceChrono (GPS)")
        ax.plot(t - a, primal, color="#2c7fb8", lw=1.4, label="PRIMAL (camera)")
        ax.axhline(0, color="#999999", lw=0.8)
        lo, hi = np.nanpercentile(np.r_[primal, rc], [0.5, 99.5])
        ax.set_ylim(lo - 0.25, hi + 0.25)
        ax.set_title(f"lap {lap_no} against lap {r + 1}: delta over the lap", fontsize=11)
        ax.set_xlabel("lap time, s"); ax.set_ylabel("delta, s")
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "racechrono_compare.png", dpi=110)
    (ROOT / "racechrono_compare.json").write_text(json.dumps({"run": run, "crossings": CROSSINGS_S, "report": report}, indent=2))


if __name__ == "__main__":
    main()
