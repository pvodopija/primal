"""
The camera's GPS speed fed to PRIMAL's tracker (the reference-time particle filter): each tick
the GPS's 2D speed (GPS5, experiments/gpmf_read.py) as of that moment (the last fix, as live),
against each particle's rate through the reference lap times the reference lap's own GPS speed
there. Both laps' speeds come from the same receiver, so its lag and scale cancel. Simulated
(train.eval stream, true speed) a speed was worth the most of anything: unseen tracks 57% ->
12-17% over 100 ms, imperfect pace 22.8% -> 1.9%.

Scored on every lap against the fastest: at the line against the transponder (independent),
and along the lap against the GPS's own position delta (gps_compare.py; not independent of
the speed, the receiver's position and speed share its filter). Conditions: camera only (as
shipped), with the GPS speed at 18 Hz trusted to SIGMAS m/s, the one-mode tracker with it, and
the speed at 1 Hz (a phone's rate, the same receiver). Also, needing no truth, how often the
delta jumps more than JUMP_S within a second.

Usage: REAL_FOOTAGE=night1 PYTHONPATH=. python experiments/gps_speed.py RUN
       (writes ROOT/gps_speed_<run>.json)
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from experiments.gps_compare import gps_delta, gps_xy
from experiments.real_demo import CONF_MIN, lap_series, setup
from experiments.real_footage import _S, ROOT, SUFFIX, VIDEO
from train.eval import REFERENCE_TIME, REFERENCE_TIME_TWO_MODES

SIGMAS = (0.3, 0.6, 1.2, 2.0, 3.0, 5.0)
JUMP_S = 0.3  # a delta moving more than this within a second: no driving does that
EDGE_S = 1.0  # along-lap scores leave a second at each line (the GPS search's wrap)


def speed_source(every: int = 1):
    """Video times -> the GPS's 2D speed as of then (the last fix; every n-th fix for a slower rate)."""
    z = np.load(ROOT / "gpmf.npz")
    t, v = z["GPS5_t"][::every], z["GPS5"][::every, 3]
    return lambda when: v[np.clip(np.searchsorted(t, when, side="right") - 1, 0, len(v) - 1)]


def score(S, fps: float, t: np.ndarray, xy: np.ndarray, **kw) -> dict:
    truth = _S["lap_times"]
    along, start, line, shown, jumps = [], [], [], [], []
    for lap_no in S.live:
        L = lap_series(S, lap_no, fps, **kw)
        g = gps_delta(t, xy, S.laps[S.r], (L.a, L.b), L.ticks)
        e = L.delta - g
        into, left = L.ticks - L.a, L.b - L.ticks
        along.append(e[(into > EDGE_S) & (left > EDGE_S)])
        start.append(e[(into > EDGE_S) & (into <= 3.0)])
        line.append(float(np.median(L.delta[-8:])) - (truth[lap_no - 1] - truth[S.r]))
        shown.append(float(np.mean(L.conf >= CONF_MIN)))
        step = int(round(1 / np.median(np.diff(L.ticks))))
        jumps.append(np.abs(L.delta[step:] - L.delta[:-step]) > JUMP_S)
    a, s, l = np.abs(np.concatenate(along)), np.abs(np.concatenate(start)), np.abs(np.array(line))
    return {"along_median_ms": 1000 * float(np.median(a)), "along_p90_ms": 1000 * float(np.percentile(a, 90)),
            "along_over_100ms": float(np.mean(a > 0.1)), "first_3s_median_ms": 1000 * float(np.median(s)),
            "line_median_ms": 1000 * float(np.median(l)), "line_max_ms": 1000 * float(l.max()),
            "shown_share": float(np.mean(shown)), "jump_share": float(np.mean(np.concatenate(jumps)))}


def main() -> None:
    run = sys.argv[1]
    S = setup(run)
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    t, xy = gps_xy()
    at18, at1 = speed_source(), speed_source(18)
    conditions = {"camera only (as shipped)": {}}
    for s in SIGMAS:
        conditions[f"+ GPS speed 18 Hz, sigma {s} m/s"] = dict(speed_at=at18, speed_sigma=s)
    conditions["+ GPS speed 18 Hz, sigma 2.0, one-mode tracker"] = dict(speed_at=at18, speed_sigma=2.0, config=REFERENCE_TIME)
    for s in (0.6, 2.0, 5.0):
        conditions[f"+ GPS speed 1 Hz (phone rate), sigma {s} m/s"] = dict(speed_at=at1, speed_sigma=s)
    report = {}
    print(f"{run}{SUFFIX}: every lap against the fastest; at the line vs the transponder, along the lap vs the GPS position")
    for name, kw in conditions.items():
        r = score(S, fps, t, xy, **kw)
        report[name] = r
        print(f"  {name:<48} line {r['line_median_ms']:4.0f} / max {r['line_max_ms']:4.0f} ms | along {r['along_median_ms']:4.0f} ms, "
              f"p90 {r['along_p90_ms']:4.0f}, {100 * r['along_over_100ms']:4.1f}% over 100 | first 3 s {r['first_3s_median_ms']:4.0f} ms "
              f"| shown {100 * r['shown_share']:3.0f}% | jumps {100 * r['jump_share']:4.1f}%", flush=True)
    (ROOT / f"gps_speed_{run}{SUFFIX}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    assert REFERENCE_TIME_TWO_MODES  # the shipped tracker, lap_series's default
    main()
