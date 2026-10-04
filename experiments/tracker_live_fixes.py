"""
Tracker settings against what the live overlay found (docs/handoff.md, 2026-10-04):
the tracker creeps forward when the kart stands still (no particle may stop), and lags
seconds behind when the live pace is far above the reference's (particles may not
exceed 1.6x its pace).

Unseen tracks' laps (G2) against their references, MobileNet, replayed as
  - steady: as recorded;
  - imperfect: experiments/pace_warp.py, ±5% pace wander and a mistake a minute;
  - stop: as recorded until 40% of the lap, then 20 s standing still, then on;
  - fast: the first 10 s at 2.5x the recording's pace (a reference with a slow start).
Share of ticks over 100 ms after acquisition, and for stop / fast the median error in
the 20 s after the stop began / during the fast stretch. Also how well "agreement" (the
share of the last second's single-clip reads within 0.25 s of the tracker) picks out
ticks more than 300 ms off (AUROC).

Usage: PYTHONPATH=. python experiments/tracker_live_fixes.py RUN   (writes runs/RUN/tracker_live_fixes.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.live_logs import auroc
from experiments.pace_warp import warped_stream
from train.estimator import EstimatorConfig
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, _run_estimator, _stream_errors, _stream_pairs, load_model
from train.model import _interp_circular
from train.train import default_device

# "current" was REFERENCE_TIME_TWO_MODES before this test; it now equals "can stop, up to 4x".
# Measured too and dropped: a particle going off-script drawing a fresh speed in 0-4x, with
# following speeds relaxing to 1 (steady 14-16%, imperfect 23-24%, stop 37-38%).
CONFIGS = {
    "current": {**REFERENCE_TIME_TWO_MODES, "speed_range": (0.05, 1.6), "free_speed_floor": 0.05},
    "can stop": {**REFERENCE_TIME_TWO_MODES, "speed_range": (0.0, 1.6), "free_speed_floor": 0.0},
    "can stop, up to 4x": {**REFERENCE_TIME_TWO_MODES, "speed_range": (0.0, 4.0), "free_speed_floor": 0.0},
}
STOP_AT, STOP_S, FAST_S, FAST_X = 0.4, 20.0, 10.0, 2.5


def clocks(duration: float) -> dict:
    play = np.arange(0.0, duration + STOP_S + 1.0, 0.01)
    stop = np.where(play < STOP_AT * duration, play, np.where(play < STOP_AT * duration + STOP_S, STOP_AT * duration, play - STOP_S))
    fast = np.where(play < FAST_S, FAST_X * play, FAST_X * FAST_S + (play - FAST_S))
    keep = lambda c: (play[c <= duration], c[c <= duration])
    return {"steady": keep(play), "imperfect": None, "stop": keep(stop), "fast": keep(fast)}


def agreement(stream: dict, bins: torch.Tensor) -> np.ndarray:
    g = stream["grid"]
    times = torch.from_numpy(g.time_s.astype(np.float32))
    est = _interp_circular(times, bins.float(), g.lap_time_s).numpy()
    single = _interp_circular(times, stream["single"].float(), g.lap_time_s).numpy()
    near = (np.abs((single - est + g.lap_time_s / 2) % g.lap_time_s - g.lap_time_s / 2) < 0.25).astype(float)
    return np.convolve(near, np.ones(15) / 15)[: near.size]


def main() -> None:
    run = sys.argv[1]
    device = torch.device(default_device())
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    clip_len = int(payload["args"]["clip_len"])
    errs = {(s, c): [] for s in ("steady", "imperfect", "stop", "fast") for c in CONFIGS}
    window = {(s, c): [] for s in ("stop", "fast") for c in CONFIGS}
    agree, bad = [], []
    rng = np.random.default_rng(0)
    for gate, reference, live in _stream_pairs(Path("data/packed_ac_v3"), "time", 99):
        if gate != "G2":
            continue
        duration = float(live.t()[-1])  # clocks count recording time from 0, as pace_warp's do
        for scenario, clock in clocks(duration).items():
            stream = warped_stream(model, clip_len, reference, live, "time", device, rng, 0.05, 1.0, clock=clock)
            for name, cfg in CONFIGS.items():
                bins = _run_estimator(stream, EstimatorConfig(**cfg), None, None, 0, reference_time=True)
                e = _stream_errors(stream, bins)[1]
                errs[(scenario, name)].append(e)
                skip = int(ACQUISITION_S * STREAM_HZ)
                t = stream["t"][skip:]
                if scenario == "stop":
                    start = STOP_AT * duration
                    window[(scenario, name)].append(e[(t >= start) & (t < start + STOP_S)])
                elif scenario == "fast":
                    window[(scenario, name)].append(e[t < FAST_S])
                if name == "current" and scenario in ("steady", "imperfect"):
                    agree.append(agreement(stream, bins)[skip:])
                    bad.append(e > 300)
        print(f"  {live.track[:24]:<24} {live.lap_id[-5:]}", flush=True)
    report = {}
    print(f"\n{run}: share of ticks over 100 ms (median |error| in the stop / fast stretch)")
    print(f"  {'':<22}" + "".join(f"{s:>22}" for s in ("steady", "imperfect", "stop", "fast")))
    for name in CONFIGS:
        row = {s: float(np.mean(np.concatenate(errs[(s, name)]) > 100)) for s in ("steady", "imperfect", "stop", "fast")}
        row.update({f"{s} window median ms": float(np.median(np.concatenate(window[(s, name)]))) for s in ("stop", "fast")})
        report[name] = row
        print(f"  {name:<22}{100 * row['steady']:>21.1f}%{100 * row['imperfect']:>21.1f}%"
              f"{100 * row['stop']:>12.1f}% ({row['stop window median ms']:>4.0f} ms){100 * row['fast']:>12.1f}% ({row['fast window median ms']:>4.0f} ms)")
    a = auroc(np.concatenate(agree), np.concatenate(bad))
    report["agreement_auroc_over_300ms"] = a
    print(f"  agreement as confidence, AUROC for ticks over 300 ms (steady + imperfect, current tracker): {a:.2f}")
    (Path("runs") / run / "tracker_live_fixes.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
