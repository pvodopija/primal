"""
Green or red on the unseen AC tracks (G2), the questions of experiments/real_signs.py
with an exact truth: is the driver ahead or behind (the delta's sign), and gaining or
losing over the last 2 s and 5 s (the sign of its change). As recorded (bot laps, which
barely gain or lose) and at an imperfect pace (experiments/pace_warp.py: ±5% wander and a
mistake a minute). Each lap's clock starts at its line crossing, extrapolated from its
first frames, as the reference grid's does.

Usage: PYTHONPATH=. python experiments/sim_signs.py RUN   (writes runs/RUN/sim_signs.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.pace_warp import warped_stream
from experiments.real_signs import cells
from experiments.tracker_live_fixes import agreement, clocks
from train.estimator import EstimatorConfig
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, _run_estimator, _stream_pairs, load_model
from train.model import _interp_circular
from train.train import default_device

CONF_MIN = 0.5


def line_time(live) -> float:
    """Recording time of the line crossing: the first frame's time less its distance past the line at its speed."""
    s, t, length = live.s().astype(np.float64), live.t().astype(np.float64), live.track_length_m
    past = ((s[0] + 0.5) % 1.0 - 0.5) * length
    speed = (((s[10] - s[0]) + 0.5) % 1.0 - 0.5) * length / (t[10] - t[0])
    return float(t[0] - past / speed)


def main() -> None:
    run = sys.argv[1]
    device = torch.device(default_device())
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    clip_len = int(payload["args"]["clip_len"])
    rng = np.random.default_rng(0)
    rows = {(pace, q): [] for pace in ("steady", "imperfect") for q in ("delta", "change 2 s", "change 5 s")}
    for gate, reference, live in _stream_pairs(Path("data/packed_ac_v3"), "time", 99):
        if gate != "G2":
            continue
        clock = clocks(float(live.t()[-1]))
        for pace in ("steady", "imperfect"):
            stream = warped_stream(model, clip_len, reference, live, "time", device, rng, 0.05, 1.0, clock=clock[pace])
            bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
            g = stream["grid"]
            times = torch.from_numpy(g.time_s.astype(np.float32))
            est = _interp_circular(times, bins.float(), g.lap_time_s).numpy().astype(np.float64)
            true = _interp_circular(times, stream["target"].float(), g.lap_time_s).numpy().astype(np.float64)
            # play time runs with recording time at the start, so the line sits at its recording time
            elapsed = stream["t"] - line_time(live)
            wrap = lambda x: (x + g.lap_time_s / 2) % g.lap_time_s - g.lap_time_s / 2
            skip = int(ACQUISITION_S * STREAM_HZ)
            d, tr = wrap(elapsed - est)[skip:], wrap(elapsed - true)[skip:]
            conf = agreement(stream, bins)[skip:]
            rows[(pace, "delta")].append((d, tr, conf))
            for w in (2, 5):
                k = int(w * STREAM_HZ)
                rows[(pace, f"change {w} s")].append((d[k:] - d[:-k], tr[k:] - tr[:-k], conf[k:]))
        print(f"  {live.track[:24]:<24} {live.lap_id[-5:]}", flush=True)
    report = {}
    for (pace, name), parts in rows.items():
        p, q, c = (np.concatenate(x) for x in zip(*parts))
        clear = np.abs(q) >= (0.1 if name == "delta" else 0.05)
        green = lambda x: x < 0
        report[f"{pace}, {name}"] = r = {"every tick": cells(green(p), green(q)), "truth clear": cells(green(p[clear]), green(q[clear])),
                                         "truth clear, shown": cells(green(p[clear & (c >= CONF_MIN)]), green(q[clear & (c >= CONF_MIN)]))}
        label = "ahead or behind" if name == "delta" else f"gaining or losing over {name[7:]}"
        print(f"\n{pace} pace, {label}")
        for part, x in r.items():
            print(f"  {part:<18} {x['ticks']:>6} ticks: agree {100 * x['agree']:5.1f}% (always one colour {100 * x['always one colour']:4.1f}%, "
                  f"balanced {100 * x['balanced']:4.1f}%)")
    (Path("runs") / run / "sim_signs.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
