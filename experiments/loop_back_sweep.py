"""
The session's loop closure (train/session.py) at lower loop_back thresholds, replayed whole on
every real-footage set as the app runs it: START LEAD_S before the first line, no hand close,
the confirmation step as shipped. Per set and model: whether the reference closed, its lap
against the true one (the set's crossings), and the laps after it within 1 s of a true lap.
A closure more than 1 s off the true first lap is a false one; laps after it count right within
1 s of any true lap. loop_margin.py showed the shipped 0.85
leaves several sets (and primal-qat8 on night footage) without a closure at all.

Usage: PYTHONPATH=. python experiments/loop_back_sweep.py RUN [RUN ...]   (writes runs/lc_logs/loop_back_sweep.json)
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from train.dataset import _to_chw
from train.eval import load_model
from train.session import LapCompleted, ReferenceClosed, Session, SessionConfig

SETS = ("night1", "indoor-pov", "BHGP", "race-chrono", "draggy", "adventure-solo")
FIRST = {"night1": 1}  # START before this crossing: night1's first is from the pit-exit lane, a view no lap repeats
EXTRA = {"rotax-night": "data/real-footage/rotax-night_148x80.npz"}  # closure_check.py's frames, START 75 s, laps ~48.3 s
THRESHOLDS = (0.85, 0.80, 0.75)
LEAD_S = 3.0


def replay(model, clip_len: int, d: torch.Tensor, t: np.ndarray, start: float, loop_back: float):
    session = Session(model, clip_len, torch.device("cpu"), config=replace(SessionConfig(), loop_back=loop_back))
    keep = np.flatnonzero(t >= start)
    session.start(float(t[keep[0]]))
    closed, laps = None, []
    for k in keep:
        for e in session.push(d[k], float(t[k])):
            closed = e if isinstance(e, ReferenceClosed) else closed
            laps += [e.lap_s] if isinstance(e, LapCompleted) else []
    return closed, laps


def main() -> None:
    runs = sys.argv[1:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = {r: load_model(Path("runs") / r / "best.pt", device) for r in runs}
    sets = {}
    for name in SETS:
        os.environ["REAL_FOOTAGE"] = name
        import experiments.real_footage as rf
        rf = importlib.reload(rf)
        f, t = rf.frames()
        cr = list(rf.CROSSINGS_S)[FIRST.get(name, 0):]
        sets[name] = (f, t, cr[0] - LEAD_S, list(np.diff(cr)))
    for name, path in EXTRA.items():
        z = np.load(path)
        sets[name] = (z["frames"], z["t"], 75.0, [49.2] + [48.3] * 12)  # closure_check: 49.2, then ~48.2-49.0
    report = {}
    for name, (f, t, start, true_laps) in sets.items():
        true_lap = true_laps[0]
        report[name] = {}
        for r, (m, payload) in models.items():
            with torch.no_grad():
                d = torch.cat([m.encoder(torch.from_numpy(_to_chw(f[i:i + 256])).to(device)).cpu() for i in range(0, len(f), 256)])
            cpu_model = load_model(Path("runs") / r / "best.pt", torch.device("cpu"))[0]
            for th in THRESHOLDS:
                closed, laps = replay(cpu_model, int(payload["args"]["clip_len"]), d, t, start, th)
                ok_laps = sum(any(abs(x - y) < 1.0 for y in true_laps[1:]) for x in laps)
                verdict = ("NEVER closed" if closed is None else
                           f"closed {closed.lap_s:6.2f} s ({'right' if abs(closed.lap_s - true_lap) < 1.0 else 'FALSE'}) by {closed.placed_by}")
                report[name][f"{r} {th}"] = {"closed_lap_s": None if closed is None else float(closed.lap_s),
                                             "true_lap_s": float(true_lap), "laps": [float(x) for x in laps],
                                             "laps_right": int(ok_laps)}
                print(f"{name:<15} {r:<26} loop_back {th:.2f}: {verdict:<36} laps right {ok_laps}/{len(laps)} "
                      f"(true reference lap {true_lap:.1f} s)", flush=True)
    Path("runs/lc_logs/loop_back_sweep.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
