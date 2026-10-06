"""
Trading delay for accuracy: show the delta of a moment L seconds ago, smoothed over the
frames on both sides of it (a centred running median over ±L), from the tracker's delta
or from the single clips'. Against the live delta, and against smoothing the past only
(a trailing median, no delay: does any gain come from seeing ahead or just from
averaging?). Scored against the truth at the moment shown: |error|, share over 100 ms,
and the green-or-red questions of experiments/real_signs.py (truth clear, balanced).

Sources: the unseen AC tracks at an imperfect pace (experiments/sim_signs.lap_deltas),
or real footage (experiments/real_demo.lap_series). Every method on the same ticks: 2 s
trimmed at each end of each lap.

Usage: PYTHONPATH=. python experiments/delayed_delta.py sim RUN
       REAL_FOOTAGE=BHGP REAL_CROP=full PYTHONPATH=. python experiments/delayed_delta.py real RUN
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.real_signs import cells

HZ = 15.0
DELAYS = (0.25, 0.5, 1.0, 2.0)
TRIM = int(2.0 * HZ)


def methods(delta: np.ndarray, single: np.ndarray) -> dict[str, np.ndarray]:
    d, s = pd.Series(delta), pd.Series(single)
    out = {"live (now)": d}
    for L in (0.5, 1.0):
        out[f"no delay, past {L:g} s median"] = d.rolling(int(round(L * HZ)) + 1).median()
    for L in DELAYS:
        k = 2 * int(round(L * HZ)) + 1
        out[f"{L:g} s late, tracker"] = d.rolling(k, center=True).median()
        out[f"{L:g} s late, single clips"] = s.rolling(k, center=True).median()
    return {name: v.to_numpy()[TRIM:-TRIM] for name, v in out.items()}


def laps(source: str, run: str):
    if source == "sim":
        import torch

        from experiments.sim_signs import lap_deltas
        from experiments.tracker_live_fixes import clocks
        from train.eval import _stream_pairs, load_model
        from train.train import default_device
        device = torch.device(default_device())
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        rng = np.random.default_rng(0)
        for gate, reference, live in _stream_pairs(Path("data/packed_ac_v3"), "time", 99):
            if gate == "G2":
                lap = lap_deltas(model, int(payload["args"]["clip_len"]), reference, live, device, rng, None)
                yield lap["delta"], lap["single"], lap["true"]
    else:
        import cv2

        from experiments.real_demo import lap_series, setup
        from experiments.real_footage import VIDEO
        S = setup(run)
        fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
        for lap_no in [k for k in range(1, len(S.laps) + 1) if k != S.r + 1]:
            L = lap_series(S, lap_no, fps)
            ok = np.isfinite(L.true)
            yield L.delta[ok], L.single[ok], L.true[ok]


def main() -> None:
    source, run = sys.argv[1], sys.argv[2]
    rows: dict[str, list] = {}
    for delta, single, true in laps(source, run):
        tr = true[TRIM:-TRIM]
        for name, est in methods(delta, single).items():
            rows.setdefault(name, []).append((est, tr))
    report = {}
    print(f"\n{source}, {run}: the delta shown against the truth at the moment it shows")
    print(f"  {'method':<30}{'median':>8}{'>100 ms':>9}{'ahead/behind':>14}{'gain/lose 2 s':>15}{'gain/lose 5 s':>15}")
    for name, parts in rows.items():
        est = np.concatenate([p[0] for p in parts]); tr = np.concatenate([p[1] for p in parts])
        err = np.abs(est - tr) * 1000
        r = {"median_ms": float(np.median(err)), "over_100ms": float(np.mean(err > 100))}
        clear = np.abs(tr) >= 0.1
        r["ahead_behind"] = cells(est[clear] < 0, tr[clear] < 0)["balanced"]
        for w in (2, 5):
            k = int(w * HZ)
            de = np.concatenate([p[0][k:] - p[0][:-k] for p in parts]); dt = np.concatenate([p[1][k:] - p[1][:-k] for p in parts])
            c = np.abs(dt) >= 0.05
            r[f"gain_lose_{w}s"] = cells(de[c] < 0, dt[c] < 0)["balanced"]
        report[name] = r
        print(f"  {name:<30}{r['median_ms']:>6.0f} ms{100 * r['over_100ms']:>8.1f}%{100 * r['ahead_behind']:>13.1f}%"
              f"{100 * r['gain_lose_2s']:>14.1f}%{100 * r['gain_lose_5s']:>14.1f}%")
    out = Path("runs") / run / f"delayed_delta_{source}.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
