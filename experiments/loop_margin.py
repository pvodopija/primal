"""
How much room a model leaves the session's loop closure (train/session.py: the last loop_k
frames against the reference's first loop_k, mean cosine, must fall under loop_away and then
rise over loop_back, 0.85, near the line): START a few seconds before a line crossing, the
score's peak within PEAK_S of the next crossing (the room is the peak less loop_back) and its
typical value mid-lap. A model whose peak sits just over 0.85 closes on clean frames and not on
a camera's (2026-10-16: primal-qat8 in the app never closed on a night video filmed off a screen,
peak 0.874 against v3's 0.943).

Usage: PYTHONPATH=. python experiments/loop_margin.py RUN [RUN ...]   (every real-footage set with crossings)
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from train.dataset import _to_chw
from train.eval import load_model
from train.session import SessionConfig

SETS = ("night1", "indoor-pov", "BHGP", "race-chrono", "draggy", "adventure-solo")
LEAD_S, PEAK_S = 3.0, 3.0


def scores(d: np.ndarray, t: np.ndarray, start: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    s0 = int(np.searchsorted(t, start))
    first = d[s0:s0 + k]
    idx = np.arange(s0 + k, len(d))
    return np.array([np.mean(np.sum(d[i - k + 1:i + 1] * first, 1)) for i in idx]), t[idx]


def main() -> None:
    runs = sys.argv[1:]
    cfg = SessionConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = {r: load_model(Path("runs") / r / "best.pt", device)[0] for r in runs}
    report = {}
    for name in SETS:
        os.environ["REAL_FOOTAGE"] = name
        import experiments.real_footage as rf
        rf = importlib.reload(rf)
        try:
            f, t = rf.frames()
        except Exception as e:  # a set whose video is not on this machine
            print(f"{name}: skipped ({e})")
            continue
        cr = list(rf.CROSSINGS_S)
        report[name] = {}
        for r, m in models.items():
            with torch.no_grad():
                d = torch.cat([m.encoder(torch.from_numpy(_to_chw(f[i:i + 256])).to(device)).cpu() for i in range(0, len(f), 256)]).numpy()
            peaks, mids = [], []
            for a, b in zip(cr[:-1], cr[1:]):
                s, st = scores(d, t, a - LEAD_S, cfg.loop_k)
                near = np.abs(st - (b - LEAD_S)) <= PEAK_S
                mid = (st > a + 5) & (st < b - 8)
                if near.any() and mid.any():
                    peaks.append(float(s[near].max()))
                    mids.append(float(np.median(s[mid])))
            report[name][r] = {"peaks": peaks, "mid_median": mids}
            p = np.array(peaks)
            print(f"{name:<15} {r:<26} peak back at the line: median {np.median(p):.3f}, min {p.min():.3f}, "
                  f"under {cfg.loop_back} on {np.sum(p < cfg.loop_back)}/{len(p)} laps; mid-lap median {np.median(mids):.3f}", flush=True)
    Path("runs/lc_logs").mkdir(parents=True, exist_ok=True)
    Path("runs/lc_logs/loop_margin.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
