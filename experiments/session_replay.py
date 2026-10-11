"""
A recorded session replayed through train/session.py's Session, as the app would run it
(frames at most 30 a second): where the reference closes, each lap's time and whether it
became the reference, against the laps' true crossings.

  race-chrono  RaceChrono's car laps (data/real-footage/race-chrono, its overlays masked);
               START at lap 1's crossing; the true crossings from RaceChrono's own clock
               (experiments/racechrono_read.py, starts_orb.json)

Usage: PYTHONPATH=. python experiments/session_replay.py [--run v3_mobilenet_lr1_s0]
Writes data/real-footage/race-chrono/session_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("REAL_FOOTAGE", "race-chrono")
from experiments.real_footage import frames as footage_frames  # noqa: E402  (reads REAL_FOOTAGE on import)
from train.eval import load_model
from train.live import encode_frames
from train.session import LapCompleted, ProposalRejected, Reading, ReferenceClosed, Session

ROOT = Path("data/real-footage/race-chrono")
MAX_FPS = 30


def at_most(t: np.ndarray, fps: float) -> np.ndarray:
    """Indices of the frames an app passes on at `fps` at most (by the frames' own times)."""
    keep, last = [], -np.inf
    for i, s in enumerate(t):
        if s - last >= 1 / fps - 0.004:
            keep.append(i)
            last = s
    return np.array(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3_mobilenet_lr1_s0")
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(Path("runs") / a.run / "best.pt", device)
    crossings = json.loads((ROOT / "starts_orb.json").read_text())
    frames, t = footage_frames()  # 148x80, overlays masked (cached by experiments/real_footage.py)
    use = at_most(t, MAX_FPS)
    use = use[(t[use] >= crossings[0] - 0.5) & (t[use] <= crossings[-1] + 6)]
    t, frames = t[use], frames[use]
    with torch.no_grad():
        descriptors = torch.cat([encode_frames(model, frames[i:i + 256], device) for i in range(0, len(frames), 256)])

    session = Session(model, int(payload["args"]["clip_len"]), device)
    events = []
    for i, s in enumerate(t):
        if session.state == "ready" and s >= crossings[0]:
            session.start(float(s))
        events += session.push(descriptors[i], float(s))
    readings = [e for e in events if isinstance(e, Reading)]
    report = {"run": a.run, "true_crossings_s": crossings,
              "true_laps_s": [round(b - a_, 3) for a_, b in zip(crossings, crossings[1:])],
              "events": [dict(type=type(e).__name__, **asdict(e)) for e in events if not isinstance(e, Reading)],
              "readings": len(readings), "shown": float(np.mean([r.shown for r in readings])) if readings else None}
    for e in events:
        if isinstance(e, ReferenceClosed):
            print(f"reference closed at {e.line_s:.2f} s (true {crossings[1]:.2f}), lap {e.lap_s:.2f} s, known at {e.t:.2f}, placed by {e.placed_by}")
        elif isinstance(e, ProposalRejected):
            print(f"proposal {e.proposal_s:.2f} s rejected at {e.t:.2f}")
        elif isinstance(e, LapCompleted):
            truth = crossings[e.lap + 2] if e.lap + 2 < len(crossings) else float("nan")
            print(f"lap {e.lap}: {e.lap_s:.2f} s, line {e.line_s:.2f} (true {truth:.2f}), shown {e.shown_share:.0%}, "
                  f"{'became the reference' if e.became_reference else 'kept the reference'}{', VOID' if e.voided else ''}")
    print(f"true laps: {report['true_laps_s']}; delta shown on {report['shown']:.0%} of {len(readings)} readings")
    (ROOT / "session_replay.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
