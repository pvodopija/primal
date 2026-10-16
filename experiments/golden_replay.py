"""
A session-replay golden from any footage, written into a copy of a release folder, so the app's
own Swift engine replays it (hotlapp's PrimalModel test "A whole session replayed ...", run in a
worktree): the release's PyTorch descriptors for the frames between START_S and END_S (as
experiments/closure_check.py caches them), Python's session runs on them (train/release.py's
seeds and summary), START at START_S. Where Python closes and the app does not, the Swift port
differs from train/session.py.

Usage: PYTHONPATH=. python experiments/golden_replay.py RELEASE_FOLDER FRAMES.npz START_S END_S
       (RELEASE_FOLDER: the copy inside the app worktree, e.g. <wt>/hotlapp_V1.0/Models/primal-...)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from train.eval import load_model
from train.release import REPLAY_SEEDS, encode_frames, session_summary, write_f32, write_f64
from train.session import Session


def main() -> None:
    folder, cache = Path(sys.argv[1]), Path(sys.argv[2])
    start_s, end_s = float(sys.argv[3]), float(sys.argv[4])
    run = json.loads((folder / "release.json").read_text())["run"]
    device = torch.device("cpu")
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    clip_len = int(payload["args"]["clip_len"])
    z = np.load(cache)
    use = np.flatnonzero((z["t"] >= start_s - 0.5) & (z["t"] <= end_s))
    t = z["t"][use].astype(np.float64)
    with torch.no_grad():
        d = torch.cat([encode_frames(model, z["frames"][use][i:i + 256], device) for i in range(0, len(use), 256)])
    runs = []
    for seed in REPLAY_SEEDS:
        session, events = Session(model, clip_len, device, seed=seed), []
        for i, s_ in enumerate(t):
            if session.state == "ready" and s_ >= start_s:
                session.start(float(s_))
            events += session.push(d[i], float(s_))
        runs.append(session_summary(events))
        ref = runs[-1]["reference"]
        print(f"{run} seed {seed}: reference {ref['lap_s'] if ref else None}, laps {[round(l['lap_s'], 2) for l in runs[-1]['laps']]}")
    index_path = folder / "golden" / "index.json"
    index = json.loads(index_path.read_text())
    index["session_replay"] = {
        "descriptors": write_f32(folder / "golden" / "replay_descriptors.f32", d.numpy()),
        "times": write_f64(folder / "golden" / "replay_times.f64", t),
        "start_s": start_s, "true_crossings_s": [], "python": runs, "source": str(cache),
        "tolerance": index["session_replay"]["tolerance"]}
    index_path.write_text(json.dumps(index, indent=2))
    print(f"wrote {folder}/golden: {len(t)} frames from {start_s} s to {end_s} s")


if __name__ == "__main__":
    main()
