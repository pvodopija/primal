"""
A release with one session setting changed and nothing else: a copy of RELEASE_DIR named NAME,
the same models and golden frames, release.json's session SETTING set to VALUE (noted under
"derived"), and the golden session replay's Python runs recomputed with it (train/release.py's
seeds and summary, on the release's own replay descriptors), so the app's golden test still
holds it to Python. 2026-10-16: primal-qat8-lb075-20261016, loop_back 0.85 -> 0.75 (loop_back_sweep.py).

Usage: PYTHONPATH=. python experiments/release_retune.py RELEASE_DIR NAME SETTING VALUE
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from train.eval import load_model
from train.release import REPLAY_SEEDS, session_summary
from train.session import Session, SessionConfig


def main() -> None:
    source, name, setting, value = Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
    out = source.parent / name
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(source, out)
    release = json.loads((out / "release.json").read_text())
    old = release["session"][setting]
    release["session"][setting] = value
    release["derived"] = {"from": source.name, "changed": {f"session.{setting}": [old, value]},
                          "made": time.strftime("%Y-%m-%d %H:%M")}
    release["name"] = name
    (out / "release.json").write_text(json.dumps(release, indent=2))

    index = json.loads((out / "golden" / "index.json").read_text())
    replay = index["session_replay"]
    d = torch.from_numpy(np.fromfile(out / "golden" / "replay_descriptors.f32", dtype=np.float32).reshape(replay["descriptors"]))
    t = np.fromfile(out / "golden" / "replay_times.f64", dtype=np.float64)
    model, payload = load_model(Path("runs") / release["run"] / "best.pt", torch.device("cpu"))
    config = replace(SessionConfig(), **{setting: value})
    runs = []
    for seed in REPLAY_SEEDS:
        session, events = Session(model, int(payload["args"]["clip_len"]), torch.device("cpu"), config=config, seed=seed), []
        for i, s_ in enumerate(t):
            if session.state == "ready" and s_ >= replay["start_s"]:
                session.start(float(s_))
            events += session.push(d[i], float(s_))
        runs.append(session_summary(events))
        ref = runs[-1]["reference"]
        print(f"seed {seed}: reference {ref['lap_s']:.2f} s by {ref['placed_by']}, laps {[round(l['lap_s'], 2) for l in runs[-1]['laps']]}")
    replay["python"] = runs
    (out / "golden" / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{out}: session.{setting} {old} -> {value}")


if __name__ == "__main__":
    main()
