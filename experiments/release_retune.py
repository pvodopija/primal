"""
A release with one session setting changed and nothing else: a copy of RELEASE_DIR named NAME,
the same models and golden frames, release.json's session SETTING set to VALUE (noted under
"derived"), and the golden session replay's Python runs recomputed with it (train/release.py's
seeds and summary, on the release's own replay descriptors), so the app's golden test still
holds it to Python. The session is the source release's own settings (not the code's defaults);
a setting it predates is off. 2026-10-16: primal-qat8-lb075-20261016, loop_back 0.85 -> 0.75
(loop_back_sweep.py); 2026-10-17: ...-swap0- (swap_shown_min 0.8 -> 0, the app's rule) and
...-swapguard- (swap_min_of_median 0.9 added, swap_rule_check.py).

Usage: PYTHONPATH=. python experiments/release_retune.py RELEASE_DIR NAME SETTING VALUE
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import fields
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
    old = release["session"].get(setting)  # None: a setting the source release did not have
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
    # the session as the release has it (code defaults moved on since some releases were made); a guard
    # the release predates is off in it
    known = {f.name for f in fields(SessionConfig)}
    legacy = {"swap_min_of_median": 0.0}
    settings = {**legacy, **{k: v for k, v in release["session"].items() if k in known}}
    config = SessionConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()})
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
