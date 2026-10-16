"""
The app's swap rule (2026-10-17, primal-qat8-lb075-swap0-20261017: any faster lap that is not voided
becomes the reference, session.swap_shown_min 0.8 -> 0.0) on every real session, replayed whole as
the app runs it (START LEAD_S before a line, loop_back 0.75, no hand close), against the shipped
0.8. Per lap: its time, whether it became the reference, its shown share; a lap that became the
reference while more than 1 s from every true lap of the set (a misplaced line: a falsely short
lap would become a broken reference) is flagged.

Usage: PYTHONPATH=. python experiments/swap_rule_check.py RUN   (writes runs/lc_logs/swap_rule_check.json)
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

from experiments.loop_back_sweep import EXTRA, FIRST, LEAD_S, SETS
from train.dataset import _to_chw
from train.eval import load_model
from train.session import LapCompleted, ReferenceClosed, Session, SessionConfig

RULES = {"shown 0.8 (before)": 0.8, "any faster lap (app now)": 0.0}


def main() -> None:
    run = sys.argv[1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    gpu, payload = load_model(Path("runs") / run / "best.pt", device)
    cpu, _ = load_model(Path("runs") / run / "best.pt", torch.device("cpu"))
    clip_len = int(payload["args"]["clip_len"])
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
        sets[name] = (z["frames"], z["t"], 75.0, [49.2] + [48.2, 48.3, 48.4, 48.6, 48.7, 48.9])
    report, flagged = {}, 0
    for name, (f, t, start, true_laps) in sets.items():
        with torch.no_grad():
            d = torch.cat([gpu.encoder(torch.from_numpy(_to_chw(f[i:i + 256])).to(device)).cpu() for i in range(0, len(f), 256)])
        report[name] = {}
        for rule, share in RULES.items():
            config = replace(SessionConfig(), loop_back=0.75, swap_shown_min=share)
            session = Session(cpu, clip_len, torch.device("cpu"), config=config)
            keep = np.flatnonzero(t >= start)
            session.start(float(t[keep[0]]))
            closed, laps = None, []
            for k in keep:
                for e in session.push(d[k], float(t[k])):
                    closed = e if isinstance(e, ReferenceClosed) else closed
                    laps += [e] if isinstance(e, LapCompleted) else []
            rows = []
            for lap in laps:
                true = min(abs(lap.lap_s - y) for y in true_laps) < 1.0
                bad = lap.became_reference and not true
                flagged += bad
                rows.append({"lap_s": float(lap.lap_s), "became_reference": bool(lap.became_reference),
                             "shown_share": float(lap.shown_share), "voided": bool(lap.voided), "true_lap": bool(true), "flag": bool(bad)})
            report[name][rule] = {"reference_lap_s": None if closed is None else float(closed.lap_s), "laps": rows}
            refs = [f"{r['lap_s']:.2f}{'!' if r['flag'] else ''}" for r in rows if r["became_reference"]]
            print(f"{name:<15} {rule:<26} reference {closed.lap_s if closed else float('nan'):6.2f} s; {len(rows)} laps, "
                  f"became the reference: {refs or '-'}; shown {np.median([r['shown_share'] for r in rows]) if rows else float('nan'):.2f} median",
                  flush=True)
    print(f"laps that became the reference while not a true lap: {flagged}")
    Path("runs/lc_logs/swap_rule_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
