"""
Pre-flight check for a new track: are its labels and telemetry sound?

    python -m capture.preflight check data/preflight/<id>

Run on a short test drive (at least one full lap) after `overlay_decode decode`.
A mod track's AI spline defines every label, and a missing or broken one
corrupts all of them, so this checks the labels against physics rather than
against themselves: the decoded track position must advance smoothly, wrap once
per lap where AC counts the lap, and advance at the speed AC reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capture import frame_log
from capture.install_overlay import find_ac_root

JUMP_M = 3.0         # a larger step between consecutive rendered frames is a spline discontinuity
BACKWARD_M = 0.3     # a step back further than this is the label running backwards
EDGE_X = 0.9         # |track x| beyond this means the camera is off the tarmac


def _wrap(ds: np.ndarray) -> np.ndarray:
    return (ds + 0.5) % 1.0 - 0.5


def check(session: Path) -> bool:
    meta = json.loads((session / "run.json").read_text())
    length = float(meta["track_length_m"])
    results: list[tuple[str, bool, str]] = []

    labels = pd.read_parquet(session / "labels.parquet")
    ok = labels.valid.mean()
    results.append(("barcode decodes", ok >= 0.999, f"{100 * ok:.2f}% of {len(labels)} frames"))

    try:
        df = frame_log.load(session)
    except SystemExit:
        df = labels.assign(sim_ms=np.nan)
    df = df[df.valid].drop_duplicates("counter").sort_values("frame_idx")
    has_log = df.sim_ms.notna().any()

    s = df.spline_pos.to_numpy()
    step = np.diff(df.counter.to_numpy())
    ds = _wrap(np.diff(s)) * length / np.maximum(step, 1)  # metres per rendered frame
    jumps = int((np.abs(ds) > JUMP_M).sum())
    back = int((ds < -BACKWARD_M).sum())
    results.append(("no jumps in s", jumps == 0, f"{jumps} steps over {JUMP_M} m per frame, largest {np.abs(ds).max():.2f} m"))
    results.append(("s never runs backwards", back == 0, f"{back} steps back over {BACKWARD_M} m"))

    wraps = np.flatnonzero((s[:-1] > 0.9) & (s[1:] < 0.1))
    bins = np.unique(np.floor(s * length / 2.0)).size / np.ceil(length / 2.0)
    results.append(("s covers the lap", bins >= 0.98, f"{100 * bins:.1f}% of 2 m bins, {len(wraps)} wraps at the finish"))

    if has_log:
        log = df[df.sim_ms.notna()]
        t = log.sim_ms.to_numpy() / 1000
        dist = np.concatenate([[0], np.cumsum(_wrap(np.diff(log.spline_pos.to_numpy())) * length)])
        # Label speed against AC's speed over half-second windows while moving.
        k = 30
        v_label = (dist[k:] - dist[:-k]) / np.maximum(t[k:] - t[:-k], 1e-6)
        v_car = log.speed_kmh.rolling(k + 1).mean().to_numpy()[k:] / 3.6
        moving = v_car > 5
        ratio = v_label[moving] / v_car[moving]
        if ratio.size:
            med, lo, hi = np.median(ratio), np.percentile(ratio, 5), np.percentile(ratio, 95)
            results.append(("s advances at AC's speed", 0.95 <= med <= 1.05,
                            f"label speed / car speed median {med:.3f}, 5-95% {lo:.2f}-{hi:.2f}"))

        # AC's lap counter must tick where s wraps. The camera sits a couple of metres
        # ahead of the car and AC's timing line need not be exactly at s = 0, so allow 10 m.
        if "lap_count" in log:
            ls = log.spline_pos.to_numpy()
            log_wraps = np.flatnonzero((ls[:-1] > 0.9) & (ls[1:] < 0.1)) + 1
            laps = np.flatnonzero(np.diff(log.lap_count.to_numpy()) > 0) + 1
            if len(laps):
                gaps = [np.abs(dist[log_wraps] - dist[i]).min() if len(log_wraps) else np.inf for i in laps]
                results.append(("s wraps where AC counts a lap", max(gaps) < 10,
                                f"{len(laps)} lap(s) counted, each within {max(gaps):.1f} m of an s wrap"))

        if "cam_trk_x" in log:
            x = log.cam_trk_x.abs()
            width = (log.side_l_m + log.side_r_m)
            results.append(("camera stays on the tarmac", (x > EDGE_X).mean() < 0.001,
                            f"|track x| p99 {x.quantile(.99):.2f}, beyond {EDGE_X}: {100 * (x > EDGE_X).mean():.2f}%, "
                            f"track {width.min():.1f}-{width.median():.1f} m wide (min-median)"))

    coverage = df.sim_ms.notna().mean()
    results.append(("telemetry joins", coverage >= 0.98, f"{100 * coverage:.1f}% of decoded frames"))
    root = find_ac_root()
    err = root / "apps" / "lua" / "locamotif_timecode" / "frame_log_error.txt" if root else None
    if err and err.exists():
        results.append(("telemetry logger ran", False, err.read_text().strip()))

    width = max(len(r[0]) for r in results)
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
    all_ok = all(r[1] for r in results)
    print(f"\n{meta['track']}/{meta['track_config'] or 'default'}, {length:.1f} m, {meta['car_model']}: "
          f"{'all checks pass' if all_ok else 'FAILED'}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    cmd = sub.add_parser("check", help="check a decoded test drive")
    cmd.add_argument("session")
    args = parser.parse_args()
    raise SystemExit(0 if check(Path(args.session)) else 1)


if __name__ == "__main__":
    main()
