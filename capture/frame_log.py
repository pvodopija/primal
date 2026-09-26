"""
Per-frame telemetry from the timecode app, joined onto decoded labels.

    python -m capture.frame_log check data/sessions/<id>

The timecode app writes, for every rendered frame, AC's speed, G-forces, local
velocity and angular velocity, and two of AC's own times: `sim_ms` for the frame
and `phys_ms` for the physics state it shows. Rows carry the same counter the
barcode encodes, so joining on it attaches telemetry to exactly the frames the
capture kept, with no clock involved. `session import` copies the log into the
session as `frame_log/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(session: Path) -> pd.DataFrame:
    """Decoded labels with telemetry columns joined on the frame counter."""
    labels = pd.read_parquet(session / "labels.parquet")
    chunks = sorted((session / "frame_log").glob("*.csv"))
    if not chunks:
        raise SystemExit(f"{session / 'frame_log'} has no chunks")
    log = pd.concat((pd.read_csv(c) for c in chunks), ignore_index=True)
    log = log.drop_duplicates("counter", keep="last")
    return labels.merge(log, on="counter", how="left", validate="many_to_one")


def cmd_check(args: argparse.Namespace) -> None:
    session = Path(args.session)
    df = load(session)
    valid = df[df.valid]
    joined = valid[valid.sim_ms.notna()]
    print(f"{len(valid)} labelled frames, {len(joined)} with telemetry "
          f"({100 * len(joined) / max(len(valid), 1):.2f}%)")
    if joined.empty:
        return
    # The logged camera position must match the barcode's to within its 20-bit
    # quantisation, or the join is wrong.
    ds = np.abs(((joined.cam_s - joined.spline_pos) + 0.5) % 1.0 - 0.5)
    print(f"logged vs decoded s: max difference {ds.max():.2e} (one barcode step is {1 / (2**20 - 1):.2e})")
    lag = joined.sim_ms - joined.phys_ms
    print(f"physics state age at render, ms: median {lag.median():.2f}, p95 {lag.quantile(.95):.2f}, max {lag.max():.2f}")
    print(f"speed km/h: {joined.speed_kmh.min():.1f} to {joined.speed_kmh.max():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    check = sub.add_parser("check", help="coverage and consistency of a session's telemetry join")
    check.add_argument("session")
    check.set_defaults(func=cmd_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
