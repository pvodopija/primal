"""
The live overlay's logs (capture/live_delta.py, data/live_logs/<run>/ticks.csv), summarised:

  - stops: ticks under 5 km/h, the tracker's and the single clips' error;
  - the lean: signed error by place on the lap (20 bands of spline position), single
    clips and tracker, in ms and in metres (error times the live speed), and per
    reference lap, so a lean tied to the scene and one tied to the reference separate;
  - confidence: how well the logged confidence, and the agreement between the tracker
    and the single clips over the last second, pick out ticks more than 300 ms off.

Error = estimated minus true reference time (positive: placed further along, the shown
delta reads faster than the truth).

Usage: PYTHONPATH=. python experiments/live_logs.py RUN [RUN ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def wrap(x: np.ndarray, period: np.ndarray) -> np.ndarray:
    return (x + period / 2) % period - period / 2


def auroc(score: np.ndarray, bad: np.ndarray) -> float:
    """Chance that a bad tick scores lower than a good one (1 = confidence separates them)."""
    if bad.all() or not bad.any():
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    nb, ng = bad.sum(), (~bad).sum()
    return float(1 - (ranks[bad].sum() - nb * (nb + 1) / 2) / (nb * ng))


def main() -> None:
    for run in sys.argv[1:]:
        d = pd.read_csv(Path("data/live_logs") / run / "ticks.csv")
        d = d[np.isfinite(d.true_ref_time_s) & np.isfinite(d.ref_time_s)].copy()
        period = d.reference_lap_s.to_numpy()
        d["err"] = wrap(d.ref_time_s.to_numpy() - d.true_ref_time_s.to_numpy(), period)
        d["err1"] = wrap(d.single_ref_time_s.to_numpy() - d.true_ref_time_s.to_numpy(), period)
        d["m"] = d.err * d.speed_kmh / 3.6
        d["m1"] = d.err1 * d.speed_kmh / 3.6
        print(f"\n== {run}: {len(d)} ticks, error median {1000 * d.err.abs().median():.0f} ms "
              f"(single clips {1000 * d.err1.abs().median():.0f} ms)")

        slow = d.speed_kmh < 5
        if slow.any():
            print(f"  under 5 km/h: {slow.sum()} ticks; tracker |error| median {1000 * d.err[slow].abs().median():.0f} ms, "
                  f"max {1000 * d.err[slow].abs().max():.0f}; single clips median {1000 * d.err1[slow].abs().median():.0f} ms")

        moving = d.speed_kmh > 10
        m = d[moving]
        print(f"  moving, signed median: tracker {1000 * m.err.median():+.0f} ms / {m.m.median():+.2f} m, "
              f"single {1000 * m.err1.median():+.0f} ms / {m.m1.median():+.2f} m; r(error, speed) {np.corrcoef(m.err, m.speed_kmh)[0, 1]:+.2f}")
        band = (m.spline_pos * 20).astype(int).clip(0, 19)
        table = m.groupby(band).agg(single_m=("m1", "median"), tracker_m=("m", "median"),
                                   single_ms=("err1", lambda x: 1000 * x.median()), kmh=("speed_kmh", "median"),
                                   ticks=("err", "size"))
        print("  by place (spline band of 5%): signed median")
        print("  " + table.round(2).to_string().replace("\n", "\n  "))
        refs = m.groupby("reference").agg(single_m=("m1", "median"), tracker_m=("m", "median"), ticks=("err", "size"))
        if len(refs) > 1:
            print("  by reference lap:")
            print("  " + refs.round(2).to_string().replace("\n", "\n  "))

        # agreement: share of the last 15 single reads (1 s) within 0.25 s of the tracker
        near = (wrap(d.single_ref_time_s.to_numpy() - d.ref_time_s.to_numpy(), period) ** 2 < 0.25 ** 2).astype(float)
        d["agree"] = pd.Series(near, index=d.index).rolling(15, min_periods=1).mean()
        bad = d.err.abs().to_numpy() > 0.3
        print(f"  ticks over 300 ms: {100 * bad.mean():.1f}%; AUROC logged confidence {auroc(d.confidence.to_numpy(), bad):.2f}, "
              f"agreement {auroc(d.agree.to_numpy(), bad):.2f}; "
              f"at agreement < 0.5: {100 * (d.agree < 0.5).mean():.1f}% of ticks, of them {100 * bad[d.agree < 0.5].mean():.0f}% over 300 ms")


if __name__ == "__main__":
    main()
