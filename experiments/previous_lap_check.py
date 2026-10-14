"""
Every lap against the lap before it (as lap timers often show), PRIMAL's delta at the line
against the transponder's difference of the two lap times: an independent check at one point
a lap, with a different reference for every lap. The same check against the session's fastest
lap is in experiments/session_bars.py.

Usage: REAL_FOOTAGE=night1 PYTHONPATH=. python experiments/previous_lap_check.py RUN [RUN ...]
"""
from __future__ import annotations

import copy
import sys

import cv2
import numpy as np
import torch

from experiments.real_demo import lap_series, setup
from experiments.real_footage import _S, BIN_S, VIDEO


class _NoTruth(dict):
    """lap_series wants an ORB truth per lap; none is used here (it fails at night)."""
    def __getitem__(self, key):
        return np.array([0.0, 1e-9])


def with_reference(S, r: int):
    """The analysis with lap r (0-based) as the reference."""
    S = copy.copy(S)
    S.r, (S.r0, r1) = r, S.laps[r]
    S.T = r1 - S.r0
    S.n = int(round(S.T / BIN_S))
    bin_frame = np.array([int(np.argmin(np.abs(S.t - (S.r0 + i * S.T / S.n)))) for i in range(S.n)])
    S.grid = copy.copy(S.grid)
    S.grid.time_s, S.grid.lap_time_s, S.grid.n_bins = np.arange(S.n) * S.T / S.n, S.T, S.n
    S.grid.pos_m, S.grid.track_length_m = np.arange(S.n, dtype=float), float(S.n)
    S.ref = S.desc[torch.from_numpy(bin_frame).to(S.device)]
    S.truth = _NoTruth()
    return S


def main() -> None:
    truth = _S["lap_times"]
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    for run in sys.argv[1:]:
        S = setup(run)
        errors = []
        print(f"\n{run}: each lap against the one before it")
        for lap_no in range(2, len(S.laps) + 1):
            L = lap_series(with_reference(S, lap_no - 2), lap_no, fps)
            primal, real = float(np.median(L.delta[-8:])), truth[lap_no - 1] - truth[lap_no - 2]
            errors.append(1000 * (primal - real))
            print(f"  lap {lap_no:2d} against lap {lap_no - 1:2d}: PRIMAL {primal:+.3f} s, transponder {real:+.3f} s, error {errors[-1]:+4.0f} ms")
        e = np.abs(np.array(errors))
        print(f"  median |error| {np.median(e):.0f} ms, max {e.max():.0f} ms over {e.size} laps")


if __name__ == "__main__":
    main()
