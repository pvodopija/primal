"""
Estimator behaviour on synthetic belief streams, where the truth is exact.

Run: python -m tests.test_estimator
"""

from __future__ import annotations

import numpy as np

from train.estimator import ProgressEstimator

LENGTH, N_BINS, DT, SPEED = 1500.0, 750, 1.0 / 15.0, 30.0


def _peak(n_bins: int, centre: float, sigma: float = 2.0) -> np.ndarray:
    offset = (np.arange(n_bins) - centre + n_bins / 2) % n_bins - n_bins / 2
    b = np.exp(-0.5 * (offset / sigma) ** 2)
    return b / b.sum()


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % LENGTH
    return min(d, LENGTH - d)


def _truth(tick: int) -> float:
    return (100.0 + SPEED * DT * tick) % LENGTH


def test_rides_through_confidently_wrong_ticks() -> None:
    rng = np.random.default_rng(0)
    est = ProgressEstimator(np.arange(N_BINS) * LENGTH / N_BINS, LENGTH, seed=1)
    errors, single = [], []
    for tick in range(450):
        true_bin = _truth(tick) / LENGTH * N_BINS
        # One tick in ten, the aligner is sure the kart is somewhere else.
        wrong = rng.random() < 0.1
        centre = (true_bin + rng.uniform(150, 600)) % N_BINS if wrong else true_bin + rng.normal(0, 0.7)
        e = est.step(_peak(N_BINS, centre), DT)
        if tick >= 30:
            errors.append(_gap(e.position_m, _truth(tick)))
            single.append(_gap(centre * LENGTH / N_BINS, _truth(tick)))
    errors = np.array(errors)
    assert np.max(single) > 250, "the stream should contain real jumps"
    assert np.median(errors) < 1.5, f"median {np.median(errors):.2f} m"
    assert errors.max() < 10.0, f"a wrong tick moved the estimate {errors.max():.1f} m"


def test_dead_reckons_through_a_blackout_then_recovers() -> None:
    est = ProgressEstimator(np.arange(N_BINS) * LENGTH / N_BINS, LENGTH, seed=2)
    flat = np.full(N_BINS, 1.0 / N_BINS)
    during, after = [], []
    for tick in range(260):
        blind = 120 <= tick < 150          # two seconds looking at the kart alongside
        belief = flat if blind else _peak(N_BINS, _truth(tick) / LENGTH * N_BINS)
        e = est.step(belief, DT)
        if blind:
            during.append(_gap(e.position_m, _truth(tick)))
        elif tick >= 170:
            after.append(_gap(e.position_m, _truth(tick)))
    # Speed was known going in, so position is carried, not lost.
    assert max(during) < 15.0, f"drifted {max(during):.1f} m while blind"
    assert max(after) < 3.0, f"not recovered: {max(after):.1f} m"


def test_resolves_an_ambiguous_start() -> None:
    est = ProgressEstimator(np.arange(N_BINS) * LENGTH / N_BINS, LENGTH, seed=3)
    for tick in range(60):
        true_bin = _truth(tick) / LENGTH * N_BINS
        belief = _peak(N_BINS, true_bin)
        if tick < 5:
            # Two corners that look alike; the evidence cannot yet tell them apart.
            belief = 0.5 * belief + 0.5 * _peak(N_BINS, (true_bin + 300) % N_BINS)
        elif tick < 25:
            # The look-alike keeps showing up, but it cannot move like a kart.
            belief = 0.7 * belief + 0.3 * _peak(N_BINS, (true_bin + 300 + tick * 7) % N_BINS)
        e = est.step(belief, DT)
    assert _gap(e.position_m, _truth(59)) < 2.0
    assert e.confidence > 0.8


def test_non_uniform_bins_map_back_to_position() -> None:
    # Time-axis bins are dense in slow sections: the first half of the lap
    # holds three quarters of the bins.
    first = np.linspace(0.0, LENGTH / 2, int(N_BINS * 0.75), endpoint=False)
    second = np.linspace(LENGTH / 2, LENGTH, N_BINS - first.size, endpoint=False)
    pos = np.concatenate([first, second])
    est = ProgressEstimator(pos, LENGTH, seed=4)
    for tick in range(120):
        truth = _truth(tick)
        centre = float(np.interp(truth, np.append(pos, LENGTH), np.arange(N_BINS + 1)))
        e = est.step(_peak(N_BINS, centre), DT)
    assert _gap(e.position_m, _truth(119)) < 1.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all estimator tests passed")
