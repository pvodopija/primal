from types import SimpleNamespace

import numpy as np

from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME, _run_estimator

N_BINS, LAP_S, HZ = 600, 60.0, 15.0


def _stream(rate: float, noise_bins: float, seed: int = 0) -> tuple[dict, np.ndarray]:
    """A live lap driven at `rate` x the reference's pace, seen through noisy beliefs."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 40.0, 1 / HZ)
    truth = (t * rate / LAP_S * N_BINS) % N_BINS
    bins = np.arange(N_BINS)
    centre = (truth + rng.normal(0, noise_bins, t.size)) % N_BINS
    offset = (bins[None, :] - centre[:, None] + N_BINS / 2) % N_BINS - N_BINS / 2
    belief = np.exp(-0.5 * (offset / 3.0) ** 2)
    belief /= belief.sum(1, keepdims=True)
    # time bins: evenly spaced in reference time; positions deliberately uneven, as on a track
    time_s = bins * LAP_S / N_BINS
    pos_m = np.cumsum(np.r_[0, 2.0 + np.sin(bins[:-1] / 20.0)])
    grid = SimpleNamespace(time_s=time_s, lap_time_s=LAP_S, pos_m=pos_m, track_length_m=float(pos_m[-1] + 2.0))
    return {"grid": grid, "belief": belief, "t": t}, truth


def _error_bins(out: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs((out - truth + N_BINS / 2) % N_BINS - N_BINS / 2)[int(2 * HZ):]


def test_reference_time_tracks_a_steady_pace_through_noisy_beliefs():
    stream, truth = _stream(rate=1.03, noise_bins=4.0)
    out = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME), None, None, 0, reference_time=True).numpy()
    assert np.median(_error_bins(out, truth)) < 1.0  # the pace prior averages the noise away


def test_reference_time_beats_metres_when_the_pace_follows_the_reference():
    stream, truth = _stream(rate=1.0, noise_bins=4.0, seed=1)
    in_time = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME), None, None, 0, reference_time=True).numpy()
    in_metres = _run_estimator(stream, EstimatorConfig(), None, None, 0).numpy()
    assert np.median(_error_bins(in_time, truth)) < np.median(_error_bins(in_metres, truth))
