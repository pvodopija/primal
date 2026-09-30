from types import SimpleNamespace

import numpy as np
import torch

from train.eval import LAG_HISTORY_S, LAG_MIN_S, STREAM_HZ, _lag_corrected

N_BINS, LAP_S, DELAY = 1000, 80.0, 11


def _stream(truth: np.ndarray, lag_bins: float, lead_bins: float) -> dict:
    grid = SimpleNamespace(time_s=np.arange(N_BINS) * LAP_S / N_BINS, lap_time_s=LAP_S)
    valid = np.ones(truth.size, bool)
    valid[-DELAY:] = False  # the lap ends before these frames' following clips
    return {
        "grid": grid,
        "single": torch.from_numpy((truth - lag_bins) % N_BINS),
        "single_backward": torch.from_numpy((truth + lead_bins) % N_BINS),
        "backward_valid": valid,
        "backward_delay_ticks": DELAY,
    }


def _circular_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + N_BINS / 2) % N_BINS - N_BINS / 2


def test_symmetric_lag_is_removed_once_the_backward_readings_arrive():
    truth = np.linspace(0, N_BINS, 1200, endpoint=False)
    stream = _stream(truth, lag_bins=4.0, lead_bins=4.0)
    out = _lag_corrected(stream, stream["single"], k=0.5).numpy()
    settled = DELAY + int(LAG_MIN_S * STREAM_HZ)
    assert np.allclose(_circular_gap(out[settled:], truth[settled:]), 0.0, atol=1e-3)
    # before any following clip has been seen, nothing is known and nothing moves
    assert np.allclose(_circular_gap(out[:DELAY], (truth[:DELAY] - 4.0) % N_BINS), 0.0, atol=1e-3)


def test_correction_wraps_through_the_finish_line():
    truth = (np.linspace(0, 300, 600) + 850) % N_BINS  # crosses bin 0 halfway
    stream = _stream(truth, lag_bins=6.0, lead_bins=6.0)
    out = _lag_corrected(stream, stream["single"], k=0.5).numpy()
    settled = DELAY + int(LAG_MIN_S * STREAM_HZ)
    assert np.abs(_circular_gap(out[settled:], truth[settled:])).max() < 1e-3


def test_only_frames_older_than_the_delay_are_used():
    truth = np.linspace(0, N_BINS, 1200, endpoint=False)
    stream = _stream(truth, lag_bins=4.0, lead_bins=4.0)
    # a burst of wild backward readings in the last DELAY ticks must not move tick i
    i = 500
    back = stream["single_backward"].numpy().copy()
    back[i - DELAY + 1 : i + 1] = (back[i - DELAY + 1 : i + 1] + 300) % N_BINS
    stream["single_backward"] = torch.from_numpy(back)
    out = _lag_corrected(stream, stream["single"], k=0.5).numpy()
    assert abs(_circular_gap(out[i : i + 1], truth[i : i + 1])[0]) < 1e-3
    assert int(LAG_HISTORY_S * STREAM_HZ) > DELAY
