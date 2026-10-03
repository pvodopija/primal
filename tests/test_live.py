"""The live loop is the offline evaluation, frame by frame."""
import numpy as np
import torch

from tests.test_train import _dataset
from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME_TWO_MODES, _lap_stream, _run_estimator
from train.live import TRACKER, LiveDelta
from train.model import SequenceAligner


def test_tracker_settings_match_the_evaluation():
    assert TRACKER == REFERENCE_TIME_TWO_MODES


def test_pushing_frames_one_by_one_reproduces_the_offline_stream():
    index = _dataset()
    track = sorted(index.by_track)[0]
    reference, live = [lap for lap in index.by_track[track] if lap.usable_as_reference(0.9, "time")][:2]
    torch.manual_seed(0)
    model = SequenceAligner(clip_len=4, dim=16, width=8, hidden=16, frame_size=(40, 64)).eval()
    device = torch.device("cpu")
    stream = _lap_stream(model, 4, reference, live, "time", 2, device, 8)
    offline = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)

    grid = reference.reference_grid("time")
    every = max(int(round(live.fps / 15.0)), 1)
    engine = LiveDelta(model, np.asarray(reference.frames()[grid.frame_idx]), grid.time_s, grid.lap_time_s, 4, device,
                       spacing_s=2 / live.fps, hz=live.fps / every)
    frames, t = live.frames(), live.t().astype(np.float64)
    readings = [r for i in range(live.n_frames) if (r := engine.push(np.asarray(frames[i]), float(t[i]))) is not None]
    assert len(readings) == len(stream["t"])
    assert np.allclose([r.t for r in readings], stream["t"])
    ref_time = np.interp(offline.numpy() % grid.n_bins, np.arange(grid.n_bins + 1), np.r_[grid.time_s, grid.lap_time_s])
    gap = (np.array([r.ref_time_s for r in readings]) - ref_time + grid.lap_time_s / 2) % grid.lap_time_s - grid.lap_time_s / 2
    assert np.abs(gap).max() < 1e-3
