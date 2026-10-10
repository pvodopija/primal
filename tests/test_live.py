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


def test_descriptors_and_reset_reproduce_a_fresh_engine():
    """A reference given as descriptors, and an engine reset after a lap, read as a fresh one."""
    index = _dataset()
    track = sorted(index.by_track)[0]
    reference, live = [lap for lap in index.by_track[track] if lap.usable_as_reference(0.9, "time")][:2]
    torch.manual_seed(0)
    model = SequenceAligner(clip_len=4, dim=16, width=8, hidden=16, frame_size=(40, 64)).eval()
    device = torch.device("cpu")
    grid = reference.reference_grid("time")
    every = max(int(round(live.fps / 15.0)), 1)
    kwargs = dict(spacing_s=2 / live.fps, hz=live.fps / every)
    ref_frames = np.asarray(reference.frames()[grid.frame_idx])
    from_frames = LiveDelta(model, ref_frames, grid.time_s, grid.lap_time_s, 4, device, **kwargs)
    from_desc = LiveDelta(model, None, grid.time_s, grid.lap_time_s, 4, device, **kwargs,
                          reference_descriptors=from_frames.ref.clone())
    frames, t = live.frames(), live.t().astype(np.float64)

    def drive(engine):
        return [(r.ref_time_s, r.confidence) for i in range(live.n_frames)
                if (r := engine.push(np.asarray(frames[i]), float(t[i]))) is not None]

    first = drive(from_frames)
    assert first and all(0.0 <= c <= 1.0 for _, c in first)
    assert drive(from_desc) == first
    from_frames.reset()
    assert drive(from_frames) == first


def test_start_at_places_the_first_reading():
    """Told where the kart is, an engine whose model knows nothing reads there from its first tick."""
    index = _dataset()
    track = sorted(index.by_track)[0]
    reference, live = [lap for lap in index.by_track[track] if lap.usable_as_reference(0.9, "time")][:2]
    torch.manual_seed(0)
    model = SequenceAligner(clip_len=4, dim=16, width=8, hidden=16, frame_size=(40, 64)).eval()
    grid = reference.reference_grid("time")
    every = max(int(round(live.fps / 15.0)), 1)
    engine = LiveDelta(model, np.asarray(reference.frames()[grid.frame_idx]), grid.time_s, grid.lap_time_s, 4,
                       torch.device("cpu"), spacing_s=2 / live.fps, hz=live.fps / every)
    frames, t = live.frames(), live.t().astype(np.float64)
    where = grid.lap_time_s / 2
    engine.start_at(float(t[0]), where, sd_s=0.5)
    first = next(r for i in range(live.n_frames) if (r := engine.push(np.asarray(frames[i]), float(t[i]))) is not None)
    expected = (where + first.t - t[0]) % grid.lap_time_s
    assert abs((first.ref_time_s - expected + grid.lap_time_s / 2) % grid.lap_time_s - grid.lap_time_s / 2) < 1.5
