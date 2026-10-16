"""The session logic's deterministic parts on made-up descriptors (the whole session on real
laps: experiments/session_replay.py)."""
import numpy as np
import torch

from train.session import ClosureCheck, LoopCloser, SessionConfig, build_reference


def _frames(times, vectors):
    return [(float(t), torch.as_tensor(v, dtype=torch.float32)) for t, v in zip(times, vectors)]


def test_reference_bins_take_the_frame_nearest_each_bins_time():
    times = np.arange(0, 10, 1 / 30)
    frames = _frames(times, np.random.default_rng(0).normal(size=(len(times), 8)))
    bin_times, descriptors, lap = build_reference(frames, 1.0, 6.2, 0.052)
    assert lap == 6.2 - 1.0
    assert len(bin_times) == round(5.2 / 0.052) == len(descriptors)
    assert np.allclose(bin_times, np.arange(len(bin_times)) * 5.2 / len(bin_times))
    nearest = [int(np.argmin(np.abs(times - (1.0 + b)))) for b in bin_times]
    assert all(torch.equal(descriptors[i], frames[k][1]) for i, k in enumerate(nearest))


def _lap_with_return(rng, lap_s=20.0, fps=30, dim=16):
    """A made-up lap: a start view, other views, then the start view again from lap_s on."""
    start = rng.normal(size=dim)
    times = np.arange(0, lap_s + 3, 1 / fps)
    vectors = []
    for t in times:
        near_start = t < 1.0 or t >= lap_s
        base = start if near_start else rng.normal(size=dim)
        vectors.append(base + 0.05 * rng.normal(size=dim))
    return times, vectors


def test_the_start_view_seen_again_proposes_the_lap_end():
    rng = np.random.default_rng(1)
    times, vectors = _lap_with_return(rng)
    closer, frames, proposal = LoopCloser(SessionConfig()), [], None
    for f in _frames(times, vectors):
        frames.append(f)
        proposal = proposal or closer.push(frames)
    assert proposal is not None and abs(proposal - 20.0) < 0.2


def test_no_proposal_before_the_view_has_moved_away():
    rng = np.random.default_rng(2)
    dim, start = 16, rng.normal(size=16)
    times = np.arange(0, 12, 1 / 30)
    vectors = [start + 0.05 * rng.normal(size=dim) for _ in times]  # standing at the start the whole time
    closer, frames = LoopCloser(SessionConfig()), []
    assert all(closer.push(frames := frames + [f]) is None for f in _frames(times, vectors))


def test_readings_must_agree_before_they_place_a_line():
    agreed = ClosureCheck._agreed
    assert agreed([10.0] * 14, 30, 15, 0.15) is None                      # too few
    assert agreed([10.0 + 0.01 * i for i in range(15)], 30, 15, 0.15) == np.median([10.0 + 0.01 * i for i in range(15)])
    scattered = [10.0, 11.0] * 10                                         # half here, half a second away
    assert agreed(scattered, 30, 15, 0.15) is None
    assert agreed([99.0] * 20 + [10.0] * 30, 30, 15, 0.15) == 10.0        # only the last window counts


class _Engine:
    def start_at(self, t, ref):
        pass

    def push_descriptor(self, d, t):
        pass


def _session_after_laps(lap_times):
    """A session past its reference lap and `lap_times` laps, its engine a stub."""
    from train.session import Session
    s = Session.__new__(Session)
    s.c, s.laps, s.lap, s.lap_start = SessionConfig(), [], 0, 0.0
    s.reference_lap_s, s.lap_times = lap_times[0], [lap_times[0]]
    s._engine = lambda frames, start, line: _Engine()
    s._new_lap_counters()
    for lap_s in lap_times[1:]:
        s.ticks, s.shown_ticks = 10, 10
        s._lap_ended(s.lap_start + lap_s, s.lap_start + lap_s)
    return s


def test_any_faster_lap_becomes_the_reference_but_not_a_line_misplaced_short():
    s = _session_after_laps([49.2, 48.6, 48.7])
    assert s.laps[0].became_reference and abs(s.reference_lap_s - 48.6) < 1e-9
    s.ticks, s.shown_ticks = 10, 2                       # its delta mostly hidden: still the reference
    s._lap_ended(s.lap_start + 48.3, s.lap_start + 48.3)
    assert s.laps[-1].became_reference and abs(s.reference_lap_s - 48.3) < 1e-9
    s.ticks, s.shown_ticks = 10, 4
    s._lap_ended(s.lap_start + 34.9, s.lap_start + 34.9)  # the pit in-lap: a misplaced line
    assert not s.laps[-1].became_reference and abs(s.reference_lap_s - 48.3) < 1e-9
