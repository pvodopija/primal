"""
Sampler invariants and a learning smoke test.

The most valuable check here is that the target actually indexes the matching
reference frame after the roll. A sign or off-by-one error in that bookkeeping
would train the model against systematically wrong answers while every loss
curve still looked healthy.

Needs a packed dataset; generates a small synthetic one if none is present.

Run: python -m tests.test_train
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from train.dataset import (
    AlignmentBatches,
    LapIndex,
    SampleConfig,
    build_reference_grid,
    circular_soft_target,
)
from train.model import (
    SequenceAligner,
    alignment_loss,
    compute_metrics,
    soft_argmax_circular,
)

ML_ROOT = Path(__file__).resolve().parents[1]
DATA = ML_ROOT / "data" / "_train_test"


def _dataset() -> LapIndex:
    if not (DATA / "index.json").exists():
        subprocess.run(
            [
                sys.executable, "-m", "train.synthetic", "packed",
                "--out", str(DATA),
                "--tracks", "2", "--laps", "3",
                "--length-m", "160", "--fps", "20",
                "--width", "64", "--height", "40",
                "--ref-spacing-m", "2.0", "--holdout-tracks", "1",
            ],
            cwd=ML_ROOT,
            check=True,
            capture_output=True,
        )
    return LapIndex.load(DATA, split="train")


def test_target_indexes_the_matching_reference_frame() -> None:
    index = _dataset()
    config = SampleConfig(batch_size=6, clip_len=4, roll_reference=True, jitter=True)
    batches = AlignmentBatches(index, config, steps=12, seed=3)

    for step in range(12):
        batch = batches[step]
        n_bins = batch["reference"].shape[0]
        roll = batch["roll"]
        target = batch["target"].numpy().astype(np.float64)
        live_s = batch["live_s"].numpy().astype(np.float64)
        ref_s = batch["ref_s"].numpy().astype(np.float64)

        # The target is the live frame's s mapped onto the rolled reference axis.
        expected = (live_s * n_bins + roll) % n_bins
        assert np.abs(target - expected).max() < 1e-3

        # Every bin holds a frame from its own position, not half a bin along.
        # Frames filed at bin centres against targets at bin starts put every
        # frame exactly half a bin off, which a looser bound let through.
        # Frames land anywhere within half a frame-spacing of their bin, so the
        # median, not the maximum, is what separates this from the bug.
        home = (np.arange(n_bins) - roll) % n_bins
        placement = np.abs(ref_s * n_bins - home) % n_bins
        placement = np.minimum(placement, n_bins - placement)
        assert np.median(placement) < 0.3, f"median placement {np.median(placement):.2f} bins"
        assert placement.max() < 0.5, f"a bin holds a frame {placement.max():.2f} bins away"

        # So the frame at the target index is the same place as the live frame,
        # within rounding to the nearest bin plus that bin's own placement.
        bin_index = np.round(target).astype(np.int64) % n_bins
        gap = np.abs(ref_s[bin_index] - live_s)
        gap = np.minimum(gap, 1.0 - gap)  # the lap is a loop
        bound = 0.5 + placement.max() + 1e-3
        assert gap.max() * n_bins <= bound, f"reference at target is {gap.max() * n_bins:.2f} bins away"


def test_frame_targets_place_every_clip_frame() -> None:
    index = _dataset()
    for axis in ("distance", "time"):
        config = SampleConfig(batch_size=6, clip_len=4, roll_reference=True, reference_axis=axis)
        batches = AlignmentBatches(index, config, steps=6, seed=6)
        for step in range(6):
            batch = batches[step]
            frames = batch["frame_targets"].numpy().astype(np.float64)
            assert frames.shape == (6, 4)
            # The last frame is the one being localised, so it is the target.
            assert np.abs(frames[:, -1] - batch["target"].numpy()).max() < 1e-3
            # A forward clip moves forward along the reference, a reversed one
            # backward, a frozen one not at all.
            n_bins = batch["reference"].shape[0]
            step_bins = (np.diff(frames, axis=1) + n_bins / 2) % n_bins - n_bins / 2
            signs = np.sign(np.round(step_bins, 3))
            assert np.all((signs == signs[:, :1]) | (signs == 0)), "a clip changed direction"

def test_time_axis_target_follows_the_reference_clock() -> None:
    index = _dataset()
    config = SampleConfig(
        batch_size=6, clip_len=4, roll_reference=True, jitter=False, reference_axis="time"
    )
    batches = AlignmentBatches(index, config, steps=8, seed=4)
    for step in range(8):
        batch = batches[step]
        grid = next(
            lap for lap in index.laps if lap.lap_id == batch["reference_lap"]
        ).reference_grid("time")
        n_bins, roll = grid.n_bins, batch["roll"]
        live_s = batch["live_s"].numpy().astype(np.float64)
        expected = (grid.target(live_s) + roll) % n_bins
        assert np.abs(batch["target"].numpy() - expected).max() < 1e-3
        # Every bin is the same slice of reference time.
        widths = np.diff(grid.time_s)
        assert np.abs(widths - grid.lap_time_s / n_bins).max() < 1e-6


def test_grid_handles_the_finish_line() -> None:
    # The lap's last frame sits just before the line and its first a little
    # after. Bin 0 must take whichever is nearer around the loop; a straight
    # search only ever looked forward from the first frame.
    s = np.linspace(0.02, 0.999, 500)
    t = np.linspace(0.0, 50.0, 500)
    speed = np.full(500, 20.0)
    for axis in ("distance", "time"):
        grid = build_reference_grid(s, t, speed, 100, 1000.0, axis)
        assert grid.frame_idx[0] == 499, f"{axis}: bin 0 took frame {grid.frame_idx[0]}"
        assert grid.placement_m.max() < 11.0


def test_a_frame_exactly_on_the_line_ends_the_lap() -> None:
    # Real laps can finish with a frame at s == 1.0. Wrapped to 0 before
    # ordering, it was filed at the start of the lap carrying the lap's last
    # timestamp, and the lap collapsed to zero duration.
    s = np.linspace(0.0005, 1.0, 1000)
    t = np.linspace(0.0, 85.0, 1000)
    grid = build_reference_grid(s, t, np.full(1000, 20.0), 200, 1700.0, "time")
    assert abs(grid.lap_time_s - 85.0) < 0.5, f"lap time {grid.lap_time_s:.2f} s"
    assert np.all(np.diff(grid.time_s) > 0)
    # And a live frame half way round lands half way along the time axis.
    assert abs(float(grid.target(0.5)) - 100.0) < 1.0

def test_grid_handles_a_stationary_stretch() -> None:
    # A lap that stops dead for a while, then continues. On the time axis the
    # stop spans many bins at one place, which must stay well-formed.
    s = np.concatenate([np.linspace(0.0, 0.4, 200), np.full(50, 0.4), np.linspace(0.4, 1.0, 300)])
    t = np.concatenate([np.linspace(0, 20, 200), np.linspace(20.1, 30, 50), np.linspace(30.1, 60, 300)])
    speed = np.full(s.size, 20.0)
    for axis in ("distance", "time"):
        grid = build_reference_grid(s, t, speed, 128, 800.0, axis)
        assert grid.frame_idx.shape == (128,)
        assert grid.frame_idx.min() >= 0 and grid.frame_idx.max() < s.size
        assert np.all(np.isfinite(grid.pos_m)) and np.all(np.isfinite(grid.time_s))
        assert np.all(np.diff(grid.pos_m) >= -1e-9) and np.all(np.diff(grid.time_s) >= -1e-9)
        assert grid.placement_m.max() < 800.0 / 128


def test_metrics_read_both_units_off_the_grid() -> None:
    # 100 bins over a 1000 m lap taking 100 s; the first half is twice as slow.
    s = np.linspace(0.0, 0.999, 2000)
    t = np.where(s < 0.5, s * 133.33, 66.67 + (s - 0.5) * 66.67)
    speed = np.where(s < 0.5, 7.5, 15.0)
    grid = build_reference_grid(s, t, speed, 100, 1000.0, "distance")
    pos, time_s = torch.tensor(grid.pos_m), torch.tensor(grid.time_s)
    # One bin (10 m) off in the slow half costs twice the time of the fast half.
    m, ms = compute_metrics(
        torch.tensor([20.0, 70.0]), torch.tensor([21.0, 71.0]),
        pos, time_s, grid.track_length_m, grid.lap_time_s,
    )
    assert np.allclose(m, [10.0, 10.0], atol=1e-3)
    assert abs(ms[0] / ms[1] - 2.0) < 0.05, f"slow/fast delta ratio {ms[0] / ms[1]:.3f}"
    # And across the finish line the error is short, not most of a lap.
    m, _ = compute_metrics(
        torch.tensor([99.5]), torch.tensor([0.5]), pos, time_s, grid.track_length_m, grid.lap_time_s
    )
    assert abs(float(m[0]) - 10.0) < 1e-2


def test_live_and_reference_are_never_the_same_lap() -> None:
    index = _dataset()
    config = SampleConfig(batch_size=4, clip_len=4)
    batches = AlignmentBatches(index, config, steps=20, seed=5)
    for step in range(20):
        batch = batches[step]
        assert batch["reference_lap"] not in batch["live_laps"]


def test_reference_only_and_live_only_are_respected() -> None:
    index = _dataset()
    all_ids = sorted(lap.lap_id for lap in index.laps)
    reserved = frozenset(all_ids[-1:])
    trainable = frozenset(all_ids) - reserved

    config = SampleConfig(
        batch_size=3, clip_len=4, reference_only=trainable, live_only=reserved
    )
    batches = AlignmentBatches(index, config, steps=10, seed=8)
    for step in range(10):
        batch = batches[step]
        assert batch["reference_lap"] in trainable
        assert set(batch["live_laps"]) <= reserved


def test_soft_target_wraps_and_normalises() -> None:
    n_bins, sigma = 64, 2.0
    for value in (0.0, 0.5, 63.5, 63.9):
        weights = circular_soft_target(np.array([value]), n_bins, sigma)[0]
        assert abs(weights.sum() - 1.0) < 1e-5
        assert int(weights.argmax()) in (int(round(value)) % n_bins, (int(round(value)) + 1) % n_bins)
    # Mass either side of the boundary, not clipped against it.
    edge = circular_soft_target(np.array([0.0]), n_bins, sigma)[0]
    assert edge[-1] > 0.05 and edge[1] > 0.05


def test_soft_argmax_recovers_a_known_peak() -> None:
    n_bins = 128
    for centre in (0.0, 3.4, 64.0, 127.5):
        bins = np.arange(n_bins)
        offset = (bins - centre + n_bins / 2) % n_bins - n_bins / 2
        logits = torch.from_numpy(-0.5 * (offset / 1.5) ** 2).float()[None, :] * 4.0
        recovered = float(soft_argmax_circular(logits, window=8)[0])
        gap = abs((recovered - centre + n_bins / 2) % n_bins - n_bins / 2)
        assert gap < 0.05, f"soft-argmax gave {recovered} for peak {centre}"


def test_model_shapes_and_loss_decreases() -> None:
    index = _dataset()
    config = SampleConfig(batch_size=4, clip_len=4, roll_reference=True, jitter=True)
    batches = AlignmentBatches(index, config, steps=30, seed=11)
    shape = np.load(index.laps[0].directory / "frames.npy", mmap_mode="r").shape
    model = SequenceAligner(
        clip_len=4, dim=64, width=16, hidden=32, frame_size=(int(shape[1]), int(shape[2]))
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first, last = None, None
    for step in range(30):
        batch = batches[step]
        logits, correlation = model(batch["live"], batch["reference"])
        assert logits.shape == (4, batch["reference"].shape[0])
        assert correlation.shape == (4, 4, batch["reference"].shape[0])
        parts = alignment_loss(logits, correlation, batch["soft_target"], batch["target"])
        optimiser.zero_grad(set_to_none=True)
        parts.total.backward()
        optimiser.step()
        if step == 0:
            first = parts.total.item()
        last = parts.total.item()

    assert last < first, f"loss did not move: {first:.4f} -> {last:.4f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all training tests passed")
