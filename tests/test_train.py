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

from train.dataset import AlignmentBatches, LapIndex, SampleConfig, circular_soft_target
from train.model import SequenceAligner, alignment_loss, soft_argmax_circular

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

        # And the reference frame sitting at that index is the same place.
        bin_index = np.round(target).astype(np.int64) % n_bins
        gap = np.abs(ref_s[bin_index] - live_s)
        gap = np.minimum(gap, 1.0 - gap)  # the lap is a loop
        assert gap.max() < 2.0 / n_bins, f"reference at target is {gap.max() * n_bins:.2f} bins away"


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
