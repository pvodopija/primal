import numpy as np

from train.dataset import AlignmentBatches, SampleConfig
from tests.test_train import _dataset


def test_folded_clips_turn_back_once_and_are_labelled_where_they_end():
    index = _dataset()
    config = SampleConfig(batch_size=8, clip_len=6, roll_reference=False, reference_axis="distance",
                          p_fold=1.0, p_reverse=0.0, p_static=0.0, strides=(2,))
    batches = AlignmentBatches(index, config, steps=6, seed=2)
    turned = 0
    for step in range(6):
        batch = batches[step]
        frames = batch["frame_targets"].numpy().astype(np.float64)
        n_bins = batch["reference"].shape[0]
        moves = np.sign(np.round((np.diff(frames, axis=1) + n_bins / 2) % n_bins - n_bins / 2, 3))
        for row in moves:
            row = row[row != 0]
            changes = np.count_nonzero(np.diff(row))
            assert changes <= 1  # forward, then back: one turn at most
            turned += changes
        # the target is still the last frame shown
        assert np.abs(frames[:, -1] - batch["target"].numpy()).max() < 1e-3
    assert turned > 0


def test_fold_off_samples_exactly_as_before():
    index = _dataset()
    a = AlignmentBatches(index, SampleConfig(batch_size=4, clip_len=4), steps=3, seed=9)[1]
    b = AlignmentBatches(index, SampleConfig(batch_size=4, clip_len=4, p_fold=0.0), steps=3, seed=9)[1]
    assert np.array_equal(a["frame_targets"].numpy(), b["frame_targets"].numpy())
