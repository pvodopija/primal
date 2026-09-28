import numpy as np

from train.dataset import AlignmentBatches, SampleConfig, _to_chw, camera_jitter, prepare_frames
from tests.test_train import _dataset


def _frames(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(6, 40, 64, 3), dtype=np.uint8)


def test_mirror_flips_left_to_right_and_nothing_else():
    raw = _frames()
    out = prepare_frames(raw, np.random.default_rng(0), camera_aug=False, mirror=True)
    assert np.array_equal(out, _to_chw(raw[:, :, ::-1]))


def test_camera_jitter_is_deterministic_and_changes_the_picture():
    raw = _frames()
    a = camera_jitter(raw, np.random.default_rng(3))
    b = camera_jitter(raw, np.random.default_rng(3))
    assert a.shape == raw.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert np.abs(a.astype(int) - raw.astype(int)).mean() > 1.0


def test_sampling_is_unchanged_with_augmentation_off():
    index = _dataset()
    plain = AlignmentBatches(index, SampleConfig(batch_size=2, clip_len=4), steps=2, seed=5)[1]
    explicit = AlignmentBatches(
        index, SampleConfig(batch_size=2, clip_len=4, camera_aug=False, mirror_p=0.0), steps=2, seed=5
    )[1]
    assert np.array_equal(plain["live"].numpy(), explicit["live"].numpy())
    assert np.array_equal(plain["reference"].numpy(), explicit["reference"].numpy())
    assert np.array_equal(plain["target"].numpy(), explicit["target"].numpy())


def test_augmented_batches_keep_their_targets():
    index = _dataset()
    config = SampleConfig(batch_size=2, clip_len=4, camera_aug=True, mirror_p=1.0, jitter=False)
    batch = AlignmentBatches(index, config, steps=1, seed=7)[0]
    n_bins = batch["reference"].shape[0]
    true_bin = np.round(batch["target"].numpy()).astype(int) % n_bins
    # Mirroring and camera jitter change pixels, never positions: the bin the
    # target names still holds the live frame's place on the lap.
    gap = np.abs(((batch["ref_s"].numpy()[true_bin] - batch["live_s"].numpy()) + 0.5) % 1.0 - 0.5)
    assert (gap * 1000 < 20).all()
