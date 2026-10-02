"""The sampler on wide renders: clips come out at the training view, targets stay right."""
import json
import shutil

import numpy as np

from tests.test_train import DATA, _dataset
from train.camera import TRAINING
from train.dataset import AlignmentBatches, LapIndex, SampleConfig, same_drive


def _wide_copy(tmp_path):
    """The synthetic set, indexed as if every lap were a wide render."""
    _dataset()
    root = tmp_path / "wide"
    shutil.copytree(DATA, root)
    index = json.loads((root / "index.json").read_text())
    for entry in index["laps"]:
        entry["fov_h_deg"] = 121.28
    (root / "index.json").write_text(json.dumps(index))
    return LapIndex.load(root, split="train")


def test_wide_laps_are_rendered_at_the_training_view_with_correct_targets(tmp_path):
    index = _wide_copy(tmp_path)
    assert all(lap.wide for lap in index.laps)
    config = SampleConfig(batch_size=4, clip_len=4, strides=(1,), jitter=False, shake_max=2.0, reference_axis="time")
    batch = AlignmentBatches(index, config, steps=4, seed=0)[1]
    assert batch["live"].shape == (4, 4, 3, TRAINING.height, TRAINING.width)
    assert batch["reference"].shape[1:] == (3, TRAINING.height, TRAINING.width)
    # the camera changes pixels, never labels: each target is the bin whose s matches the clip's
    ref_s, live_s, target = batch["ref_s"].numpy(), batch["live_s"].numpy(), batch["target"].numpy()
    for s, b in zip(live_s, target):
        gap = abs((ref_s[int(round(b)) % ref_s.size] - s + 0.5) % 1.0 - 0.5)
        assert gap < 0.03


def test_shake_off_and_no_wide_laps_draw_exactly_what_they_always_did():
    index = _dataset()
    plain = AlignmentBatches(index, SampleConfig(batch_size=2, clip_len=4), steps=2, seed=5)[0]
    again = AlignmentBatches(index, SampleConfig(batch_size=2, clip_len=4, shake_max=0.0, head_yaw_deg=20.0), steps=2, seed=5)[0]
    assert np.array_equal(plain["live"].numpy(), again["live"].numpy())


def test_same_drive_finds_a_rerender_and_not_another_track():
    index = _dataset()
    whole = [lap for lap in index.laps if lap.s_span > 0.98]
    assert same_drive(whole[0], whole[0])
    other = next((lap for lap in whole if lap.track != whole[0].track), None)
    if other is not None:
        assert not same_drive(whole[0], other)
