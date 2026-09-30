import numpy as np
import torch

from train.dataset import camera_jitter
from train.model import SequenceAligner


def test_norm_none_has_no_groupnorm_and_runs():
    model = SequenceAligner(clip_len=4, norm="none")
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in model.modules())
    logits, corr = model(torch.rand(2, 4, 3, 80, 148), torch.rand(20, 3, 80, 148), use_checkpoint=False)
    assert logits.shape == (2, 20) and torch.isfinite(logits).all()


def test_default_yaw_draws_as_before():
    frames = np.random.default_rng(0).integers(0, 256, (3, 80, 148, 3), dtype=np.uint8)
    a = camera_jitter(frames, np.random.default_rng(5), parts=frozenset({"pose"}))
    b = camera_jitter(frames, np.random.default_rng(5), parts=frozenset({"pose"}), yaw_deg=4.0)
    c = camera_jitter(frames, np.random.default_rng(5), parts=frozenset({"pose"}), yaw_deg=10.0)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
