import numpy as np

from tests.test_train import _dataset
from train.dataset import AlignmentBatches, SampleConfig, draw_traffic, traffic_episodes


def test_a_kart_ahead_covers_part_of_the_road_and_stays_put_through_the_clip():
    frames = np.full((6, 80, 148, 3), 128, np.uint8)
    times = np.arange(6) / 30.0
    for style in ("blocks", "kart"):
        out = draw_traffic(frames, np.random.default_rng(3), times, style=style)
        changed = np.any(out != frames, axis=-1)
        assert out.shape == frames.shape and out.dtype == np.uint8
        assert changed.any(), style
        # on or below the horizon (the road), never across the whole frame
        rows = np.flatnonzero(changed.any(axis=(0, 2)))
        assert rows.min() >= 0.6 * 80 - 25 and changed.mean() < 0.5
        # consecutive frames 1/30 s apart barely move it
        assert np.mean(changed[0] ^ changed[1]) < 0.02


def test_traffic_episodes_cover_about_the_share_asked():
    frames = np.full((600, 80, 148, 3), 128, np.uint8)
    times = np.arange(600) / 60.0
    out = traffic_episodes(frames, times, 0.3, np.random.default_rng(0))
    share = np.mean(np.any(out != frames, axis=(1, 2, 3)))
    assert 0.15 < share < 0.75


def test_traffic_off_samples_exactly_as_before():
    index = _dataset()
    a = AlignmentBatches(index, SampleConfig(batch_size=4, clip_len=4), steps=3, seed=9)[1]
    b = AlignmentBatches(index, SampleConfig(batch_size=4, clip_len=4, p_traffic=0.0), steps=3, seed=9)[1]
    assert np.array_equal(a["live"].numpy(), b["live"].numpy())
