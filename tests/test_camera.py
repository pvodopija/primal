import numpy as np

from train.camera import HALO, TRAINING, WIDE, Shake, render, render_clip


def test_a_direction_lands_where_the_turned_view_points():
    img = np.zeros((WIDE.height, WIDE.width, 3), np.uint8)
    for yaw, pitch in ((10.0, 0.0), (-8.0, 5.0)):
        u = WIDE.cx + WIDE.focal * np.tan(np.radians(yaw))
        v = WIDE.cy - WIDE.focal * np.tan(np.radians(pitch)) / np.cos(np.radians(yaw))
        img[:] = 0
        img[int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = 255
        out = render(img, WIDE, TRAINING, yaw=yaw, pitch=pitch)
        ys, xs = np.nonzero(out[..., 0] > 64)
        assert abs(xs.mean() - TRAINING.cx) < 1.5 and abs(ys.mean() - TRAINING.cy) < 1.5


def test_the_unturned_view_keeps_the_training_framing():
    # the horizon (a ray straight ahead) sits where the packed captures have it
    assert abs(TRAINING.cy - 41.6) < 0.2 and abs(WIDE.cy - 75.4) < 0.2
    assert HALO.width / HALO.height == 4 / 3


def test_shake_is_reproducible_and_scales_with_level():
    a = Shake(1.0, np.random.default_rng(0)).at(np.linspace(-1, 0, 200))
    b = Shake(1.0, np.random.default_rng(0)).at(np.linspace(-1, 0, 200))
    c = Shake(2.0, np.random.default_rng(0)).at(np.linspace(-1, 0, 200))
    assert np.array_equal(a, b) and np.allclose(c, 2 * a)


def test_render_clip_shapes():
    frames = np.random.default_rng(1).integers(0, 255, (4, WIDE.height, WIDE.width, 3), dtype=np.uint8)
    out = render_clip(frames, np.arange(4) / 30, np.random.default_rng(2), view=HALO, shake_level=1.0)
    assert out.shape == (4, HALO.height, HALO.width, 3) and out.dtype == np.uint8
