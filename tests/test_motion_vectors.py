import numpy as np
import pytest

from capture.motion_vectors import pool_vectors, radial_flow

FIELDS = [
    ("source", "<i4"), ("w", "u1"), ("h", "u1"), ("dst_x", "<i2"), ("dst_y", "<i2"),
    ("motion_x", "<i4"), ("motion_y", "<i4"), ("motion_scale", "<u2"),
]


def _vectors(rows):
    return np.array(rows, dtype=FIELDS)


def test_pool_is_area_weighted_and_reads_content_motion():
    vectors = _vectors(
        [
            # source, w, h, dst_x, dst_y, motion_x, motion_y, scale
            (-1, 16, 16, 8, 8, -8, 0, 4),  # came from 2 px to the left: content moved +2
            (-1, 8, 8, 20, 4, -16, 4, 4),  # +4 x, -1 y, a quarter of the area
            (1, 16, 16, 8, 24, 400, 400, 4),  # future reference: ignored
        ]
    )
    flow, cover = pool_vectors(vectors, width=48, height=32, cell=32)
    assert flow.shape == (2, 1, 2) and cover.shape == (1, 2)
    assert flow[0, 0, 0] == pytest.approx((2 * 256 + 4 * 64) / 320)
    assert flow[1, 0, 0] == pytest.approx(-64 / 320)
    assert cover[0, 0] == pytest.approx(320 / 1024)
    assert cover[0, 1] == 0 and flow[:, 0, 1].tolist() == [0, 0]


def test_edge_cells_measure_cover_against_their_own_size():
    vectors = _vectors([(-1, 16, 16, 40, 8, 0, 0, 4), (-1, 16, 16, 40, 24, 0, 0, 4)])
    _, cover = pool_vectors(vectors, width=48, height=32, cell=32)
    assert cover[0, 1] == pytest.approx(1.0)  # 512 px of a 16x32 edge cell


def test_radial_flow_ignores_a_pure_sideways_shift():
    grid_h, grid_w, cell = 9, 16, 32
    size = (grid_w * cell, grid_h * cell)
    cx = (np.arange(grid_w) + 0.5) * cell - size[0] / 2
    cy = (np.arange(grid_h) + 0.5) * cell - size[1] / 2
    expand = np.stack([np.broadcast_to(cx[None, :], (grid_h, grid_w)), np.broadcast_to(cy[:, None], (grid_h, grid_w))])
    shift = np.stack([np.full((grid_h, grid_w), 5.0), np.zeros((grid_h, grid_w))])
    flow = np.stack([0.01 * expand, 0.02 * expand, shift, 0.02 * expand + shift])
    cover = np.ones((4, grid_h, grid_w))
    r = radial_flow(flow, cover, cell, size, keep_rows=(0, size[1]))
    assert r[1] == pytest.approx(2 * r[0])
    assert abs(r[2]) < 1e-6
    assert r[3] == pytest.approx(r[1])


def test_encoder_recovers_a_known_translation():
    av = pytest.importorskip("av")
    from capture.motion_vectors import encoder_motion

    width, height, step = 320, 192, (3, 1)
    rng = np.random.default_rng(0)
    noise = rng.random((height + 64, width + 96))
    kernel = np.ones(5) / 5
    texture = np.apply_along_axis(np.convolve, 1, noise, kernel, "same")
    texture = np.apply_along_axis(np.convolve, 0, texture, kernel, "same")
    texture = ((texture - texture.min()) / np.ptp(texture) * 255).astype(np.uint8)

    def frames():
        for i in range(10):
            # The window slides right and down, so the content moves left and up.
            window = texture[i * step[1] : i * step[1] + height, i * step[0] : i * step[0] + width]
            yield av.VideoFrame.from_ndarray(np.dstack([window] * 3), format="rgb24")

    out = list(encoder_motion(frames(), width, height, fps=60.0, cell=32))
    assert [index for index, _, _ in out] == list(range(10))
    assert out[0][2].max() == 0  # the keyframe carries no vectors
    for _, flow, cover in out[1:]:
        inner = (slice(1, -1), slice(1, -1))
        assert np.median(cover[inner]) > 0.9
        assert np.median(flow[0][inner]) == pytest.approx(-step[0], abs=0.25)
        assert np.median(flow[1][inner]) == pytest.approx(-step[1], abs=0.25)
