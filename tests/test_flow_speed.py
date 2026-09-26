import cv2
import numpy as np
import pytest

from capture.flow_speed import HALO_FOCAL_PX, RoadStrip, fit_motion, motion_bases

WIDTH, HEIGHT, CAM_H = 640, 480, 1.15
PX_PER_M = 200.0


def _texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.random((1024, 1024)).astype(np.float32)
    grain = cv2.GaussianBlur(noise, (0, 0), 3.0)  # ~1.5 cm, as tarmac reads at VGA
    patches = cv2.GaussianBlur(rng.random((1024, 1024)).astype(np.float32), (0, 0), 12)
    return cv2.normalize(grain + 2 * patches, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _render(texture: np.ndarray, forward_m: float, sideways_m: float) -> np.ndarray:
    """A flat textured road seen by a level camera CAM_H above it, at a given pose."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    x, y = xs - WIDTH / 2 + 0.5, ys - HEIGHT / 2 + 0.5
    g = np.maximum(y, 1.0)
    depth = HALO_FOCAL_PX * CAM_H / g
    lateral = x * depth / HALO_FOCAL_PX
    u = (((lateral + sideways_m) * PX_PER_M) % texture.shape[1]).astype(np.float32)
    v = (((depth + forward_m) * PX_PER_M) % texture.shape[0]).astype(np.float32)
    ground = cv2.remap(texture, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    ground[y < 2] = 128  # sky
    return cv2.cvtColor(ground, cv2.COLOR_GRAY2BGR)


def _project(ground: np.ndarray, forward_m: float = 0.0, sideways_m: float = 0.0) -> np.ndarray:
    """Exact image position of road points (lateral X, depth Z) from a moved camera."""
    lateral, depth = ground[:, 0] - sideways_m, ground[:, 1] - forward_m
    return np.stack([HALO_FOCAL_PX * lateral / depth, HALO_FOCAL_PX * CAM_H / depth], 1)


def test_fit_recovers_translation_exactly_despite_outliers():
    rng = np.random.default_rng(1)
    ground = np.stack([rng.uniform(-3, 3, 300), rng.uniform(3, 10, 300)], 1)
    before = _project(ground)
    after = _project(ground, forward_m=0.5, sideways_m=0.03)
    flow = after - before
    flow[:20] += rng.normal(0, 30, (20, 2))  # a passing kart: gross outliers
    motion, rms, inlier = fit_motion(before, flow, HALO_FOCAL_PX, 0.0, CAM_H)
    assert motion[0] == pytest.approx(0.5, rel=0.005)
    assert motion[1] == pytest.approx(0.03, abs=1e-3)
    assert np.abs(motion[2:]).max() < 1e-4
    assert inlier[20:].mean() > 0.95 and inlier[:20].mean() < 0.3


def test_fit_separates_a_head_turn_from_forward_motion():
    rng = np.random.default_rng(2)
    ground = np.stack([rng.uniform(-3, 3, 300), rng.uniform(3, 10, 300)], 1)
    before = _project(ground)
    after = _project(ground, forward_m=0.25)
    yaw = np.radians(100 / 120)  # a 100 deg/s glance, one frame at 120 fps
    after = after + yaw * motion_bases(before, HALO_FOCAL_PX, 0.0, CAM_H)[2]
    motion, _, _ = fit_motion(before, after - before, HALO_FOCAL_PX, 0.0, CAM_H)
    assert motion[0] == pytest.approx(0.25, rel=0.03)
    assert motion[2] == pytest.approx(yaw, rel=0.05)


# 30 m/s is 0.25 m per frame at 120 fps and 0.5 m at 60 fps. The bigger step
# zooms the near road more, so fewer points survive the round trip.
@pytest.mark.parametrize("step, drift, min_kept", [(0.25, 0.0, 0.6), (0.5, 0.03, 0.35)])
def test_rendered_road_gives_forward_motion_and_drift(step, drift, min_kept):
    texture = _texture()
    strip = RoadStrip(WIDTH, HEIGHT, HALO_FOCAL_PX, HALO_FOCAL_PX, CAM_H)
    a = strip.cut(_render(texture, 10.0, 0.0))
    b = strip.cut(_render(texture, 10.0 + step, drift))
    motion, points, kept, rms = strip.measure(a, b)
    assert motion is not None and points > 50 and kept > min_kept
    assert motion[0] == pytest.approx(step, rel=0.05)
    assert motion[1] == pytest.approx(drift, abs=0.01)
    assert rms < 0.5


def test_band_covers_three_to_ten_metres():
    strip = RoadStrip(1280, 720, 360 / np.tan(np.radians(30)), HALO_FOCAL_PX, CAM_H)
    assert strip.size == (round(1280 * strip.scale), round(720 * strip.scale))
    near_row = strip.size[1] / 2 + HALO_FOCAL_PX * CAM_H / 3.0
    far_row = strip.size[1] / 2 + HALO_FOCAL_PX * CAM_H / 10.0
    assert abs(strip.rows[0] - far_row) <= 1 and abs(strip.rows[1] - near_row) <= 1
