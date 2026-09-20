"""
Round-trip checks for the timecode wire format and the grid locator.

Run: python -m tests.test_timecode
"""

from __future__ import annotations

import cv2
import numpy as np

from capture.overlay_decode import refine
from capture.timecode import (
    CELL_PX,
    COUNTER_WRAP,
    SPLINE_MAX,
    OverlayGeometry,
    decode,
    decode_frame,
    encode,
    render_cells,
)


def test_bit_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2000):
        counter = int(rng.integers(0, COUNTER_WRAP))
        spline = float(rng.random())
        got_counter, got_spline, ok = decode(encode(counter, spline))
        assert ok
        assert got_counter == counter
        # Quantisation is one part in 2^20, i.e. ~3 mm on a 3 km track.
        assert abs(got_spline - spline) <= 1.0 / SPLINE_MAX


def test_checksum_catches_bit_flips() -> None:
    rng = np.random.default_rng(1)
    caught = 0
    for _ in range(500):
        bits = encode(int(rng.integers(0, COUNTER_WRAP)), float(rng.random()))
        bits[int(rng.integers(0, len(bits)))] ^= 1
        if not decode(bits)[2]:
            caught += 1
    # A 6-bit parity fold catches every single-bit flip.
    assert caught == 500


def _render_scene(geom: OverlayGeometry, counter: int, spline: float) -> np.ndarray:
    """A noisy 1280x720 frame with the grid drawn into it, then JPEG-degraded."""
    rng = np.random.default_rng(counter)
    frame = rng.integers(20, 200, size=(720, 1280), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 3.0)
    render_cells(encode(counter, spline), geom, frame)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def test_decode_survives_compression() -> None:
    geom = OverlayGeometry(x0=24.0, y0=676.0, cell=float(CELL_PX))
    rng = np.random.default_rng(2)
    for _ in range(40):
        counter = int(rng.integers(0, COUNTER_WRAP))
        spline = float(rng.random())
        gray = _render_scene(geom, counter, spline)
        got_counter, got_spline, ok = decode_frame(gray, geom)
        assert ok, "checksum failed on a compressed frame"
        assert got_counter == counter
        assert abs(got_spline - spline) <= 2.0 / SPLINE_MAX


def test_refine_recovers_offset_geometry() -> None:
    truth = OverlayGeometry(x0=31.0, y0=664.0, cell=17.0)
    frames = [_render_scene(truth, 1000 + i, i / 40.0) for i in range(12)]
    # Seed as if the user dragged a sloppy box: off by a few pixels, wrong scale.
    seed = OverlayGeometry(x0=truth.x0 - 5, y0=truth.y0 + 4, cell=truth.cell - 0.6)
    found, score = refine(frames, seed)
    assert score == 1.0, f"pass rate {score} with geometry {found}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all timecode tests passed")
