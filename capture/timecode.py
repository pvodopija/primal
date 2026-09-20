"""
Wire format for the on-screen timecode grid.

Single source of truth for the Python side. Must stay identical to
`ac_overlay/locamotif_timecode.lua`; see that file for the layout comment.

Layout: 2 rows of 27 square cells, row-major.

    cell  0       black   low luminance reference
    cell  1       white   high luminance reference
    cell  2       black   finder
    cell  3       white   finder
    cells 4..51   48 data bits, LSB first
    cell 52       white   tail finder
    cell 53       black   tail finder

Data bits:

    bits  0..21   frame counter, LSB first, wraps at 2^22
    bits 22..41   spline position as round(clamp(s, 0, 1) * (2^20 - 1))
    bits 42..47   checksum; bit j is the parity of payload bits j, j+6, ... j+36
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CELL_PX = 14
COLS = 27
ROWS = 2
N_CELLS = COLS * ROWS

COUNTER_BITS = 22
SPLINE_BITS = 20
CHECK_BITS = 6
PAYLOAD_BITS = COUNTER_BITS + SPLINE_BITS
DATA_BITS = PAYLOAD_BITS + CHECK_BITS

COUNTER_WRAP = 1 << COUNTER_BITS
SPLINE_MAX = (1 << SPLINE_BITS) - 1

DATA_OFFSET = 4
LOW_REF_CELLS = (0, 2, N_CELLS - 1)
HIGH_REF_CELLS = (1, 3, N_CELLS - 2)

# Below this black/white separation (0..255) the grid is judged unreadable
# rather than silently decoded from noise.
MIN_CONTRAST = 40.0

# Fraction of each cell's width sampled at its centre, to stay clear of
# compression ringing on the cell borders.
SAMPLE_FRACTION = 0.4


@dataclass(frozen=True)
class OverlayGeometry:
    """Pixel placement of the grid within a captured frame."""

    x0: float
    y0: float
    cell: float

    @property
    def width(self) -> float:
        return self.cell * COLS

    @property
    def height(self) -> float:
        return self.cell * ROWS

    def as_dict(self) -> dict:
        return {"x0": float(self.x0), "y0": float(self.y0), "cell": float(self.cell)}

    @staticmethod
    def from_dict(d: dict) -> "OverlayGeometry":
        return OverlayGeometry(x0=float(d["x0"]), y0=float(d["y0"]), cell=float(d["cell"]))


def checksum_bits(payload: np.ndarray) -> np.ndarray:
    """Parity-fold `PAYLOAD_BITS` payload bits into `CHECK_BITS` check bits."""
    out = np.zeros(CHECK_BITS, dtype=np.uint8)
    for j in range(CHECK_BITS):
        out[j] = payload[j::CHECK_BITS].sum() % 2
    return out


def encode(counter: int, spline: float) -> np.ndarray:
    """Build the 48 data bits. Mirrors the Lua writer; used by tests."""
    bits = np.zeros(DATA_BITS, dtype=np.uint8)
    value = int(counter) % COUNTER_WRAP
    for i in range(COUNTER_BITS):
        bits[i] = (value >> i) & 1
    fixed = int(round(min(max(spline, 0.0), 1.0) * SPLINE_MAX))
    for i in range(SPLINE_BITS):
        bits[COUNTER_BITS + i] = (fixed >> i) & 1
    bits[PAYLOAD_BITS:] = checksum_bits(bits[:PAYLOAD_BITS])
    return bits


def decode(bits: np.ndarray) -> tuple[int, float, bool]:
    """Recover (counter, spline, checksum_ok) from 48 data bits."""
    expected = checksum_bits(bits[:PAYLOAD_BITS])
    ok = bool(np.array_equal(expected, bits[PAYLOAD_BITS:]))
    weights = 1 << np.arange(COUNTER_BITS, dtype=np.uint64)
    counter = int((bits[:COUNTER_BITS].astype(np.uint64) * weights).sum())
    weights = 1 << np.arange(SPLINE_BITS, dtype=np.uint64)
    fixed = int((bits[COUNTER_BITS:PAYLOAD_BITS].astype(np.uint64) * weights).sum())
    return counter, fixed / SPLINE_MAX, ok


def sample_cells(gray: np.ndarray, geom: OverlayGeometry) -> np.ndarray | None:
    """
    Mean luminance of each cell's central box.

    Returns None if the grid does not fit inside the frame.
    """
    height, width = gray.shape[:2]
    margin = geom.cell * (1.0 - SAMPLE_FRACTION) / 2.0
    out = np.empty(N_CELLS, dtype=np.float32)
    for index in range(N_CELLS):
        col, row = index % COLS, index // COLS
        left = geom.x0 + col * geom.cell + margin
        top = geom.y0 + row * geom.cell + margin
        right = left + geom.cell * SAMPLE_FRACTION
        bottom = top + geom.cell * SAMPLE_FRACTION
        x1, y1 = int(round(left)), int(round(top))
        x2, y2 = max(int(round(right)), x1 + 1), max(int(round(bottom)), y1 + 1)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return None
        out[index] = float(gray[y1:y2, x1:x2].mean())
    return out


def decode_frame(gray: np.ndarray, geom: OverlayGeometry) -> tuple[int, float, bool]:
    """
    Decode one grayscale frame.

    Thresholds against the frame's own black and white reference cells rather
    than a fixed level, so any gamma or tonemapping shift is absorbed.
    """
    values = sample_cells(gray, geom)
    if values is None:
        return 0, 0.0, False
    low = float(values[list(LOW_REF_CELLS)].mean())
    high = float(values[list(HIGH_REF_CELLS)].mean())
    if high - low < MIN_CONTRAST:
        return 0, 0.0, False
    bits = (values[DATA_OFFSET : DATA_OFFSET + DATA_BITS] > (low + high) / 2.0).astype(np.uint8)
    return decode(bits)


def render_cells(bits: np.ndarray, geom: OverlayGeometry, canvas: np.ndarray) -> None:
    """Draw a grid onto `canvas` (grayscale or BGR). Used by tests and synthetic data."""
    values = np.zeros(N_CELLS, dtype=np.uint8)
    values[list(HIGH_REF_CELLS)] = 255
    values[DATA_OFFSET : DATA_OFFSET + DATA_BITS] = bits.astype(np.uint8) * 255
    for index in range(N_CELLS):
        col, row = index % COLS, index // COLS
        x1 = int(round(geom.x0 + col * geom.cell))
        y1 = int(round(geom.y0 + row * geom.cell))
        x2 = int(round(geom.x0 + (col + 1) * geom.cell))
        y2 = int(round(geom.y0 + (row + 1) * geom.cell))
        canvas[y1:y2, x1:x2] = values[index]
