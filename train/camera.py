"""
A virtual camera on the wide renders: any head pose and any camera shake, rendered exactly.

`packed_ac_v3_wide` was rendered at 121° x 87° from the same rig position as the normal
captures. Turning a camera about its own centre needs no depth, so the view of a camera
turned by any yaw, pitch and roll is a homography of the wide frame,
`K_wide @ R @ inv(K_view)`, exact wherever the turned view stays inside the wide frame
(at the 148x80 training view, ±15° of yaw and ±14° of pitch).

Two kinds of motion are rendered:
- head pose, held over a clip: looking into a corner, a lean (`pose`);
- camera shake (`Shake`): kart vibration (no suspension, 8-25 Hz, mostly pitch and roll),
  occasional kerb hits (a few degrees, gone within ~0.2 s), and slow head wobble
  (0.3-2 Hz). Exposure blur is the average of the views along the motion during the
  exposure, which is what a global shutter records.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class View:
    """A pinhole camera: size in pixels and intrinsics (square pixels)."""

    width: int
    height: int
    focal: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1]], dtype=np.float64)

    @staticmethod
    def rendered(width: int, height: int, hfov_deg: float, render_vfov_deg: float | None = None) -> "View":
        """
        A capture from AC: horizontally centred. AC renders with the principal point at the
        centre and the barcode band is then cut off the bottom, so the top keeps half the
        render's vertical FOV; `render_vfov_deg` None means nothing was cut (centred).
        """
        focal = (width / 2) / np.tan(np.radians(hfov_deg / 2))
        cy = height / 2 if render_vfov_deg is None else focal * np.tan(np.radians(render_vfov_deg / 2))
        return View(width, height, float(focal), width / 2, float(cy))


# The packed sets, and the views to render.
WIDE = View.rendered(268, 144, 121.28, render_vfov_deg=90.0)  # packed_ac_v3_wide
TRAINING = View.rendered(148, 80, 91.49, render_vfov_deg=60.0)  # packed_ac_v3, what the models see
HALO = View.rendered(128, 96, 81.2)  # Halo's camera (81.2° x ~65.5°, 4:3), at model size


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera turned right by yaw, up by pitch, and rolled clockwise; x right, y down, z forward."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def render(frame: np.ndarray, source: View, view: View, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """One frame as seen by `view` turned by (yaw, pitch, roll) relative to the camera that took it."""
    to_source = source.K @ rotation(yaw, pitch, roll) @ np.linalg.inv(view.K)
    return cv2.warpPerspective(frame, to_source, (view.width, view.height),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)


@dataclass
class Shake:
    """
    One clip's camera shake, as yaw, pitch and roll in degrees at any time. Drawn once per
    clip; `level` scales everything (0 = none, 1 = a typical kart, 2 = a rough one). Kerb
    hits are drawn from `start` to 2 s.
    """

    level: float
    rng: np.random.Generator
    start: float = -1.0

    def __post_init__(self) -> None:
        rng, a = self.rng, self.level
        self.f_vib = rng.uniform(8.0, 25.0)  # kart vibration
        self.vib = [(self.f_vib * rng.uniform(0.7, 1.3), rng.uniform(0, 2 * np.pi), rng.normal(size=3)) for _ in range(4)]
        self.vib_amp = a * rng.uniform(0.2, 0.8) * np.array([0.3, 1.0, 0.6])  # deg; yaw, pitch, roll
        self.wob = [(rng.uniform(0.3, 2.0), rng.uniform(0, 2 * np.pi), rng.normal(size=3)) for _ in range(3)]
        self.wob_amp = a * rng.uniform(0.2, 0.8) * np.array([0.8, 0.6, 0.6])
        self.kerbs = []  # (time, amplitude deg, decay s), at ~0.3 per second
        t = rng.exponential(1 / 0.3) + self.start
        while t < 2.0:
            self.kerbs.append((t, a * rng.uniform(1.0, 3.0), rng.uniform(0.08, 0.2)))
            t += rng.exponential(1 / 0.3)

    def at(self, t: np.ndarray) -> np.ndarray:
        """[len(t), 3] yaw, pitch, roll in degrees; t in seconds, from `start` to 2."""
        t = np.asarray(t, dtype=np.float64)[:, None]
        vib = sum(np.sin(2 * np.pi * f * t + ph) * d for f, ph, d in self.vib) / 2.0
        wob = sum(np.sin(2 * np.pi * f * t + ph) * d for f, ph, d in self.wob) / np.sqrt(3.0)
        out = vib * self.vib_amp + wob * self.wob_amp
        for t0, amp, decay in self.kerbs:
            dt = t[:, 0] - t0
            hit = np.where(dt >= 0, amp * np.exp(-np.maximum(dt, 0) / decay) * np.sin(2 * np.pi * self.f_vib * np.maximum(dt, 0)), 0.0)
            out[:, 1] += hit
            out[:, 2] += 0.5 * hit
        return out


def render_clip(frames: np.ndarray, times: np.ndarray, rng: np.random.Generator, source: View = WIDE,
                view: View = TRAINING, pose: tuple[float, float, float] = (0.0, 0.0, 0.0), shake_level: float = 0.0,
                exposure_s: float | None = None, blur_samples: int = 4) -> np.ndarray:
    """
    Frames [T, H, W, 3] of one clip, taken at `times` (seconds), seen by `view` with a held
    head `pose` (yaw, pitch, roll) plus camera shake, with exposure blur from the shake.
    Works for a clip or a whole lap: the shake covers every time given.
    """
    times = np.asarray(times, dtype=np.float64)
    rel = times - times[-1]
    shake = Shake(shake_level, rng, start=min(float(rel.min()), 0.0) - 1.0) if shake_level > 0 else None
    exposure = rng.uniform(0.002, 0.010) if exposure_s is None else exposure_s
    out = np.empty((len(frames), view.height, view.width, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        if shake is None:
            out[i] = render(frame, source, view, *pose)
            continue
        offsets = np.linspace(0.0, exposure, blur_samples) if exposure > 0 else np.zeros(1)
        angles = shake.at(rel[i] - offsets) + np.asarray(pose)
        acc = np.zeros((view.height, view.width, 3), dtype=np.float32)
        for yaw, pitch, roll in angles:
            acc += render(frame, source, view, yaw, pitch, roll)
        out[i] = np.clip(acc / len(angles) + 0.5, 0, 255).astype(np.uint8)
    return out
