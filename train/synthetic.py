"""
Procedural track renderer, for checking the pipeline and the architecture
before any AC footage exists.

Not a substitute for AC data. It is a deliberately *easier* version of the task:
landmark colours are distinct per track so perceptual aliasing is mild, and
there is no weather, no tyre smoke, no other cars. Its purpose is to answer
"does the correlation-alignment architecture learn this at all, and does the
data plumbing work end to end", which are exactly the questions the G0 overfit
gate asks.

Two outputs, both matching the real pipeline byte for byte downstream:

    python -m train.synthetic packed    --out data/packed_synth
    python -m train.synthetic recording --out data/fake_session

`packed` writes the same per-lap npz files that `capture.pack` writes, so
`train.dataset` cannot tell the difference. `recording` writes an mp4 with the
timecode grid burned in plus a `run.json`, so the calibrate / decode / pack
chain can be tested without AC.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from capture.pack import reference_index_map, write_lap

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass
class Track:
    name: str
    length_m: float
    centre: np.ndarray  # [M, 2] world (x, z) at ~1 m spacing
    tangent: np.ndarray  # [M, 2] unit
    half_width: float
    post_s: np.ndarray
    post_lateral: np.ndarray
    post_height: np.ndarray
    post_width: np.ndarray
    post_colour: np.ndarray  # [P, 3] BGR
    banner_s: np.ndarray
    banner_colour: np.ndarray
    sky_colour: np.ndarray
    ground_colour: np.ndarray


def build_track(name: str, length_m: float, seed: int) -> Track:
    """A closed loop from a few low-order harmonics, arc-length resampled."""
    rng = np.random.default_rng(seed)

    theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    radius = np.ones_like(theta)
    for order in range(2, 6):
        radius += rng.uniform(0.04, 0.18) * np.cos(order * theta + rng.uniform(0, 2 * np.pi))
    raw = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    step = np.linalg.norm(np.diff(raw, axis=0, append=raw[:1]), axis=1)
    raw *= length_m / step.sum()

    step = np.linalg.norm(np.diff(raw, axis=0, append=raw[:1]), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)[:-1]])
    total = arc[-1] + step[-1]
    grid = np.arange(0.0, total, 1.0)
    closed_arc = np.concatenate([arc, [total]])
    closed_raw = np.concatenate([raw, raw[:1]], axis=0)
    centre = np.stack(
        [np.interp(grid, closed_arc, closed_raw[:, 0]), np.interp(grid, closed_arc, closed_raw[:, 1])],
        axis=1,
    )
    tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    half_width = float(rng.uniform(3.2, 4.6))

    # Dense alternating kerb posts on both edges: continuous structure, so
    # ego-motion is visible even where there is no distinctive landmark.
    kerb_s = np.arange(0.0, total, 6.0)
    kerb = np.concatenate([kerb_s, kerb_s + 3.0])
    kerb_lateral = np.concatenate(
        [np.full(kerb_s.size, half_width + 0.6), np.full(kerb_s.size, -half_width - 0.6)]
    )
    kerb_colour = np.where(
        (np.arange(kerb.size) % 2)[:, None] == 0,
        np.array([[40, 40, 220]]),
        np.array([[240, 240, 240]]),
    ).astype(np.float64)

    # Sparse distinctive landmarks: these carry track identity.
    count = int(total / 18.0)
    mark_s = rng.uniform(0.0, total, count)
    mark_lateral = rng.uniform(half_width + 2.0, half_width + 14.0, count) * rng.choice(
        [-1.0, 1.0], count
    )
    mark_colour = rng.integers(30, 235, size=(count, 3)).astype(np.float64)

    return Track(
        name=name,
        length_m=float(total),
        centre=centre,
        tangent=tangent,
        half_width=half_width,
        post_s=np.concatenate([kerb, mark_s]),
        post_lateral=np.concatenate([kerb_lateral, mark_lateral]),
        post_height=np.concatenate(
            [np.full(kerb.size, 0.8), rng.uniform(1.8, 5.0, count)]
        ),
        post_width=np.concatenate([np.full(kerb.size, 0.22), rng.uniform(0.5, 1.6, count)]),
        post_colour=np.concatenate([kerb_colour, mark_colour], axis=0),
        banner_s=rng.uniform(0.0, total, max(int(total / 120.0), 2)),
        banner_colour=rng.integers(30, 235, size=(max(int(total / 120.0), 2), 3)).astype(np.float64),
        sky_colour=np.array([225.0, 190.0, 150.0]),
        ground_colour=np.array([70.0, 105.0, 75.0]),
    )


def sample_track(track: Track, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolated centre point and unit tangent at arc length `s`."""
    count = track.centre.shape[0]
    position = np.mod(s, track.length_m) * (count / track.length_m)
    i0 = np.floor(position).astype(np.int64) % count
    i1 = (i0 + 1) % count
    frac = (position - np.floor(position))[:, None]
    centre = track.centre[i0] * (1 - frac) + track.centre[i1] * frac
    tangent = track.tangent[i0] * (1 - frac) + track.tangent[i1] * frac
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return centre, tangent


class Camera:
    """Pinhole camera; `project` returns pixel coordinates and camera-space depth."""

    def __init__(self, position: np.ndarray, forward: np.ndarray, size: tuple[int, int], fov_deg: float):
        self.position = position
        self.forward = forward / np.linalg.norm(forward)
        self.right = np.cross(WORLD_UP, self.forward)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.forward, self.right)
        self.width, self.height = size
        self.fx = (self.width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        self.cx, self.cy = self.width / 2.0, self.height / 2.0

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = points - self.position
        depth = delta @ self.forward
        safe = np.maximum(depth, 1e-3)
        u = self.cx + self.fx * (delta @ self.right) / safe
        v = self.cy - self.fx * (delta @ self.up) / safe
        return np.stack([u, v], axis=1), depth


def _poly(image: np.ndarray, pixels: np.ndarray, colour: np.ndarray) -> None:
    clipped = np.clip(pixels, -1e4, 1e4)
    cv2.fillPoly(image, [np.round(clipped).astype(np.int32)], tuple(float(c) for c in colour))


def render_frame(
    track: Track,
    s: float,
    size: tuple[int, int],
    lateral: float,
    camera_height: float,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    brightness: float,
    tint: np.ndarray,
) -> np.ndarray:
    width, height = size
    centre, tangent = sample_track(track, np.array([s]))
    centre, tangent = centre[0], tangent[0]
    normal = np.array([-tangent[1], tangent[0]])

    eye_xz = centre + normal * lateral
    position = np.array([eye_xz[0], camera_height, eye_xz[1]])

    yaw = np.radians(yaw_deg)
    heading = np.array(
        [
            tangent[0] * np.cos(yaw) - tangent[1] * np.sin(yaw),
            0.0,
            tangent[0] * np.sin(yaw) + tangent[1] * np.cos(yaw),
        ]
    )
    heading /= np.linalg.norm(heading)
    heading[1] = np.tan(np.radians(pitch_deg))
    camera = Camera(position, heading, size, fov_deg)

    image = np.empty((height, width, 3), dtype=np.float64)

    # The horizon is the projection of the ground plane at infinity. With no
    # roll it is a single row, so fill sky and ground as two bands instead of
    # projecting a huge quad whose near corners land at absurd coordinates.
    horizon = np.array([[position[0] + heading[0] * 1e6, 0.0, position[2] + heading[2] * 1e6]])
    row = int(np.clip(round(float(camera.project(horizon)[0][0, 1])), 0, height))
    image[:row] = track.sky_colour
    image[row:] = track.ground_colour

    # Road ribbon as independent 2 m quads, painted far to near by *depth*.
    # Arc-length order is not depth order: on a short loop the road 200 m ahead
    # along the track can be a few metres to the side of the camera, and drawing
    # it before the road immediately ahead leaves it smeared across the frame.
    ahead = s + np.arange(-4.0, 170.0, 2.0)
    road_centre, road_tangent = sample_track(track, ahead)
    road_normal = np.stack([-road_tangent[:, 1], road_tangent[:, 0]], axis=1)
    edge_y = np.zeros(ahead.size)
    tarmac = np.array([62.0, 62.0, 66.0])
    gravel = np.array([120.0, 140.0, 150.0])
    line = np.array([218.0, 218.0, 218.0])
    bands = []
    for inner, outer, colour in (
        (track.half_width + 3.0, -track.half_width - 3.0, gravel),
        (track.half_width, -track.half_width, tarmac),
        (track.half_width, track.half_width - 0.3, line),
        (-track.half_width + 0.3, -track.half_width, line),
    ):
        left = road_centre + road_normal * inner
        right = road_centre + road_normal * outer
        pixels_l, depth_l = camera.project(np.stack([left[:, 0], edge_y, left[:, 1]], axis=1))
        pixels_r, depth_r = camera.project(np.stack([right[:, 0], edge_y, right[:, 1]], axis=1))
        bands.append((pixels_l, depth_l, pixels_r, depth_r, colour))

    _, centre_depth = camera.project(
        np.stack([road_centre[:, 0], edge_y, road_centre[:, 1]], axis=1)
    )
    segment_depth = 0.5 * (centre_depth[:-1] + centre_depth[1:])
    for i in np.argsort(-segment_depth):
        if segment_depth[i] < 2.0:
            continue
        for pixels_l, depth_l, pixels_r, depth_r, colour in bands:
            if min(depth_l[i], depth_l[i + 1], depth_r[i], depth_r[i + 1]) < 1.5:
                continue
            quad = np.array([pixels_l[i], pixels_r[i], pixels_r[i + 1], pixels_l[i + 1]])
            if np.abs(quad[:, 0]).max() > 8 * width:
                continue
            _poly(image, quad, colour)

    # Posts and banners, also painted by depth rather than by arc length.
    visible = np.nonzero(np.mod(track.post_s - s, track.length_m) < 200.0)[0]
    if visible.size:
        base_centre, base_tangent = sample_track(track, track.post_s[visible])
        base_normal = np.stack([-base_tangent[:, 1], base_tangent[:, 0]], axis=1)
        foot = base_centre + base_normal * track.post_lateral[visible, None]
        base = np.stack([foot[:, 0], np.zeros(visible.size), foot[:, 1]], axis=1)
        pixels, depth = camera.project(base)
        top = camera.project(base + WORLD_UP * track.post_height[visible, None])[0]
        for i in np.argsort(-depth):
            if depth[i] <= 1.5:
                continue
            index = visible[i]
            half = camera.fx * track.post_width[index] / (2.0 * depth[i])
            if half < 0.35:
                continue
            quad = np.array(
                [
                    [pixels[i, 0] - half, pixels[i, 1]],
                    [pixels[i, 0] + half, pixels[i, 1]],
                    [top[i, 0] + half, top[i, 1]],
                    [top[i, 0] - half, top[i, 1]],
                ]
            )
            _poly(image, quad, track.post_colour[index])

    visible = np.nonzero(np.mod(track.banner_s - s, track.length_m) < 200.0)[0]
    if visible.size:
        banner_centre, banner_tangent = sample_track(track, track.banner_s[visible])
        banner_normal = np.stack([-banner_tangent[:, 1], banner_tangent[:, 0]], axis=1)
        span = track.half_width + 1.5
        _, banner_depth = camera.project(
            np.stack([banner_centre[:, 0], np.full(visible.size, 5.4), banner_centre[:, 1]], axis=1)
        )
        for i in np.argsort(-banner_depth):
            left = banner_centre[i] + banner_normal[i] * span
            right = banner_centre[i] - banner_normal[i] * span
            quad = np.array(
                [
                    [left[0], 4.6, left[1]],
                    [right[0], 4.6, right[1]],
                    [right[0], 6.2, right[1]],
                    [left[0], 6.2, left[1]],
                ]
            )
            pixels, depth = camera.project(quad)
            if (depth > 1.5).all():
                _poly(image, pixels, track.banner_colour[visible[i]])

    image *= brightness * tint
    return np.clip(image, 0, 255).astype(np.uint8)


def signed_curvature(track: Track) -> np.ndarray:
    """
    Radians per metre along the centreline, positive for a left-hand corner.

    Smoothed over ~40 m because a driver reads a corner as one shape and starts
    moving across the track before the curvature actually arrives.
    """
    heading = np.unwrap(np.arctan2(track.tangent[:, 1], track.tangent[:, 0]))
    raw = np.gradient(heading)
    kernel = np.hanning(41)
    kernel /= kernel.sum()
    padded = np.concatenate([raw[-20:], raw, raw[:20]])
    return np.convolve(padded, kernel, mode="same")[20:-20]


def lateral_profile(
    track: Track,
    rng: np.random.Generator,
    bias_m: float,
    apex_gain: float,
    wander_m: float,
) -> np.ndarray:
    """
    Offset from the centreline in metres, sampled at 1 m of arc length.

    Three independent parts, because they fail differently. `bias_m` is a
    constant side-of-the-track preference, which is the knob the cross-line
    experiment turns. `apex_gain` pulls toward the inside of corners, which is
    what separates a racing line from a wet line. `wander_m` is smooth
    low-frequency noise so no two laps trace the same path exactly.

    Clipped to keep the camera on the tarmac; a viewpoint out in the gravel
    would test scene extrapolation rather than line invariance.
    """
    count = track.centre.shape[0]
    grid = np.arange(count)
    limit = max(track.half_width - 0.8, 0.5)

    apex = np.tanh(signed_curvature(track) * 120.0)

    wander = np.zeros(count)
    for order in range(1, 7):
        wander += rng.uniform(-1, 1) * np.cos(
            2 * np.pi * order * grid / count + rng.uniform(0, 2 * np.pi)
        )
    wander *= wander_m / 6.0

    return np.clip(bias_m + apex_gain * limit * apex + wander, -limit, limit)


# A head turns; a bonnet camera does not. These bound how far.
YAW_LIMIT_DEG = 30.0
LOOK_AHEAD_M = 20
LOOK_AHEAD_MAX_DEG = 25.0


def yaw_profile(track: Track, bias_deg: float, look_ahead_gain: float) -> np.ndarray:
    """
    Camera yaw in degrees at 1 m of arc length, positive toward the inside of a
    left-hand corner -- the same sense `lateral_profile` calls positive.

    Mirrors `lateral_profile`, and splits into parts for the same reason: they
    fail differently. `bias_deg` is a constant off-axis heading -- glasses worn
    askew, or a rider who habitually holds their head turned -- and is the knob
    the cross-yaw experiment turns. `look_ahead_gain` turns the camera into a
    corner before arriving at it, which is what a head does and a fixed mount
    does not; it is most of what separates head-worn footage from a bonnet cam.

    Per-frame yaw wander is added by `render_lap`, which already owns the
    temporal noise, so there is none here.

    Clipped to +-30 deg: past that a 73 deg camera has the track leaving frame
    and the sample tests scene extrapolation rather than yaw invariance, the
    same reason `lateral_profile` keeps the camera on the tarmac.
    """
    turn = np.tanh(signed_curvature(track) * 120.0)
    # Look where you are going, not where you are: the curvature LOOK_AHEAD_M
    # further on. The profile is at 1 m spacing, so the roll is in metres.
    look = np.roll(turn, -LOOK_AHEAD_M)
    return np.clip(
        bias_deg + look_ahead_gain * LOOK_AHEAD_MAX_DEG * look,
        -YAW_LIMIT_DEG,
        YAW_LIMIT_DEG,
    )


@dataclass
class LapStyle:
    """Everything that differs between two laps of the same track."""

    lateral: np.ndarray  # [M] metres, indexed at 1 m spacing
    yaw: np.ndarray  # [M] degrees, indexed at 1 m spacing
    speed: np.ndarray  # [M] m/s
    camera_height: float
    yaw_noise_deg: float
    fov_deg: float
    brightness: float
    tint: np.ndarray
    bias_m: float
    apex_gain: float
    yaw_bias_deg: float
    look_ahead_gain: float

    @staticmethod
    def sample(
        track: Track,
        seed: int,
        bias_m: float = 0.0,
        apex_gain: float = 0.0,
        wander_m: float | None = None,
        yaw_bias_deg: float = 0.0,
        look_ahead_gain: float = 0.0,
    ) -> "LapStyle":
        rng = np.random.default_rng(seed)
        count = track.centre.shape[0]
        grid = np.arange(count)

        def smooth(amplitude: float, harmonics: int) -> np.ndarray:
            out = np.zeros(count)
            for order in range(1, harmonics + 1):
                phase = rng.uniform(0, 2 * np.pi)
                out += rng.uniform(-1, 1) * np.cos(2 * np.pi * order * grid / count + phase)
            return out * amplitude / max(harmonics, 1)

        wander = rng.uniform(1.2, 3.0) if wander_m is None else wander_m
        return LapStyle(
            lateral=lateral_profile(track, rng, bias_m, apex_gain, wander),
            speed=np.clip(rng.uniform(16.0, 30.0) + smooth(7.0, 4), 6.0, 45.0),
            camera_height=float(rng.uniform(1.15, 1.85)),
            yaw_noise_deg=float(rng.uniform(0.5, 4.0)),
            fov_deg=float(rng.uniform(62.0, 78.0)),
            brightness=float(rng.uniform(0.72, 1.15)),
            tint=rng.uniform(0.88, 1.12, size=3),
            bias_m=float(bias_m),
            apex_gain=float(apex_gain),
            yaw=yaw_profile(track, yaw_bias_deg, look_ahead_gain),
            yaw_bias_deg=float(yaw_bias_deg),
            look_ahead_gain=float(look_ahead_gain),
        )


def render_lap(
    track: Track, style: LapStyle, size: tuple[int, int], fps: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (frames [N,H,W,3] uint8, s [N] normalised 0..1, t [N] seconds)."""
    rng = np.random.default_rng(seed)
    count = track.centre.shape[0]

    arc: list[float] = [0.0]
    while arc[-1] < track.length_m:
        index = int(arc[-1]) % count
        arc.append(arc[-1] + style.speed[index] / fps)
    arc_array = np.array(arc[:-1])

    # Head yaw wanders slowly and is zero-mean, unlike vehicle heading.
    yaw = np.zeros(arc_array.size)
    for order in range(1, 5):
        yaw += rng.uniform(-1, 1) * np.cos(
            2 * np.pi * order * arc_array / track.length_m + rng.uniform(0, 2 * np.pi)
        )
    yaw *= style.yaw_noise_deg / 4.0
    # Per-lap bias and corner-seeking, on top of that wander. Zero for a
    # dataset generated without --yaw-spread or --look-ahead-gain, so those
    # datasets reproduce exactly.
    yaw = yaw + style.yaw[arc_array.astype(int) % count]
    pitch = -1.5 + 0.6 * np.sin(2 * np.pi * arc_array / max(track.length_m, 1.0) * 3.0)

    frames = np.empty((arc_array.size, size[1], size[0], 3), dtype=np.uint8)
    for i, s in enumerate(arc_array):
        frames[i] = render_frame(
            track,
            float(s),
            size,
            lateral=float(style.lateral[int(s) % count]),
            camera_height=style.camera_height,
            yaw_deg=float(yaw[i]),
            pitch_deg=float(pitch[i]),
            fov_deg=style.fov_deg,
            brightness=style.brightness,
            tint=style.tint,
        )
    s = (arc_array / track.length_m).astype(np.float32)
    t = (np.arange(arc_array.size) / fps).astype(np.float32)
    return frames, s, t


def cmd_packed(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "laps").mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)
    index: list[dict] = []

    for track_i in range(args.tracks):
        name = f"synth{track_i:02d}"
        track = build_track(name, args.length_m, seed=1000 + track_i)
        split = "holdout" if track_i >= args.tracks - args.holdout_tracks else "train"
        n_bins = max(int(round(track.length_m / args.ref_spacing_m)), 8)
        # Yaw walks its own rung of the ladder, permuted per track, so the
        # lap furthest left is not also reliably the lap looking left. Two
        # correlated axes could be satisfied by one cue, and each would
        # contaminate the separation the other's gate sweeps.
        yaw_order = np.random.default_rng(7000 + track_i).permutation(args.laps)
        for lap_i in range(args.laps):
            seed = 100000 + track_i * 100 + lap_i
            # A deterministic ladder of biases rather than random draws, so
            # every dataset is guaranteed to contain widely separated pairs
            # instead of leaving it to chance.
            ladder = (2.0 * lap_i / (args.laps - 1) - 1.0) if args.laps > 1 else 0.0
            bias = args.lateral_spread * ladder
            apex = float(np.random.default_rng(seed).uniform(*args.apex_gain))
            yaw_rung = yaw_order[lap_i]
            yaw_ladder = (2.0 * yaw_rung / (args.laps - 1) - 1.0) if args.laps > 1 else 0.0
            yaw_bias = args.yaw_spread * yaw_ladder
            look_ahead = float(
                np.random.default_rng(seed + 50000).uniform(*args.look_ahead_gain)
            )
            style = LapStyle.sample(
                track,
                seed,
                bias_m=bias,
                apex_gain=apex,
                yaw_bias_deg=yaw_bias,
                look_ahead_gain=look_ahead,
            )
            frames, s, t = render_lap(track, style, size, args.fps, seed)
            lap_id = f"{name}__lap{lap_i:02d}"
            write_lap(
                out / "laps" / lap_id,
                frames=frames,
                s=s,
                t=t,
                ref_idx=reference_index_map(s, n_bins),
            )
            index.append(
                {
                    "lap_id": lap_id,
                    "track": name,
                    "track_config": "",
                    "session_id": f"{name}__synthetic",
                    "car_model": f"synthcar{lap_i % 2}",
                    "split": split,
                    "n_frames": int(frames.shape[0]),
                    "s_span": float(s.max() - s.min()),
                    "track_length_m": float(track.length_m),
                    "ref_bins": int(n_bins),
                    "ref_spacing_m": float(track.length_m / n_bins),
                    "fps": float(args.fps),
                    "frame_size": [size[0], size[1]],
                    "line_bias_m": float(style.bias_m),
                    "line_mean_m": float(style.lateral.mean()),
                    "line_std_m": float(style.lateral.std()),
                    "apex_gain": float(style.apex_gain),
                    "yaw_bias_deg": float(style.yaw_bias_deg),
                    "yaw_mean_deg": float(style.yaw.mean()),
                    "yaw_std_deg": float(style.yaw.std()),
                    "look_ahead_gain": float(style.look_ahead_gain),
                    "path": f"laps/{lap_id}",
                }
            )
            print(
                f"{lap_id}: {frames.shape[0]} frames, {track.length_m:.0f} m, split={split}, "
                f"line mean {style.lateral.mean():+.2f} m (bias {style.bias_m:+.2f}, apex {style.apex_gain:.2f}), "
                f"yaw mean {style.yaw.mean():+.1f} deg (bias {style.yaw_bias_deg:+.1f}, look {style.look_ahead_gain:.2f})"
            )

    (out / "index.json").write_text(json.dumps({"schema": 1, "laps": index}, indent=2))
    print(f"\nwrote {len(index)} laps to {out}")


def cmd_recording(args: argparse.Namespace) -> None:
    """A fake OBS capture: rendered frames with the timecode grid burned in."""
    from capture.timecode import CELL_PX, COLS, ROWS, OverlayGeometry, encode, render_cells

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)
    track = build_track("fake", args.length_m, seed=7)

    geometry = OverlayGeometry(x0=20.0, y0=float(args.height - CELL_PX * ROWS - 20), cell=float(CELL_PX))
    writer = cv2.VideoWriter(
        str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size
    )
    if not writer.isOpened():
        raise SystemExit("could not open an mp4 writer")

    truth: list[tuple[int, float]] = []
    counter = 0
    try:
        for lap_i in range(args.laps):
            style = LapStyle.sample(track, seed=555 + lap_i)
            frames, s, _ = render_lap(track, style, size, args.fps, seed=555 + lap_i)
            for i in range(frames.shape[0]):
                frame = frames[i].copy()
                counter += 1
                grid = np.zeros((CELL_PX * ROWS, CELL_PX * COLS), dtype=np.uint8)
                render_cells(
                    encode(counter, float(s[i])),
                    OverlayGeometry(0.0, 0.0, float(CELL_PX)),
                    grid,
                )
                x0, y0 = int(geometry.x0), int(geometry.y0)
                frame[y0 : y0 + grid.shape[0], x0 : x0 + grid.shape[1]] = grid[:, :, None]
                writer.write(frame)
                truth.append((counter, float(s[i])))
    finally:
        writer.release()

    (out / "run.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "session_id": out.name,
                "video": "video.mp4",
                "track": "fake",
                "track_config": "",
                "track_length_m": float(track.length_m),
                "car_model": "fakecar",
                "ac_version": "synthetic",
                "fps": float(args.fps),
                "resolution": [size[0], size[1]],
                "frame_count": len(truth),
                "split": "train",
                "notes": "synthetic recording for pipeline tests",
                "created": "synthetic",
            },
            indent=2,
        )
    )
    np.savez(out / "truth.npz", counter=np.array([c for c, _ in truth]), s=np.array([v for _, v in truth]))
    (out / "overlay_truth.json").write_text(json.dumps(geometry.as_dict(), indent=2))
    print(f"wrote {len(truth)} frames to {out / 'video.mp4'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--length-m", type=float, default=800.0)
    common.add_argument("--fps", type=float, default=30.0)
    common.add_argument("--laps", type=int, default=5)

    packed = sub.add_parser("packed", parents=[common], help="write packed laps for training")
    packed.add_argument("--out", default="data/packed_synth")
    packed.add_argument("--tracks", type=int, default=4)
    packed.add_argument("--holdout-tracks", type=int, default=1)
    packed.add_argument("--width", type=int, default=160)
    packed.add_argument("--height", type=int, default=96)
    packed.add_argument("--ref-spacing-m", type=float, default=1.0)
    packed.add_argument(
        "--lateral-spread",
        type=float,
        default=0.0,
        help="metres; laps are spread evenly across [-spread, +spread] of the centreline",
    )
    packed.add_argument(
        "--apex-gain",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("LO", "HI"),
        help="per-lap racing-line strength, 0 ignores corners and 1 hugs every apex",
    )
    packed.add_argument(
        "--yaw-spread",
        type=float,
        default=0.0,
        help="degrees; laps are spread evenly across [-spread, +spread] of camera yaw",
    )
    packed.add_argument(
        "--look-ahead-gain",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("LO", "HI"),
        help="per-lap corner-seeking yaw; 0 stares straight ahead, 1 looks hard into apexes",
    )
    packed.set_defaults(func=cmd_packed)

    recording = sub.add_parser("recording", parents=[common], help="write a fake OBS capture")
    recording.add_argument("--out", default="data/fake_session")
    recording.add_argument("--width", type=int, default=1280)
    recording.add_argument("--height", type=int, default=720)
    recording.set_defaults(func=cmd_recording)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
