"""
Alignment samples: one live clip plus one reference lap, target = where in the
reference lap the clip's final frame is.

Three properties of the sampler carry most of the design:

**One reference per batch.** A reference lap at 1 m spacing is ~1000 frames. If
every sample carried its own reference, a batch of 16 would need 16000 encoder
passes. Batching around a single (track, reference lap, roll) triple brings that
to one reference pass plus B*K live passes.

**The reference is rolled.** The target is an index into the reference array, and
the array is rolled by a random offset every batch. A model that has learned any
absolute notion of track position is therefore wrong on every sample, which
makes the shortcut unlearnable rather than merely discouraged.

**Live and reference are jittered independently.** Otherwise matched exposure is
a valid matching cue and the model will use it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Lap:
    lap_id: str
    track: str
    session_id: str
    car_model: str
    split: str
    n_frames: int
    s_span: float
    track_length_m: float
    ref_bins: int
    ref_spacing_m: float
    fps: float
    root: Path
    path: str
    # Mean offset from the centreline in metres. Synthetic data sets this
    # deliberately so cross-line generalization can be measured; real captures
    # leave it at zero until a driving line is logged.
    line_mean_m: float = 0.0
    line_bias_m: float = 0.0
    # Mean camera yaw in degrees. Synthetic data sets this so cross-yaw
    # generalization can be measured; real captures leave it at zero until
    # a head pose is logged.
    yaw_mean_deg: float = 0.0
    yaw_bias_deg: float = 0.0
    # "live" for a recorded drive; "look" for a re-render of one from its replay
    # with the rig turning the head into corners. A re-render repeats its source
    # drive exactly, so it must never be paired with it as a test.
    variant: str = "live"
    # Horizontal field of view of the render. Wide renders (packed_ac_v3_wide, 121 deg)
    # are not seen as they are: the sampler turns a virtual camera inside them.
    fov_h_deg: float = 91.49

    @property
    def directory(self) -> Path:
        return self.root / self.path

    @property
    def wide(self) -> bool:
        return self.fov_h_deg > 100.0

    def frames(self) -> np.ndarray:
        return np.load(self.directory / "frames.npy", mmap_mode="r")

    def s(self) -> np.ndarray:
        return np.load(self.directory / "s.npy")

    def t(self) -> np.ndarray:
        return np.load(self.directory / "t.npy")

    def reference_grid(self, axis: str = "distance") -> "ReferenceGrid":
        return _cached_grid(self, axis)

    def usable_as_reference(self, min_span: float, axis: str = "distance") -> bool:
        """
        Covers the lap and holds a frame within one bin of every bin.

        A hole in a reference is a bin whose picture shows somewhere else, which
        is a wrong answer baked into the map. A live lap with the same hole is
        harmless, so this only restricts the reference role.
        """
        if self.s_span < min_span:
            return False
        return float(self.reference_grid(axis).placement_m.max()) <= self.ref_spacing_m

    def speed_mps(self) -> np.ndarray:
        """Per-frame speed from the labels."""
        return _speed_from_labels(self.s(), self.t(), self.track_length_m)


def _speed_from_labels(s: np.ndarray, t: np.ndarray, track_length_m: float) -> np.ndarray:
    s, t = np.asarray(s, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if s.size < 3:
        return np.full(s.size, 1e-3)
    ds = np.gradient(np.unwrap(s * 2 * np.pi) / (2 * np.pi)) * track_length_m
    dt = np.gradient(t)
    return np.abs(ds / np.maximum(dt, 1e-6)).clip(0.5, 120.0)


@dataclass
class LapIndex:
    """The packed dataset, grouped by track."""

    laps: list[Lap]
    by_track: dict[str, list[Lap]] = field(init=False)

    def __post_init__(self) -> None:
        self.by_track = {}
        for lap in self.laps:
            self.by_track.setdefault(lap.track, []).append(lap)

    @staticmethod
    def load(
        root: str | Path,
        split: str | None = None,
        tracks: list[str] | None = None,
        max_laps_per_track: int | None = None,
        variants: tuple[str, ...] | None = ("live",),
    ) -> "LapIndex":
        root = Path(root)
        payload = json.loads((root / "index.json").read_text())
        laps: list[Lap] = []
        for entry in payload["laps"]:
            if split is not None and entry.get("split", "train") != split:
                continue
            if variants is not None and entry.get("variant", "live") not in variants:
                continue
            if tracks is not None and entry["track"] not in tracks:
                continue
            laps.append(
                Lap(
                    lap_id=entry["lap_id"],
                    track=entry["track"],
                    session_id=entry.get("session_id", ""),
                    car_model=entry.get("car_model", ""),
                    split=entry.get("split", "train"),
                    n_frames=int(entry["n_frames"]),
                    s_span=float(entry.get("s_span", 1.0)),
                    track_length_m=float(entry["track_length_m"]),
                    ref_bins=int(entry["ref_bins"]),
                    ref_spacing_m=float(entry["ref_spacing_m"]),
                    fps=float(entry["fps"]),
                    root=root,
                    path=entry["path"],
                    line_mean_m=float(entry.get("line_mean_m", 0.0)),
                    line_bias_m=float(entry.get("line_bias_m", 0.0)),
                    yaw_mean_deg=float(entry.get("yaw_mean_deg", 0.0)),
                    yaw_bias_deg=float(entry.get("yaw_bias_deg", 0.0)),
                    variant=entry.get("variant", "live"),
                    fov_h_deg=float(entry.get("fov_h_deg", 91.49)),
                )
            )
        index = LapIndex(laps)
        if max_laps_per_track is not None:
            index = LapIndex(
                [lap for group in index.by_track.values() for lap in group[:max_laps_per_track]]
            )
        return index

    def pairable_tracks(
        self,
        min_ref_span: float,
        reference_only: frozenset[str] | None = None,
        live_only: frozenset[str] | None = None,
        axis: str = "distance",
    ) -> list[str]:
        """Tracks that can supply a full-coverage reference and a different live lap."""
        out = []
        for track in self.by_track:
            references, lives = self.split_group(
                track, min_ref_span, reference_only, live_only, axis
            )
            if any(live.lap_id != reference.lap_id for reference in references for live in lives):
                out.append(track)
        return sorted(out)

    def split_group(
        self,
        track: str,
        min_ref_span: float,
        reference_only: frozenset[str] | None = None,
        live_only: frozenset[str] | None = None,
        axis: str = "distance",
    ) -> tuple[list[Lap], list[Lap]]:
        """(laps usable as reference, laps usable as live) for one track."""
        group = self.by_track[track]
        references = [
            lap
            for lap in group
            if lap.usable_as_reference(min_ref_span, axis)
            and (reference_only is None or lap.lap_id in reference_only)
        ]
        lives = [lap for lap in group if live_only is None or lap.lap_id in live_only]
        return references, lives


def same_drive(a: Lap, b: Lap, frames: float = 2.0, rms_m: float = 0.6) -> bool:
    """
    Whether two whole laps could be one drive rendered twice: a re-render of a replay
    repeats the lap time to the frame and the progress along the track to centimetres.
    AC's bots lap within a few hundredths of each other, so this also catches some
    different drives; it is used to exclude, where erring that way is safe.
    """
    if a.track != b.track or a.s_span < 0.98 or b.s_span < 0.98:
        return False
    ta, tb = a.t().astype(np.float64), b.t().astype(np.float64)
    if abs((ta[-1] - ta[0]) - (tb[-1] - tb[0])) > frames / min(a.fps, b.fps):
        return False
    grid = np.linspace(0.0, min(ta[-1] - ta[0], tb[-1] - tb[0]), 400)
    progress = [np.interp(grid, t - t[0], np.unwrap(lap.s().astype(np.float64), period=1.0) * lap.track_length_m)
                for lap, t in ((a, ta), (b, tb))]
    gap = progress[0] - progress[1]
    return float(np.sqrt(np.mean((gap - np.median(gap)) ** 2))) <= rms_m


REFERENCE_AXES = ("distance", "time")


def _circular_nearest(sorted_u: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index into `sorted_u` (period 1) of the value circularly nearest each query."""
    n = sorted_u.size
    right = np.searchsorted(sorted_u, queries) % n
    left = (right - 1) % n

    def gap(i: np.ndarray) -> np.ndarray:
        d = np.abs(sorted_u[i] - queries)
        return np.minimum(d, 1.0 - d)

    return np.where(gap(left) <= gap(right), left, right)


@dataclass
class ReferenceGrid:
    """
    One reference lap resampled onto N bins along a chosen axis.

    Bin k sits at axis coordinate k/N and holds the frame nearest to it, found
    around the loop so the finish line is not a boundary. The target for a live
    frame is its own axis coordinate times N, so a live frame at the same place
    as bin k's frame targets exactly k.

    `distance` spaces bins evenly along the track. `time` spaces them evenly in
    the reference lap's own elapsed time, so one bin is the same slice of delta
    everywhere: dense in slow corners, sparse on straights. Either way a bin is
    a place on the track. The live clock never moves it.

    Every bin carries its position in metres and its reference time in seconds,
    so errors in both units are read off the grid exactly on either axis.
    """

    axis: str
    frame_idx: np.ndarray
    frame_s: np.ndarray
    speed_mps: np.ndarray
    pos_m: np.ndarray
    time_s: np.ndarray
    placement_m: np.ndarray
    track_length_m: float
    lap_time_s: float
    s_knots: np.ndarray
    tau_knots: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.frame_idx.size)

    def target(self, s: np.ndarray | float) -> np.ndarray:
        """Continuous bin coordinate of track position `s` on this grid."""
        s = np.asarray(s, dtype=np.float64) % 1.0
        if self.axis == "distance":
            u = s
        else:
            u = np.interp(s, self.s_knots, self.tau_knots) / self.lap_time_s
        return (u * self.n_bins) % self.n_bins


def build_reference_grid(
    s: np.ndarray,
    t: np.ndarray,
    speed_mps: np.ndarray,
    n_bins: int,
    track_length_m: float,
    axis: str = "distance",
) -> ReferenceGrid:
    if axis not in REFERENCE_AXES:
        raise ValueError(f"axis must be one of {REFERENCE_AXES}, got {axis!r}")
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    length = float(track_length_m)

    # Walk the lap in recording order and unwrap position along it. Sorting by
    # s instead files a frame that sits exactly on the finish line (s == 1.0,
    # wrapping to 0) at the start of the lap with the lap's latest timestamp.
    x = np.unwrap(s * 2 * np.pi) / (2 * np.pi)
    x -= np.floor(np.median(x))
    # Monotonic, so the time axis stays a valid reparametrisation of position
    # through a stationary or briefly reversing stretch.
    x = np.maximum.accumulate(x)
    x_knots, t_knots = x, t
    # A lap's first and last frames sit a little either side of the line;
    # extend to it at the local speed where the recording stops short.
    if x_knots[0] > 0.0:
        t0 = t_knots[0] - x_knots[0] * length / max(float(speed_mps[0]), 0.5)
        x_knots, t_knots = np.concatenate([[0.0], x_knots]), np.concatenate([[t0], t_knots])
    if x_knots[-1] < 1.0:
        t1 = t_knots[-1] + (1.0 - x_knots[-1]) * length / max(float(speed_mps[-1]), 0.5)
        x_knots, t_knots = np.concatenate([x_knots, [1.0]]), np.concatenate([t_knots, [t1]])
    t_line = float(np.interp(0.0, x_knots, t_knots))
    lap_time = float(np.interp(1.0, x_knots, t_knots)) - t_line
    tau_knots = t_knots - t_line

    u_bins = np.arange(n_bins, dtype=np.float64) / n_bins
    if axis == "distance":
        u_frame = x % 1.0
        pos_m = u_bins * length
        time_s = np.interp(u_bins, x_knots, tau_knots)
    else:
        u_frame = ((t - t_line) / lap_time) % 1.0
        time_s = u_bins * lap_time
        pos_m = np.interp(time_s, tau_knots, x_knots) * length

    u_order = np.argsort(u_frame, kind="stable")
    frame_idx = u_order[_circular_nearest(u_frame[u_order], u_bins)].astype(np.int64)
    frame_s = x[frame_idx] % 1.0
    gap = np.abs(frame_s - pos_m / length)
    return ReferenceGrid(
        axis=axis,
        frame_idx=frame_idx,
        frame_s=frame_s,
        speed_mps=np.asarray(speed_mps, dtype=np.float64)[frame_idx],
        pos_m=pos_m,
        time_s=time_s,
        placement_m=np.minimum(gap, 1.0 - gap) * length,
        track_length_m=length,
        lap_time_s=lap_time,
        s_knots=x_knots,
        tau_knots=tau_knots,
    )


@lru_cache(maxsize=4096)
def _grid_for(directory: Path, n_bins: int, track_length_m: float, axis: str) -> ReferenceGrid:
    s, t = np.load(directory / "s.npy"), np.load(directory / "t.npy")
    speed = _speed_from_labels(s, t, track_length_m)
    return build_reference_grid(s, t, speed, n_bins, track_length_m, axis)


def _cached_grid(lap: Lap, axis: str) -> ReferenceGrid:
    return _grid_for(lap.directory, lap.ref_bins, lap.track_length_m, axis)


AUG_PARTS = ("pose", "blur", "occlude", "vignette", "jpeg")


@dataclass
class SampleConfig:
    batch_size: int = 8
    clip_len: int = 12
    strides: tuple[int, ...] = (1, 2, 3, 4)
    # Which clip frame is localised: "last" is causal, as the live tracker runs;
    # "middle" is halfway between the two central frames, with context on both
    # sides, for placing a finished lap on another after the fact.
    target_at: str = "last"
    p_reverse: float = 0.25
    # Share of clips that turn back at a random frame and retrace it, [a b c d c b]:
    # pace changes inside the clip, so the last frame cannot be found by
    # extrapolating the clip's slope (the pace prior behind the stride lag).
    p_fold: float = 0.0
    # Share of live clips with a kart ahead drawn in (draw_traffic, "blocks").
    p_traffic: float = 0.0
    p_static: float = 0.03
    sigma_bins: float = 2.0
    roll_reference: bool = True
    jitter: bool = True
    min_ref_span: float = 0.9
    reference_axis: str = "distance"
    # Restrict which laps may serve each role. Used to hold out whole laps: the
    # reference stays in the training set while the live lap has never been seen.
    reference_only: frozenset[str] | None = None
    live_only: frozenset[str] | None = None
    # Camera-side augmentation (camera_jitter), drawn separately for every live
    # clip and for the reference, like the photometric jitter.
    camera_aug: bool = False
    # Zoom range and pitch amplitude for camera_jitter; see its docstring.
    aug_zoom: tuple[float, float] = (1.03, 1.12)
    aug_pitch_deg: float = 3.0
    aug_yaw_deg: float = 4.0
    aug_parts: frozenset[str] = frozenset(AUG_PARTS)
    # Probability that a step mirrors its reference and all its clips together:
    # a mirror-image circuit, a plausible track the data does not contain.
    mirror_p: float = 0.0
    # Wide laps go through the virtual camera (train/camera.py): every live clip is seen
    # with a head pose held over the clip, drawn uniformly within these amplitudes and
    # exact (no borders) up to about 14 deg of yaw; the reference with camera_jitter's
    # amplitudes (aug_yaw_deg, aug_pitch_deg, 6 deg roll), exactly too.
    head_yaw_deg: float = 14.0
    head_pitch_deg: float = 4.0
    head_roll_deg: float = 8.0
    # Camera shake on every live clip, wide or not: vibration, kerb hits and head wobble
    # with exposure blur, at a level drawn uniformly from [0, shake_max] (1 = a typical
    # kart); the reference gets half that, without blur. 0 = off.
    shake_max: float = 0.0


def _to_chw(frames: np.ndarray) -> np.ndarray:
    """uint8 [..., H, W, 3] BGR -> float32 [..., 3, H, W] in [0, 1]."""
    return np.ascontiguousarray(np.moveaxis(frames, -1, -3)).astype(np.float32) / 255.0


def photometric_jitter(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    One exposure per clip plus mild per-frame flicker.

    Applied independently to live and reference so that matched exposure cannot
    serve as a matching cue.
    """
    gamma = rng.uniform(0.8, 1.3)
    gain = rng.uniform(0.75, 1.3)
    bias = rng.uniform(-0.08, 0.08)
    channel = rng.uniform(0.9, 1.1, size=(3, 1, 1)).astype(np.float32)
    flicker_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    out = np.power(np.clip(x, 0.0, 1.0), gamma, dtype=np.float32)
    out *= (gain * channel).astype(np.float32)
    out *= rng.uniform(0.97, 1.03, size=flicker_shape).astype(np.float32)
    out += np.float32(bias)
    out += rng.normal(0.0, 0.012, size=x.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0, out=out)


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    size = max(3, int(np.ceil(length)) | 1)
    kernel = np.zeros((size, size), np.float32)
    c = size // 2
    dx, dy = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    for t in np.linspace(-length / 2, length / 2, 4 * size):
        kernel[int(round(c + t * dy)), int(round(c + t * dx))] = 1.0
    return kernel / kernel.sum()


def camera_jitter(
    frames: np.ndarray,
    rng: np.random.Generator,
    hfov_deg: float = 91.5,
    zoom: tuple[float, float] = (1.03, 1.12),
    pitch_deg: float = 3.0,
    parts: frozenset[str] = frozenset(AUG_PARTS),
    yaw_deg: float = 4.0,
) -> np.ndarray:
    """
    Camera-side augmentation for uint8 frames [T, H, W, 3] of one clip, or of a
    whole reference lap. Only what a different camera or head would change, never
    the scene, since on a kart track the scene is the signal:

    - head pose: one roll (+-6 deg), yaw (+-`yaw_deg`, default 4), pitch (+-`pitch_deg`) and zoom
      (drawn from `zoom`) for the sequence, plus per-frame shake (0.3 px, 0.3 deg).
      Zoom and pitch are also distance cues: things look bigger, and the road
      sits lower, the closer they are. Drawn independently for live and
      reference they teach the model to ignore its own sense of distance, so a
      fixed zoom (only to hide the borders) and a small pitch keep precision;
    - blur (30%): out of focus, or a short streak as from vibration or motion;
    - occluders (25%): one or two blank patches fixed in the frame, as a hand, a
      steering wheel or a kart ahead;
    - vignetting (30%) and JPEG compression (30%, quality 25-85).

    `parts` switches effects off individually (pose, blur, occlude, vignette,
    jpeg); every random draw is still made, so the others are unchanged.
    """
    import cv2

    count, height, width = frames.shape[:3]
    focal = (width / 2) / np.tan(np.radians(hfov_deg / 2))
    roll = rng.uniform(-6, 6)
    shift_x = focal * np.tan(np.radians(rng.uniform(-yaw_deg, yaw_deg)))
    shift_y = focal * np.tan(np.radians(rng.uniform(-pitch_deg, pitch_deg)))
    zoom = rng.uniform(*zoom)
    shake = rng.normal(0.0, [0.3, 0.3, 0.3], size=(count, 3))
    blur = rng.random() < 0.3
    blur_kernel = None
    if blur and rng.random() < 0.5:
        blur_kernel = _motion_kernel(rng.uniform(1.5, 4.0), rng.uniform(60, 120))
    blur_sigma = rng.uniform(0.3, 1.0)
    patches = []
    if rng.random() < 0.25:
        for _ in range(int(rng.integers(1, 3))):
            area = rng.uniform(0.05, 0.2) * height * width
            aspect = rng.uniform(0.5, 2.0)
            h = int(min(height, np.sqrt(area / aspect)))
            w = int(min(width, np.sqrt(area * aspect)))
            y, x = int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1))
            patches.append((y, x, h, w, rng.integers(0, 256, size=3)))
    vignette = None
    if rng.random() < 0.3:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        vignette = (1.0 - rng.uniform(0.1, 0.4) * r2 / 2.0)[..., None]
    jpeg = rng.random() < 0.3
    quality = int(rng.integers(25, 86))

    blur = blur and "blur" in parts
    patches = patches if "occlude" in parts else []
    vignette = vignette if "vignette" in parts else None
    jpeg = jpeg and "jpeg" in parts

    out = np.empty_like(frames)
    center = (width / 2, height / 2)
    for i in range(count):
        if "pose" in parts:
            matrix = cv2.getRotationMatrix2D(center, roll + shake[i, 2], zoom)
            matrix[0, 2] += shift_x + shake[i, 0]
            matrix[1, 2] += shift_y + shake[i, 1]
            frame = cv2.warpAffine(frames[i], matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
        else:
            frame = frames[i].copy()
        if blur:
            frame = (cv2.filter2D(frame, -1, blur_kernel) if blur_kernel is not None
                     else cv2.GaussianBlur(frame, (0, 0), blur_sigma))
        for y, x, h, w, colour in patches:
            frame[y : y + h, x : x + w] = colour
        if vignette is not None:
            frame = np.clip(frame.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
        if jpeg:
            ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if ok else frame
        out[i] = frame
    return out


def draw_traffic(frames: np.ndarray, rng: np.random.Generator, times_s: np.ndarray, style: str = "blocks",
                 hfov_deg: float = 91.5, camera_height_m: float = 1.15) -> np.ndarray:
    """
    A kart ahead on the road, drawn into uint8 frames [T, H, W, 3] of one stretch of a lap.
    It stands on the road in perspective (smaller and higher the farther away) and moves
    as a kart one follows does: closing or pulling away, drifting sideways, consistently
    over the frames' times.

    "blocks" builds it from a few dark rectangles, for training. "kart" is a different
    silhouette (a coloured body, dark wheels, driver and helmet), for testing robustness
    without testing the training generator.
    """
    import cv2

    out = np.ascontiguousarray(frames.copy())
    count, height, width = frames.shape[:3]
    focal = (width / 2) / np.tan(np.radians(hfov_deg / 2))
    horizon = 0.6 * height  # where AC's road meets the sky at the rig's height and pitch
    distance = float(np.exp(rng.uniform(np.log(2.5), np.log(25.0))))  # close karts as often as far ones
    closing = rng.uniform(-1.5, 1.5)  # m/s; positive closes in
    lateral = rng.uniform(-2.5, 2.5)
    drift = rng.uniform(-0.8, 0.8)  # m/s sideways
    if style == "blocks":
        shade = rng.integers(10, 70, size=3)
        # (centre, half width, bottom, top) in metres, relative to the kart and the road
        parts = [(rng.uniform(-0.4, 0.4), rng.uniform(0.2, 0.7), rng.uniform(0.0, 0.3), rng.uniform(0.4, 1.1),
                  np.clip(shade + rng.integers(-10, 11, size=3), 0, 255)) for _ in range(int(rng.integers(2, 5)))]
    else:
        body = rng.integers(40, 230, size=3)
        helmet = rng.integers(20, 240, size=3)
        dark = np.array([20, 20, 20])
        parts = [(0.0, 0.65, 0.10, 0.45, body), (-0.6, 0.13, 0.0, 0.28, dark), (0.6, 0.13, 0.0, 0.28, dark),
                 (0.0, 0.25, 0.40, 0.85, dark + 30)]
    last = float(times_s[-1])
    for k in range(count):
        ago = last - float(times_s[k])
        d = float(np.clip(distance + closing * ago, 2.0, 40.0))
        x0 = lateral - drift * ago
        for centre, half, bottom, top, colour in parts:
            u0 = int(round(width / 2 + focal * (x0 + centre - half) / d))
            u1 = int(round(width / 2 + focal * (x0 + centre + half) / d))
            v0 = int(round(horizon + focal * (camera_height_m - top) / d))
            v1 = int(round(horizon + focal * (camera_height_m - bottom) / d))
            if u1 >= 0 and u0 < width and v1 >= 0 and v0 < height and u1 > u0 and v1 > v0:
                cv2.rectangle(out[k], (u0, v0), (u1, v1), tuple(int(c) for c in colour), thickness=-1)
        if style != "blocks":
            centre_uv = (int(round(width / 2 + focal * x0 / d)), int(round(horizon + focal * (camera_height_m - 0.98) / d)))
            cv2.circle(out[k], centre_uv, max(1, int(round(focal * 0.14 / d))), tuple(int(c) for c in helmet), thickness=-1)
    return out


def traffic_episodes(frames: np.ndarray, times_s: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    """A whole lap with a kart ahead for about `share` of it, in 3-8 s stretches ("kart" style)."""
    out = np.asarray(frames).copy()
    total = float(times_s[-1] - times_s[0])
    covered, guard = 0.0, 0
    while covered < share * total and guard < 1000:
        guard += 1
        length = rng.uniform(3.0, 8.0)
        start = rng.uniform(float(times_s[0]), max(float(times_s[0]), float(times_s[-1]) - length))
        span = np.flatnonzero((times_s >= start) & (times_s <= start + length))
        if span.size < 2:
            continue
        out[span] = draw_traffic(out[span], rng, times_s[span].astype(np.float64), style="kart")
        covered += length
    return out


def prepare_frames(
    raw: np.ndarray, rng: np.random.Generator, camera_aug: bool, mirror: bool,
    zoom: tuple[float, float] = (1.03, 1.12), pitch_deg: float = 3.0,
    parts: frozenset[str] = frozenset(AUG_PARTS), yaw_deg: float = 4.0,
) -> np.ndarray:
    """uint8 [T, H, W, 3] -> float32 [T, 3, H, W], mirrored and camera-jittered as asked."""
    if mirror:
        raw = np.ascontiguousarray(raw[:, :, ::-1])
    if camera_aug:
        raw = camera_jitter(raw, rng, zoom=zoom, pitch_deg=pitch_deg, parts=parts, yaw_deg=yaw_deg)
    return _to_chw(raw)


def circular_soft_target(target: np.ndarray, n_bins: int, sigma_bins: float) -> np.ndarray:
    """Gaussian over bins, wrapped at the lap boundary, normalised per row."""
    bins = np.arange(n_bins, dtype=np.float64)[None, :]
    offset = (bins - target[:, None] + n_bins / 2.0) % n_bins - n_bins / 2.0
    weight = np.exp(-0.5 * (offset / sigma_bins) ** 2)
    return (weight / weight.sum(axis=1, keepdims=True)).astype(np.float32)


class AlignmentBatches(Dataset):
    """
    Yields whole batches, not samples: every item is one (reference, B clips)
    group, so `DataLoader(..., batch_size=None)` is the right way to consume it.
    """

    def __init__(
        self,
        index: LapIndex,
        config: SampleConfig,
        steps: int,
        seed: int = 0,
        reference_cache: int = 8,
        reference_cache_bytes: float = 1.5e9,
    ) -> None:
        self.index = index
        self.config = config
        self.steps = steps
        self.seed = seed
        self.tracks = index.pairable_tracks(
            config.min_ref_span, config.reference_only, config.live_only, config.reference_axis
        )
        if not self.tracks:
            raise SystemExit(
                "no track has both a full-coverage reference lap and a different live lap; "
                "capture more laps per track"
            )
        self._cache_limit = max(reference_cache, 1)
        # A wide reference is ~4x a normal one (up to ~430 MB for Black Cat County).
        self._cache_bytes = reference_cache_bytes
        # Cached as uint8: a 1000-bin reference is ~10 MB packed against ~120 MB
        # as float32, and the float conversion is negligible next to the encoder.
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.steps

    def _reference(self, lap: Lap) -> tuple[np.ndarray, ReferenceGrid]:
        """(frames [N,H,W,3] uint8, grid) for one reference lap."""
        hit = self._cache.get(lap.lap_id)
        if hit is not None:
            return hit
        grid = lap.reference_grid(self.config.reference_axis)
        value = (np.asarray(lap.frames()[grid.frame_idx]), grid)
        while self._cache and (len(self._cache) >= self._cache_limit or
                               sum(v[0].nbytes for v in self._cache.values()) + value[0].nbytes > self._cache_bytes):
            self._cache.pop(next(iter(self._cache)))
        self._cache[lap.lap_id] = value
        return value

    def _camera(self, lap: Lap, raw: np.ndarray, times: np.ndarray, rng: np.random.Generator,
                live: bool) -> tuple[np.ndarray, frozenset[str]]:
        """
        Frames as the training camera sees them, and the camera_jitter parts still to
        apply. A wide lap is rendered through the virtual camera, which replaces
        camera_jitter's approximate pose; a normal lap only gets the shake. Draws nothing
        when neither applies, so runs without them sample exactly what they always did.
        """
        config = self.config
        if not lap.wide and config.shake_max <= 0.0:
            return raw, config.aug_parts
        from train import camera

        level = rng.uniform(0.0, config.shake_max) * (1.0 if live else 0.5) if config.shake_max > 0.0 else 0.0
        pose = (0.0, 0.0, 0.0)
        if lap.wide:
            amplitude = ((config.head_yaw_deg, config.head_pitch_deg, config.head_roll_deg) if live
                         else (config.aug_yaw_deg, config.aug_pitch_deg, 6.0))
            pose = tuple(float(rng.uniform(-a, a)) for a in amplitude)
        elif level == 0.0:
            return raw, config.aug_parts
        frames = camera.render_clip(raw, times, rng, source=camera.WIDE if lap.wide else camera.TRAINING,
                                    view=camera.TRAINING, pose=pose, shake_level=level,
                                    exposure_s=None if live else 0.0)
        return frames, (config.aug_parts - {"pose"} if lap.wide else config.aug_parts)

    def _clip_indices(self, lap: Lap, rng: np.random.Generator) -> np.ndarray:
        clip_len = self.config.clip_len
        if rng.random() < self.config.p_static:
            start = int(rng.integers(0, lap.n_frames))
            return np.full(clip_len, start, dtype=np.int64)

        usable = [d for d in self.config.strides if (clip_len - 1) * d + 1 <= lap.n_frames]
        stride = int(rng.choice(usable)) if usable else 1
        span = (clip_len - 1) * stride + 1
        start = int(rng.integers(0, max(lap.n_frames - span + 1, 1)))
        indices = start + np.arange(clip_len, dtype=np.int64) * stride
        indices = np.clip(indices, 0, lap.n_frames - 1)
        if rng.random() < self.config.p_reverse:
            indices = indices[::-1].copy()
        # Drawn only when enabled, so runs without folds sample exactly what they always did.
        if self.config.p_fold > 0.0 and rng.random() < self.config.p_fold:
            pivot = int(rng.integers(1, clip_len - 1))
            k = np.arange(clip_len)
            steps = np.where(k <= pivot, k, 2 * pivot - k)  # forward to the pivot, then back past it
            step = int(indices[1] - indices[0]) if clip_len > 1 else 0
            indices = np.clip(indices[0] + steps * step, 0, lap.n_frames - 1).astype(np.int64)
        return indices

    def __getitem__(self, step: int) -> dict:
        config = self.config
        rng = np.random.default_rng((self.seed, step))

        track = str(rng.choice(self.tracks))
        references, lives = self.index.split_group(
            track, config.min_ref_span, config.reference_only, config.live_only,
            config.reference_axis,
        )
        reference_lap = references[int(rng.integers(0, len(references)))]
        live_pool = [lap for lap in lives if lap.lap_id != reference_lap.lap_id]
        if not live_pool:
            # The only eligible live lap is the chosen reference; pick another
            # reference so live and reference are never the same recording.
            reference_lap = next(r for r in references if any(l.lap_id != r.lap_id for l in lives))
            live_pool = [lap for lap in lives if lap.lap_id != reference_lap.lap_id]

        ref_raw, grid = self._reference(reference_lap)
        n_bins = grid.n_bins
        roll = int(rng.integers(0, n_bins)) if config.roll_reference else 0
        ref_raw = np.roll(ref_raw, roll, axis=0)
        ref_raw, ref_parts = self._camera(reference_lap, ref_raw,
                                          np.roll(reference_lap.t()[grid.frame_idx].astype(np.float64), roll), rng, live=False)
        ref_s, ref_speed = np.roll(grid.frame_s, roll), np.roll(grid.speed_mps, roll)
        ref_pos, ref_time = np.roll(grid.pos_m, roll), np.roll(grid.time_s, roll)
        # Drawn only when enabled, so runs without these augmentations sample
        # exactly what they always did.
        mirror = bool(config.mirror_p > 0.0 and rng.random() < config.mirror_p)
        ref_frames = prepare_frames(
            ref_raw, rng, config.camera_aug, mirror, config.aug_zoom, config.aug_pitch_deg, ref_parts,
            yaw_deg=config.aug_yaw_deg,
        )

        clips = np.empty(
            (config.batch_size, config.clip_len) + ref_frames.shape[1:], dtype=np.float32
        )
        target = np.empty(config.batch_size, dtype=np.float64)
        frame_targets = np.empty((config.batch_size, config.clip_len), dtype=np.float64)
        live_s = np.empty(config.batch_size, dtype=np.float64)
        live_ids: list[str] = []
        for i in range(config.batch_size):
            live_lap = live_pool[int(rng.integers(0, len(live_pool)))]
            indices = self._clip_indices(live_lap, rng)
            raw = np.asarray(live_lap.frames()[indices])
            raw, parts = self._camera(live_lap, raw, live_lap.t()[indices].astype(np.float64), rng, live=True)
            # Drawn only when enabled, so runs without traffic sample exactly what they always did.
            if config.p_traffic > 0.0 and rng.random() < config.p_traffic:
                raw = draw_traffic(raw, rng, live_lap.t()[indices].astype(np.float64))
            clip = prepare_frames(
                raw, rng, config.camera_aug, mirror,
                config.aug_zoom, config.aug_pitch_deg, parts, yaw_deg=config.aug_yaw_deg,
            )
            clips[i] = photometric_jitter(clip, rng) if config.jitter else clip
            # The final frame is the one being localised: causal, matching runtime.
            frame_s = live_lap.s()[indices]
            s_now = float(frame_s[-1])
            if config.target_at == "middle":
                a, b = frame_s[config.clip_len // 2 - 1], frame_s[config.clip_len // 2]
                s_now = float((a + 0.5 * (((b - a) + 0.5) % 1.0 - 0.5)) % 1.0)
            live_s[i] = s_now
            target[i] = (float(grid.target(s_now)) + roll) % n_bins
            # Where every clip frame sits, for supervising each frame's own row.
            frame_targets[i] = (grid.target(frame_s) + roll) % n_bins
            live_ids.append(live_lap.lap_id)

        reference = photometric_jitter(ref_frames, rng) if config.jitter else ref_frames

        return {
            "live": torch.from_numpy(clips),
            "reference": torch.from_numpy(np.ascontiguousarray(reference)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "frame_targets": torch.from_numpy(frame_targets.astype(np.float32)),
            "soft_target": torch.from_numpy(
                circular_soft_target(target, n_bins, config.sigma_bins)
            ),
            "ref_speed_mps": torch.from_numpy(ref_speed.astype(np.float32)),
            # Position and reference time of every bin after the roll: errors
            # in metres and milliseconds are both read off these.
            "ref_pos_m": torch.from_numpy(ref_pos.astype(np.float32)),
            "ref_time_s": torch.from_numpy(ref_time.astype(np.float32)),
            "track_length_m": grid.track_length_m,
            "lap_time_s": grid.lap_time_s,
            # Diagnostics: `ref_s` is the true s at each reference bin after the
            # roll, `live_s` the true s of each clip's final frame. Tests assert
            # that the target actually indexes the matching reference frame.
            "ref_s": torch.from_numpy(ref_s.astype(np.float32)),
            "live_s": torch.from_numpy(live_s.astype(np.float32)),
            "ref_spacing_m": float(reference_lap.ref_spacing_m),
            "track": track,
            "reference_lap": reference_lap.lap_id,
            "live_laps": live_ids,
            "roll": roll,
        }
