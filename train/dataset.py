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

    @property
    def directory(self) -> Path:
        return self.root / self.path

    def frames(self) -> np.ndarray:
        return np.load(self.directory / "frames.npy", mmap_mode="r")

    def s(self) -> np.ndarray:
        return np.load(self.directory / "s.npy")

    def t(self) -> np.ndarray:
        return np.load(self.directory / "t.npy")

    def ref_idx(self) -> np.ndarray:
        return np.load(self.directory / "ref_idx.npy")

    def speed_mps(self) -> np.ndarray:
        """Per-frame speed from the labels, used to report delta error in ms."""
        s, t = self.s().astype(np.float64), self.t().astype(np.float64)
        if s.size < 3:
            return np.full(s.size, 1e-3)
        ds = np.gradient(np.unwrap(s * 2 * np.pi) / (2 * np.pi)) * self.track_length_m
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
    ) -> "LapIndex":
        root = Path(root)
        payload = json.loads((root / "index.json").read_text())
        laps: list[Lap] = []
        for entry in payload["laps"]:
            if split is not None and entry.get("split", "train") != split:
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
    ) -> list[str]:
        """Tracks that can supply a full-coverage reference and a different live lap."""
        out = []
        for track in self.by_track:
            references, lives = self.split_group(track, min_ref_span, reference_only, live_only)
            if any(live.lap_id != reference.lap_id for reference in references for live in lives):
                out.append(track)
        return sorted(out)

    def split_group(
        self,
        track: str,
        min_ref_span: float,
        reference_only: frozenset[str] | None = None,
        live_only: frozenset[str] | None = None,
    ) -> tuple[list[Lap], list[Lap]]:
        """(laps usable as reference, laps usable as live) for one track."""
        group = self.by_track[track]
        references = [
            lap
            for lap in group
            if lap.s_span >= min_ref_span
            and (reference_only is None or lap.lap_id in reference_only)
        ]
        lives = [lap for lap in group if live_only is None or lap.lap_id in live_only]
        return references, lives


@dataclass
class SampleConfig:
    batch_size: int = 8
    clip_len: int = 12
    strides: tuple[int, ...] = (1, 2, 3, 4)
    p_reverse: float = 0.25
    p_static: float = 0.03
    sigma_bins: float = 2.0
    roll_reference: bool = True
    jitter: bool = True
    min_ref_span: float = 0.9
    # Restrict which laps may serve each role. Used to hold out whole laps: the
    # reference stays in the training set while the live lap has never been seen.
    reference_only: frozenset[str] | None = None
    live_only: frozenset[str] | None = None


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
    ) -> None:
        self.index = index
        self.config = config
        self.steps = steps
        self.seed = seed
        self.tracks = index.pairable_tracks(
            config.min_ref_span, config.reference_only, config.live_only
        )
        if not self.tracks:
            raise SystemExit(
                "no track has both a full-coverage reference lap and a different live lap; "
                "capture more laps per track"
            )
        self._cache_limit = max(reference_cache, 1)
        # Cached as uint8: a 1000-bin reference is ~10 MB packed against ~120 MB
        # as float32, and the float conversion is negligible next to the encoder.
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.steps

    def _reference_arrays(self, lap: Lap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(frames [N,H,W,3] uint8, bin s [N], bin speed [N]) on the uniform s grid."""
        hit = self._cache.get(lap.lap_id)
        if hit is not None:
            return hit
        ref_idx = lap.ref_idx()
        gathered = np.asarray(lap.frames()[ref_idx])
        value = (gathered, lap.s()[ref_idx], lap.speed_mps()[ref_idx])
        if len(self._cache) >= self._cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[lap.lap_id] = value
        return value

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
        return indices

    def __getitem__(self, step: int) -> dict:
        config = self.config
        rng = np.random.default_rng((self.seed, step))

        track = str(rng.choice(self.tracks))
        references, lives = self.index.split_group(
            track, config.min_ref_span, config.reference_only, config.live_only
        )
        reference_lap = references[int(rng.integers(0, len(references)))]
        live_pool = [lap for lap in lives if lap.lap_id != reference_lap.lap_id]
        if not live_pool:
            # The only eligible live lap is the chosen reference; pick another
            # reference so live and reference are never the same recording.
            reference_lap = next(r for r in references if any(l.lap_id != r.lap_id for l in lives))
            live_pool = [lap for lap in lives if lap.lap_id != reference_lap.lap_id]

        ref_raw, ref_s, ref_speed = self._reference_arrays(reference_lap)
        n_bins = ref_raw.shape[0]
        roll = int(rng.integers(0, n_bins)) if config.roll_reference else 0
        if roll:
            ref_raw = np.roll(ref_raw, roll, axis=0)
            ref_s = np.roll(ref_s, roll)
            ref_speed = np.roll(ref_speed, roll)
        ref_frames = _to_chw(ref_raw)

        clips = np.empty(
            (config.batch_size, config.clip_len) + ref_frames.shape[1:], dtype=np.float32
        )
        target = np.empty(config.batch_size, dtype=np.float64)
        live_s = np.empty(config.batch_size, dtype=np.float64)
        live_ids: list[str] = []
        for i in range(config.batch_size):
            live_lap = live_pool[int(rng.integers(0, len(live_pool)))]
            indices = self._clip_indices(live_lap, rng)
            clip = _to_chw(np.asarray(live_lap.frames()[indices]))
            clips[i] = photometric_jitter(clip, rng) if config.jitter else clip
            # The final frame is the one being localised: causal, matching runtime.
            s_now = float(live_lap.s()[indices[-1]])
            live_s[i] = s_now
            target[i] = (s_now * n_bins + roll) % n_bins
            live_ids.append(live_lap.lap_id)

        reference = photometric_jitter(ref_frames, rng) if config.jitter else ref_frames

        return {
            "live": torch.from_numpy(clips),
            "reference": torch.from_numpy(np.ascontiguousarray(reference)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "soft_target": torch.from_numpy(
                circular_soft_target(target, n_bins, config.sigma_bins)
            ),
            "ref_speed_mps": torch.from_numpy(ref_speed.astype(np.float32)),
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
