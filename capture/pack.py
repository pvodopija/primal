"""
Turn decoded sessions into per-lap arrays ready for training.

    python -m capture.pack data/sessions/<id>
    python -m capture.pack --all

For each session this splits the recording into laps, drops stretches where the
frame counter is not contiguous, crops away the timecode band, resizes, and
writes one npz per lap plus a merged `index.json`.

Laps are split only on the spline wrap and on counter discontinuities. `s` is
kept exactly as decoded, including stationary or slightly backwards stretches:
those are useful training samples, and dropping interior frames would break the
constant-stride assumption that clip sampling relies on.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from capture.timecode import ROWS, OverlayGeometry

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "packed"
CROP_MARGIN_PX = 4
# Largest counter gap kept inside one lap; matches the widest stride in
# train.dataset, so a missed render frame stays inside the training distribution.
MAX_COUNTER_GAP = 4


def write_lap(
    directory: Path, frames: np.ndarray, s: np.ndarray, t: np.ndarray, ref_idx: np.ndarray
) -> None:
    """
    One directory of plain .npy per lap.

    Plain arrays rather than a single npz so `frames` can be opened with
    `mmap_mode='r'`; a 30 minute session at 60 fps is several GB and should not
    have to fit in RAM.
    """
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "frames.npy", frames)
    np.save(directory / "s.npy", s.astype(np.float32))
    np.save(directory / "t.npy", t.astype(np.float32))
    np.save(directory / "ref_idx.npy", ref_idx.astype(np.int32))


def reference_index_map(s: np.ndarray, n_bins: int) -> np.ndarray:
    """
    For each uniform-`s` bin, the index of the nearest frame by `s`.

    Nearest rather than blended: averaging two frames would produce an image no
    camera ever saw. Handles non-monotonic `s` because it sorts first.
    """
    order = np.argsort(s)
    sorted_s = s[order]
    targets = (np.arange(n_bins) + 0.5) / n_bins
    right = np.searchsorted(sorted_s, targets)
    left = np.clip(right - 1, 0, sorted_s.size - 1)
    right = np.clip(right, 0, sorted_s.size - 1)
    take_left = np.abs(sorted_s[left] - targets) <= np.abs(sorted_s[right] - targets)
    return order[np.where(take_left, left, right)].astype(np.int32)


@dataclass
class LapPlan:
    index: int
    start_frame: int
    end_frame: int  # exclusive
    frame_idx: np.ndarray
    s: np.ndarray
    t: np.ndarray

    @property
    def span(self) -> float:
        return float(self.s.max() - self.s.min())


def plan_laps(
    labels: pd.DataFrame, fps: float, min_frames: int, max_gap: int = MAX_COUNTER_GAP
) -> list[LapPlan]:
    good = labels[labels.valid].sort_values("frame_idx")
    if good.empty:
        return []
    frame_idx = good.frame_idx.to_numpy()
    counter = good.counter.to_numpy()
    spline = good.spline_pos.to_numpy().astype(np.float32)

    # AC and OBS run on independent clocks, so a capture either repeats a render
    # frame or misses one. A repeat is a genuine duplicate and is dropped. A miss
    # leaves the surviving frames exactly labelled and evenly spaced in capture
    # time, so it only ends a lap once it exceeds the sampler's largest stride.
    fresh = np.concatenate([[True], np.diff(counter) != 0])
    frame_idx, counter, spline = frame_idx[fresh], counter[fresh], spline[fresh]

    counter_step = np.diff(counter)
    spline_step = np.diff(spline)
    breaks = np.nonzero((counter_step < 1) | (counter_step > max_gap) | (spline_step < -0.5))[0] + 1
    bounds = np.concatenate([[0], breaks, [frame_idx.size]])

    plans: list[LapPlan] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < min_frames:
            continue
        segment = slice(lo, hi)
        plans.append(
            LapPlan(
                index=len(plans),
                start_frame=int(frame_idx[lo]),
                end_frame=int(frame_idx[hi - 1]) + 1,
                frame_idx=frame_idx[segment],
                s=spline[segment],
                t=((frame_idx[segment] - frame_idx[lo]) / fps).astype(np.float32),
            )
        )
    return plans


def crop_rows(height: int, geometry: OverlayGeometry | None) -> tuple[int, int]:
    """Rows to keep, removing the horizontal band the timecode grid sits in."""
    if geometry is None:
        return 0, height
    top = int(np.floor(geometry.y0)) - CROP_MARGIN_PX
    bottom = int(np.ceil(geometry.y0 + geometry.cell * ROWS)) + CROP_MARGIN_PX
    if (top + bottom) / 2 < height / 2:
        return max(bottom, 0), height
    return 0, min(max(top, 1), height)


def rig_line_stats(rig_log: Path) -> tuple[float, float]:
    """
    Mean and spread of the sideways offset a rig render actually applied, from
    samples where the car was moving. The spread is not noise: it is where the
    rig pulled the camera in from a track edge.
    """
    log = pd.read_csv(rig_log)
    moving = log[log["car_spline"].diff().abs() > 1e-6]
    return float(moving["applied_lateral_m"].mean()), float(moving["applied_lateral_m"].std())


def pack_session(
    session: Path,
    out: Path,
    size: tuple[int, int],
    ref_spacing_m: float,
    min_frames: int,
) -> list[dict]:
    meta = json.loads((session / "run.json").read_text())
    labels_path = session / "labels.parquet"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} missing; run `overlay_decode decode` first")
    labels = pd.read_parquet(labels_path)
    # Sessions recorded before the timecode encoded the camera's own position carry
    # the car's centre instead; `label_offset_m` moves those labels to the viewpoint.
    offset_m = float(meta.get("label_offset_m", 0.0))
    if offset_m:
        labels["spline_pos"] = (labels["spline_pos"] + offset_m / float(meta["track_length_m"])) % 1.0

    geometry = None
    overlay_path = session / "overlay.json"
    if overlay_path.exists():
        geometry = OverlayGeometry.from_dict(json.loads(overlay_path.read_text()))

    rig_log = session / "rig_log.csv"
    if rig_log.exists():
        line_mean_m, line_std_m = rig_line_stats(rig_log)
    else:
        line_mean_m, line_std_m = float(meta.get("line_offset_m", 0.0)), 0.0

    fps = float(meta["fps"])
    track_length = float(meta["track_length_m"])
    n_bins = max(int(round(track_length / ref_spacing_m)), 8)
    plans = plan_laps(labels, fps, min_frames)
    if not plans:
        print(f"  {session.name}: no usable laps")
        return []

    video = session / meta["video"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    keep_lo, keep_hi = crop_rows(height, geometry)

    (out / "laps").mkdir(parents=True, exist_ok=True)
    wanted = {int(i): (plan, position) for plan in plans for position, i in enumerate(plan.frame_idx)}
    buffers = {plan.index: np.empty((plan.frame_idx.size, size[1], size[0], 3), np.uint8) for plan in plans}

    frame_i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hit = wanted.get(frame_i)
            if hit is not None:
                plan, position = hit
                cropped = frame[keep_lo:keep_hi]
                buffers[plan.index][position] = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
            frame_i += 1
    finally:
        cap.release()

    entries: list[dict] = []
    for plan in plans:
        lap_id = f"{meta['session_id']}__lap{plan.index:02d}"
        write_lap(
            out / "laps" / lap_id,
            frames=buffers[plan.index],
            s=plan.s,
            t=plan.t,
            ref_idx=reference_index_map(plan.s, n_bins),
        )
        entries.append(
            {
                "lap_id": lap_id,
                # Layouts of one circuit share an AC track name but not a spline, so
                # grouping by name alone would pair laps whose s values do not correspond.
                "track": "__".join(filter(None, (meta["track"], meta.get("track_config", "")))),
                "track_config": meta.get("track_config", ""),
                "session_id": meta["session_id"],
                "car_model": meta.get("car_model", ""),
                # Sideways offset of the rendering camera from the car, for renders
                # made with the camera rig; recorded laps ride on the car, at 0.
                "line_mean_m": line_mean_m,
                "line_std_m": line_std_m,
                "split": meta.get("split", "train"),
                "n_frames": int(plan.frame_idx.size),
                "s_span": plan.span,
                "track_length_m": track_length,
                "ref_bins": int(n_bins),
                "ref_spacing_m": track_length / n_bins,
                "fps": fps,
                "frame_size": [size[0], size[1]],
                "path": f"laps/{lap_id}",
            }
        )
        print(
            f"  {lap_id}: {plan.frame_idx.size} frames, s span {plan.span:.3f}, "
            f"{plan.t[-1]:.1f} s"
        )
    return entries


def merge_index(out: Path, entries: list[dict], session_ids: set[str]) -> None:
    path = out / "index.json"
    existing = []
    if path.exists():
        existing = [
            lap
            for lap in json.loads(path.read_text()).get("laps", [])
            if lap["session_id"] not in session_ids
        ]
    laps = existing + entries
    path.write_text(json.dumps({"schema": 1, "laps": laps}, indent=2))
    tracks = sorted({lap["track"] for lap in laps})
    print(f"\nindex now holds {len(laps)} laps across {len(tracks)} tracks: {', '.join(tracks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", help="session directories")
    parser.add_argument("--all", action="store_true", help="every decoded session under data/sessions")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--ref-spacing-m", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=120)
    args = parser.parse_args()

    paths = [Path(p) for p in args.sessions]
    if args.all:
        paths = sorted(p.parent for p in SESSIONS_DIR.glob("*/labels.parquet"))
    if not paths:
        raise SystemExit("nothing to pack; pass session directories or --all")

    out = Path(args.out)
    entries: list[dict] = []
    session_ids: set[str] = set()
    for session in paths:
        print(f"{session.name}:")
        packed = pack_session(
            session,
            out,
            (args.width, args.height),
            args.ref_spacing_m,
            args.min_frames,
        )
        entries.extend(packed)
        if packed:
            session_ids.add(packed[0]["session_id"])
    merge_index(out, entries, session_ids)


if __name__ == "__main__":
    main()
