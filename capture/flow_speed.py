"""
Speed from the road's motion between frames: tracked points and a ground plane.

    python -m capture.flow_speed run data/sessions/<run>/video.mp4
    python -m capture.flow_speed run data/sessions/<run>/video.mp4 --skip 2   # as if 30 fps

The planned speed source for camera glasses (docs/ml-pivot.md, "The speed
lane"), measured on AC recordings first. Each frame is scaled to a target
camera's focal length (Halo by default, so the test sees Halo's detail, not
the recording's), and a band of road 3-10 m ahead is cut out. Textured points
there are tracked into the next frame and back again; points that do not
return to where they started are dropped. The survivors are fitted to a flat
road seen by a camera moving forward and sideways and turning in yaw, pitch
and roll. Forward motion over the elapsed time is the speed.

The camera's height above the road sets the scale, and it is not known in the
product. The filter's scale state absorbs that, so what this measures is
whether the speed is right *up to one scale*: flat across speed bands.

Writes `flow_speed[_skipN].npz` next to the video (or `--out`):

    frame     [F]     video frame index, as `labels.parquet` counts them
    speed     [F]     m/s at the camera height given; NaN where there was no fit
    motion    [F, 5]  forward m, sideways m, yaw, pitch, roll rad over the pair
    points    [F]     points tracked; `kept` [F] the share that tracked back
    residual  [F]     px, RMS of the fitted flow's error over kept points
    dt        [F]     seconds between the pair (AC's render counter when labels exist)

With `labels.parquet` and `run.json` next to the video it prints a check
against the labelled speed, per frame and per 15 Hz tick.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HALO_FOCAL_PX = 320.0 / np.tan(np.radians(81.2 / 2))  # 640 px across 81.2°
NEAR_M, FAR_M = 3.0, 10.0
SPEED_WINDOW = 6  # label frames either side for the labelled speed
TICK_FRAMES = 4  # 60 fps frames per 15 Hz tick


def motion_bases(
    points: np.ndarray, focal: float, horizon: float, height: float, destination: np.ndarray | None = None
) -> np.ndarray:
    """
    Image flow per unit of each motion, at `points` (N, 2) given relative to the
    principal point, x right and y down: [5, N, 2] for forward (per metre),
    sideways (per metre), yaw, pitch, roll (per radian). Forward and sideways
    motion over a flat road `height` below the camera; small rotations anywhere.

    With `destination` (where each point was found in the next frame), the
    translation terms are exact rather than first-order: a road point at depth
    Z seen from d metres closer moves by g0 * g1 * d / (f h), not g0^2 * d / (f h),
    which reads 5-10% fast for the near road at 0.25-0.5 m per frame.
    """
    x, y = points[:, 0], points[:, 1]
    g = y - horizon
    g1 = g if destination is None else destination[:, 1] - horizon
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([x * g1 / (focal * height), g * g1 / (focal * height)], -1),
            np.stack([-g1 / height, zero], -1),
            np.stack([focal + x * x / focal, x * y / focal], -1),
            np.stack([x * y / focal, focal + y * y / focal], -1),
            np.stack([-y, x], -1),
        ]
    )


def fit_motion(
    points: np.ndarray, flow: np.ndarray, focal: float, horizon: float, height: float, iters: int = 4
) -> tuple[np.ndarray, float, np.ndarray]:
    """Huber-weighted least squares: (motion [5], RMS residual px, inlier mask)."""
    bases = motion_bases(points, focal, horizon, height, destination=points + flow)
    design = np.concatenate([bases[:, :, 0].T, bases[:, :, 1].T])
    target = np.concatenate([flow[:, 0], flow[:, 1]])
    weight = np.ones_like(target)
    for _ in range(iters):
        root = np.sqrt(weight)
        motion, *_ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
        error = target - design @ motion
        scale = 1.4826 * np.median(np.abs(error)) + 1e-3
        k = np.abs(error) / (1.5 * scale)
        weight = np.where(k <= 1.0, 1.0, 1.0 / k)
    n = len(points)
    inlier = (weight[:n] >= 0.5) & (weight[n:] >= 0.5)
    rms = float(np.sqrt(np.mean(error.reshape(2, n).T[inlier] ** 2))) if inlier.any() else float("nan")
    return motion, rms, inlier


def track(previous: np.ndarray, current: np.ndarray, max_points: int = 200, fb_px: float = 0.5):
    """Points in `previous`, where they went in `current`, and whether they tracked back."""
    found = cv2.goodFeaturesToTrack(previous, max_points, 0.005, 6, blockSize=5)
    if found is None:
        empty = np.zeros((0, 2), np.float32)
        return empty, empty, np.zeros(0, bool)
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    ahead, ok_ahead, _ = cv2.calcOpticalFlowPyrLK(previous, current, found, None, **lk)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, ahead, None, **lk)
    returned = np.linalg.norm((back - found).reshape(-1, 2), axis=1) < fb_px
    kept = ok_ahead.ravel().astype(bool) & ok_back.ravel().astype(bool) & returned
    return found.reshape(-1, 2), ahead.reshape(-1, 2), kept


class RoadStrip:
    """Scales frames to the target focal length and cuts the road band out."""

    def __init__(self, width: int, height: int, source_focal: float, focal: float,
                 camera_height: float, horizon: float = 0.0, keep_rows: tuple[int, int] | None = None):
        self.scale = focal / source_focal
        self.size = (int(round(width * self.scale)), int(round(height * self.scale)))
        self.focal = focal
        self.horizon = horizon
        self.camera_height = camera_height
        cy = self.size[1] / 2
        top = int(cy + horizon + focal * camera_height / FAR_M)
        bottom = int(cy + horizon + focal * camera_height / NEAR_M)
        if keep_rows is not None:
            bottom = min(bottom, int(keep_rows[1] * self.scale) - 4)
        self.rows = (max(top, 0), min(bottom, self.size[1]))
        margin = int(self.size[0] * 0.15)
        self.cols = (margin, self.size[0] - margin)
        # the band's own pixel (0, 0), relative to the principal point
        self.offset = np.array([self.cols[0] - self.size[0] / 2, self.rows[0] - cy], np.float32)

    def cut(self, frame_bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return grey[self.rows[0] : self.rows[1], self.cols[0] : self.cols[1]]

    def measure(self, previous: np.ndarray, current: np.ndarray):
        p0, p1, kept = track(previous, current)
        if kept.sum() < 12:
            return None, len(p0), float(kept.mean()) if len(p0) else 0.0, float("nan")
        points = p0[kept] + self.offset
        motion, rms, _ = fit_motion(points, p1[kept] - p0[kept], self.focal, self.horizon, self.camera_height)
        return motion, len(p0), float(kept.mean()), rms


def _labelled(session: Path, fps: float):
    """(frame -> labelled speed m/s, frame -> render counter, render Hz) from labels.parquet."""
    import pandas as pd

    labels = pd.read_parquet(session / "labels.parquet").sort_values("frame_idx")
    length = float(json.loads((session / "run.json").read_text())["track_length_m"])
    good = labels[labels.valid.to_numpy()]
    frames, counter = good.frame_idx.to_numpy(), good.counter.to_numpy().astype(np.float64)
    render_hz = (counter[-1] - counter[0]) / ((frames[-1] - frames[0]) / fps)
    fresh = np.concatenate([[True], np.diff(counter) != 0])
    f2, c2 = frames[fresh], counter[fresh]
    x = np.unwrap(good.spline_pos.to_numpy(np.float64)[fresh] * 2 * np.pi) / (2 * np.pi) * length
    w = SPEED_WINDOW
    speed = np.full(f2.size, np.nan)
    speed[w:-w] = (x[2 * w :] - x[: -2 * w]) / ((c2[2 * w :] - c2[: -2 * w]) / render_hz)
    return dict(zip(f2.tolist(), speed.tolist())), dict(zip(frames.tolist(), counter.tolist())), render_hz


def _report(name: str, est: np.ndarray, true: np.ndarray) -> None:
    ok = np.isfinite(est) & np.isfinite(true) & (true > 3) & (true < 120)
    if ok.sum() < 50:
        print(f"  {name}: only {int(ok.sum())} usable values")
        return
    e, t = est[ok], true[ok]
    rel = np.abs(e - t) / t
    bands = []
    for lo, hi in [(0, 15), (15, 25), (25, 35), (35, 45), (45, 120)]:
        m = (t >= lo) & (t < hi)
        if m.sum() >= 30:
            bands.append(f"{lo}-{hi}: {np.median(e[m] / t[m]):.3f}")
    print(f"  {name}: n {ok.sum()}, median |error| {100 * np.median(rel):.1f}%, p90 {100 * np.percentile(rel, 90):.1f}%, "
          f"correlation {np.corrcoef(e, t)[0, 1]:.3f}")
    print(f"    estimate / true by speed band (flat = unbiased up to one scale): {', '.join(bands)}")


def cmd_run(args: argparse.Namespace) -> None:
    video = Path(args.video)
    session = video.parent
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta = json.loads((session / "run.json").read_text()) if (session / "run.json").exists() else {}
    vfov = float(meta.get("camera_vfov_deg", 60.0))
    source_focal = (height / 2) / np.tan(np.radians(vfov / 2))
    keep = None
    if (session / "overlay.json").exists():
        from capture.pack import crop_rows
        from capture.timecode import OverlayGeometry

        keep = crop_rows(height, OverlayGeometry.from_dict(json.loads((session / "overlay.json").read_text())))
    strip = RoadStrip(width, height, source_focal, args.focal, args.camera_height, args.horizon, keep)

    labelled = counter = None
    render_hz = fps
    if (session / "labels.parquet").exists() and (session / "run.json").exists():
        labelled, counter, render_hz = _labelled(session, fps)

    print(f"{video}: {width}x{height} at {fps:.2f} fps; scaled x{strip.scale:.3f} to focal {args.focal:.1f} px, "
          f"road band rows {strip.rows} of {strip.size[1]} ({NEAR_M:g}-{FAR_M:g} m at {args.camera_height} m), "
          f"pairs {args.skip} frame(s) apart")

    frames, speed, motion, points, kept, residual, dts = [], [], [], [], [], [], []
    history: deque = deque(maxlen=args.skip + 1)
    started = time.monotonic()
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.limit_frames and index >= args.limit_frames):
            break
        history.append((index, strip.cut(frame)))
        if len(history) == args.skip + 1:
            (i0, a), (i1, b) = history[0], history[-1]
            if counter is not None and i0 in counter and i1 in counter:
                dt = (counter[i1] - counter[i0]) / render_hz
            else:
                dt = (i1 - i0) / fps
            m = n = share = rms = None
            if dt > 0:  # a repeated render has no motion to measure
                m, n, share, rms = strip.measure(a, b)
            frames.append(i1)
            motion.append(m if m is not None else np.full(5, np.nan))
            speed.append(m[0] / dt if m is not None and dt > 0 else np.nan)
            points.append(n or 0)
            kept.append(share if share is not None else np.nan)
            residual.append(rms if rms is not None else np.nan)
            dts.append(dt)
        index += 1
        if index % 5000 == 0:
            rate = index / (time.monotonic() - started)
            left = f", ~{(total - index) / rate / 60:.0f} min left" if total else ""
            print(f"  {index} frames, {rate:.0f} frames/s{left}")
    cap.release()

    frames_arr = np.asarray(frames, np.int32)
    speed_arr = np.asarray(speed, np.float32)
    suffix = f"_skip{args.skip}" if args.skip > 1 else ""
    out = Path(args.out) if args.out else session / f"flow_speed{suffix}.npz"
    np.savez_compressed(
        out, frame=frames_arr, speed=speed_arr, motion=np.asarray(motion, np.float32),
        points=np.asarray(points, np.int16), kept=np.asarray(kept, np.float32),
        residual=np.asarray(residual, np.float32), dt=np.asarray(dts, np.float32),
        focal=args.focal, scale=strip.scale, camera_height=args.camera_height, horizon=args.horizon,
        band_rows=np.asarray(strip.rows), skip=args.skip, fps=fps,
    )
    print(f"{frames_arr.size} pairs in {(time.monotonic() - started) / 60:.1f} min -> {out}")
    print(f"  fitted {100 * np.isfinite(speed_arr).mean():.1f}% of pairs; median points {np.median(points):.0f}, "
          f"median share tracked back {np.nanmedian(kept):.2f}, median residual {np.nanmedian(residual):.2f} px")

    if labelled is None:
        return
    true = np.array([labelled.get(int(f), np.nan) for f in frames_arr])
    print("check against the labelled speed:")
    _report("per frame pair", speed_arr, true)
    n = (frames_arr.size // TICK_FRAMES) * TICK_FRAMES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # ticks with no fit or no label
        tick_est = np.nanmedian(speed_arr[:n].reshape(-1, TICK_FRAMES), 1)
        tick_true = np.nanmedian(true[:n].reshape(-1, TICK_FRAMES), 1)
    _report("per 15 Hz tick", tick_est, tick_true)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="measure speed from road motion and check it against the labels")
    run.add_argument("video")
    run.add_argument("--skip", type=int, default=1, help="frames between the pair: 2 is as if 30 fps from 60")
    run.add_argument("--focal", type=float, default=HALO_FOCAL_PX, help="target focal length in px (Halo by default)")
    run.add_argument("--camera-height", type=float, default=1.15, help="metres above the road; sets the scale")
    run.add_argument("--horizon", type=float, default=0.0, help="horizon row relative to the image centre, px")
    run.add_argument("--limit-frames", type=int, default=0, help="stop after this many frames (a quick trial)")
    run.add_argument("--out", default=None)
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
