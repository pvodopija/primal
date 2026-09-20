"""
Recover per-frame labels from the on-screen timecode grid in a recorded mp4.

Two steps, both offline:

    python -m capture.overlay_decode calibrate data/sessions/<run>/video.mp4
    python -m capture.overlay_decode decode    data/sessions/<run>/video.mp4

Calibration locates the grid once per session and writes `overlay.json` next to
the video. Decoding walks every frame and writes `labels.parquet`. Because the
counter is burned into the render, dropped or duplicated capture frames show up
as counter gaps rather than as silent label error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from capture.timecode import COLS, OverlayGeometry, decode_frame


def _sample_frame_indices(count: int, wanted: int) -> list[int]:
    if count <= wanted:
        return list(range(count))
    # Skip the first and last 5%; recordings often start or end mid-menu.
    lo, hi = int(count * 0.05), int(count * 0.95)
    return [int(round(x)) for x in np.linspace(lo, max(hi - 1, lo), wanted)]


def _read_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    out = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    if not out:
        raise SystemExit(f"could not read any frames from {path}")
    return out


def _pass_rate(frames: list[np.ndarray], geom: OverlayGeometry) -> float:
    good = 0
    for i, gray in enumerate(frames):
        if decode_frame(gray, geom)[2]:
            good += 1
        elif i >= 3 and good == 0:
            # Nothing decodable in the first few frames; not worth finishing.
            return 0.0
    return good / len(frames)


def refine(frames: list[np.ndarray], seed: OverlayGeometry) -> tuple[OverlayGeometry, float]:
    """
    Coarse-to-fine search around an approximate placement.

    A hand-drawn ROI is within a few pixels and the cell size is only known up
    to CSP's UI scaling, so both are searched. Score is the checksum pass rate,
    which makes the search self-validating: a wrong geometry scores zero.
    """
    coarse_frames = frames[: max(6, len(frames) // 4)]
    best, best_score = seed, _pass_rate(coarse_frames, seed)

    for dx in range(-8, 9, 2):
        for dy in range(-8, 9, 2):
            candidate = OverlayGeometry(seed.x0 + dx, seed.y0 + dy, seed.cell)
            score = _pass_rate(coarse_frames, candidate)
            if score > best_score:
                best, best_score = candidate, score

    fine, fine_score = best, _pass_rate(frames, best)
    for dcell in np.arange(-1.0, 1.01, 0.25):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                candidate = OverlayGeometry(best.x0 + dx, best.y0 + dy, best.cell + float(dcell))
                if candidate.cell <= 3.0:
                    continue
                score = _pass_rate(frames, candidate)
                if score > fine_score:
                    fine, fine_score = candidate, score
        if fine_score >= 1.0:
            break
    return fine, fine_score


def _select_roi(gray: np.ndarray) -> tuple[int, int, int, int]:
    window = "Drag a box around the timecode grid - ENTER to confirm"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI(window, gray, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    if w <= 0 or h <= 0:
        raise SystemExit("ROI selection cancelled")
    return int(x), int(y), int(w), int(h)


def cmd_calibrate(args: argparse.Namespace) -> None:
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames = _read_frames(video, _sample_frame_indices(count, args.frames))

    if args.roi:
        x, y, w, _ = (int(v) for v in args.roi.split(","))
    else:
        x, y, w, _ = _select_roi(frames[len(frames) // 2])

    seed = OverlayGeometry(x0=float(x), y0=float(y), cell=w / COLS)
    geom, score = refine(frames, seed)
    print(f"geometry x0={geom.x0:.1f} y0={geom.y0:.1f} cell={geom.cell:.2f}")
    print(f"checksum pass rate {score * 100:.1f}% over {len(frames)} sampled frames")
    if score < 0.98:
        print(
            "WARNING: low pass rate. Check the overlay is visible and unobstructed, "
            "and that the box covered the whole 2x27 grid."
        )

    out = video.parent / "overlay.json"
    out.write_text(json.dumps({**geom.as_dict(), "pass_rate": score}, indent=2))
    print(f"wrote {out}")


def cmd_decode(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if args.roi:
        x, y, w, _ = (float(v) for v in args.roi.split(","))
        geom = OverlayGeometry(x0=x, y0=y, cell=w / COLS)
    else:
        path = video.parent / "overlay.json"
        if not path.exists():
            raise SystemExit(f"{path} missing; run `calibrate` first")
        geom = OverlayGeometry.from_dict(json.loads(path.read_text()))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)

    rows = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            counter, spline, valid = decode_frame(gray, geom)
            rows.append((index, counter, spline, valid))
            index += 1
            if index % 5000 == 0:
                print(f"  {index} frames")
    finally:
        cap.release()

    frame_df = pd.DataFrame(rows, columns=["frame_idx", "counter", "spline_pos", "valid"])
    out = video.parent / "labels.parquet"
    frame_df.to_parquet(out, index=False)

    good = frame_df[frame_df.valid]
    print(f"\n{len(frame_df)} frames at {fps:.2f} fps -> {out}")
    print(f"checksum ok: {len(good)} ({len(good) / max(len(frame_df), 1) * 100:.2f}%)")
    if good.empty:
        print("nothing decoded; geometry is probably wrong")
        return

    steps = np.diff(good.counter.to_numpy())
    print(f"counter step: min {steps.min()} median {int(np.median(steps))} max {steps.max()}")
    print(f"duplicate counters: {int((steps == 0).sum())}, gaps > 1: {int((steps > 1).sum())}")
    wraps = int((np.diff(good.spline_pos.to_numpy()) < -0.5).sum())
    print(f"spline range [{good.spline_pos.min():.4f}, {good.spline_pos.max():.4f}], {wraps} lap wraps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cal = sub.add_parser("calibrate", help="locate the grid and write overlay.json")
    cal.add_argument("video")
    cal.add_argument("--roi", help="x,y,w,h instead of interactive selection")
    cal.add_argument("--frames", type=int, default=24, help="frames sampled for scoring")
    cal.set_defaults(func=cmd_calibrate)

    dec = sub.add_parser("decode", help="decode every frame to labels.parquet")
    dec.add_argument("video")
    dec.add_argument("--roi", help="x,y,w,h override")
    dec.set_defaults(func=cmd_decode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
