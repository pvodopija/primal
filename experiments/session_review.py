"""
Videos of a recorded live session: a folder with video.mp4 (every frame the session saw),
video_times.csv (each frame's time on the session's clock), readings.csv (PRIMAL's
readings: pts, ref_time_s, confidence, single_ref_time_s, delta_s) and summary.json
(reference_start_pts, reference_lap_s):

  reference.mp4  the reference lap as PRIMAL got it, with its clock;
  review.mp4     from the reference's end: the reference frame PRIMAL matched (left) beside the
                 live frame (right), with the delta, the confidence and the reference time.

Usage: PYTHONPATH=. python experiments/session_review.py data/gopro/<time>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd



def session_frames(run: Path, scale: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
    """The session's frame times and frames (downscaled by `scale`)."""
    t = pd.read_csv(run / "video_times.csv").t.to_numpy()
    cap, imgs = cv2.VideoCapture(str(run / "video.mp4")), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        imgs.append(img if scale == 1 else cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA))
    return t[:len(imgs)], imgs


def writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30, size)
        if w.isOpened():
            return w
    raise RuntimeError("no video writer")


def main() -> None:
    run = Path(sys.argv[1])
    s = json.loads((run / "summary.json").read_text())
    r = pd.read_csv(run / "readings.csv")
    t, imgs = session_frames(run)
    h, w = imgs[0].shape[:2]
    r0, T = s["reference_start_pts"], s["reference_lap_s"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    ref = np.where((t >= r0) & (t <= r0 + T))[0]
    out = writer(run / "reference.mp4", (w, h))
    for i in ref:
        img = imgs[i].copy()
        cv2.putText(img, f"REFERENCE  {t[i] - r0:5.2f} / {T:.2f} s", (12, 36), font, 1.0, (255, 255, 255), 2)
        out.write(img)
    out.release()

    live = np.where(t > r0 + T)[0]
    half = (w // 2, h // 2)
    out = writer(run / "review.mp4", (w, half[1] + 70))
    rt = r.pts.to_numpy()
    for i in live:
        k = np.searchsorted(rt, t[i], side="right") - 1
        canvas = np.zeros((half[1] + 70, w, 3), np.uint8)
        canvas[70:, half[0]:] = cv2.resize(imgs[i], half, interpolation=cv2.INTER_AREA)
        if k >= 0:
            x = r.iloc[k]
            j = ref[np.argmin(np.abs(t[ref] - (r0 + x.ref_time_s)))]
            canvas[70:, :half[0]] = cv2.resize(imgs[j], half, interpolation=cv2.INTER_AREA)
            shown = x.confidence >= 0.5
            text = f"{x.delta_s:+.2f} s" if shown else "unavailable"
            colour = ((80, 200, 80) if x.delta_s < 0 else (80, 80, 230)) if shown else (170, 170, 170)
            cv2.putText(canvas, text, (12, 48), font, 1.4, colour, 3)
            cv2.putText(canvas, f"confidence {x.confidence:.2f}   matched reference {x.ref_time_s:5.2f} s   run {t[i]:6.1f} s",
                        (260, 44), font, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, "REFERENCE (matched)", (8, 88), font, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "LIVE", (half[0] + 8, 88), font, 0.5, (255, 255, 255), 1)
        out.write(canvas)
    out.release()
    print(f"written {run / 'reference.mp4'} ({len(ref)} frames) and {run / 'review.mp4'} ({len(live)} frames)")


if __name__ == "__main__":
    main()
