"""
Videos of a run of experiments/gopro_preview.py, decoded from its saved preview
(stream.ts, through ios/tools/gopro_probe --file):

  reference.mp4  the reference lap as PRIMAL got it, with its clock;
  review.mp4     from the reference's end: the reference frame PRIMAL matched (left) beside the
                 live frame (right), with the delta, the confidence and the reference time.

The stream's own timestamps restart now and then; each stretch between restarts gets the
offset the live run gave it (frames.csv), so the times are the run's.

Usage: PYTHONPATH=. python experiments/gopro_review.py data/gopro/<time>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments.gopro_preview import HEADER, build, read_exact


def stretches(stream_pts: np.ndarray) -> np.ndarray:
    """Index of the stretch between timestamp restarts for each frame (as StreamClock finds them)."""
    d = np.diff(stream_pts, prepend=stream_pts[0])
    return np.cumsum(~((-0.5 < d) & (d < 1.0)))


def writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30, size)
        if w.isOpened():
            return w
    raise RuntimeError("no video writer")


def main() -> None:
    run = Path(sys.argv[1])
    s = json.loads((run / "summary.json").read_text())
    f, r = pd.read_csv(run / "frames.csv"), pd.read_csv(run / "readings.csv")
    f["stretch"] = stretches(f.stream_pts.to_numpy())
    offset = (f.pts - f.stream_pts).groupby(f.stretch).median()
    probe = subprocess.Popen([str(build("gopro_probe")), "--file", str(run / "stream.ts")], stdout=subprocess.PIPE)
    pts, imgs = [], []
    while True:
        head = read_exact(probe.stdout, HEADER.size)
        if len(head) < HEADER.size:
            break
        _, p, _, _, w, h = HEADER.unpack(head)
        imgs.append(np.frombuffer(read_exact(probe.stdout, w * h * 4), np.uint8).reshape(h, w, 4)[:, :, :3])
        pts.append(p)
    raw = np.array(pts)
    t = raw + offset.reindex(stretches(raw)).to_numpy()
    print(f"{len(t)} frames decoded, {len(offset)} stretches between timestamp restarts", flush=True)
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
