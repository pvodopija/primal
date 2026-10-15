"""
Dragy's own numbers off a YouTube video's overlay (data/real-footage/draggy): the live delta
(its red box) and the lap clock, at 10 Hz over a stretch, with macOS's Vision text recognition
(experiments/tools/vision_ocr.swift). Writes ocr.csv: t (video s), clock_s, delta_s.

Usage: PYTHONPATH=. python experiments/draggy_read.py START_S END_S
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/real-footage/draggy")
VIDEO = ROOT / "draggy.mp4"
DELTA, CLOCK = (1185, 1262, 38, 72), (1090, 1260, 64, 108)  # x0, x1, y0, y1 of the 1280x720 frame
HZ, UPSCALE = 10, 3


def number(text: str) -> float:
    m = re.search(r"([+-]?)\s*(\d+)[.,](\d+)", text.replace("−", "-"))
    return float(f"{m.group(1) or '+'}{m.group(2)}.{m.group(3)}") if m else np.nan


def clock(text: str) -> float:
    m = re.search(r"(\d+):(\d{2})[.,](\d+)", text)
    return int(m.group(1)) * 60 + float(f"{m.group(2)}.{m.group(3)}") if m else np.nan


def main() -> None:
    start, end = float(sys.argv[1]), float(sys.argv[2])
    crops = ROOT / "ocr"
    crops.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    times = []
    for k, t in enumerate(np.arange(start, end, 1 / HZ)):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        for name, (x0, x1, y0, y1) in (("delta", DELTA), ("clock", CLOCK)):
            cv2.imwrite(str(crops / f"{k:05d}_{name}.png"), cv2.resize(f[y0:y1, x0:x1], None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC))
        times.append(float(t))
    paths = "\n".join(str(crops / f"{k:05d}_{n}.png") for k in range(len(times)) for n in ("delta", "clock"))
    out = subprocess.run(["xcrun", "swift", "experiments/tools/vision_ocr.swift"], input=paths, capture_output=True, text=True).stdout
    text = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)
    rows = [(t, clock(text.get(str(crops / f"{k:05d}_clock.png"), "")), number(text.get(str(crops / f"{k:05d}_delta.png"), "")))
            for k, t in enumerate(times)]
    d = pd.DataFrame(rows, columns=["t", "clock_s", "delta_s"])
    d.to_csv(ROOT / "ocr.csv", index=False)
    print(f"{len(d)} readings; readable: clock {100 * d.clock_s.notna().mean():.0f}%, delta {100 * d.delta_s.notna().mean():.0f}%")


if __name__ == "__main__":
    main()
