"""
RaceChrono's own numbers, read off its video overlay (data/real-footage/race-chrono):
the current lap and its clock, and the live delta against its comparison lap (R1,
1:26.5, not in the video), at 10 Hz, with macOS's Vision text recognition
(experiments/tools/vision_ocr.swift). Writes ocr.csv: t (video s), lap, clock_s, delta_s.

Usage: PYTHONPATH=. python experiments/racechrono_read.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/real-footage/race-chrono")
VIDEO = ROOT / "race-chrono.mp4"
CURRENT, DELTA = (540, 692, 0, 90), (700, 870, 0, 90)  # x0, x1, y0, y1 of the 1280x720 frame
HZ = 10


def seconds(text: str) -> float:
    """'42.5' or '1:25.3' or '+3:42.4' -> seconds, NaN if unreadable."""
    m = re.fullmatch(r"([+-]?)(?:(\d+):)?(\d+(?:\.\d+)?)", text.replace(" ", "").replace("−", "-"))
    if not m:
        return float("nan")
    value = (int(m.group(2)) * 60 if m.group(2) else 0) + float(m.group(3))
    return -value if m.group(1) == "-" else value


def main() -> None:
    crops = ROOT / "ocr"
    crops.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    every, i, times = max(int(round(fps / HZ)), 1), 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            k = len(times)
            for name, (x0, x1, y0, y1) in (("cur", CURRENT), ("delta", DELTA)):
                cv2.imwrite(str(crops / f"{k:05d}_{name}.png"), f[y0:y1, x0:x1])
            times.append(i / fps)
        i += 1
    paths = "\n".join(str(crops / f"{k:05d}_{n}.png") for k in range(len(times)) for n in ("cur", "delta"))
    out = subprocess.run(["xcrun", "swift", "experiments/tools/vision_ocr.swift"], input=paths, capture_output=True, text=True).stdout
    text = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)
    rows = []
    for k, t in enumerate(times):
        cur = [x.strip() for x in text.get(str(crops / f"{k:05d}_cur.png"), "").split("|")]
        nums = [x for x in cur if re.fullmatch(r"\d+(:\d+)?(\.\d+)?", x.replace(" ", ""))]
        lap = int(nums[-2]) if len(nums) >= 2 and nums[-2].isdigit() else -1
        rows.append((t, lap, seconds(nums[-1]) if nums else np.nan, seconds(text.get(str(crops / f"{k:05d}_delta.png"), "").strip())))
    d = pd.DataFrame(rows, columns=["t", "lap", "clock_s", "delta_s"])
    d.to_csv(ROOT / "ocr.csv", index=False)
    print(d.describe().round(2).to_string())
    print("readable: clock", f"{100 * d.clock_s.notna().mean():.0f}%", "delta", f"{100 * d.delta_s.notna().mean():.0f}%", "laps", sorted(d.lap.unique()))


if __name__ == "__main__":
    main()
