"""
The field of view of a real lap video, from the laps themselves: over one lap the car
turns exactly 360°, so the scenery's sideways slide, summed frame to frame (phase
correlation on the upper picture, far scenery), is 360° worth of pixels. Head turns
cancel over a lap. Summed in a centre strip it gives the focal length at the centre;
the edge strips (left and right averaged, which cancels the outward flow of driving
forward) tell the projection: on a pinhole lens the edges slide 1/cos² faster, on a
fisheye about as fast as the centre.

Usage: REAL_FOOTAGE=BHGP PYTHONPATH=. python experiments/real_fov.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from experiments.real_footage import CROP, ROOT, VIDEO

W, H = 640, 360  # analysed at half size
ROWS = (10, 140)  # above the horizon and the cockpit, at half size
STRIPS = {"left": (0, 160), "centre": (224, 416), "right": (480, 640)}


def main() -> None:
    starts = json.loads((ROOT / "starts_orb.json").read_text())
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(starts[0] * fps)))
    windows = {k: cv2.createHanningWindow((x1 - x0, ROWS[1] - ROWS[0]), cv2.CV_32F) for k, (x0, x1) in STRIPS.items()}
    cache = ROOT / "fov_shifts.npz"
    if cache.exists():
        z = np.load(cache)
        shifts, laps = {k: z[k] for k in STRIPS}, z["lap"]
    else:
        shifts, laps = read_shifts(cap, fps, starts, windows)
        np.savez(cache, lap=laps, **shifts)
    sums = {k: np.bincount(laps, weights=v, minlength=len(starts) - 1) for k, v in shifts.items()}
    centre = np.abs(sums["centre"]) * (1280 / W) / (2 * np.pi)  # focal length, px of the 1280 frame
    # left and right averaged with their signs: driving forward pushes them apart equally
    edges = np.abs((sums["left"] + sums["right"]) / 2) / np.abs(sums["centre"])
    print("focal length at the centre, per lap (px of 1280):", np.round(centre, 0).tolist())
    print("edge strips slide x the centre's, per lap:", np.round(edges, 2).tolist())
    f = float(np.median(centre))
    x_edge = (np.mean([abs(np.mean(STRIPS[s]) - W / 2) for s in ("left", "right")])) * (1280 / W)  # strip centre from the middle
    pinhole_ratio = 1 / np.cos(np.arctan(x_edge / f)) ** 2
    print(f"focal {f:.0f} px (laps {centre.min():.0f}-{centre.max():.0f}); a pinhole lens would slide its edge strips "
          f"{pinhole_ratio:.2f}x the centre's, a fisheye about 1.0x; measured {np.median(edges):.2f}x")
    for name, half in (("whole frame", 640), ("model's crop", (CROP[1] - CROP[0]) / 2)):
        print(f"  {name} ({2 * half:.0f} px): pinhole {np.degrees(2 * np.arctan(half / f)):.0f}°, "
              f"fisheye (equidistant) {np.degrees(2 * half / f):.0f}°")


def read_shifts(cap, fps, starts, windows):
    shifts, laps = {k: [] for k in STRIPS}, []
    prev, i = None, int(round(starts[0] * fps))
    while i < int(round(starts[-1] * fps)):
        ok, f = cap.read()
        if not ok:
            break
        g = np.float32(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))[ROWS[0]:ROWS[1]]
        lap = int(np.searchsorted(starts, i / fps, side="right")) - 1
        if prev is not None:
            for k, (x0, x1) in STRIPS.items():
                (dx, _), _ = cv2.phaseCorrelate(prev[:, x0:x1], g[:, x0:x1], windows[k])
                shifts[k].append(dx)
            laps.append(lap)
        prev, i = g, i + 1
    return {k: np.array(v) for k, v in shifts.items()}, np.array(laps)


if __name__ == "__main__":
    main()
