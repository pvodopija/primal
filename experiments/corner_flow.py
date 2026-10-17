"""
Corners from the camera's own turning on real footage, against GPS labels (night1: the HERO7's
GPS5 sideways acceleration, as experiments/corner_probe.py, its 0.5 s lag taken out): optical flow
on the original video at 960x540 (LK features in the scenery band, a similarity per pair of distinct
pictures, as experiments/gyro_stabilise.py: its horizontal shift is the camera turning), and the
camera's gyro as the ceiling. Per 0.2 s: corner vs straight from the turning's size (AUROC), left
vs right from its sign. corner_probe.py's image slide on the model's 148x80 crop failed here
(+0.06 against the gyro): this asks whether the signal is there at a camera's resolution.

Usage: PYTHONPATH=. python experiments/corner_flow.py   (writes runs/lc_logs/corner_flow_night1.json)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ["REAL_FOOTAGE"] = "night1"
from experiments import corner_probe as cp
from experiments import real_footage as rf
from experiments.gyro_stabilise import REPEAT

STEP_S = 0.2


def flow(video: Path, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Times and horizontal shift (px at 960 wide, per pair of distinct pictures) between a and b."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps))
    i = int(a * fps)
    prev, times, shifts = None, [], []
    mask = np.zeros((540, 960), np.uint8)
    mask[10:270, 173:787] = 255
    while i / fps < b:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        t = i / fps
        i += 1
        if prev is not None and np.abs(cv2.resize(g, (64, 36)).astype(np.float32) - cv2.resize(prev, (64, 36))).mean() < REPEAT:
            continue  # a repeated picture (Auto Low Light): no motion
        if prev is not None:
            p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) > 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok_ = st.ravel() == 1
                A, inl = cv2.estimateAffinePartial2D(p0[ok_] - [480, 270], p1[ok_] - [480, 270], method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
                if A is not None and inl.sum() > 10:
                    times.append(t)
                    shifts.append(A[0, 2])
        prev = g
    return np.array(times), np.array(shifts)


def main() -> None:
    a, b = rf.CROSSINGS_S[0], rf.CROSSINGS_S[-1]
    t, dx = flow(rf.ROOT / "GH010018_sd.MP4", a, b)
    grid = np.arange(a + 1, b - 1, STEP_S)
    rate = np.interp(grid, t, cp.smooth(dx, 6))  # px per picture pair, ~0.2 s average
    z = np.load(rf.ROOT / "gpmf.npz")
    g, gt = z["GPS5"], z["GPS5_t"]
    lat0, lon0 = g[0, :2]
    x = (g[:, 1] - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (g[:, 0] - lat0) * 110540
    k = 4
    heading = np.arctan2(np.roll(x, -k) - np.roll(x, k), np.roll(y, -k) - np.roll(y, k))
    acc, _ = cp.labels_from_path(heading, cp.smooth(g[:, 3], k), float(np.median(np.diff(gt))))
    lat = np.interp(grid + cp.GPS_LAG_S, gt, acc)
    cls = np.where(np.abs(lat) > cp.CORNER_MS2, 1, np.where(np.abs(lat) < cp.STRAIGHT_MS2, 0, -1))
    gyro = np.interp(grid, z["GYRO_t"] + 0.028, cp.smooth(z["GYRO"][:, 2], 40))
    report = {"pairs": int(len(t)), "corr_flow_gyro": float(np.corrcoef(rate, gyro)[0, 1]),
              "corr_flow_gps": float(np.corrcoef(rate, lat)[0, 1])}
    cs, corner = cls >= 0, cls == 1
    for name, sig in (("optical flow 960x540", rate), ("gyro", gyro)):
        sign = np.sign(np.corrcoef(sig[corner], lat[corner])[0, 1])
        report[name] = {"corner_auroc": cp.auroc(np.abs(sig)[cs], cls[cs]),
                        "left_right_accuracy": float(np.mean(np.sign(sig[corner]) * sign == np.sign(lat[corner])))}
    print(f"night1, {len(t)} picture pairs with flow; corr(flow, gyro) {report['corr_flow_gyro']:+.2f}, "
          f"corr(flow, GPS sideways accel) {report['corr_flow_gps']:+.2f}")
    for name in ("optical flow 960x540", "gyro"):
        r = report[name]
        print(f"  {name:<22} corner vs straight AUROC {r['corner_auroc']:.3f} | left vs right {r['left_right_accuracy']:.1%}")
    Path("runs/lc_logs/corner_flow_night1.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
