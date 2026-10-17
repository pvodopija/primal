"""
Does the encoder know a corner from a straight, and left from right? Labels from the car's own
path: AC's new-circuit recordings (wide renders, turned into the model's training view with the
camera's turn against the car undone, as training sees them) carry the car's position, heading and speed per frame
(pose.npz); its sideways acceleration (speed x yaw rate, yaw rate smoothed over SMOOTH_S) over
CORNER_MS2 is a corner, under STRAIGHT_MS2 a straight (between: left out), its sign left or right.
Probes, each a logistic regression (as experiments/track_probe.py), trained on the eight
training circuits (and their mirror images, which swap left and right) and tested on the two the
model never saw (deutschlandring, daytona), every 0.2 s of every lap:
  model, one frame: the 128-d descriptor (does the view itself say "corner"?);
  model, two frames: the descriptor and its change over 0.2 s (the motion as well);
  raw pixels: the grey 37x20 picture and its change (the control: does the model add anything?);
  image slide: the picture's horizontal shift over 0.2 s (phase correlation), the camera turning,
  no learning beyond a threshold (the classic way to mark corners from video).
Then on real footage, labelled by its GPS (night1: the HERO7's GPS5, sideways acceleration from
its speed and the turning of its track), the probes trained on AC alone.

Usage: PYTHONPATH=. python experiments/corner_probe.py RUN   (writes runs/RUN/corner_probe.json)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.track_or_not import describe
from train import camera
from train.eval import load_model

ROOT = Path("data/packed_ac_b1007_new_live")
STEP_S, SMOOTH_S, CORNER_MS2, STRAIGHT_MS2 = 0.2, 0.3, 4.0, 1.0
GPS_LAG_S = 0.5  # night1's GPS turning trails the camera's (its gyro): best match 0.3-0.6 s


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    n = max(int(n), 1)
    return np.convolve(np.pad(x, n // 2, mode="edge"), np.ones(n) / n, mode="valid")[: len(x)]


def labels_from_path(heading: np.ndarray, speed: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Sideways acceleration (m/s^2, signed) and the class: 1 corner, 0 straight, -1 left out."""
    yaw_rate = smooth(np.gradient(np.unwrap(heading)) / dt, SMOOTH_S / dt)
    lat = speed * yaw_rate
    cls = np.where(np.abs(lat) > CORNER_MS2, 1, np.where(np.abs(lat) < STRAIGHT_MS2, 0, -1))
    return lat, cls


def grey(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2GRAY).astype(np.float32)


def pixels(f: np.ndarray) -> np.ndarray:
    g = cv2.resize(grey(f), (37, 20), interpolation=cv2.INTER_AREA).reshape(-1)
    return (g - g.mean()) / (g.std() + 1e-6)


def slide(a: np.ndarray, b: np.ndarray) -> float:
    """Horizontal shift (px at 148 wide) from frame a to frame b, upper two thirds (the scenery)."""
    win = cv2.createHanningWindow((148, 53), cv2.CV_32F)
    (dx, _), _ = cv2.phaseCorrelate(grey(a)[:53], grey(b)[:53], win)
    return float(dx)


def features(model, frames: np.ndarray, device) -> dict:
    """Per pair (frame t-STEP, frame t): every probe's input."""
    d = describe(model, frames, device)
    px = np.stack([pixels(f) for f in frames])
    sl = np.array([slide(frames[i - 1], frames[i]) for i in range(1, len(frames))])
    return {"model, one frame": d[1:], "model, two frames": np.hstack([d[1:], d[1:] - d[:-1]]),
            "raw pixels": np.hstack([px[1:], px[1:] - px[:-1]]), "image slide": np.stack([sl, np.abs(sl)], 1)}


def ac_set(model, device, split: str) -> tuple[dict, np.ndarray, np.ndarray, list]:
    index = json.loads((ROOT / "index.json").read_text())["laps"]
    feats, lat, cls, tracks = {}, [], [], []
    for lap in index:
        if lap["split"] != split:
            continue
        lap_dir = ROOT / lap["path"]
        frames = np.load(lap_dir / "frames.npy", mmap_mode="r")
        pose = np.load(lap_dir / "pose.npz")
        fps = float(lap["fps"])
        n = min(len(frames), len(pose["car_fwd"]))
        fwd = pose["car_fwd"][:n]
        heading = np.arctan2(fwd[:, 0], fwd[:, 2])
        a, c = labels_from_path(heading, pose["speed_kmh"][:n] / 3.6, 1 / fps)
        pick = np.arange(0, n, int(round(STEP_S * fps)))
        # the wide render seen as the model was trained to see it: the training view, the camera's
        # turn against the car undone (train/dataset.py, derotate_wide)
        source = camera.View.rendered(*lap["frame_size"], lap["fov_h_deg"], render_vfov_deg=lap.get("render_vfov_deg"))
        f = np.stack([camera.render(np.asarray(frames[i]), source, camera.TRAINING, yaw=-float(pose["cam_yaw_deg"][i])) for i in pick])
        for mirror in (False, True):
            ff = f[:, :, ::-1].copy() if mirror else f
            for k, v in features(model, ff, device).items():
                feats.setdefault(k, []).append(v)
            lat.append(-a[pick][1:] if mirror else a[pick][1:])
            cls.append(c[pick][1:])
            tracks += [lap["track"]] * (len(pick) - 1)
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(lat), np.concatenate(cls), tracks


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, steps: int = 400) -> tuple:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd).float()
    yt = torch.from_numpy(y).float()
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = float(y.mean())
    weight = torch.where(yt > 0.5, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))  # balanced classes
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt, reduction="none")).mean() + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mu, sd, w.detach().numpy(), float(b.detach())


def predict(probe, x: np.ndarray) -> np.ndarray:
    mu, sd, w, b = probe
    return 1 / (1 + np.exp(-(((x - mu) / sd) @ w + b)))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scores(probes: dict, feats: dict, lat: np.ndarray, cls: np.ndarray) -> dict:
    out = {}
    corner_or_straight = cls >= 0
    corner = cls == 1
    for name, (cs, lr) in probes.items():
        x = feats[name]
        p = predict(cs, x[corner_or_straight])
        y = cls[corner_or_straight]
        acc = float(np.mean([np.mean((p[y == k] > 0.5) == k) for k in (0, 1)]))  # balanced
        side = predict(lr, x[corner]) > 0.5
        out[name] = {"corner_auroc": auroc(p, y), "corner_balanced_accuracy": acc,
                     "left_right_accuracy": float(np.mean(side == (lat[corner] > 0)))}
    return out


def night_set(model, device) -> tuple[dict, np.ndarray, np.ndarray]:
    os.environ["REAL_FOOTAGE"] = "night1"
    from experiments import real_footage as rf
    f, t = rf.frames()
    z = np.load(rf.ROOT / "gpmf.npz")
    g, gt = z["GPS5"], z["GPS5_t"]
    lat0, lon0 = g[0, :2]
    x = (g[:, 1] - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (g[:, 0] - lat0) * 110540
    dt = float(np.median(np.diff(gt)))
    k = max(int(round(0.25 / dt)), 1)  # course over 0.5 s: the fixes' own jitter averaged
    heading = np.arctan2(np.roll(x, -k) - np.roll(x, k), np.roll(y, -k) - np.roll(y, k))
    a, _ = labels_from_path(heading, smooth(g[:, 3], k), dt)
    moving = smooth(g[:, 3], k) > 5.0
    keep = (t >= rf.CROSSINGS_S[0]) & (t <= rf.CROSSINGS_S[-1])
    pick = np.flatnonzero(keep)[:: int(round(STEP_S * 30))]
    feats = features(model, f[pick], device)
    a_at = np.interp(t[pick][1:] + GPS_LAG_S, gt, a)
    ok = np.interp(t[pick][1:] + GPS_LAG_S, gt, moving.astype(float)) > 0.5
    cls = np.where(~ok, -1, np.where(np.abs(a_at) > CORNER_MS2, 1, np.where(np.abs(a_at) < STRAIGHT_MS2, 0, -1)))
    return feats, a_at, cls


def main() -> None:
    run = sys.argv[1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_model(Path("runs") / run / "best.pt", device)
    model.eval()
    with torch.no_grad():
        train_f, train_lat, train_cls, _ = ac_set(model, device, "train")
        test_f, test_lat, test_cls, _ = ac_set(model, device, "holdout")
    print(f"{run}: AC training circuits {len(train_cls)} pairs ({np.mean(train_cls == 1):.0%} corner, {np.mean(train_cls == 0):.0%} straight); "
          f"unseen circuits {len(test_cls)} pairs ({np.mean(test_cls == 1):.0%} corner, {np.mean(test_cls == 0):.0%} straight)", flush=True)
    probes = {}
    for name, x in train_f.items():
        cs_rows = train_cls >= 0
        corner = train_cls == 1
        probes[name] = (fit_logistic(x[cs_rows], train_cls[cs_rows].astype(float)),
                        fit_logistic(x[corner], (train_lat[corner] > 0).astype(float)))
    report = {"unseen AC circuits": scores(probes, test_f, test_lat, test_cls)}
    with torch.no_grad():
        night_f, night_lat, night_cls = night_set(model, device)
    print(f"night1 (real, GPS labels): {len(night_cls)} pairs ({np.mean(night_cls == 1):.0%} corner, {np.mean(night_cls == 0):.0%} straight)")
    report["night1, real (trained on AC only)"] = scores(probes, night_f, night_lat, night_cls)
    for where, rows in report.items():
        print(f"{where}:")
        for name, r in rows.items():
            print(f"  {name:<20} corner vs straight AUROC {r['corner_auroc']:.3f}, balanced accuracy {r['corner_balanced_accuracy']:.1%} | "
                  f"left vs right {r['left_right_accuracy']:.1%}", flush=True)
    (Path("runs") / run / "corner_probe.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
