"""
A "track or not" check from the encoder's descriptor: logistic regression on the 128-d
descriptor (128 weights and a bias: one dot product per frame on what the encoder already
computes), decided over a 2 s window (the mean probability of its 30 frames at 15 Hz).

Trained on AC frames from the training tracks (and mirrored) and real track footage
against Open Images photos
(CC BY 2.0, data/negatives/openimages: no road, track or vehicle labels; 80% of the photos,
three random crops each at the model's aspect), synthetic pictures and macOS wallpapers.
Rendered frames alone as the positives teach "rendered or photographed", so real track
footage joins them, tested leave one out: each real video (outdoor kart GoPro, indoor kart
GoPro, BHGP helmet camera, RaceChrono car) scored by a probe trained without it. Also tested:
AC laps of the held-out tracks, the other 20% of the photos, and the GoPro's home sessions
(rooms and screens, a kind of "not track" with no example in training). The probe to ship
is then trained with every real video.
Writes runs/RUN/track_probe.json (the weights, bias, threshold and every result) and .png.

Usage: PYTHONPATH=. python experiments/track_probe.py RUN [WALLPAPER_DIR]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.track_or_not import H, W, ac_frames, describe, fit, synthetic, wallpaper_frames
from train.dataset import LapIndex
from train.eval import load_model
from train.train import default_device

HZ, WINDOW = 15.0, 30  # 2 s windows
TRAIN_AC, CROPS = 8000, 3
PHOTOS = Path("data/negatives/openimages/images")
rng = np.random.default_rng(1)


def photo_crops(paths: list[Path], crops: int) -> np.ndarray:
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for _ in range(crops):
            s = rng.uniform(0.5, 1.0)
            ch, cw = int(h * s), int(min(w, h * s * W / H))
            ch = min(ch, int(cw * H / W) + 1)
            y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
            out.append(fit(img[y:y + ch, x:x + cw]))
    return np.stack(out)


def ac_sequences(split: str, laps_per_track: int) -> list[np.ndarray]:
    out = []
    for track, laps in sorted(LapIndex.load(Path("data/packed_ac_v3"), split=split).by_track.items()):
        for lap in [l for l in laps if l.s_span > 0.98][:laps_per_track]:
            step = max(int(round(lap.fps / HZ)), 1)
            out.append(np.asarray(lap.frames()[::step]))
    return out


def cache_sequence(path: str) -> list[np.ndarray]:
    z = np.load(path)
    step = max(int(round((1 / HZ) / np.median(np.diff(z["t"])))), 1)
    return [z["frames"][::step]]


def video_sequences(paths: list[str]) -> list[np.ndarray]:
    out = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            continue
        step, frames, i = max(int(round(cap.get(cv2.CAP_PROP_FPS) / HZ)), 1), [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(fit(f))
            i += 1
        if frames:
            out.append(np.stack(frames))
    return out


def windows(p: np.ndarray) -> np.ndarray:
    n = len(p) // WINDOW
    return p[: n * WINDOW].reshape(n, WINDOW).mean(1) if n else np.empty(0)


def train_probe(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, float]:
    """Logistic regression, the two classes weighted equally."""
    x, y = np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(4000):
        prob = 1 / (1 + np.exp(-(x @ w + b)))
        grad = (prob - y) * weight
        w -= 1.0 * (x.T @ grad / len(y) + 1e-4 * w)
        b -= 1.0 * grad.mean()
    return w, b


def main() -> None:
    run = sys.argv[1]
    walls = sys.argv[2] if len(sys.argv) > 2 else None
    device = torch.device(default_device())
    model, _ = load_model(Path("runs") / run / "best.pt", device)
    model.eval()
    d = lambda frames: describe(model, frames, device)

    photos = sorted(PHOTOS.glob("*.jpg"))
    order = rng.permutation(len(photos))
    split = int(0.8 * len(photos))
    train_photos, test_photos = [photos[i] for i in order[:split]], [photos[i] for i in order[split:]]
    ac = ac_frames("train", TRAIN_AC)
    ac_pos = np.concatenate([d(ac), d(ac[:, :, ::-1].copy())])
    neg_parts = [d(photo_crops(train_photos, CROPS)), d(synthetic(600))]
    if walls:
        neg_parts.append(d(wallpaper_frames(walls, 300)))
    neg = np.concatenate(neg_parts)
    # Real track footage, a sequence per video at 15 Hz: positives for training when not the
    # one tested (rendered frames alone teach "rendered or photographed", not "track or not").
    real = {"real: outdoor kart": video_sequences(["data/real-footage/adventure-solo/adventure-pov.mp4"]),
            "real: indoor kart": cache_sequence("data/real-footage/indoor-pov/frames_148x80_c80.npz"),
            "real: BHGP helmet": cache_sequence("data/real-footage/BHGP/frames_148x80_full.npz"),
            "real: RaceChrono car": cache_sequence("data/real-footage/race-chrono/frames_148x80.npz")}
    real_desc = {name: [d(s) for s in seqs] for name, seqs in real.items()}
    tests = {"AC, held-out tracks": ("track", [d(s) for s in ac_sequences("holdout", 2)]),
             "home GoPro sessions": ("not", [d(s) for s in video_sequences(sorted(glob.glob("data/gopro/2026*/video.mp4")))]),
             "photos, held out": ("not", [d(photo_crops(test_photos, 1))])}

    def score(w: np.ndarray, b: float, seqs: list[np.ndarray], sequence: bool = True) -> dict:
        ps = [1 / (1 + np.exp(-(s @ w + b))) for s in seqs]
        frames_p = np.concatenate(ps)
        win = np.concatenate([windows(p) for p in ps]) if sequence else np.empty(0)
        return {"frames": int(frames_p.size), "frames_called_track": float(np.mean(frames_p >= 0.5)),
                "windows": int(win.size), "windows_called_track": float(np.mean(win >= 0.5)) if win.size else None,
                "_windows": win}

    def show(name: str, kind: str, r: dict) -> None:
        wtxt = f", windows {100 * r['windows_called_track']:5.1f}% of {r['windows']}" if r["windows"] else ""
        print(f"    {name:<22} {kind:<5} frames {100 * r['frames_called_track']:5.1f}% of {r['frames']}{wtxt}")

    sample = lambda seqs, n: np.concatenate(seqs)[np.sort(rng.choice(sum(len(s) for s in seqs), min(n, sum(len(s) for s in seqs)), replace=False))]
    report = {"leave_one_out": {}}
    print(f"{run}: leave one real video out (trained on AC, the other real videos, photos and synthetic pictures)")
    for held in real:
        pos = np.concatenate([ac_pos] + [sample(real_desc[n], 2000) for n in real if n != held])
        w, b = train_probe(pos, neg)
        print(f"  without {held}:")
        rows = {held: ("track", score(w, b, real_desc[held]))}
        rows |= {n: (k, score(w, b, seqs, n != "photos, held out")) for n, (k, seqs) in tests.items()}
        for n, (k, r) in rows.items():
            show(n, k, r)
        report["leave_one_out"][held] = {n: {k2: v for k2, v in r.items() if not k2.startswith("_")} for n, (k, r) in rows.items()}

    # the probe to ship: every real video in
    pos = np.concatenate([ac_pos] + [sample(real_desc[n], 2000) for n in real])
    w, b = train_probe(pos, neg)
    print("  all real videos in (the probe to ship; track sets are now partly trained on):")
    final = {n: (k, score(w, b, seqs, n != "photos, held out")) for n, (k, seqs) in tests.items()}
    for n, (k, r) in final.items():
        show(n, k, r)
    report["final"] = {n: {k2: v for k2, v in r.items() if not k2.startswith("_")} for n, (k, r) in final.items()}
    (Path("runs") / run / "track_probe.json").write_text(json.dumps(
        {"weights": w.tolist(), "bias": b, "threshold": 0.5, "window_s": WINDOW / HZ, "hz": HZ,
         "trained_on": {"ac_frames": len(ac_pos), "real_videos": list(real), "negatives": len(neg),
                        "photos": "Open Images validation, CC BY 2.0, data/negatives/openimages/attribution.csv"},
         **report}, indent=2))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    rows = [(f"{h} (left out)", "track", report_w) for h, report_w in
            ((h, score(*train_probe(np.concatenate([ac_pos] + [sample(real_desc[n], 2000) for n in real if n != h]), neg),
                       real_desc[h])["_windows"]) for h in real)]
    rows += [(n, k, r["_windows"]) for n, (k, r) in final.items() if r["windows"]]
    for i, (n, k, v) in enumerate(rows):
        ax.scatter(v, np.full(v.size, i) + rng.uniform(-0.25, 0.25, v.size), s=6, alpha=0.6,
                   color="#2c7fb8" if k == "track" else "#d9534f")
    ax.axvline(0.5, color="#222222", lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("probability of 'track', mean over a 2 s window")
    ax.set_title(f"{run}: track or not, from the descriptor (128 weights)")
    fig.tight_layout()
    fig.savefig(Path("runs") / run / "track_probe.png", dpi=110)


if __name__ == "__main__":
    main()
