"""
Does the encoder know a track when it sees one? It was trained only to tell places on a
lap apart, never "track or not". Each picture's 128-d descriptor is scored by how close it
sits to a bank of AC frames from the training tracks: the mean cosine of its 10 nearest
bank frames. Groups:
  track:     AC frames from the held-out tracks; real track footage (indoor kart GoPro,
             BHGP helmet camera, RaceChrono car), never trained on;
  not track: the GoPro's home test sessions (data/gopro, rooms and screens), macOS's
             wallpapers (random crops), synthetic pictures (black, white, grey, noise,
             smoothed noise, sky-to-ground gradients, checkerboards, colour blocks);
  probes:    AC held-out frames upside down, and mirrored.
Reports each group's scores, the AUROC of track against not track, and the share of each
group above the threshold that keeps 95% of track pictures; then a linear probe trained on
AC against synthetic pictures and wallpapers only, tested on the real tracks and the home
sessions it never saw. Writes
runs/RUN/track_or_not.json and .png (scores, and the descriptors' first two principal
components; no home frames are drawn).

Usage: PYTHONPATH=. python experiments/track_or_not.py RUN [WALLPAPER_DIR]
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

from train.dataset import LapIndex, _to_chw
from train.eval import load_model
from train.train import default_device

W, H = 148, 80
BANK, PER_GROUP, K = 3000, 300, 10
rng = np.random.default_rng(0)


def fit(img: np.ndarray) -> np.ndarray:
    """A centre crop at the model's aspect, area-resized to its input size."""
    h, w = img.shape[:2]
    if w / h > W / H:
        cw = int(h * W / H); img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = int(w * H / W); img = img[(h - ch) // 2:(h - ch) // 2 + ch]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def ac_frames(split: str, n: int) -> np.ndarray:
    laps = [lap for lap in LapIndex.load(Path("data/packed_ac_v3"), split=split).laps if lap.s_span > 0.98]
    picks = [(laps[i], int(rng.integers(0, laps[i].n_frames))) for i in rng.integers(0, len(laps), n)]
    return np.stack([np.asarray(lap.frames()[k]) for lap, k in picks])


def cache_frames(path: str, n: int) -> np.ndarray:
    f = np.load(path)["frames"]
    return f[np.sort(rng.choice(len(f), min(n, len(f)), replace=False))]


def video_frames(paths: list[str], n: int) -> np.ndarray:
    out = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in np.linspace(0, count - 1, n // len(paths)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            if ok:
                out.append(fit(f))
    return np.stack(out)


def wallpaper_frames(folder: str, n: int) -> np.ndarray:
    images = [cv2.imread(p) for p in sorted(glob.glob(f"{folder}/*.jpg"))]
    out = []
    for i in range(n):
        img = images[i % len(images)]
        h, w = img.shape[:2]
        s = rng.uniform(0.4, 1.0)
        ch, cw = int(h * s), int(min(w, h * s * W / H))
        y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        out.append(fit(img[y:y + ch, x:x + cw]))
    return np.stack(out)


def synthetic(n: int) -> np.ndarray:
    out = []
    for i in range(n):
        kind = i % 8
        if kind == 0:
            img = np.zeros((H, W, 3), np.uint8)
        elif kind == 1:
            img = np.full((H, W, 3), 255, np.uint8)
        elif kind == 2:
            img = np.full((H, W, 3), int(rng.integers(40, 216)), np.uint8)
        elif kind == 3:
            img = rng.integers(0, 256, (H, W, 3)).astype(np.uint8)
        elif kind == 4:
            img = cv2.GaussianBlur(rng.integers(0, 256, (H, W, 3)).astype(np.uint8), (0, 0), float(rng.uniform(2, 8)))
        elif kind == 5:  # sky above, ground below
            t = np.linspace(0, 1, H)[:, None, None]
            sky, ground = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            img = np.broadcast_to(sky * (1 - t) + ground * t, (H, W, 3)).astype(np.uint8)
        elif kind == 6:
            c = int(rng.integers(4, 20))
            img = ((np.indices((H, W)).sum(0) // c) % 2 * 255).astype(np.uint8)[..., None].repeat(3, -1)
        else:
            img = cv2.resize(rng.integers(0, 256, (4, 6, 3)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(img)
    return np.stack(out)


@torch.no_grad()
def describe(model, frames: np.ndarray, device) -> np.ndarray:
    d = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(frames[i:i + 256])).to(device), use_checkpoint=False)
                   for i in range(0, len(frames), 256)])
    return torch.nn.functional.normalize(d, dim=-1).cpu().numpy()


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def main() -> None:
    run = sys.argv[1]
    walls = sys.argv[2] if len(sys.argv) > 2 else None
    device = torch.device(default_device())
    model, _ = load_model(Path("runs") / run / "best.pt", device)
    model.eval()
    held = ac_frames("holdout", PER_GROUP)
    groups = {
        "AC, held-out tracks": ("track", held),
        "real: indoor kart": ("track", cache_frames("data/real-footage/indoor-pov/frames_148x80_c80.npz", PER_GROUP)),
        "real: BHGP helmet": ("track", cache_frames("data/real-footage/BHGP/frames_148x80_full.npz", PER_GROUP)),
        "real: RaceChrono car": ("track", cache_frames("data/real-footage/race-chrono/frames_148x80.npz", PER_GROUP)),
        "home GoPro sessions": ("not", video_frames([p for p in sorted(glob.glob("data/gopro/2026*/video.mp4"))
                                                     if cv2.VideoCapture(p).isOpened()], PER_GROUP)),
        "synthetic": ("not", synthetic(PER_GROUP)),
        "AC upside down": ("probe", held[:, ::-1].copy()),
        "AC mirrored": ("probe", held[:, :, ::-1].copy()),
    }
    if walls:
        groups["macOS wallpapers"] = ("not", wallpaper_frames(walls, PER_GROUP))
    bank = describe(model, ac_frames("train", BANK), device)
    desc = {name: describe(model, frames, device) for name, (_, frames) in groups.items()}
    score = {name: np.sort(d @ bank.T, axis=1)[:, -K:].mean(1) for name, d in desc.items()}
    track = np.concatenate([score[n] for n, (k, _) in groups.items() if k == "track"])
    not_track = np.concatenate([score[n] for n, (k, _) in groups.items() if k == "not"])
    threshold = float(np.percentile(track, 5))
    report = {"auroc_track_vs_not": auroc(track, not_track), "threshold_keeps_95pct_track": threshold, "groups": {}}
    print(f"{run}: score = mean cosine to the 10 nearest of {BANK} AC training frames")
    for name, s in score.items():
        report["groups"][name] = r = {"kind": groups[name][0], "median": float(np.median(s)), "p5": float(np.percentile(s, 5)),
                                      "p95": float(np.percentile(s, 95)), "above_threshold": float(np.mean(s >= threshold))}
        print(f"  {name:<24} {groups[name][0]:<6} median {r['median']:.3f} (p5 {r['p5']:.3f}, p95 {r['p95']:.3f}), "
              f"called track {100 * r['above_threshold']:5.1f}%")
    print(f"  AUROC track vs not track: {report['auroc_track_vs_not']:.3f}; threshold {threshold:.3f} keeps 95% of track pictures")
    for name, (kind, _) in groups.items():
        if kind == "track":
            report["groups"][name]["auroc_vs_not"] = auroc(score[name], not_track)
    # A linear probe: logistic regression on the descriptors, trained on AC training frames
    # against synthetic pictures and wallpapers only, then tested on what it never saw: the
    # held-out and real tracks, and the home sessions (a kind of "not track" it has no example of).
    train_neg = [n for n in ("synthetic", "macOS wallpapers") if n in desc]
    x = np.concatenate([bank] + [desc[n] for n in train_neg])
    y = np.concatenate([np.ones(len(bank)), np.zeros(sum(len(desc[n]) for n in train_neg))])
    w, bias = np.zeros(x.shape[1]), 0.0
    weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # classes balanced
    for _ in range(3000):
        prob = 1 / (1 + np.exp(-(x @ w + bias)))
        grad = (prob - y) * weight
        w -= 0.5 * (x.T @ grad / len(y) + 1e-3 * w)
        bias -= 0.5 * grad.mean()
    probe = {name: 1 / (1 + np.exp(-(d @ w + bias))) for name, d in desc.items()}
    report["probe"] = {name: {"median_p_track": float(np.median(p)), "called_track": float(np.mean(p >= 0.5))}
                       for name, p in probe.items()}
    report["probe_auroc_track_vs_home"] = auroc(np.concatenate([probe[n] for n, (k, _) in groups.items() if k == "track"]),
                                                probe["home GoPro sessions"])
    print(f"  linear probe (trained on AC vs {', '.join(train_neg)}), share called track at p >= 0.5:")
    for name, r in report["probe"].items():
        print(f"    {name:<24} {100 * r['called_track']:5.1f}%  (median p {r['median_p_track']:.2f})")
    print(f"    AUROC tracks vs home sessions (never trained on): {report['probe_auroc_track_vs_home']:.3f}")
    (Path("runs") / run / "track_or_not.json").write_text(json.dumps(report, indent=2))

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    colours = {"track": "#2c7fb8", "not": "#d9534f", "probe": "#999999"}
    for i, (name, s) in enumerate(score.items()):
        a.scatter(s, np.full(s.size, i) + rng.uniform(-0.25, 0.25, s.size), s=4, alpha=0.5, color=colours[groups[name][0]])
    a.axvline(threshold, color="#222222", lw=1, ls="--")
    a.set_yticks(range(len(score)), list(score)); a.set_xlabel("closeness to AC track frames (mean cosine of 10 nearest)")
    a.set_title("Does the encoder know a track when it sees one?")
    everything = np.concatenate([bank] + list(desc.values()))
    mean = everything.mean(0)
    _, _, vt = np.linalg.svd(everything - mean, full_matrices=False)
    for name, d in desc.items():
        p = (d - mean) @ vt[:2].T
        b.scatter(p[:, 0], p[:, 1], s=4, alpha=0.5, label=name)
    b.set_title("descriptors, first two principal components"); b.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(Path("runs") / run / "track_or_not.png", dpi=110)


if __name__ == "__main__":
    main()
