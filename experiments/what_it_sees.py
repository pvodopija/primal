"""
Where in the picture does the encoder look? Occlusion maps: a grey 16x16 patch slides over
the 148x80 frame (8 px steps); at each place, how far the frame's 128-d descriptor moves
(1 - cosine to the unoccluded descriptor). Averaged over frames spread along held-out AC
laps (the training view) and along a real lap (indoor GoPro, the c80 crop), with a few
single frames for illustration. Writes runs/RUN/what_it_sees.png and .json.

Usage: PYTHONPATH=. python experiments/what_it_sees.py RUN
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.yaw_search import encode
from train import camera
from train.dataset import LapIndex
from train.eval import load_model
from train.train import default_device

PATCH, STEP, N_SIM, N_REAL = 16, 8, 240, 160


@torch.no_grad()
def occlusion(model, frames: np.ndarray, device) -> np.ndarray:
    """[F, rows, cols]: 1 - cosine between each frame's descriptor and its occluded versions."""
    base = torch.nn.functional.normalize(encode(model, frames, device), dim=-1)
    ys, xs = range(0, 80 - PATCH + 1, STEP), range(0, 148 - PATCH + 1, STEP)
    out = np.zeros((len(frames), len(ys), len(xs)), np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            occ = frames.copy()
            occ[:, y:y + PATCH, x:x + PATCH] = 128
            d = torch.nn.functional.normalize(encode(model, occ, device), dim=-1)
            out[:, i, j] = (1 - (d * base).sum(-1)).cpu().numpy()
    return out


def upsample(m: np.ndarray) -> np.ndarray:
    """A patch map back to 148x80: each pixel gets the mean of the patches covering it."""
    acc, n = np.zeros((80, 148)), np.zeros((80, 148))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            acc[i * STEP:i * STEP + PATCH, j * STEP:j * STEP + PATCH] += m[i, j]
            n[i * STEP:i * STEP + PATCH, j * STEP:j * STEP + PATCH] += 1
    return acc / np.maximum(n, 1)


def main() -> None:
    run = sys.argv[1]
    device = torch.device(default_device())
    model, _ = load_model(Path("runs") / run / "best.pt", device)
    model.eval()
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    sim = []
    for track, group in sorted(wide.by_track.items()):
        lap = max(group, key=lambda x: x.s_span)
        idx = np.linspace(0, lap.n_frames - 1, N_SIM // len(wide.by_track)).astype(int)
        sim.append(camera.render_clip(np.asarray(lap.frames()[idx]), np.zeros(idx.size), np.random.default_rng(zlib.crc32(lap.lap_id.encode()))))
    sim = np.concatenate(sim)
    real_path = Path("data/real-footage/indoor-pov/frames_148x80_c80.npz")
    real = np.load(real_path)["frames"] if real_path.exists() else None
    if real is not None:
        real = real[np.linspace(0, len(real) - 1, N_REAL).astype(int)]
    maps = {"AC held-out laps": occlusion(model, sim, device)}
    if real is not None:
        maps["indoor GoPro (real)"] = occlusion(model, real, device)
    report = {}
    fig, axes = plt.subplots(len(maps), 4, figsize=(14, 2.6 * len(maps)), squeeze=False)
    for r, (name, m) in enumerate(maps.items()):
        mean = upsample(m.mean(0))
        frames = sim if r == 0 else real
        thirds = [float(mean[a:b].mean()) for a, b in ((0, 27), (27, 54), (54, 80))]
        report[name] = {"mean_change": float(mean.mean()), "rows_top_middle_bottom": thirds,
                        "cols_left_middle_right": [float(mean[:, a:b].mean()) for a, b in ((0, 49), (49, 99), (99, 148))]}
        ax = axes[r, 0]
        ax.imshow(cv2.cvtColor(np.median(frames, 0).astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax.imshow(mean, cmap="magma", alpha=0.6)
        ax.set_title(f"{name}: average", fontsize=9)
        for c, k in enumerate(np.linspace(0, len(frames) - 1, 3).astype(int)):
            ax = axes[r, c + 1]
            ax.imshow(cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB))
            ax.imshow(upsample(m[k]), cmap="magma", alpha=0.55)
            ax.set_title(f"{name}: frame {k}", fontsize=9)
    for ax in fig.axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{run}: where a grey patch moves the descriptor most (brighter = the encoder relies on it)", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path("runs") / run / "what_it_sees.png", dpi=110, bbox_inches="tight")
    (Path("runs") / run / "what_it_sees.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
