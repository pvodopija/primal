"""
A demo under stress: one lap of a held-out new circuit (unseen by every model) against
another lap of it, the live camera shaken hard all lap (train/camera.py's kart vibration,
kerbs and head wobble at level 2) and pitched on a schedule: a nodding head, braking dips,
glances down at the wheel and a stretch with the helmet tilted up. Several models side by
side: the camera view as the models see it, each model's delta error against the truth over
the lap, and its share of ticks over 100 ms. Writes runs/demos/stress_<track>.mp4 and .json.

Usage: PYTHONPATH=. python experiments/stress_demo.py RUN [RUN ...]
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

from experiments.newtracks_eval import DATA, turned_back
from experiments.yaw_bias import signed_error
from experiments.yaw_search import encode, per_offset
from train import camera
from train.dataset import LapIndex
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, load_model
from train.train import default_device

TRACK, STRIDE, AXIS, SHAKE = "deutschlandring", 2, "time", 2.0
COLOURS = ["#999999", "#2c7fb8", "#d95f02", "#7570b3"]


def pitch_schedule(t: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Degrees (positive = up) and a label per frame: what the head is doing."""
    t = t - t[0]
    pitch = 3.0 * np.sin(2 * np.pi * 0.3 * t)  # nodding all lap
    label = np.array(["nodding"] * len(t), dtype=object)
    brake = (t % 9.0) < 1.5  # braking dips
    pitch[brake] += -6.0 * np.sin(np.pi * (t[brake] % 9.0) / 1.5)
    label[brake] = "braking dip"
    glance = ((t % 11.0) > 5.0) & ((t % 11.0) < 5.8)  # a look down at the wheel
    pitch[glance] = -15.0
    label[glance] = "glance at the wheel"
    tilted = (t > 0.6 * t[-1]) & (t < 0.8 * t[-1])  # helmet tilted up for a stretch
    pitch[tilted] += 8.0
    label[tilted] = label[tilted] + " + helmet tilted up"
    return pitch, list(label)


def main() -> None:
    runs = sys.argv[1:]
    device = torch.device(default_device())
    laps = LapIndex.load(Path(DATA), split="holdout").by_track[TRACK]
    whole = [lap for lap in laps if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
    reference, live = whole[0], whole[1]
    grid = reference.reference_grid(AXIS)
    ref_raw = np.asarray(reference.frames()[grid.frame_idx])
    t = live.t().astype(np.float64)
    every = max(int(round(live.fps / STREAM_HZ)), 1)
    clip_len = 12
    ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
    clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
    used = np.unique(clip_idx)
    row = np.full(live.n_frames, -1)
    row[used] = np.arange(used.size)
    pitch, label = pitch_schedule(t)
    poses = turned_back(live, used)
    poses[:, 1] = pitch[used]
    frames = camera.render_clip(np.asarray(live.frames()[used]), t[used], np.random.default_rng(0), source=live.source_view,
                                frame_poses=poses, shake_level=SHAKE)
    target = torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))
    skip = int(ACQUISITION_S * STREAM_HZ)
    errors = {}
    for run in runs:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
        ref = [encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng, source=reference.source_view,
                                                frame_poses=turned_back(reference, grid.frame_idx)), device)]
        beliefs, _ = per_offset(model, encode(model, frames, device), row[clip_idx], ref)
        stream = {"grid": grid, "t": t[ticks], "belief": beliefs[0].astype(np.float64), "target": target}
        bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
        errors[run] = np.abs(signed_error(stream, bins)) * 1000
        print(f"  {run:<28} over 100 ms {100 * np.mean(errors[run][skip:] > 100):5.1f}%, median {np.median(errors[run][skip:]):.0f} ms", flush=True)

    out = Path("runs/demos"); out.mkdir(parents=True, exist_ok=True)
    path = out / f"stress_{TRACK}.mp4"
    lap_t = t[ticks] - t[ticks][0]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = None
    for k in range(skip, len(ticks)):
        fig.clf()
        ax_img = fig.add_axes([0.03, 0.47, 0.5, 0.5])
        img = frames[row[ticks[k]]]
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="nearest"); ax_img.axis("off")
        ax_img.set_title(f"camera as the models see it  |  pitch {pitch[ticks[k]]:+.0f}°, shake 2  |  {label[ticks[k]]}", fontsize=11)
        ax_txt = fig.add_axes([0.56, 0.47, 0.42, 0.5]); ax_txt.axis("off")
        ax_txt.text(0, 0.95, f"{TRACK}, never trained on  |  lap time {lap_t[k]:5.1f} s", fontsize=12, weight="bold")
        for i, run in enumerate(runs):
            e = errors[run][k]
            so_far = errors[run][skip:k + 1]
            ax_txt.text(0, 0.78 - 0.17 * i, run, fontsize=11, color=COLOURS[i % len(COLOURS)], weight="bold")
            ax_txt.text(0, 0.70 - 0.17 * i, f"error now {e:6.0f} ms   " + ("OK" if e <= 100 else "OFF"),
                        fontsize=11, color="#1a9850" if e <= 100 else "#d73027")
            ax_txt.text(0.55, 0.70 - 0.17 * i, f"over 100 ms so far {100 * np.mean(so_far > 100):4.1f}%", fontsize=10)
        ax = fig.add_axes([0.06, 0.07, 0.9, 0.33])
        for i, run in enumerate(runs):
            ax.plot(lap_t[skip:], np.minimum(errors[run][skip:], 1000), color=COLOURS[i % len(COLOURS)], lw=1.2, label=run)
        ax.axhline(100, color="#222222", ls="--", lw=0.8)
        ax.axvline(lap_t[k], color="#222222", lw=0.8)
        ax.set_ylim(0, 1000); ax.set_xlim(lap_t[skip], lap_t[-1])
        ax.set_ylabel("delta error, ms (capped at 1 s)"); ax.set_xlabel("lap time, s")
        ax.legend(loc="upper left", fontsize=9, ncol=len(runs))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), STREAM_HZ, (rgb.shape[1], rgb.shape[0]))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    (out / f"stress_{TRACK}.json").write_text(json.dumps(
        {run: {"over_100ms": float(np.mean(e[skip:] > 100)), "median_ms": float(np.median(e[skip:]))} for run, e in errors.items()}, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
