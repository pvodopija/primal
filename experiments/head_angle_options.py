"""
Two questions about the head-angle search (experiments/yaw_search.py):

1. What would a gyroscope give it at best? The search picks the reference angle that
   matched best over the last 10 s, so it follows a held turn but lags a glance into a
   corner. A gyro (with the kart's own turn taken out) would report the head angle at
   once. Its ceiling: at every tick, the reference angle nearest the head's true angle
   over the clip ("perfect gyro").
2. What does the search's room cost on Halo? With one camera for both laps, the model's
   view must be narrower than the camera's: Halo's 81.2° x 65.5° (128x96 at model size)
   cropped to 112x96 (73.7°) leaves ±3.75°, room for references at -3.5°, 0° and +3.5°.
   Live: the centre crop of the glasses' view. Reference: the same lap's Halo frames,
   cropped turned.

Held-out tracks, wide laps against another lap of the same session (straight ahead),
heads: level, held 3.5° and 7°, turning into corners (yaw_search.corner_look), and into
corners with the glasses 7° crooked. Share of ticks over 100 ms, two-mode tracker.

Usage: PYTHONPATH=. python experiments/head_angle_options.py RUN [RUN ...]
       (writes runs/RUN/head_angle_options.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.yaw_search import corner_look, encode, per_offset, searched
from train import camera
from train.dataset import LapIndex
from train.eval import (REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, _stream_errors,
                        load_model)
from train.train import default_device

LAPS, STRIDE, AXIS, WINDOW_S = 4, 2, "time", 10.0
WIDE_OFFSETS = np.arange(-10.5, 10.51, 3.5)
HALO_OFFSETS = np.array([-3.5, 0.0, 3.5])
HALO_CROP = camera.View(112, 96, camera.HALO.focal, 56.0, camera.HALO.cy)  # 73.7° x 65.5°
HEADS = ("level", "held 3.5°", "held 7°", "into corners", "into corners + 7° crooked")
METHODS = {
    "training view, wide reference": ["no search", "search (10 s)", "perfect gyro"],
    "Halo": ["81° full view, no search", "74° crop, no search", "74° crop, search (10 s)", "74° crop, perfect gyro"],
}


def turned(raw: np.ndarray, yaw: np.ndarray, view: camera.View, source: camera.View = camera.WIDE) -> np.ndarray:
    return np.stack([camera.render(f, source, view, yaw=float(y)) for f, y in zip(raw, yaw)])


def gyro(beliefs: np.ndarray, offsets: np.ndarray, clip_yaw: np.ndarray) -> np.ndarray:
    """Per tick, the belief of the reference angle nearest the head's mean angle over the clip."""
    pick = np.abs(offsets[:, None] - clip_yaw[None, :]).argmin(0)
    return beliefs[pick, np.arange(beliefs.shape[1])]


def main() -> None:
    device = torch.device(default_device())
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    probe = camera.HALO.K @ camera.rotation(3.5, 0, 0) @ np.linalg.inv(HALO_CROP.K) @ np.array(
        [[0, 0, 1], [112, 0, 1], [0, 96, 1], [112, 96, 1]], dtype=np.float64).T
    print(f"Halo crop {np.degrees(2 * np.arctan(56 / HALO_CROP.focal)):.1f}° wide; turned 3.5°, its corners land at "
          f"x {np.round(probe[0] / probe[2], 1)} y {np.round(probe[1] / probe[2], 1)} in the 128x96 frame")
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        errs = {(setup, head, m): [] for setup, ms in METHODS.items() for head in HEADS for m in ms}
        for track, group in sorted(wide.by_track.items()):
            whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
            for k, live in enumerate(whole[:LAPS]):
                reference = whole[(k + 1) % len(whole)]
                grid = reference.reference_grid(AXIS)
                ref_raw = np.asarray(reference.frames()[grid.frame_idx])
                zeros = np.zeros(len(ref_raw))
                refs_wide = [encode(model, turned(ref_raw, zeros + o, camera.TRAINING), device) for o in WIDE_OFFSETS]
                ref_halo = turned(ref_raw, zeros, camera.HALO)
                ref_halo_full = [encode(model, ref_halo, device)]
                refs_halo_crop = [encode(model, turned(ref_halo, zeros + o, HALO_CROP, source=camera.HALO), device)
                                  for o in HALO_OFFSETS]

                every = max(int(round(live.fps / STREAM_HZ)), 1)
                ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                used = np.unique(clip_idx)
                row = np.full(live.n_frames, -1)
                row[used] = np.arange(used.size)
                raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
                stream = {"grid": grid, "t": t[ticks], "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
                sign = 1.0 if k % 2 == 0 else -1.0
                level = turned(raw, np.zeros(len(raw)), camera.TRAINING)
                look = corner_look(level, t[used])
                yaws = {"level": np.zeros(len(raw)), "held 3.5°": np.full(len(raw), sign * 3.5),
                        "held 7°": np.full(len(raw), sign * 7.0), "into corners": look,
                        "into corners + 7° crooked": look + sign * 7.0}
                window = int(WINDOW_S * STREAM_HZ)

                def score(setup: str, head: str, m: str, belief: np.ndarray) -> None:
                    bins = _run_estimator({**stream, "belief": belief.astype(np.float64)},
                                          EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                    errs[(setup, head, m)].append(_stream_errors(stream, bins)[1])

                for head, yaw in yaws.items():
                    clip_yaw = yaw[row[clip_idx]].mean(1)
                    # the training view, references turned from the wide render
                    b, s = per_offset(model, encode(model, level if head == "level" else turned(raw, yaw, camera.TRAINING), device),
                                      row[clip_idx], refs_wide)
                    setup, ms = "training view, wide reference", METHODS["training view, wide reference"]
                    score(setup, head, ms[0], b[int(np.argmin(np.abs(WIDE_OFFSETS)))])
                    score(setup, head, ms[1], searched(b, s, np.ones(len(WIDE_OFFSETS), bool), window)[0])
                    score(setup, head, ms[2], gyro(b, WIDE_OFFSETS, clip_yaw))
                    # Halo: the full view, or its centre crop against the turned crops of the reference
                    setup, ms = "Halo", METHODS["Halo"]
                    b, _ = per_offset(model, encode(model, turned(raw, yaw, camera.HALO), device), row[clip_idx], ref_halo_full)
                    score(setup, head, ms[0], b[0])
                    b, s = per_offset(model, encode(model, turned(raw, yaw, HALO_CROP), device), row[clip_idx], refs_halo_crop)
                    score(setup, head, ms[1], b[1])
                    score(setup, head, ms[2], searched(b, s, np.ones(len(HALO_OFFSETS), bool), window)[0])
                    score(setup, head, ms[3], gyro(b, HALO_OFFSETS, clip_yaw))
                print(f"  {run} {track[:26]:<26} live {live.lap_id[-5:]} reference {reference.lap_id[-5:]}, "
                      f"into corners |yaw| median {np.median(np.abs(look)):.1f}°, p90 {np.percentile(np.abs(look), 90):.1f}°", flush=True)

        report = {}
        for setup, ms in METHODS.items():
            print(f"\n{run}, {setup}: share of ticks over 100 ms")
            print(f"  {'head':<28}" + "".join(f"{m:>27}" for m in ms))
            report[setup] = {}
            for head in HEADS:
                r = {m: float(np.mean(np.concatenate(errs[(setup, head, m)]) > 100)) for m in ms}
                report[setup][head] = r
                print(f"  {head:<28}" + "".join(f"{100 * r[m]:>26.1f}%" for m in ms))
        (Path("runs") / run / "head_angle_options.json").write_text(json.dumps(
            {"wide_offsets": WIDE_OFFSETS.tolist(), "halo_offsets": HALO_OFFSETS.tolist(), "window_s": WINDOW_S,
             "halo_crop_px": [HALO_CROP.width, HALO_CROP.height], "report": report}, indent=2))


if __name__ == "__main__":
    main()
