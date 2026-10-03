"""
Why a held head turn hurts: the lost edge, or the shift of everything?

A 7° turn at the training view (91.5°) drops the outer 16 px of one side (30 px at 14°),
shows 16 px of new scenery on the other that the reference never saw, and moves
everything between by 9 px in the centre to 16 px at the edges. Taken apart, on the
held-out tracks' wide laps against another lap of the same session (straight ahead):
  - level;
  - edge lost only: the level view with one side's outer band hidden (filled with the
    mirror image of its neighbour, as camera_jitter's borders were), no shift;
  - turned, new side hidden: the exact 7° turn with the newly seen band hidden the same way;
  - turned: the exact 7° turn.
Signs alternate lap to lap.

Usage: PYTHONPATH=. python experiments/edge_test.py RUN [RUN ...]   (writes runs/RUN/edge_test.json)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.yaw_search import encode, per_offset
from train import camera
from train.dataset import LapIndex
from train.eval import (ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator,
                        _stream_errors, load_model)
from train.train import default_device

LAPS, STRIDE, AXIS = 4, 2, "time"


def band(turn: float) -> int:
    """Pixels a turn moves out of (and into) the training view at its edge."""
    f, half = camera.TRAINING.focal, camera.TRAINING.width / 2
    return int(round(half - f * np.tan(np.arctan(half / f) - np.radians(abs(turn)))))


def hide(frames: np.ndarray, width: int, side: str) -> np.ndarray:
    """One side's outer band replaced by the mirror image of the band next to it."""
    out = frames.copy()
    if side == "left":
        out[:, :, :width] = frames[:, :, width:2 * width][:, :, ::-1]
    else:
        out[:, :, -width:] = frames[:, :, -2 * width:-width][:, :, ::-1]
    return out


def main() -> None:
    device = torch.device(default_device())
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    names = ["level", "edge lost only (7° worth)", "edge lost only (14° worth)",
             "turned 7°, new side hidden", "turned 7°", "turned 14°, new side hidden", "turned 14°"]
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        errs = {name: [] for name in names}
        for track, group in sorted(wide.by_track.items()):
            whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
            for k, live in enumerate(whole[:LAPS]):
                reference = whole[(k + 1) % len(whole)]
                grid = reference.reference_grid(AXIS)
                rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
                ref = [encode(model, camera.render_clip(np.asarray(reference.frames()[grid.frame_idx]),
                                                        np.zeros(grid.n_bins), rng), device)]
                every = max(int(round(live.fps / STREAM_HZ)), 1)
                ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                used = np.unique(clip_idx)
                row = np.full(live.n_frames, -1)
                row[used] = np.arange(used.size)
                raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
                stream = {"grid": grid, "t": t[ticks], "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
                sign = 1.0 if k % 2 == 0 else -1.0
                lost, gained = ("left", "right") if sign > 0 else ("right", "left")  # turning right loses the left edge
                level = camera.render_clip(raw, t[used], np.random.default_rng(k))
                turned = {a: camera.render_clip(raw, t[used], np.random.default_rng(k), pose=(sign * a, 0, 0)) for a in (7, 14)}
                frames = {
                    "level": level,
                    "edge lost only (7° worth)": hide(level, band(7), lost),
                    "edge lost only (14° worth)": hide(level, band(14), lost),
                    "turned 7°, new side hidden": hide(turned[7], band(7), gained),
                    "turned 7°": turned[7],
                    "turned 14°, new side hidden": hide(turned[14], band(14), gained),
                    "turned 14°": turned[14],
                }
                for name, f in frames.items():
                    beliefs, _ = per_offset(model, encode(model, f, device), row[clip_idx], ref)
                    bins = _run_estimator({**stream, "belief": beliefs[0].astype(np.float64)},
                                          EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                    errs[name].append(_stream_errors(stream, bins)[1])
                print(f"  {run} {track[:26]:<26} live {live.lap_id[-5:]} reference {reference.lap_id[-5:]}", flush=True)
        report = {name: float(np.mean(np.concatenate(v) > 100)) for name, v in errs.items()}
        print(f"\n{run}: share of ticks over 100 ms (band at 7°: {band(7)} px, at 14°: {band(14)} px of {camera.TRAINING.width})")
        for name in names:
            print(f"  {name:<32} {100 * report[name]:5.1f}%")
        (Path("runs") / run / "edge_test.json").write_text(json.dumps({"band_px": {"7": band(7), "14": band(14)}, "report": report}, indent=2))


if __name__ == "__main__":
    main()
