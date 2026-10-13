"""
The ten new circuits' two held-out tracks (Deutschlandring, Daytona road course), unseen by
every model: references and live laps are rig renders (full-width wander, its turn undone
from pose.npz), seen through the virtual camera at the training view. Each live lap is
scored level and with the view held pitched (down / up 4 and 8 deg: braking, a nodding
head, a camera tilted on the helmet), against each reference lap of the same track. Per
tick the error and how far sideways the live camera is from the reference's at the same
place. All laps of a track come from one session (same car and weather), so this is the
easier, same-session kind of pair.

Usage: PYTHONPATH=. python experiments/newtracks_eval.py RUN [RUN ...]   (writes runs/RUN/newtracks_eval.json)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.yaw_bias import signed_error
from experiments.yaw_search import encode, per_offset
from train import camera
from train.dataset import LapIndex
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, load_model
from train.train import default_device

DATA, STRIDE, AXIS = "data/packed_ac_b1007_new_live", 2, "time"
REFS_PER_TRACK, MIN_LIVE_S = 2, 30.0
PITCHES = (0.0, -4.0, 4.0, -8.0, 8.0)  # deg, positive = up
GAP_BANDS = ((0, 1), (1, 3), (3, 6), (6, 99))


def turned_back(lap, idx: np.ndarray, pitch: float = 0.0) -> np.ndarray:
    yaw = lap.camera_yaw()
    poses = np.zeros((len(idx), 3))
    if yaw is not None:
        poses[:, 0] = -yaw[idx]
    poses[:, 1] = pitch
    return poses


def main() -> None:
    device = torch.device(default_device())
    index = LapIndex.load(Path(DATA), split="holdout")
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        rows = []  # (pitch, gap m, |error| s)
        for track, laps in sorted(index.by_track.items()):
            refs = [lap for lap in laps if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)][:REFS_PER_TRACK]
            for reference in refs:
                grid = reference.reference_grid(AXIS)
                rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
                ref_raw = np.asarray(reference.frames()[grid.frame_idx])
                ref = [encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng, source=reference.source_view,
                                                        frame_poses=turned_back(reference, grid.frame_idx)), device)]
                ref_lat = reference.lateral()[grid.frame_idx]
                for live in laps:
                    t = live.t().astype(np.float64)
                    if live.lap_id == reference.lap_id or t[-1] - t[0] < MIN_LIVE_S:
                        continue
                    every = max(int(round(live.fps / STREAM_HZ)), 1)
                    ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                    clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                    used = np.unique(clip_idx)
                    row = np.full(live.n_frames, -1)
                    row[used] = np.arange(used.size)
                    raw = np.asarray(live.frames()[used])
                    target = grid.target(live.s()[ticks])
                    gap = np.abs(live.lateral()[ticks] - ref_lat[np.round(target).astype(int) % grid.n_bins])
                    skip = int(ACQUISITION_S * STREAM_HZ)
                    for pitch in PITCHES:
                        frames = camera.render_clip(raw, t[used], np.random.default_rng(0), source=live.source_view,
                                                    frame_poses=turned_back(live, used, pitch))
                        beliefs, _ = per_offset(model, encode(model, frames, device), row[clip_idx], ref)
                        stream = {"grid": grid, "t": t[ticks], "belief": beliefs[0].astype(np.float64),
                                  "target": torch.from_numpy(target.astype(np.float32))}
                        bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                        err = np.abs(signed_error(stream, bins))[skip:]
                        rows += [(pitch, float(g), float(e)) for g, e in zip(gap[skip:], err)]
            print(f"  {run} {track[:28]:<28} done", flush=True)
        pitch, gap, err = (np.array([r[k] for r in rows]) for k in range(3))
        cell = lambda m: {"ticks": int(m.sum()), "over_100ms": float(np.mean(err[m] > 0.1)), "median_ms": float(np.median(err[m]) * 1000)}
        report = {"by_pitch": {}, "by_pitch_within_3m": {}, "by_gap_level": {}}
        print(f"\n{run}: new held-out tracks, two-mode tracker, over 100 ms (median ms)")
        for p in PITCHES:
            report["by_pitch"][f"{p:+g}"] = a = cell(pitch == p)
            report["by_pitch_within_3m"][f"{p:+g}"] = b = cell((pitch == p) & (gap <= 3))
            print(f"  pitch {p:+3.0f} deg   all {100 * a['over_100ms']:5.1f}% ({a['median_ms']:.0f} ms)   "
                  f"within 3 m sideways {100 * b['over_100ms']:5.1f}% ({b['median_ms']:.0f} ms)")
        for lo, hi in GAP_BANDS:
            m = (pitch == 0) & (gap >= lo) & (gap < hi)
            if m.sum() > 50:
                report["by_gap_level"][f"{lo}-{hi} m"] = r = cell(m)
                print(f"  level, {lo}-{hi} m sideways: {100 * r['over_100ms']:5.1f}% ({r['median_ms']:.0f} ms, {r['ticks']} ticks)")
        (Path("runs") / run / "newtracks_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
