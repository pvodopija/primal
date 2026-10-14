"""
The ten new circuits' two held-out tracks (Deutschlandring, Daytona road course), unseen by
every model: references and live laps are rig renders (full-width wander, its turn undone
from pose.npz), seen through the virtual camera at the training view. Each live lap is
scored level, with the view held pitched (down / up 4 to 16 deg: braking, a nodding head,
a glance down at the wheel, a camera tilted on the helmet) and shaken (train/camera.py's
kart vibration, kerb hits and head wobble), against each reference lap of the same track;
also how large the errors over 100 ms are. Per
tick the error and how far sideways the live camera is from the reference's at the same
place. All laps of a track come from one session (same car and weather), so this is the
easier, same-session kind of pair.

Usage: [PRIMAL_SIMULATE=int8|fp16] PYTHONPATH=. python experiments/newtracks_eval.py RUN [RUN ...]
       (writes runs/RUN/newtracks_eval[_int8|_fp16].json)
"""
from __future__ import annotations

import json
import os
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
# (name, held pitch deg (positive = up), shake level: train/camera.py's kart vibration, kerbs and head wobble)
CONDITIONS = [("level", 0.0, 0.0)] + [(f"pitch {p:+g}", p, 0.0) for p in (-4.0, 4.0, -8.0, 8.0, -12.0, 12.0, -16.0, 16.0)] \
    + [("shake 1", 0.0, 1.0), ("shake 2", 0.0, 2.0), ("shake 2, pitch -8", -8.0, 2.0)]
# how large the errors over 100 ms are
ERROR_BANDS = ((0.1, 0.15), (0.15, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1e9))
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
        rows = []  # (condition, gap m, |error| s)
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
                    for name, pitch, level in CONDITIONS:
                        frames = camera.render_clip(raw, t[used], np.random.default_rng(0), source=live.source_view,
                                                    frame_poses=turned_back(live, used, pitch), shake_level=level)
                        beliefs, _ = per_offset(model, encode(model, frames, device), row[clip_idx], ref)
                        stream = {"grid": grid, "t": t[ticks], "belief": beliefs[0].astype(np.float64),
                                  "target": torch.from_numpy(target.astype(np.float32))}
                        bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                        err = np.abs(signed_error(stream, bins))[skip:]
                        rows += [(name, float(g), float(e)) for g, e in zip(gap[skip:], err)]
            print(f"  {run} {track[:28]:<28} done", flush=True)
        cond = np.array([r[0] for r in rows]); gap, err = (np.array([r[k] for r in rows]) for k in (1, 2))

        def cell(m: np.ndarray) -> dict:
            out = {"ticks": int(m.sum()), "over_100ms": float(np.mean(err[m] > 0.1)), "median_ms": float(np.median(err[m]) * 1000),
                   "p90_ms": float(np.percentile(err[m], 90) * 1000)}
            out["bands"] = {f"{lo:g}-{hi:g} s" if hi < 1e8 else f"over {lo:g} s": float(np.mean((err[m] >= lo) & (err[m] < hi)))
                            for lo, hi in ERROR_BANDS}
            return out

        report = {"by_condition": {}, "by_condition_within_3m": {}, "by_gap_level": {}}
        print(f"\n{run}: new held-out tracks, two-mode tracker: over 100 ms, then the share of ticks off by "
              + ", ".join(f"{lo:g}-{hi:g} s" if hi < 1e8 else f"over {lo:g} s" for lo, hi in ERROR_BANDS))
        for name, _, _ in CONDITIONS:
            report["by_condition"][name] = a = cell(cond == name)
            report["by_condition_within_3m"][name] = cell((cond == name) & (gap <= 3))
            print(f"  {name:<18} {100 * a['over_100ms']:5.1f}%  (median {a['median_ms']:3.0f} ms, p90 {a['p90_ms']:4.0f} ms)  "
                  + "  ".join(f"{100 * v:4.1f}%" for v in a["bands"].values()))
        for lo, hi in GAP_BANDS:
            m = (cond == "level") & (gap >= lo) & (gap < hi)
            if m.sum() > 50:
                report["by_gap_level"][f"{lo}-{hi} m"] = r = cell(m)
                print(f"  level, {lo}-{hi} m sideways: {100 * r['over_100ms']:5.1f}% ({r['median_ms']:.0f} ms, {r['ticks']} ticks)")
        sim = os.environ.get("PRIMAL_SIMULATE")  # load_model's "as on the NPU" switch
        (Path("runs") / run / f"newtracks_eval{'_' + sim if sim else ''}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
