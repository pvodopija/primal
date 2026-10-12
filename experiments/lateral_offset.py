"""
How much does driving off the reference's line cost? Held-out tracks: each reference is a
recorded lap on the AI line (packed_ac_v3, holdout split); the live laps are the batch
b1007 renders of the holdouts, seen through the virtual camera at the training view with
no head pose: full-width wander in two weathers (the camera's offset from the AI line
logged per frame in pose.npz) and fixed +3 m / -3 m laps, with the v3 holdout laps
themselves (0 m) as the baseline, and packed_ac_v3_wide (the same drives re-rendered at
121 deg, 0 m) as the control for what re-rendering alone costs. Live laps that re-render the reference's own drive are
left out. The wandering camera also turns with its path (up to ~17 deg off the car's
heading), so the wander sets are scored twice: as rendered (offset and turn) and turned
back to the car's heading (pose.npz's cam_yaw_deg undone: the offset alone). Two-mode
reference-time tracker; per tick the error, the camera's lateral offset and its turn;
reported by set, by offset band and by turn band.

Usage: PYTHONPATH=. python experiments/lateral_offset.py RUN [RUN ...]   (writes runs/RUN/lateral_offset.json)
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
from train.dataset import LapIndex, same_drive
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, load_model
from train.train import default_device

STRIDE, AXIS = 2, "time"
REFS_PER_TRACK = 2
MIN_LIVE_S = 30.0  # shorter pieces leave too little after the tracker's acquisition
SETS = {"v3 (0 m)": "data/packed_ac_v3", "v3_wide (0 m, re-rendered)": "data/packed_ac_v3_wide", "wander, own weather": "data/packed_ac_b1007_full_own",
        "wander, other weather": "data/packed_ac_b1007_full_alt", "+3 m": "data/packed_ac_b1007_off_plus3",
        "-3 m": "data/packed_ac_b1007_off_minus3"}
BANDS = ((0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 99.0))
TURN_BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 99.0))
WANDER = ("wander, own weather", "wander, other weather")


def lateral(lap) -> np.ndarray:
    """The camera's offset from the AI line per frame, m (0 for laps recorded on it)."""
    pose = lap.directory / "pose.npz"
    return np.load(pose)["lateral_m"].astype(np.float64) if pose.exists() else np.zeros(lap.n_frames)


def main() -> None:
    device = torch.device(default_device())
    sets = {name: LapIndex.load(Path(path), split="holdout") for name, path in SETS.items()}
    refs = {}
    for track, group in sorted(sets["v3 (0 m)"].by_track.items()):
        whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
        by_session = {}
        for lap in whole:
            by_session.setdefault(lap.session_id, lap)
        refs[track] = list(by_session.values())[:REFS_PER_TRACK]
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        rows = []  # (set, mode, |lateral| m, |turn| deg, |error| s)
        for track, references in refs.items():
            for reference in references:
                grid = reference.reference_grid(AXIS)
                rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
                ref = [encode(model, camera.render_clip(np.asarray(reference.frames()[grid.frame_idx]), np.zeros(grid.n_bins), rng,
                                                        source=camera.TRAINING), device)]
                for name, index in sets.items():
                    for live in index.by_track.get(track, []):
                        t = live.t().astype(np.float64)
                        if live.lap_id == reference.lap_id or t[-1] - t[0] < MIN_LIVE_S or same_drive(live, reference, rms_m=1.5):
                            continue
                        every = max(int(round(live.fps / STREAM_HZ)), 1)
                        ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                        clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                        used = np.unique(clip_idx)
                        row = np.full(live.n_frames, -1)
                        row[used] = np.arange(used.size)
                        raw = np.asarray(live.frames()[used])
                        yaw = live.camera_yaw() if live.wide else None
                        modes = {"as rendered": None}
                        if name in WANDER and yaw is not None:
                            modes["turned back"] = np.stack([-yaw[used], 0 * yaw[used], 0 * yaw[used]], 1)
                        skip = int(ACQUISITION_S * STREAM_HZ)
                        lat = np.abs(lateral(live)[ticks])[skip:]
                        turn = np.abs(yaw[ticks])[skip:] if yaw is not None else np.zeros(lat.size)
                        for mode, frame_poses in modes.items():
                            frames = camera.render_clip(raw, t[used], np.random.default_rng(0), frame_poses=frame_poses,
                                                        source=live.source_view if live.wide else camera.TRAINING)
                            beliefs, _ = per_offset(model, encode(model, frames, device), row[clip_idx], ref)
                            stream = {"grid": grid, "t": t[ticks], "belief": beliefs[0].astype(np.float64),
                                      "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
                            bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                            err = np.abs(signed_error(stream, bins))[skip:]
                            rows += [(name, mode, float(a), float(y), float(e)) for a, y, e in zip(lat, turn, err)]
            print(f"  {run} {track[:28]:<28} done", flush=True)
        names = np.array([r[0] for r in rows]); modes = np.array([r[1] for r in rows])
        lat, turn, err = (np.array([r[k] for r in rows]) for k in (2, 3, 4))

        def cell(m: np.ndarray) -> dict:
            return {"ticks": int(m.sum()), "over_100ms": float(np.mean(err[m] > 0.1)), "median_ms": float(np.median(err[m]) * 1000)}

        def show(label: str, r: dict) -> None:
            print(f"    {label:<34} {r['ticks']:>7} ticks  {100 * r['over_100ms']:5.1f}%  ({r['median_ms']:.0f} ms)")

        report = {"by_set": {}, "by_offset": {}, "by_turn": {}}
        print(f"\n{run}: two-mode tracker, share of ticks over 100 ms (median ms)")
        for name in SETS:
            for mode in ("as rendered", "turned back"):
                m = (names == name) & (modes == mode)
                if m.any():
                    report["by_set"][f"{name}, {mode}"] = r = cell(m)
                    show(f"{name}, {mode}", r)
        base = names == "v3 (0 m)"
        for mode in ("as rendered", "turned back"):
            wander = np.isin(names, WANDER) & (modes == mode)
            print(f"  by offset from the AI line (wander {mode}, with v3 at 0 m):")
            for lo, hi in BANDS:
                m = (wander | base) & (lat >= lo) & (lat < hi)
                if m.sum() > 50:
                    report["by_offset"][f"{mode}, {lo:g}-{hi:g} m"] = r = cell(m)
                    show(f"{lo:g}-{hi:g} m", r)
        print("  by the camera's turn off the car's heading (wander as rendered):")
        for lo, hi in TURN_BANDS:
            m = np.isin(names, WANDER) & (modes == "as rendered") & (turn >= lo) & (turn < hi)
            if m.sum() > 50:
                report["by_turn"][f"{lo:g}-{hi:g} deg"] = r = cell(m)
                show(f"{lo:g}-{hi:g} deg", r)
        (Path("runs") / run / "lateral_offset.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
