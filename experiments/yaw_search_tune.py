"""
Tuning the head-angle search on the trained tracks (G1), never on the held-out ones.

The search as first run (experiments/yaw_search.py) leaves straight ahead on a quarter of
the ticks when there is no turn to find. Rule: follow the angle with the best mean match
over the last W seconds, but only when it beats straight ahead by a margin. Tuned on
  - held turns: the wide laps training left out as G1's drives, turned ±3.5° and ±7°,
    against another wide lap of the same track;
  - no turn: G1's held-out recorded laps, straight ahead, against a wide lap that is not
    their drive;
  - looking into corners (the left-out wide laps with a head that follows the corners,
    `yaw_search.corner_look`), alone and on top of a held 7° turn;
with Halo's budget (references at -3.5°, 0°, +3.5°).

Usage: PYTHONPATH=. python experiments/yaw_search_tune.py RUN   (writes runs/RUN/yaw_search_tune.json)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

import experiments.yaw_search as ys
from train import camera
from train.dataset import LapIndex, same_drive
from train.eval import (ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator,
                        _stream_errors, load_model)
from train.train import default_device, holdout_live_laps

ys.OFFSETS = np.array([-3.5, 0.0, 3.5])
WINDOWS_S = (10.0, 30.0)
MARGINS = (0.0, 0.002, 0.005, 0.01, 0.02)


def streams(model, clip_len, device):
    """(kind, turn, beliefs [3, T, N], scores [3, T], stream) for every tuning case."""
    v3 = LapIndex.load(Path("data/packed_ac_v3"), split="train")
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="train")
    reserved = holdout_live_laps(v3, 1)
    held = [lap for lap in v3.laps if lap.lap_id in reserved]
    for track, group in sorted(wide.by_track.items()):
        whole = [w for w in group if w.s_span > 0.98 and w.usable_as_reference(0.98, ys.AXIS)]
        left_out = [w for w in whole if any(same_drive(w, h) for h in held)]
        trained = [w for w in whole if w not in left_out]
        if not trained:
            continue
        reference = trained[0]
        grid = reference.reference_grid(ys.AXIS)
        ref_raw = np.asarray(reference.frames()[grid.frame_idx])
        rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
        refs = [ys.encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng, pose=(float(o), 0, 0)), device)
                for o in ys.OFFSETS]
        cases = [("turn", lap, k) for k, lap in enumerate(left_out[:2])]
        cases += [("no turn", h, 0) for h in held if h.track == track and not same_drive(h, reference)]
        for kind, live, k in cases:
            every = max(int(round(live.fps / STREAM_HZ)), 1)
            ticks = np.arange((clip_len - 1) * ys.STRIDE, live.n_frames, every)
            clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * ys.STRIDE
            used = np.unique(clip_idx)
            row = np.full(live.n_frames, -1)
            row[used] = np.arange(used.size)
            raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
            stream = {"grid": grid, "t": t[ticks], "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
            turns = [(1.0 if k % 2 == 0 else -1.0) * a for a in (3.5, 7.0)] if kind == "turn" else [0.0]
            for turn in turns:
                frames = camera.render_clip(raw, t[used], np.random.default_rng(k), pose=(turn, 0, 0)) if live.wide else raw
                beliefs, scores = ys.per_offset(model, ys.encode(model, frames, device), row[clip_idx], refs)
                yield kind, turn, beliefs, scores, stream
            if kind == "turn":
                look = ys.corner_look(camera.render_clip(raw, t[used], np.random.default_rng(k)), t[used])
                for extra in (0.0, (1.0 if k % 2 == 0 else -1.0) * 7.0):
                    frames = ys.render_turning(raw, look + extra)
                    beliefs, scores = ys.per_offset(model, ys.encode(model, frames, device), row[clip_idx], refs)
                    yield "look", extra, beliefs, scores, stream
            print(f"  {track[:28]:<28} {kind:<8} {live.lap_id[-22:]}", flush=True)


def main() -> None:
    run = sys.argv[1]
    device = torch.device(default_device())
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    rules = [("no search", None, None)] + [(f"W {w:g} s, margin {m:g}", w, m) for w in WINDOWS_S for m in MARGINS]
    groups = ("turn 3.5", "turn 7", "no turn", "look", "look + 7")
    errs = {(kind, r[0]): [] for kind in groups for r in rules}
    gaps = {g: [] for g in groups}
    allowed = np.ones(3, bool)
    for kind, turn, beliefs, scores, stream in streams(model, int(payload["args"]["clip_len"]), device):
        group = {"no turn": "no turn", "look": "look + 7" if turn else "look"}.get(kind, f"turn {abs(turn):g}")
        skip = int(ACQUISITION_S * STREAM_HZ)
        gaps[group].append((scores.max(0) - scores[1])[skip:])
        for name, w, m in rules:
            b = beliefs[1] if w is None else ys.searched(beliefs, scores, allowed, int(w * STREAM_HZ), m)[0]
            bins = _run_estimator({**stream, "belief": b.astype(np.float64)}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES),
                                  None, None, 0, reference_time=True)
            errs[(group, name)].append(_stream_errors(stream, bins)[1])
    print("\nper-tick gap, best angle minus straight ahead (percentiles 50/90/99):")
    for g, v in gaps.items():
        v = np.concatenate(v)
        print(f"  {g:<9} " + " ".join(f"{np.percentile(v, p):.4f}" for p in (50, 90, 99)))
    report = {}
    print(f"\n{run}, trained tracks, share of ticks over 100 ms:\n{'rule':<26}" + "".join(f"{g:>10}" for g in groups + ("mean",)))
    for name, *_ in rules:
        row = {g: float(np.mean(np.concatenate(errs[(g, name)]) > 100)) for g in groups}
        row["mean"] = float(np.mean(list(row.values())))
        report[name] = row
        print(f"{name:<26}" + "".join(f"{100 * row[g]:>9.1f}%" for g in groups + ("mean",)))
    (Path("runs") / run / "yaw_search_tune.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
