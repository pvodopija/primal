"""
Head-angle search: does trying the reference at a few head angles undo a held head turn?

A held sideways turn triples the errors of every model, on trained tracks too (a design
limit: the descriptor keeps the picture's left-right layout, which is the place signal).
Instead of asking the model to ignore turns, prepare the reference at several head angles
(possible when the reference is captured wider than the model's view: Halo's 81° against a
~70° view leaves about ±5°) and, at every tick, use the angle whose reference has matched
the live frames best over the last WINDOW_S seconds. Live frames cost nothing extra.

Held-out tracks, wide renders: live laps turned by THETA (sign alternating lap to lap),
references another wide lap of the same session (the only exact way to turn a reference).
Scored with the two-mode reference-time tracker against
  - no search (the reference straight ahead),
  - the right angle given (the oracle),
  - the search over a wide grid (±10.5° in 3.5° steps) and over Halo's budget (±3.5°).

With --cross, the check that the search does no harm across sessions: references are wide
laps (turnable), lives recorded laps from another session in the other car, straight ahead
(`live`) or turning into corners (`look` re-renders, which exist only in the wide laps' car,
so any that could be the reference's own drive are left out); the right angle is then 0°.

Usage: PYTHONPATH=. python experiments/yaw_search.py [--cross] RUN [RUN ...]
       (writes runs/RUN/yaw_search.json or yaw_search_cross.json)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from train import camera
from train.dataset import LapIndex, _to_chw, same_drive
from train.eval import (ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator,
                        _stream_errors, load_model)
from train.train import default_device

OFFSETS = np.arange(-10.5, 10.51, 3.5)
HALO_BUDGET = 3.5
THETAS = (0.0, 3.5, 7.0, 14.0)
WINDOW_S, LAPS, STRIDE, AXIS = 10.0, 4, 2, "time"


@torch.no_grad()
def encode(model, frames: np.ndarray, device) -> torch.Tensor:
    return torch.cat([model.encode_reference(torch.from_numpy(_to_chw(frames[i:i + 256])).to(device), use_checkpoint=False)
                      for i in range(0, len(frames), 256)])


@torch.no_grad()
def per_offset(model, live_desc: torch.Tensor, clip_rows: np.ndarray, refs: list[torch.Tensor]):
    """For each reference angle: beliefs [T, N] and a match score per tick (mean best cosine over the clip)."""
    beliefs, scores = [], []
    scale = model.logit_scale.exp()
    for ref in refs:
        b, s = [], []
        for chunk in np.array_split(np.arange(len(clip_rows)), max(1, len(clip_rows) // 64)):
            corr = torch.einsum("tkd,nd->tkn", live_desc[torch.from_numpy(clip_rows[chunk]).to(live_desc.device)], ref)
            b.append(model.head(corr * scale).softmax(-1).float().cpu())
            s.append(corr.max(-1).values.mean(-1).float().cpu())
        beliefs.append(torch.cat(b).numpy())
        scores.append(torch.cat(s).numpy())
    return np.stack(beliefs), np.stack(scores)  # [K, T, N], [K, T]


def searched(beliefs: np.ndarray, scores: np.ndarray, allowed: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per tick, the belief of the allowed angle with the best mean score over the last `window` ticks."""
    run = np.cumsum(np.pad(scores, ((0, 0), (1, 0))), axis=1)
    t = np.arange(scores.shape[1])
    mean = (run[:, t + 1] - run[:, np.maximum(t + 1 - window, 0)]) / np.minimum(t + 1, window)
    mean[~allowed] = -np.inf
    pick = mean.argmax(0)
    return beliefs[pick, t], pick


def cross(runs: list[str]) -> None:
    device = torch.device(default_device())
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    recorded = LapIndex.load(Path("data/packed_ac_v3"), split="holdout", variants=("live", "look"))
    for run in runs:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        methods = ["no search", f"search ±{OFFSETS.max():g}°", f"search ±{HALO_BUDGET:g}° (Halo)"]
        errs = {(v, m): [] for v in ("live", "look") for m in methods}
        zero = {v: [] for v in ("live", "look")}
        for track, group in sorted(wide.by_track.items()):
            reference = next(lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS))
            grid = reference.reference_grid(AXIS)
            ref_raw = np.asarray(reference.frames()[grid.frame_idx])
            rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
            refs = [encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng, pose=(float(o), 0, 0)), device)
                    for o in OFFSETS]
            for variant in ("live", "look"):
                lives = [lap for lap in recorded.by_track[track] if lap.variant == variant and lap.s_span > 0.98
                         and (variant == "look" or lap.car_model != reference.car_model)
                         and not same_drive(lap, reference)][:LAPS]
                for live in lives:
                    every = max(int(round(live.fps / STREAM_HZ)), 1)
                    ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                    clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                    used = np.unique(clip_idx)
                    row = np.full(live.n_frames, -1)
                    row[used] = np.arange(used.size)
                    stream = {"grid": grid, "t": live.t().astype(np.float64)[ticks],
                              "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
                    beliefs, scores = per_offset(model, encode(model, np.asarray(live.frames()[used]), device), row[clip_idx], refs)
                    window = int(WINDOW_S * STREAM_HZ)
                    centre = int(np.argmin(np.abs(OFFSETS)))
                    full, pick = searched(beliefs, scores, np.ones(len(OFFSETS), bool), window)
                    streams = {"no search": beliefs[centre], methods[1]: full,
                               methods[2]: searched(beliefs, scores, np.abs(OFFSETS) <= HALO_BUDGET + 1e-6, window)[0]}
                    zero[variant].append((pick == centre)[int(ACQUISITION_S * STREAM_HZ):])
                    for m, b in streams.items():
                        bins = _run_estimator({**stream, "belief": b.astype(np.float64)}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES),
                                              None, None, 0, reference_time=True)
                        errs[(variant, m)].append(_stream_errors(stream, bins)[1])
                    print(f"  {run} {track[:26]:<26} {variant} {live.lap_id[-22:]} reference {reference.lap_id[-5:]}", flush=True)
        report = {}
        print(f"\n{run}: across sessions, share of ticks over 100 ms (two-mode reference-time tracker)")
        print(f"{'live laps':<26}" + "".join(f"{m:>22}" for m in methods) + f"{'search stays at 0°':>22}")
        for variant, name in (("live", "straight ahead"), ("look", "turning into corners")):
            row_ = {m: float(np.mean(np.concatenate(errs[(variant, m)]) > 100)) for m in methods}
            row_["search stays at 0"] = float(np.mean(np.concatenate(zero[variant])))
            report[name] = row_
            print(f"{name:<26}" + "".join(f"{100 * row_[m]:>21.1f}%" for m in methods) + f"{100 * row_['search stays at 0']:>21.1f}%")
        (Path("runs") / run / "yaw_search_cross.json").write_text(json.dumps({"offsets": OFFSETS.tolist(), "window_s": WINDOW_S,
                                                                               "report": report}, indent=2))


def main() -> None:
    if sys.argv[1] == "--cross":
        return cross(sys.argv[2:])
    device = torch.device(default_device())
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        methods = ["no search", "right angle given", f"search ±{OFFSETS.max():g}°", f"search ±{HALO_BUDGET:g}° (Halo)"]
        errs = {(theta, m): [] for theta in THETAS for m in methods}
        picked_right = {theta: [] for theta in THETAS}
        for track, group in sorted(wide.by_track.items()):
            whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
            for k, live in enumerate(whole[:LAPS]):
                reference = whole[(k + 1) % len(whole)]
                grid = reference.reference_grid(AXIS)
                ref_raw = np.asarray(reference.frames()[grid.frame_idx])
                rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
                refs = [encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng, pose=(float(o), 0, 0)), device)
                        for o in OFFSETS]
                every = max(int(round(live.fps / STREAM_HZ)), 1)
                ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                used = np.unique(clip_idx)
                row = np.full(live.n_frames, -1)
                row[used] = np.arange(used.size)
                raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
                stream = {"grid": grid, "t": t[ticks], "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
                sign = 1.0 if k % 2 == 0 else -1.0
                for theta in THETAS:
                    turn = sign * theta
                    frames = camera.render_clip(raw, t[used], np.random.default_rng(k), pose=(turn, 0, 0))
                    beliefs, scores = per_offset(model, encode(model, frames, device), row[clip_idx], refs)
                    window = int(WINDOW_S * STREAM_HZ)
                    right = int(np.argmin(np.abs(OFFSETS - turn)))
                    streams = {
                        "no search": beliefs[int(np.argmin(np.abs(OFFSETS)))],
                        "right angle given": beliefs[right],
                        methods[2]: searched(beliefs, scores, np.ones(len(OFFSETS), bool), window)[0],
                        methods[3]: searched(beliefs, scores, np.abs(OFFSETS) <= HALO_BUDGET + 1e-6, window)[0],
                    }
                    pick = searched(beliefs, scores, np.ones(len(OFFSETS), bool), window)[1]
                    # right = the grid angle nearest the turn (14° is off the grid: 10.5° is right)
                    picked_right[theta].append((pick == right)[int(ACQUISITION_S * STREAM_HZ):])
                    for m, b in streams.items():
                        bins = _run_estimator({**stream, "belief": b.astype(np.float64)}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES),
                                              None, None, 0, reference_time=True)
                        errs[(theta, m)].append(_stream_errors(stream, bins)[1])
                print(f"  {run} {track[:26]:<26} live {live.lap_id[-5:]} reference {reference.lap_id[-5:]}", flush=True)
        report = {}
        print(f"\n{run}: share of ticks over 100 ms, two-mode reference-time tracker (same-session references)")
        print(f"{'held turn':<12}" + "".join(f"{m:>22}" for m in methods) + f"{'search picks right':>22}")
        for theta in THETAS:
            row_ = {m: float(np.mean(np.concatenate(errs[(theta, m)]) > 100)) for m in methods}
            row_["search picks the right angle"] = float(np.mean(np.concatenate(picked_right[theta])))
            report[f"{theta:g}"] = row_
            print(f"{theta:>5g}°      " + "".join(f"{100 * row_[m]:>21.1f}%" for m in methods)
                  + f"{100 * row_['search picks the right angle']:>21.1f}%")
        (Path("runs") / run / "yaw_search.json").write_text(json.dumps({"offsets": OFFSETS.tolist(), "window_s": WINDOW_S,
                                                                         "report": report}, indent=2))


if __name__ == "__main__":
    main()
