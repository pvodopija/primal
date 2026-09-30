"""
An imperfect driver, made from the bot's laps: how do the trackers cope when the live lap
does not repeat the reference's rhythm?

Every live lap is replayed on a warped clock. Play time t maps to recording time W(t),
whose rate dW/dt wanders smoothly around 1 (about 1 s correlation), like braking
points and corner speeds that differ from lap to lap, and optionally slows to a fraction
of the pace for a few seconds now and then, like a mistake. Labels stay exact: each
shown frame keeps its own position. The clip frames are sampled at the runtime spacing
in play time (stride 2 at 60 fps, 1/30 s), so the model sees the varying pace as it
would live.

Scored like `train.eval stream` (15 Hz, 2 s acquisition left out, every full lap of the
unseen tracks, cross-session references): the metre filter, the reference-time filter,
and the metre filter with the true (warped) speed.

Usage: PYTHONPATH=. python experiments/pace_warp.py CHECKPOINT [--wander 0.05] [--mistakes 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train.dataset import _to_chw
from train.estimator import EstimatorConfig
from train.eval import (ACQUISITION_S, REFERENCE_TIME, SPEED_FED, STREAM_HZ, _run_estimator,
                        _stream_errors, _stream_pairs, load_model, reference_axis_of)
from train.model import soft_argmax_circular


def warp_clock(duration_s: float, wander: float, mistakes: float, rng: np.random.Generator,
               dt: float = 1 / 120) -> tuple[np.ndarray, np.ndarray]:
    """Play times, and the recording time shown at each; stops at the end of the recording."""
    n = int(duration_s / dt * 1.6) + 10
    a = np.exp(-dt / 1.0)  # ~1 s correlation
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = a * noise[i - 1] + np.sqrt(1 - a * a) * rng.normal()
    rate = 1.0 + wander * noise
    if mistakes > 0:  # a slowdown to 35-60% of the pace for 1.5-3 s, `mistakes` per minute on average
        t = 0.0
        while True:
            t += rng.exponential(60.0 / mistakes)
            start = int(t / dt)
            if start >= n:
                break
            length = int(rng.uniform(1.5, 3.0) / dt)
            depth = rng.uniform(0.35, 0.6)
            ramp = np.sin(np.linspace(0, np.pi, length)) ** 2
            rate[start:start + length] *= 1 - (1 - depth) * ramp[: max(0, min(length, n - start))]
    rate = np.clip(rate, 0.05, 2.0)
    clock = np.concatenate([[0.0], np.cumsum(rate[:-1] * dt)])
    play = np.arange(n) * dt
    keep = clock <= duration_s
    return play[keep], clock[keep]


@torch.no_grad()
def warped_stream(model, clip_len: int, reference, live, axis: str, device, rng, wander: float,
                  mistakes: float) -> dict:
    grid = reference.reference_grid(axis)
    ref = model.encode_reference(torch.from_numpy(_to_chw(np.asarray(reference.frames()[grid.frame_idx]))).to(device),
                                 use_checkpoint=False)
    frames = live.frames()
    desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames[i:i + 256]))).to(device),
                                             use_checkpoint=False) for i in range(0, live.n_frames, 256)])
    t_rec = live.t().astype(np.float64)
    play, clock = warp_clock(float(t_rec[-1]), wander, mistakes, rng)
    frame_at = lambda seconds: np.clip(np.searchsorted(t_rec, seconds), 0, live.n_frames - 1)
    spacing = 1 / 30  # stride 2 at 60 fps, in play time
    ticks = np.arange((clip_len - 1) * spacing, play[-1], 1 / STREAM_HZ)
    clip_play = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * spacing
    clip_idx = frame_at(np.interp(clip_play, play, clock))
    scale = model.logit_scale.exp()
    beliefs, single = [], []
    for chunk in np.array_split(np.arange(ticks.size), max(1, ticks.size // 64)):
        logits = model.head(torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(clip_idx[chunk]).to(device)], ref) * scale)
        single.append(soft_argmax_circular(logits, window=8).float().cpu())
        beliefs.append(logits.softmax(-1).cpu().double())
    last = clip_idx[:, -1]
    s = live.s().astype(np.float64)
    # true speed in play time: the recording's speed times the clock's rate at this tick
    rate = np.gradient(np.interp(ticks, play, clock), ticks)
    ds = ((s[np.clip(last + 3, 0, s.size - 1)] - s[np.clip(last - 3, 0, s.size - 1)] + 0.5) % 1.0 - 0.5) * live.track_length_m
    dts = t_rec[np.clip(last + 3, 0, s.size - 1)] - t_rec[np.clip(last - 3, 0, s.size - 1)]
    return {"grid": grid, "belief": torch.cat(beliefs).numpy(), "single": torch.cat(single),
            "target": torch.from_numpy(grid.target(s[last]).astype(np.float32)), "t": ticks,
            "speed": ds / np.maximum(dts, 1e-3) * rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="data/packed_ac_v3")
    parser.add_argument("--wander", type=float, default=0.05)
    parser.add_argument("--mistakes", type=float, default=0.0, help="per minute")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(Path(args.checkpoint), device)
    clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
    rng = np.random.default_rng(args.seed)
    kinds = {
        "filter": lambda st: _run_estimator(st, EstimatorConfig(), None, None, 0),
        "filter, reference time": lambda st: _run_estimator(st, EstimatorConfig(**REFERENCE_TIME), None, None, 0,
                                                            reference_time=True),
        "filter + speed": lambda st: _run_estimator(st, EstimatorConfig(**SPEED_FED), st["speed"], 2.0, 0),
    }
    scores: dict[str, list] = {}
    for gate, reference, live in _stream_pairs(Path(args.data), axis, 99):
        st = warped_stream(model, clip_len, reference, live, axis, device, rng, args.wander, args.mistakes)
        groups = [gate] + ([f"{gate} same car"] if gate == "G2" and reference.car_model == live.car_model else [])
        for kind, run in kinds.items():
            _, ms = _stream_errors(st, run(st))
            for g in groups:
                scores.setdefault(f"{g} {kind}", []).append(ms)
    report = {k: float(np.mean(np.concatenate(v) > 100)) for k, v in scores.items()}
    print(f"{Path(args.checkpoint).parent.name}, pace wander ±{args.wander:.0%}, {args.mistakes} mistakes/min:")
    for k, v in sorted(report.items()):
        print(f"  {k:40s} {100 * v:5.1f}% over 100 ms")
    out = Path(args.checkpoint).parent / f"pace_warp_w{args.wander}_m{args.mistakes}.json"
    out.write_text(json.dumps({"args": vars(args), "report": report}, indent=2))


if __name__ == "__main__":
    main()
