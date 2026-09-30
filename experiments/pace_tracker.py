"""
Make the reference-time tracker hold up when the driver does not repeat the reference's
rhythm. Streams come from `pace_warp.py`: every lap of the v3 protocol unwarped, as an
imperfect driver (pace wandering ±5% with ~1 s correlation, one mistake a minute), and
as a rough one (±10%). Tracker variants are tuned on the known tracks' imperfect laps (G1) and scored
on the unseen tracks (G2), imperfect and unwarped, so a gain for humans cannot hide a
loss for steady driving.

- two modes: each particle follows the reference's rhythm or is off-script (freer pace,
  down to a near stop), switching at set rates;
- braking zones: pace noise scaled up where the reference's own speed changes fast,
  from the reference grid's speed profile;
- plain retunes of the single-mode tracker (pace noise, weight of vision).

Usage: PYTHONPATH=. python experiments/pace_tracker.py CHECKPOINT
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.pace_warp import warped_stream
from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME, _run_estimator, _stream_errors, _stream_pairs, load_model, reference_axis_of


def braking_profile(grid, k: float) -> np.ndarray:
    """1 + k x |acceleration of the reference| in g, smoothed over ~0.5 s of reference time."""
    v = np.asarray(grid.speed_mps, dtype=np.float64)
    dt = grid.lap_time_s / grid.n_bins
    accel = np.abs(np.roll(v, -1) - np.roll(v, 1)) / (2 * dt) / 9.81
    width = max(1, int(round(0.5 / dt)))
    kernel = np.ones(width) / width
    smooth = np.convolve(np.r_[accel[-width:], accel, accel[:width]], kernel, mode="same")[width:-width]
    return 1.0 + k * smooth


def main() -> None:
    ckpt = sys.argv[1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(Path(ckpt), device)
    clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
    pairs = _stream_pairs(Path("data/packed_ac_v3"), axis, 99)
    streams = {}
    for level, (wander, mistakes) in {"imperfect": (0.05, 1.0), "steady": (0.0, 0.0), "rough": (0.10, 0.0)}.items():
        rng = np.random.default_rng(0)
        streams[level] = []
        for gate, ref, live in pairs:
            st = warped_stream(model, clip_len, ref, live, axis, device, rng, wander, mistakes)
            st["belief"] = st["belief"].astype(np.float32)
            streams[level].append((gate, ref.car_model == live.car_model, st))
        print(f"{level}: {len(streams[level])} streams", flush=True)

    def score(level, gate, cfg, k=0.0):
        rows = []
        for g, _, st in streams[level]:
            if g == gate:
                profile = braking_profile(st["grid"], k) if k > 0 else None
                _, ms = _stream_errors(st, _run_estimator(st, cfg, None, None, 0, reference_time=True, noise_profile=profile))
                rows.append(ms)
        return float(np.mean(np.concatenate(rows) > 100))

    base = dict(REFERENCE_TIME)
    variants = {"reference time (today)": (base, 0.0)}
    for accel, power in itertools.product((0.03, 0.06, 0.12), (0.15, 0.3, 0.5)):
        variants[f"retuned, pace noise {accel}, vision {power}"] = (dict(base, accel_noise=accel, likelihood_power=power), 0.0)
    for free, to_free in itertools.product((0.1, 0.3), (0.1, 0.5)):
        variants[f"two modes, free {free}, to free {to_free}/s"] = (dict(base, accel_noise_free=free, to_free_per_s=to_free,
                                                                        to_follow_per_s=1.0, speed_range=(0.05, 1.6)), 0.0)
    for k in (2.0, 5.0):
        variants[f"braking zones, k {k}"] = (base, k)
    results = {}
    for name, (cfg, k) in variants.items():
        results[name] = {"G1 imperfect": score("imperfect", "G1", EstimatorConfig(**cfg), k)}
        print(f"  {name:38s} G1 imperfect {100 * results[name]['G1 imperfect']:5.1f}%", flush=True)
    retuned = min((n for n in results if n.startswith("retuned")), key=lambda n: results[n]["G1 imperfect"])
    two = min((n for n in results if n.startswith("two")), key=lambda n: results[n]["G1 imperfect"])
    brake = min((n for n in results if n.startswith("braking")), key=lambda n: results[n]["G1 imperfect"])
    cfg2, _ = variants[two]
    _, kb = variants[brake]
    variants["both"] = (cfg2, kb)
    results["both"] = {"G1 imperfect": score("imperfect", "G1", EstimatorConfig(**cfg2), kb)}
    finalists = ["reference time (today)", retuned, two, brake, "both"]
    print(f"\n{'% ticks over 100 ms':44s}{'G1 imperfect':>14s}{'G2 imperfect':>14s}{'G2 steady':>12s}{'G2 rough':>11s}{'G1 steady':>12s}")
    for name in finalists:
        cfg, k = variants[name]
        results[name].update({"G2 imperfect": score("imperfect", "G2", EstimatorConfig(**cfg), k),
                              "G2 steady": score("steady", "G2", EstimatorConfig(**cfg), k),
                              "G2 rough": score("rough", "G2", EstimatorConfig(**cfg), k),
                              "G1 steady": score("steady", "G1", EstimatorConfig(**cfg), k)})
        r = results[name]
        print(f"{name:44s}{100*r['G1 imperfect']:13.1f}%{100*r['G2 imperfect']:13.1f}%{100*r['G2 steady']:11.1f}%{100*r['G2 rough']:10.1f}%{100*r['G1 steady']:11.1f}%", flush=True)
    Path(ckpt).parent.joinpath("pace_tracker.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
