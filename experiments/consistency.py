"""
Is the delta consistently wrong, or wrong by a different amount every moment?

A driver reads the delta's changes ("I gained a tenth through that corner"), and a
constant offset cancels out of every change. So the error is split into
  - the lap's offset: the median error over the lap;
  - the wobble around that offset;
  - the error in the delta's change over the last W seconds (W = 1, 2, 5, 10): what a
    driver reading "gained / lost since the corner entry" would get wrong, in corners
    (the slowest 40% of the lap) and on straights.
On the bot's laps as recorded (steady: the true delta hardly moves, so every change shown
is a phantom), and replayed at an imperfect pace (experiments/pace_warp.py: ±5% wander, one
mistake a minute), where the true delta really moves, set against how much it moved.

Delta error = shown delta - true delta, in ms (positive: shows you slower than you are).
Two-mode reference-time tracker, runtime stride 2, every full lap of the unseen tracks
against cross-session references (G2), and G1's held-out laps.

Usage: PYTHONPATH=. python experiments/consistency.py RUN [RUN ...]
       (writes runs/RUN/consistency.json and consistency_series.npz)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.pace_warp import warped_stream
from train.estimator import EstimatorConfig
from train.eval import (ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, _lap_stream, _run_estimator, _stream_pairs,
                        load_model, reference_axis_of)
from train.model import _interp_circular
from train.train import default_device

WINDOWS_S = (1, 2, 5, 10)
PACES = {"steady": None, "imperfect": (0.05, 1.0)}


def delta_error(stream: dict, bins: torch.Tensor) -> np.ndarray:
    """Shown minus true delta, ms, per tick after acquisition."""
    g = stream["grid"]
    times = torch.from_numpy(g.time_s.astype(np.float32))
    est = _interp_circular(times, bins.float(), g.lap_time_s).numpy().astype(np.float64)
    true = _interp_circular(times, stream["target"].float(), g.lap_time_s).numpy().astype(np.float64)
    e = (est - true + g.lap_time_s / 2) % g.lap_time_s - g.lap_time_s / 2
    return -1000.0 * e[int(ACQUISITION_S * STREAM_HZ):]


def true_delta(stream: dict) -> np.ndarray:
    """The true delta, ms, up to a constant: elapsed live time minus the reference time of the place."""
    g = stream["grid"]
    times = torch.from_numpy(g.time_s.astype(np.float32))
    ref_t = np.unwrap(_interp_circular(times, stream["target"].float(), g.lap_time_s).numpy().astype(np.float64),
                      period=g.lap_time_s)
    return 1000.0 * (stream["t"] - ref_t)[int(ACQUISITION_S * STREAM_HZ):]


def summarise(laps: list[dict]) -> dict:
    d = np.concatenate([lap["d"] for lap in laps])
    offsets = np.array([np.median(lap["d"]) for lap in laps])
    around = np.concatenate([lap["d"] - np.median(lap["d"]) for lap in laps])
    out = {
        "ticks": int(d.size), "laps": len(laps),
        "error": {"median_ms": float(np.median(np.abs(d))), "p90_ms": float(np.percentile(np.abs(d), 90)),
                  "over_100ms": float(np.mean(np.abs(d) > 100))},
        "offset": {"mean_abs_ms": float(np.mean(np.abs(offsets))), "p90_abs_ms": float(np.percentile(np.abs(offsets), 90)),
                   "share_of_variance": float(np.var(offsets) / max(np.var(d), 1e-9))},
        "around_offset": {"median_ms": float(np.median(np.abs(around))), "p90_ms": float(np.percentile(np.abs(around), 90)),
                          "over_100ms": float(np.mean(np.abs(around) > 100))},
        "jitter_ms": float(np.median(np.abs(np.concatenate([np.diff(lap["d"]) for lap in laps])))),
        "change": {},
    }
    for w in WINDOWS_S:
        k = int(w * STREAM_HZ)
        rows = {"all": [], "corners": [], "straights": [], "true": []}
        for lap in laps:
            if lap["d"].size <= k:
                continue
            change = lap["d"][k:] - lap["d"][:-k]
            slow = lap["speed"][k:] < np.percentile(lap["speed"], 40)
            rows["all"].append(change)
            rows["corners"].append(change[slow])
            rows["straights"].append(change[~slow])
            rows["true"].append(lap["true"][k:] - lap["true"][:-k])
        row = {}
        for part in ("all", "corners", "straights"):
            c = np.abs(np.concatenate(rows[part]))
            row[part] = {"median_ms": float(np.median(c)), "p90_ms": float(np.percentile(c, 90)),
                         "over_50ms": float(np.mean(c > 50)), "over_100ms": float(np.mean(c > 100))}
        tr = np.abs(np.concatenate(rows["true"]))
        row["true_change"] = {"median_ms": float(np.median(tr)), "p90_ms": float(np.percentile(tr, 90))}
        out["change"][f"{w}s"] = row
    return out


def main() -> None:
    device = torch.device(default_device())
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
        report, series = {}, {}
        for pace, warp in PACES.items():
            rng = np.random.default_rng(0)
            laps: dict[str, list[dict]] = {"G1": [], "G2": []}
            for n, (gate, reference, live) in enumerate(_stream_pairs(Path("data/packed_ac_v3"), axis, 99)):
                if warp is None:
                    stream = _lap_stream(model, clip_len, reference, live, axis, 2, device, 8)
                else:
                    stream = warped_stream(model, clip_len, reference, live, axis, device, rng, *warp)
                bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                skip = int(ACQUISITION_S * STREAM_HZ)
                lap = {"d": delta_error(stream, bins), "true": true_delta(stream), "speed": np.asarray(stream["speed"])[skip:]}
                laps[gate].append(lap)
                series[f"{pace}/{gate}/{n}/{live.track}"] = np.stack([lap["d"], lap["true"], lap["speed"]])
            report[pace] = {gate: summarise(v) for gate, v in laps.items() if v}
            for gate, r in report[pace].items():
                print(f"\n{run}, {pace} pace, {gate} ({r['laps']} laps): error median {r['error']['median_ms']:.0f} ms, "
                      f">100 ms {100 * r['error']['over_100ms']:.1f}%; lap offset mean |{r['offset']['mean_abs_ms']:.0f}| ms "
                      f"(p90 {r['offset']['p90_abs_ms']:.0f}, {100 * r['offset']['share_of_variance']:.0f}% of the variance); "
                      f"around it median {r['around_offset']['median_ms']:.0f} ms, >100 ms {100 * r['around_offset']['over_100ms']:.1f}%; "
                      f"tick-to-tick {r['jitter_ms']:.1f} ms")
                print(f"  {'change over':<12}{'true change':>14}{'error: median':>15}{'p90':>8}{'>50 ms':>8}{'>100 ms':>9}"
                      f"{'corners >50':>13}{'straights >50':>15}")
                for w, c in r["change"].items():
                    print(f"  {w:<12}{c['true_change']['median_ms']:>11.0f} ms{c['all']['median_ms']:>12.0f} ms{c['all']['p90_ms']:>5.0f} ms"
                          f"{100 * c['all']['over_50ms']:>7.1f}%{100 * c['all']['over_100ms']:>8.1f}%"
                          f"{100 * c['corners']['over_50ms']:>12.1f}%{100 * c['straights']['over_50ms']:>14.1f}%", flush=True)
        (Path("runs") / run / "consistency.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(Path("runs") / run / "consistency_series.npz", **series)


if __name__ == "__main__":
    main()
