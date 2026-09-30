"""Track progress in reference-lap time instead of metres: a speed prior from the map.

The same particle filter, fed bin *times* instead of bin positions, so its "speed" is the
rate of progress through the reference lap (~1.0: braking is already in the reference).
Tuned on G1 (held-out laps of trained tracks), scored on the unseen tracks, against the
metre filter with and without the true speed. Stride 2, v3 protocol.
Usage: reftime_filter.py CHECKPOINT"""
import itertools, json, sys
from pathlib import Path
import numpy as np, torch
from train.eval import (SPEED_FED, _lap_stream, _stream_errors, _stream_pairs, load_model, reference_axis_of)
from train.estimator import EstimatorConfig, ProgressEstimator

ckpt = sys.argv[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path(ckpt), device)
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
pairs = _stream_pairs(Path("data/packed_ac_v3"), axis, 99)
streams = []
for gate, ref, live in pairs:
    st = _lap_stream(model, clip_len, ref, live, axis, 2, device, 8)
    st["belief"] = st["belief"].astype(np.float32)
    streams.append((gate, live.track, ref.car_model == live.car_model, st))
print(f"{len(streams)} streams cached", flush=True)


def run(st, cfg, time_mode, speed=None):
    g = st["grid"]
    est = ProgressEstimator(g.time_s if time_mode else g.pos_m, g.lap_time_s if time_mode else g.track_length_m, cfg, seed=0)
    dts = np.diff(st["t"], prepend=st["t"][0] - 1 / 15)
    out = np.empty(dts.size)
    for i, (b, dt) in enumerate(zip(st["belief"], dts)):
        e = est.step(b.astype(np.float64), dt, speed_obs=None if speed is None else float(speed[i]), speed_sigma=None if speed is None else 2.0)
        out[i] = est.bin_of(e.position_m)
    return torch.from_numpy(out)


def time_cfg(accel, power, pos_noise):
    return EstimatorConfig(accel_noise=accel, likelihood_power=power, position_noise=pos_noise,
                           speed_range=(0.6, 1.6), reinject_speed_sd=0.05, cluster_m=0.5)


def score(which, make_bins):
    rows = {}
    for gate, track, same, st in streams:
        if which(gate):
            m, ms = _stream_errors(st, make_bins(st))
            for key in (gate, f"{gate} {track}") + ((f"{gate} same car",) if gate == "G2" and same else ()):
                rows.setdefault(key, []).append(ms)
    return {k: float(np.mean(np.concatenate(v) > 100)) for k, v in rows.items()}


grid = list(itertools.product((0.003, 0.006, 0.01, 0.02), (0.1, 0.15, 0.2, 0.3), (0.01,)))
best, best_val = None, 9.9
for accel, power, pn in grid:
    r = score(lambda g: g == "G1", lambda st: run(st, time_cfg(accel, power, pn), True))["G1"]
    print(f"  G1 ref-time accel {accel} power {power} pos {pn}: {100 * r:.1f}% over 100 ms", flush=True)
    if r < best_val:
        best, best_val = (accel, power, pn), r
print("best on G1:", best, f"{100 * best_val:.1f}%", flush=True)

results = {
    "metre filter, no speed": score(lambda g: True, lambda st: run(st, EstimatorConfig(), False)),
    "metre filter, true speed": score(lambda g: True, lambda st: run(st, EstimatorConfig(**SPEED_FED), False, st["speed"])),
    f"reference-time filter {best}, no speed": score(lambda g: True, lambda st: run(st, time_cfg(*best), True)),
}
keys = sorted({k for r in results.values() for k in r})
print(f"\n{'% ticks over 100 ms':44s}" + "".join(f"{k[:26]:>28s}" for k in keys))
for name, r in results.items():
    print(f"{name:44s}" + "".join(f"{100 * r.get(k, float('nan')):27.1f}%" for k in keys))
Path(ckpt).parent.joinpath("reftime_filter_wide.json").write_text(json.dumps({"best": best, "results": results}, indent=2))
