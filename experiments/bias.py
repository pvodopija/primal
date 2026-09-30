"""Is the live error mostly a steady lag? Signed errors on the stream-eval pairs.

Signed = estimate minus truth, so negative means the estimate trails the kart. Reports the
median signed error per gate and per lap, and what removing a constant, calibrated on the
known tracks (G1) and applied unchanged to the unseen one (G2), would do to ticks over 100 ms.
Usage: bias.py CHECKPOINT"""
import json, sys
from pathlib import Path
import numpy as np, torch
from train.eval import (ACQUISITION_S, SPEED_FED, STREAM_HZ, _lap_stream, _run_estimator,
                        _stream_pairs, load_model, reference_axis_of)
from train.estimator import EstimatorConfig
from train.model import _interp_circular

ckpt = sys.argv[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path(ckpt), device)
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
skip = int(ACQUISITION_S * STREAM_HZ)


def signed(stream, bins):
    g = stream["grid"]
    pos, tim = torch.from_numpy(g.pos_m.astype(np.float32)), torch.from_numpy(g.time_s.astype(np.float32))
    out = []
    for table, period in ((pos, g.track_length_m), (tim, g.lap_time_s)):
        d = _interp_circular(table, bins.float(), period) - _interp_circular(table, stream["target"], period)
        out.append(((d + period / 2) % period - period / 2).numpy()[skip:])
    return out[0], out[1] * 1000.0


kinds = ("single", "filter", "filter + speed")
rows = {g: {k: [[], []] for k in kinds} for g in ("G1", "G2")}
per_lap = {g: [] for g in ("G1", "G2")}
for gate, ref, live in _stream_pairs(Path("data/packed_ac_v2"), axis, 3):
    stream = _lap_stream(model, clip_len, ref, live, axis, 4, device, 8)
    runs = {"single": stream["single"],
            "filter": _run_estimator(stream, EstimatorConfig(), None, None, 0),
            "filter + speed": _run_estimator(stream, EstimatorConfig(**SPEED_FED), stream["speed"], 2.0, 0)}
    for k, bins in runs.items():
        m, ms = signed(stream, bins)
        rows[gate][k][0].append(m); rows[gate][k][1].append(ms)
    m, ms = signed(stream, runs["filter"])
    per_lap[gate].append((live.track[:22], ref.session_id[-7:], live.session_id[-7:], float(np.median(m)), float(np.median(ms))))

report = {}
for gate in ("G1", "G2"):
    print(f"\n{gate} per lap, filter: median signed error")
    for t, r, l, m, ms in per_lap[gate]:
        print(f"  {t:22s} ref {r} live {l}  {m:+6.2f} m  {ms:+5.0f} ms")
cal = {k: float(np.median(np.concatenate(rows["G1"][k][1]))) for k in kinds}
print(f"\n{Path(ckpt).parent.name}: signed = estimate - truth; correction = G1 median signed ms")
for gate in ("G1", "G2"):
    for k in kinds:
        m, ms = (np.concatenate(v) for v in rows[gate][k])
        row = dict(signed_m=float(np.median(m)), signed_ms=float(np.median(ms)),
                   over_100ms=float(np.mean(np.abs(ms) > 100)), over_100ms_corrected=float(np.mean(np.abs(ms - cal[k]) > 100)),
                   median_abs_ms=float(np.median(np.abs(ms))), median_abs_ms_corrected=float(np.median(np.abs(ms - cal[k]))))
        report[f"{gate} {k}"] = row
        print(f"  {gate} {k:15s} signed {row['signed_m']:+5.2f} m {row['signed_ms']:+4.0f} ms | over 100 ms {100*row['over_100ms']:4.1f}% -> {100*row['over_100ms_corrected']:4.1f}% after {cal[k]:+.0f} ms | median |ms| {row['median_abs_ms']:.0f} -> {row['median_abs_ms_corrected']:.0f}")
Path(ckpt).parent.joinpath("bias.json").write_text(json.dumps({"correction_ms": cal, "report": report, "per_lap": per_lap}, indent=2))
