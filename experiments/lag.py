"""Where does the lag come from, and can the phone remove it without labels?

On the stream-eval pairs, per tick: the forward readout (clip ends at the frame, as live),
the backward readout of the same frame (clip starts at it and runs back from 0.73 s later;
training's reversed clips cover this), the forward belief's confidence, and the filters.
Corrections, all calibrated on the known tracks (G1) and applied unchanged to Silverstone:
  constant     - subtract G1's median lag
  confidence   - lag as a linear function of the last 2 s of belief confidence
  self (fwd/bwd) - lag = k * running median of (forward - backward) readouts of frames
                 0.73 s old, which the phone has without labels; k fitted on G1
  oracle       - each lap's own median lag (the prize, not achievable)
Usage: lag.py CHECKPOINT [G2_LIVE_LAPS] [STRIDE]"""
import json, sys
from pathlib import Path
import numpy as np, torch
from train.dataset import _to_chw
from train.eval import (ACQUISITION_S, SPEED_FED, STREAM_HZ, _lap_stream, _run_estimator,
                        _stream_pairs, load_model, reference_axis_of)
from train.estimator import EstimatorConfig
from train.model import _interp_circular, soft_argmax_circular

ckpt = sys.argv[1]
g2_lives = int(sys.argv[2]) if len(sys.argv) > 2 else 3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path(ckpt), device)
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
skip, SPAN = int(ACQUISITION_S * STREAM_HZ), (clip_len - 1) * STRIDE


def wrap(d, p):
    return (d + p / 2) % p - p / 2


@torch.no_grad()
def backward_single(ref, live, ticks, grid):
    ref_desc = model.encode_reference(torch.from_numpy(_to_chw(np.asarray(ref.frames()[grid.frame_idx]))).to(device), use_checkpoint=False)
    frames = live.frames()
    desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames[i:i + 256]))).to(device), use_checkpoint=False)
                      for i in range(0, live.n_frames, 256)])
    idx = np.clip(ticks[:, None] + np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE, 0, live.n_frames - 1)
    out = []
    for chunk in np.array_split(np.arange(ticks.size), max(1, ticks.size // 64)):
        corr = torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(idx[chunk]).to(device)], ref_desc) * model.logit_scale.exp()
        out.append(soft_argmax_circular(model.head(corr), window=8).float().cpu())
    return torch.cat(out)


def ref_ms(grid, bins):
    return _interp_circular(torch.from_numpy(grid.time_s.astype(np.float32)), torch.as_tensor(bins).float(), grid.lap_time_s).numpy() * 1000


laps = {"G1": [], "G2": []}
for gate, ref, live in _stream_pairs(Path("data/packed_ac_v2"), axis, g2_lives):
    st = _lap_stream(model, clip_len, ref, live, axis, STRIDE, device, 8)
    g, ticks = st["grid"], np.arange(SPAN, live.n_frames, max(int(round(live.fps / STREAM_HZ)), 1))
    P = g.lap_time_s * 1000
    truth = ref_ms(g, st["target"])
    fwd = ref_ms(g, st["single"])
    bwd = ref_ms(g, backward_single(ref, live, ticks, g))
    bwd_ok = ticks + SPAN < live.n_frames
    b = st["belief"]; peak = b.argmax(1)
    near = (peak[:, None] + np.arange(-3, 4)[None, :]) % b.shape[1]
    conf = np.take_along_axis(b, near, 1).sum(1)
    filt = ref_ms(g, _run_estimator(st, EstimatorConfig(), None, None, 0))
    fspd = ref_ms(g, _run_estimator(st, EstimatorConfig(**SPEED_FED), st["speed"], 2.0, 0))
    # what the phone knows at tick i: forward-minus-backward of ticks whose backward clip has already been seen
    lagged = int(np.ceil(SPAN / max(int(round(live.fps / STREAM_HZ)), 1)))
    gap = np.where(bwd_ok, wrap(fwd - bwd, P), np.nan)
    running_gap, running_conf = np.full(ticks.size, np.nan), np.zeros(ticks.size)
    for i in range(ticks.size):
        past = gap[max(0, i - lagged - 150):max(0, i - lagged)]      # up to 10 s of settled frames
        past = past[np.isfinite(past)]
        running_gap[i] = np.median(past) if past.size >= 15 else np.nan
        running_conf[i] = conf[max(0, i - 30):i + 1].mean()
    laps[gate].append(dict(track=live.track[:22], P=P, fwd=wrap(fwd - truth, P), bwd=np.where(bwd_ok, wrap(bwd - truth, P), np.nan),
                           conf=conf, rconf=running_conf, rgap=running_gap,
                           filter=wrap(filt - truth, P), speed=wrap(fspd - truth, P)))
    print(f"  {gate} {live.track[:24]:24s} fwd {np.median(laps[gate][-1]['fwd'][skip:]):+5.0f} ms  bwd {np.nanmedian(laps[gate][-1]['bwd'][skip:]):+5.0f} ms  "
          f"filter {np.median(laps[gate][-1]['filter'][skip:]):+5.0f}  +speed {np.median(laps[gate][-1]['speed'][skip:]):+5.0f}", flush=True)


def cat(gate, key):
    return np.concatenate([l[key][skip:] for l in laps[gate]])


report = {}
for gate in ("G1", "G2"):
    c, f = cat(gate, "conf"), cat(gate, "fwd")
    q = np.quantile(c, [0, .25, .5, .75, 1])
    print(f"\n{gate} forward lag by confidence quartile: " + "  ".join(
        f"{q[i]:.2f}-{q[i+1]:.2f}: {np.median(f[(c >= q[i]) & (c <= q[i+1])]):+.0f} ms" for i in range(4)))
# fits on G1
k_fit = {}
for kind in ("filter", "speed"):
    e, rg, rc = cat("G1", kind), cat("G1", "rgap"), cat("G1", "rconf")
    ok = np.isfinite(rg)
    k_fit[kind] = dict(const=float(np.median(e)),
                       conf=np.polyfit(rc, e, 1).tolist(),
                       self=float(np.sum(rg[ok] * e[ok]) / np.sum(rg[ok] ** 2)))
print("\nfits on G1:", json.dumps(k_fit))
for gate in ("G1", "G2"):
    for kind, label in (("filter", "filter"), ("speed", "filter + speed")):
        e, rg, rc = cat(gate, kind), cat(gate, "rgap"), cat(gate, "rconf")
        fit = k_fit[kind]
        oracle = np.concatenate([l[kind][skip:] - np.median(l[kind][skip:]) for l in laps[gate]])
        variants = {"none": e, "constant": e - fit["const"], "confidence": e - np.polyval(fit["conf"], rc),
                    "self (fwd/bwd)": np.where(np.isfinite(rg), e - fit["self"] * rg, e - fit["const"]),
                    "self, k=0.5": np.where(np.isfinite(rg), e - 0.5 * rg, e),
                    "oracle per lap": oracle}
        for name, v in variants.items():
            report[f"{gate} {label} {name}"] = dict(over_100ms=float(np.mean(np.abs(v) > 100)), median_abs_ms=float(np.median(np.abs(v))), signed_ms=float(np.median(v)))
        print(f"  {gate} {label:15s} " + " | ".join(f"{n}: {100*np.mean(np.abs(v) > 100):4.1f}% ({np.median(v):+.0f})" for n, v in variants.items()))
Path(ckpt).parent.joinpath(f"lag_{g2_lives}_stride{STRIDE}.json").write_text(json.dumps({"fits": k_fit, "report": report}, indent=2))
