"""Voting across reference laps, with the references aligned by the model instead of labels.

multiref.py aligned every extra reference to the first by their labels, an upper bound.
The phone has no labels: here each extra reference lap is run as a live lap against the
first, a monotone hindsight path (Viterbi) places its frames on the first one's bins, and
its beliefs are resampled through that placement. "both ways" places each frame at the
midpoint of the forward and backward paths, cancelling the matcher's lag.
Live laps and alignment both at runtime stride STRIDE (default 2).
Usage: multiref2.py CHECKPOINT [MAX_REFS] [STRIDE]"""
import dataclasses, json, sys
from pathlib import Path
import numpy as np, torch
from train.dataset import LapIndex
from train.eval import (SPEED_FED, _lap_stream, _run_estimator, _stream_errors,
                        load_model, reference_axis_of)
from train.estimator import EstimatorConfig
from train.model import soft_argmax_circular
from train.train import holdout_live_laps

DATA = Path("data/packed_ac_v2")
ckpt = sys.argv[1]
max_refs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 2
max_pairs = 10**9
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path(ckpt), device)
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)


def cases():
    out = []
    for split, gate in (("train", "G1"), ("holdout", "G2")):
        index = LapIndex.load(DATA, split=split)
        reserved = holdout_live_laps(index, 1) if gate == "G1" else None
        for _, group in sorted(index.by_track.items()):
            lives = [l for l in group if l.s_span > 0.98 and (reserved is None or l.lap_id in reserved)]
            if gate == "G2":
                lives = lives[:3]
            for live in lives:
                by = {}
                for r in group:
                    if r.session_id == live.session_id or (reserved is not None and r.lap_id in reserved):
                        continue
                    if r.usable_as_reference(0.98, axis):
                        by.setdefault(r.session_id, []).append(r)
                refs = []  # interleave sessions, so extra references differ in conditions
                for i in range(max((len(v) for v in by.values()), default=0)):
                    refs += [by[sid][i] for sid in sorted(by) if i < len(by[sid])]
                if len(refs) >= 2:
                    out.append((gate, live, refs[:max_refs]))
    return out[:max_pairs]


def onto(belief, grid_from, grid_to):
    """Resample a belief over grid_from's bins onto grid_to's bins by track position."""
    L, n = grid_to.track_length_m, belief.shape[1]
    pos = np.maximum.accumulate(grid_from.pos_m.astype(np.float64))
    xp = np.r_[pos, pos[0] + L]
    idx = np.interp((grid_to.pos_m - pos[0]) % L + pos[0], xp, np.r_[np.arange(n), n].astype(np.float64))
    lo = np.floor(idx).astype(int)
    w = idx - lo
    out = belief[:, lo % n] * (1 - w) + belief[:, (lo + 1) % n] * w
    return out / out.sum(1, keepdims=True)


def viterbi(belief, max_step=4):
    n = belief.shape[1]
    logp = np.log(belief + 1e-12)
    score, back = logp[0].copy(), np.zeros(belief.shape, np.int8)
    for i in range(1, len(belief)):
        options = np.stack([np.roll(score, k) for k in range(max_step + 1)])
        back[i] = options.argmax(0)
        score = options.max(0) + logp[i]
    path = np.empty(len(belief), int)
    path[-1] = int(score.argmax())
    for i in range(len(belief) - 1, 0, -1):
        path[i - 1] = (path[i] - back[i, path[i]]) % n
    return path


@torch.no_grad()
def placed_grid(primary_lap, primary_grid, ref_lap, both):
    """ref_lap's grid with positions read off the primary by hindsight, not from labels."""
    st = _lap_stream(model, clip_len, primary_lap, ref_lap, axis, STRIDE, device, 8, backward=both)
    n = primary_grid.n_bins
    path = viterbi(st["belief"]).astype(np.float64)
    ticks = np.arange((clip_len - 1) * STRIDE, ref_lap.n_frames, max(int(round(ref_lap.fps / 15.0)), 1))
    if both:  # shift by half the median forward-backward gap, one constant per lap
        f, b = st["single"].numpy(), st["single_backward"].numpy()
        ok = st["backward_valid"]
        gap = ((f - b) + n / 2) % n - n / 2
        path = path - 0.5 * np.median(gap[ok])
    L = primary_grid.track_length_m
    pos_cycle = np.r_[primary_grid.pos_m, primary_grid.pos_m[0] + L]
    unwrapped = np.unwrap(path * 2 * np.pi / n) * n / (2 * np.pi)
    g = ref_lap.reference_grid(axis)
    at_bins = np.interp(g.frame_idx, ticks, unwrapped) % n
    placed = np.interp(at_bins, np.arange(n + 1), pos_cycle)
    truth = ref_lap.s()[g.frame_idx] * L
    err = np.abs((placed - truth + L / 2) % L - L / 2)
    return dataclasses.replace(g, pos_m=np.unwrap(placed * 2 * np.pi / L) * L / (2 * np.pi)), err


def fuse(beliefs, how):
    b = np.stack(beliefs)
    if how == "mean":
        out = b.mean(0)
    else:  # product of experts, tempered by the count so sharpness stays comparable
        out = np.exp(np.log(b + 1e-12).mean(0))
    return out / out.sum(1, keepdims=True)


variants = [("1 reference", 1, "labels"), (f"{max_refs} references, labels", max_refs, "labels"),
            (f"{max_refs} references, model", max_refs, "model"), (f"{max_refs} references, model both ways", max_refs, "both")]
place_err = {"model": [], "both": []}
scores = {g: {v[0]: {k: [[], []] for k in ("single", "filter", "filter + speed")} for v in variants} for g in ("G1", "G2")}
for gate, live, refs in cases():
    streams = [_lap_stream(model, clip_len, r, live, axis, STRIDE, device, 8) for r in refs]
    primary = streams[0]
    aligned = {"labels": [primary["belief"]] + [onto(s["belief"], s["grid"], primary["grid"]) for s in streams[1:]]}
    for mode in ("model", "both"):
        aligned[mode] = [primary["belief"]]
        for r, s in zip(refs[1:], streams[1:]):
            g, err = placed_grid(refs[0], primary["grid"], r, mode == "both")
            place_err[mode].append(err)
            aligned[mode].append(onto(s["belief"], g, primary["grid"]))
    for name, k, how in variants:
        if k > len(aligned[how]):
            continue
        stream = dict(primary)
        stream["belief"] = fuse(aligned[how][:k], "mean")
        runs = {
            "single": soft_argmax_circular(torch.log(torch.from_numpy(stream["belief"]) + 1e-12), window=8),
            "filter": _run_estimator(stream, EstimatorConfig(), None, None, 0),
            "filter + speed": _run_estimator(stream, EstimatorConfig(**SPEED_FED), stream["speed"], 2.0, 0),
        }
        for kind, bins in runs.items():
            m, ms = _stream_errors(stream, bins)
            scores[gate][name][kind][0].append(m)
            scores[gate][name][kind][1].append(ms)
    print(f"  {gate} {live.lap_id[-40:]}  {len(refs)} references", flush=True)

report = {}
print(f"\n{Path(ckpt).parent.name}: whole laps at 15 Hz, stride {STRIDE}, scored on the first reference")
for mode, errs in place_err.items():
    e = np.concatenate(errs)
    print(f"  reference placement by {mode}: median {np.median(e):.2f} m, p90 {np.percentile(e, 90):.2f} m, beyond 10 m {100 * np.mean(e > 10):.1f}%")
for gate in ("G1", "G2"):
    for name, *_ in variants:
        line = f"  {gate} {name:24s}"
        for kind in ("single", "filter", "filter + speed"):
            if not scores[gate][name][kind][0]:
                continue
            m, ms = (np.concatenate(v) for v in scores[gate][name][kind])
            row = dict(median_m=float(np.median(m)), over_100ms=float(np.mean(ms > 100)),
                       over_10m=float(np.mean(m > 10)), worst_m=float(m.max()))
            report[f"{gate} {name} {kind}"] = row
            line += f" | {kind}: {row['median_m']:.2f} m {100*row['over_100ms']:4.1f}% {100*row['over_10m']:4.1f}%"
        print(line, flush=True)
Path(ckpt).parent.joinpath(f"multiref2_{max_refs}_stride{STRIDE}.json").write_text(json.dumps(report, indent=2))
