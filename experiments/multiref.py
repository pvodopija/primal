"""Voting across reference laps: does matching a live lap against several references,
combined on the first reference's positions, beat matching it against one?

Uses the same live laps, scoring and filter as `train.eval stream`. References are
aligned to the first by their labels (an upper bound for a learned alignment).
Usage: multiref.py CHECKPOINT [MAX_REFS] [MAX_PAIRS]"""
import json, sys
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
max_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
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


def fuse(beliefs, how):
    b = np.stack(beliefs)
    if how == "mean":
        out = b.mean(0)
    else:  # product of experts, tempered by the count so sharpness stays comparable
        out = np.exp(np.log(b + 1e-12).mean(0))
    return out / out.sum(1, keepdims=True)


variants = [("1 reference", 1, "mean")] + [(f"{k} references, {h}", k, h) for k in sorted({3, max_refs}) for h in ("mean", "product") if k <= max_refs]
scores = {g: {v[0]: {k: [[], []] for k in ("single", "filter", "filter + speed")} for v in variants} for g in ("G1", "G2")}
for gate, live, refs in cases():
    streams = [_lap_stream(model, clip_len, r, live, axis, 4, device, 8) for r in refs]
    primary = streams[0]
    aligned = [primary["belief"]] + [onto(s["belief"], s["grid"], primary["grid"]) for s in streams[1:]]
    for name, k, how in variants:
        if k > len(aligned):
            continue
        stream = dict(primary)
        stream["belief"] = fuse(aligned[:k], how)
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
print(f"\n{Path(ckpt).parent.name}: whole laps at 15 Hz, scored on the first reference")
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
Path(ckpt).parent.joinpath(f"multiref_{max_refs}.json").write_text(json.dumps(report, indent=2))
