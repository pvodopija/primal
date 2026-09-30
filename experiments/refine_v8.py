"""A reference that improves lap by lap, placed against the raw anchor only (v8).

v8: place each finished lap with the lap aligner, a fine-tune whose target is the clip's
middle frame (context on both sides): clips centred on each tick, monotone path.
Usage: refine_v8.py CHECKPOINT TRACK MAX_LAPS ALIGNER_CHECKPOINT. Older notes follow.

v7: place each finished lap by its frames' own fingerprints, not the clip head. Every
frame is compared with every anchor bin (the per-frame rows training supervises), and a
monotone path through the whole lap picks the placement. No clip, so no clip lag.
Older notes follow.

v4 showed placement noise is harmless (2 m of wandering error costs about a point) and a
steady offset is not (-1.5 m makes refinement worse than none). So here the forward
hindsight path is shifted by one constant per lap: half the lap's median gap between the
forward and backward single readings, the same measurement as the live lag correction.
"lag-free, oracle" removes the path's true median offset instead (the upper bound).

Simulates one track's laps in recording order. The first usable lap is the raw anchor:
it defines positions and timing and is never changed. Every later lap is first localized
against the current map (as the driver would experience it), then used to update the map:

  anchor only   - today's single reference
  blend         - anchor + one refined view, R <- normalise((1-a) R + a V_lap)
  multi-view    - anchor + the last K laps' views
Each view is read by the head on its own and the beliefs are averaged, the way the head
was trained (one lap per reference).

Updates assign a lap's frames to anchor bins from the labels (upper bound) or by hindsight:
a monotone Viterbi path through the whole lap's beliefs against the RAW ANCHOR ALONE, never
against the refined views, so one lap's placement error cannot feed the next (v1 placed
laps against the refined map and drifted like an integrated gyro). "both ways" also reads
each frame from reversed clips (the frame as the last of the following 12, which training's
reversed clips cover) and multiplies the two beliefs. "midpoint" runs the monotone path
through forward and backward beliefs separately and takes the point halfway between: the
forward reading trails the kart and the backward one leads it by about as much (v2 found
the placement lag, not noise, was what kept refinement from helping).
Usage: refine_map.py CHECKPOINT TRACK [MAX_LAPS]"""
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from train.dataset import LapIndex, _to_chw
from train.estimator import EstimatorConfig, ProgressEstimator
from train.eval import load_model, reference_axis_of
from train.model import compute_metrics, soft_argmax_circular

DATA = Path("data/packed_ac_v2")
ckpt, track = sys.argv[1], sys.argv[2]
max_laps = int(sys.argv[3]) if len(sys.argv) > 3 else 12
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path(ckpt), device)
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
import os
STRIDE, HZ, SKIP = int(os.environ.get("STRIDE", 4)), 15.0, 30
scale = model.logit_scale.exp()

laps = [l for l in LapIndex.load(DATA).by_track[track] if l.s_span > 0.98]
laps.sort(key=lambda l: l.lap_id)          # session timestamp, then lap number: recording order
anchor = next(l for l in laps if l.usable_as_reference(0.98, axis))
sequence = [l for l in laps if l.lap_id != anchor.lap_id][:max_laps]
grid = anchor.reference_grid(axis)
n_bins = grid.n_bins
pos_t = torch.from_numpy(grid.pos_m.astype(np.float32)); time_t = torch.from_numpy(grid.time_s.astype(np.float32))


@torch.no_grad()
def encode(lap):
    frames = lap.frames()
    return torch.cat([model.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames[i:i + 256]))).to(device),
                                             use_checkpoint=False) for i in range(0, lap.n_frames, 256)])


anchor_view = encode(anchor)[torch.from_numpy(grid.frame_idx).to(device)]


@torch.no_grad()
def localise(desc, lap, views, ticks=None, reverse=False):
    every = max(int(round(lap.fps / HZ)), 1)
    if ticks is None:
        ticks = np.arange((clip_len - 1) * STRIDE, lap.n_frames, every)
    offsets = np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
    idx = ticks[:, None] + offsets if reverse else ticks[:, None] - offsets
    beliefs = []
    for chunk in np.array_split(np.arange(ticks.size), max(1, ticks.size // 64)):
        clips = desc[torch.from_numpy(idx[chunk]).to(device)]
        per_view = [model.head(torch.einsum("tkd,nd->tkn", clips, v) * scale).softmax(-1) for v in views]
        beliefs.append(torch.stack(per_view).mean(0).cpu().double())
    return ticks, torch.cat(beliefs).numpy()


def errors(bins, lap, ticks):
    target = torch.from_numpy(grid.target(lap.s()[ticks]).astype(np.float32))
    m, ms = compute_metrics(torch.as_tensor(bins).float(), target, pos_t, time_t, grid.track_length_m, grid.lap_time_s)
    return m[SKIP:], ms[SKIP:]


def filter_bins(belief, lap, ticks):
    est = ProgressEstimator(grid.pos_m, grid.track_length_m, EstimatorConfig(), seed=0)
    t = lap.t()[ticks].astype(np.float64)
    dts = np.diff(t, prepend=t[0] - 1 / HZ)
    return np.array([est.bin_of(est.step(b, dt).position_m) for b, dt in zip(belief, dts)])


def hindsight_path(belief, max_step=3):
    """Monotone Viterbi over bins: each tick advances 0..max_step bins, wrapping at the line."""
    logp = np.log(belief + 1e-12)
    score, back = logp[0].copy(), np.zeros(belief.shape, np.int8)
    for i in range(1, len(belief)):
        options = np.stack([np.roll(score, k) for k in range(max_step + 1)])
        back[i] = options.argmax(0)
        score = options.max(0) + logp[i]
    path = np.empty(len(belief), int)
    path[-1] = int(score.argmax())
    for i in range(len(belief) - 1, 0, -1):
        path[i - 1] = (path[i] - back[i, path[i]]) % n_bins
    return path


def confident(belief, path, radius=2, threshold=0.5):
    near = (path[:, None] + np.arange(-radius, radius + 1)[None, :]) % n_bins
    return np.take_along_axis(belief, near, 1).sum(1) > threshold


def lap_view(desc, lap, ticks, assigned_bins, keep=None):
    """One view of the lap on anchor bins: the frame placed nearest each bin, else the anchor's."""
    frame_bins = np.interp(np.arange(lap.n_frames), ticks, np.unwrap(assigned_bins * 2 * np.pi / n_bins) * n_bins / (2 * np.pi)) % n_bins
    frame_bins[(np.arange(lap.n_frames) < ticks[0]) | (np.arange(lap.n_frames) > ticks[-1])] = np.inf
    if keep is not None:  # frames next to a rejected tick keep no place
        near_tick = np.clip(np.searchsorted(ticks, np.arange(lap.n_frames)), 0, len(ticks) - 1)
        frame_bins[~keep[near_tick]] = np.inf
    view = anchor_view.clone()
    order = np.argsort(frame_bins)
    nearest = np.searchsorted(frame_bins[order], np.arange(n_bins), side="left").clip(0, lap.n_frames - 1)
    frame = order[nearest]
    with np.errstate(invalid="ignore"):
        ok = np.abs(((frame_bins[frame] - np.arange(n_bins)) + n_bins / 2) % n_bins - n_bins / 2) < 0.75
    view[torch.from_numpy(np.flatnonzero(ok)).to(device)] = desc[torch.from_numpy(frame[ok]).to(device)]
    return view


NAMES = ["anchor only", "blend, labels", "blend, aligner", "blend, aligner, oracle"]
aligner, _ = load_model(Path(sys.argv[4]), device)
variants = {name: {"views": [], "blend": anchor_view.clone(), "scores": []} for name in NAMES}
placement = {"aligner": [], "aligner, oracle": []}
signed = {"aligner": [], "aligner, oracle": []}
with torch.no_grad():
    aligner_anchor = aligner.encode_reference(torch.from_numpy(_to_chw(np.asarray(anchor.frames()[grid.frame_idx]))).to(device), use_checkpoint=False)
L = grid.track_length_m
print(f"{track}: anchor {anchor.lap_id[-26:]}, {len(sequence)} laps in order", flush=True)
for n, lap in enumerate(sequence):
    desc = encode(lap)
    # placement: hindsight against the raw anchor alone, on ticks where both directions exist
    every = max(int(round(lap.fps / HZ)), 1)
    pticks = np.arange((clip_len - 1) * STRIDE, lap.n_frames - (clip_len - 1) * STRIDE, every)
    _, fwd = localise(desc, lap, [anchor_view], pticks)
    _, bwd = localise(desc, lap, [anchor_view], pticks, reverse=True)
    both = np.sqrt(fwd * bwd); both /= both.sum(1, keepdims=True)
    truth_p = grid.target(lap.s()[pticks])
    with torch.no_grad():
        frames = lap.frames()
        adesc = torch.cat([aligner.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames[i:i + 256]))).to(device), use_checkpoint=False)
                           for i in range(0, lap.n_frames, 256)])
        centred = (pticks[:, None] + ((np.arange(clip_len) - (clip_len - 1) / 2) * STRIDE)[None, :]).astype(int)
        rows = []
        for chunk in np.array_split(np.arange(pticks.size), max(1, pticks.size // 64)):
            corr = torch.einsum("tkd,nd->tkn", adesc[torch.from_numpy(centred[chunk]).to(device)], aligner_anchor) * aligner.logit_scale.exp()
            rows.append(aligner.head(corr).softmax(-1).cpu().double())
        rows = torch.cat(rows).numpy()
    fpath = hindsight_path(rows, max_step=4)
    true_off = np.median(((fpath - truth_p) + n_bins / 2) % n_bins - n_bins / 2)
    paths = {"aligner": fpath.astype(float), "aligner, oracle": (fpath - true_off) % n_bins}
    pos_cycle = np.r_[grid.pos_m, grid.pos_m[0] + L]
    for k, path in paths.items():
        at = np.interp(path, np.arange(n_bins + 1), pos_cycle)
        d = (at - lap.s()[pticks] * L + L / 2) % L - L / 2
        placement[k].append(np.abs(d)); signed[k].append(np.median(d))
    for name, v in variants.items():
        views = [anchor_view] if name == "anchor only" else [anchor_view, v["blend"]] if name.startswith("blend") else [anchor_view] + v["views"][-4:]
        ticks, belief = localise(desc, lap, views)
        v["scores"].append(errors(filter_bins(belief, lap, ticks), lap, ticks))
        if name == "anchor only":
            continue
        assigned = truth_p if name.endswith("labels") else paths[name.split(", ", 1)[1]]
        new = lap_view(desc, lap, pticks, np.asarray(assigned))
        if name.startswith("blend"):
            v["blend"] = F.normalize(0.7 * v["blend"] + 0.3 * new, dim=-1)
        else:
            v["views"].append(new)
    line = f"  lap {n + 1:2d} {lap.lap_id[-18:]}"
    for name, v in variants.items():
        m, ms = v["scores"][-1]
        line += f" | {name}: {np.median(m):.2f} m {100 * np.mean(ms > 100):3.0f}%"
    for k in placement:
        line += f" | place {k}: {np.median(placement[k][-1]):.1f} m (bias {signed[k][-1]:+.1f})"
    print(line, flush=True)

report = {}
print(f"\n{track}, filter over laps 2..{len(sequence)} (the first lap has nothing to learn from yet):")
for name, v in variants.items():
    m = np.concatenate([s[0] for s in v["scores"][1:]]); ms = np.concatenate([s[1] for s in v["scores"][1:]])
    report[name] = dict(median_m=float(np.median(m)), over_100ms=float(np.mean(ms > 100)), over_10m=float(np.mean(m > 10)))
    print(f"  {name:32s} median {np.median(m):.2f} m   over 100 ms {100 * np.mean(ms > 100):4.1f}%   over 10 m {100 * np.mean(m > 10):4.1f}%")
for k in placement:
    e = np.concatenate(placement[k])
    report[f"placement {k}"] = dict(median_m=float(np.median(e)), over_10m=float(np.mean(e > 10)))
    print(f"  placement {k:22s} median {np.median(e):.2f} m   beyond 10 m {100 * np.mean(e > 10):4.1f}%")
Path(ckpt).parent.joinpath(f"refine8_stride{STRIDE}_{track.split('__')[0]}.json").write_text(json.dumps(report, indent=2))
