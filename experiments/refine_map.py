"""A reference that improves lap by lap: does it help, and does it drift?

Simulates one track's laps in recording order. The first usable lap is the raw anchor:
it defines positions and timing and is never changed. Every later lap is first localized
against the current map (as the driver would experience it), then used to update the map:

  anchor only   - today's single reference
  blend         - anchor + one refined view, R <- normalise((1-a) R + a V_lap)
  multi-view    - anchor + the last K laps' views
Each view is read by the head on its own and the beliefs are averaged, the way the head
was trained (one lap per reference).

Updates assign a lap's frames to anchor bins either from the labels (upper bound) or by
hindsight from the model itself: a monotone Viterbi path through the whole lap's beliefs,
so progress only moves forward, optionally keeping only ticks where that path is confident.
The gap between labels and hindsight is the drift.
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
STRIDE, HZ, SKIP = 4, 15.0, 30
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
def localise(desc, lap, views):
    every = max(int(round(lap.fps / HZ)), 1)
    ticks = np.arange((clip_len - 1) * STRIDE, lap.n_frames, every)
    idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
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


variants = {name: {"views": [], "blend": anchor_view.clone(), "scores": []}
            for name in ["anchor only", "blend, labels", "blend, hindsight",
                         "multi-view, labels", "multi-view, hindsight", "multi-view, confident"]}
align_err = []
print(f"{track}: anchor {anchor.lap_id[-26:]}, {len(sequence)} laps in order", flush=True)
for n, lap in enumerate(sequence):
    desc = encode(lap)
    for name, v in variants.items():
        if name == "anchor only":
            views = [anchor_view]
        elif name.startswith("blend"):
            views = [anchor_view, v["blend"]]
        else:
            views = [anchor_view] + v["views"][-4:]
        ticks, belief = localise(desc, lap, views)
        bins = filter_bins(belief, lap, ticks)
        m, ms = errors(bins, lap, ticks)
        v["scores"].append((m, ms))
        if name == "anchor only":
            continue
        truth = grid.target(lap.s()[ticks])
        keep = None
        if name.endswith("labels"):
            assigned = truth
        else:
            assigned = hindsight_path(belief)
            if name == "multi-view, hindsight":
                L = grid.track_length_m
                align_err.append(np.abs((grid.pos_m[assigned] - lap.s()[ticks] * L + L / 2) % L - L / 2)[SKIP:])
            if name.endswith("confident"):
                keep = confident(belief, assigned)
        new = lap_view(desc, lap, ticks, np.asarray(assigned), keep)
        if name.startswith("blend"):
            v["blend"] = F.normalize(0.7 * v["blend"] + 0.3 * new, dim=-1)
        else:
            v["views"].append(new)
    line = f"  lap {n + 1:2d} {lap.lap_id[-26:]}"
    for name, v in variants.items():
        m, ms = v["scores"][-1]
        line += f" | {name}: {np.median(m):.2f} m {100 * np.mean(ms > 100):3.0f}%"
    e = align_err[-1]
    print(line + f" | hindsight placement: median {np.median(e):.1f} m, {100 * np.mean(e > 10):.0f}% beyond 10 m", flush=True)

report = {}
print(f"\n{track}, filter over laps 2..{len(sequence)} (the first lap has nothing to learn from yet):")
for name, v in variants.items():
    m = np.concatenate([s[0] for s in v["scores"][1:]]); ms = np.concatenate([s[1] for s in v["scores"][1:]])
    report[name] = dict(median_m=float(np.median(m)), over_100ms=float(np.mean(ms > 100)), over_10m=float(np.mean(m > 10)))
    print(f"  {name:24s} median {np.median(m):.2f} m   over 100 ms {100 * np.mean(ms > 100):4.1f}%   over 10 m {100 * np.mean(m > 10):4.1f}%")
Path(ckpt).parent.joinpath(f"refine_{track.split('__')[0]}.json").write_text(json.dumps(report, indent=2))
