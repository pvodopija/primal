"""How precisely can a finished lap be placed on another? The lap aligner's own test.

Every Silverstone lap is placed on the first usable one (the anchor) at stride 2:
  forward  - the live matcher: clips ending at each frame (causal)
  middle   - the aligner: clips centred on each frame, context on both sides
each read two ways: a monotone path through the lap (Viterbi), and each tick's own
windowed soft-argmax. Errors against the labels, in metres; signed = placed - true.
Usage: aligner_test.py FORWARD_CKPT MIDDLE_CKPT [TRACK]"""
import sys
from pathlib import Path
import numpy as np, torch
from train.dataset import LapIndex, _to_chw
from train.eval import load_model, reference_axis_of
from train.model import soft_argmax_circular

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
models = {name: load_model(Path(p), device) for name, p in (("forward", sys.argv[1]), ("middle", sys.argv[2]))}
track = sys.argv[3] if len(sys.argv) > 3 else "ks_silverstone__national"
_, payload = models["forward"]
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
import os
STRIDE, HZ, SKIP = int(os.environ.get("STRIDE", 2)), 15.0, 30
laps = sorted([l for l in LapIndex.load(Path("data/packed_ac_v2")).by_track[track] if l.s_span > 0.98], key=lambda l: l.lap_id)
anchor = next(l for l in laps if l.usable_as_reference(0.98, axis))
grid = anchor.reference_grid(axis)
n, L = grid.n_bins, grid.track_length_m
pos_cycle = np.r_[grid.pos_m, grid.pos_m[0] + L]


def viterbi(belief, max_step=4):
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
def place(model, lap, centred):
    ref = model.encode_reference(torch.from_numpy(_to_chw(np.asarray(anchor.frames()[grid.frame_idx]))).to(device), use_checkpoint=False)
    frames = lap.frames()
    desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames[i:i + 256]))).to(device), use_checkpoint=False)
                      for i in range(0, lap.n_frames, 256)])
    half = (clip_len - 1) * STRIDE
    ticks = np.arange(half, lap.n_frames - half, max(int(round(lap.fps / HZ)), 1))
    k = np.arange(clip_len)
    offsets = (k - (clip_len - 1) / 2) * STRIDE if centred else (k - (clip_len - 1)) * STRIDE
    idx = (ticks[:, None] + offsets[None, :]).astype(int)
    beliefs, singles = [], []
    for chunk in np.array_split(np.arange(ticks.size), max(1, ticks.size // 64)):
        logits = model.head(torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(idx[chunk]).to(device)], ref) * model.logit_scale.exp())
        beliefs.append(logits.softmax(-1).cpu().double()); singles.append(soft_argmax_circular(logits, window=8).float().cpu())
    return ticks, torch.cat(beliefs).numpy(), torch.cat(singles).numpy()


def metres(bins, lap, ticks):
    at = np.interp(np.asarray(bins, float) % n, np.arange(n + 1), pos_cycle)
    return ((at - lap.s()[ticks] * L + L / 2) % L - L / 2)[SKIP:]


rows = {f"{m}, {r}": [] for m in models for r in ("path", "each tick")}
for lap in laps:
    if lap.lap_id == anchor.lap_id:
        continue
    line = f"  {lap.lap_id[-18:]}"
    for name, (model, _) in models.items():
        ticks, belief, single = place(model, lap, name == "middle")
        for r, bins in (("path", viterbi(belief)), ("each tick", single)):
            d = metres(bins, lap, ticks)
            rows[f"{name}, {r}"].append(d)
            line += f" | {name} {r}: {np.median(np.abs(d)):.2f} m ({np.median(d):+.2f})"
    print(line, flush=True)
print(f"\n{track}, {len(laps) - 1} laps placed on {anchor.lap_id[-18:]}, stride {STRIDE}: |error| median / p90 / >10 m, signed median")
for k, v in rows.items():
    d = np.concatenate(v)
    print(f"  {k:20s} {np.median(np.abs(d)):.2f} m / {np.percentile(np.abs(d), 90):.2f} m / {100 * np.mean(np.abs(d) > 10):.1f}%   signed {np.median(d):+.2f} m")
