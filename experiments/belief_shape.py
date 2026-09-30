"""Average belief around the true bin, forward clips, standard stream pairs. A second bump
about one clip span behind the truth would mean the head hedges on the clip's direction;
a skewed main peak would mean it reads a faint stripe's end early.
Usage: belief_shape.py CHECKPOINT"""
import sys
from pathlib import Path
import numpy as np, torch
from train.eval import ACQUISITION_S, STREAM_HZ, _lap_stream, _stream_pairs, load_model, reference_axis_of

ckpt = sys.argv[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path(ckpt), device)
clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
skip = int(ACQUISITION_S * STREAM_HZ)
OFF = np.arange(-40, 21)
for gate in ("G1", "G2"):
    prof, n = np.zeros(OFF.size), 0
    weak = np.zeros(OFF.size); nw = 0
    for g, ref, live in _stream_pairs(Path("data/packed_ac_v2"), axis, 3):
        if g != gate:
            continue
        st = _lap_stream(model, clip_len, ref, live, axis, 4, device, 8)
        b, tgt = st["belief"][skip:], np.round(st["target"].numpy()[skip:]).astype(int)
        nb = b.shape[1]
        rows = np.take_along_axis(b, (tgt[:, None] + OFF[None, :]) % nb, 1)
        prof += rows.sum(0); n += len(rows)
        mass = np.take_along_axis(b, (b.argmax(1)[:, None] + np.arange(-3, 4)[None, :]) % nb, 1).sum(1)
        w = mass < np.quantile(mass, 0.25)
        weak += rows[w].sum(0); nw += w.sum()
    prof /= n; weak /= nw
    print(f"\n{gate}: mean belief by bin offset from truth (one bin = 1/{STREAM_HZ:.0f} s of reference lap time? see note)")
    for o, p, q in zip(OFF, prof, weak):
        if o % 2 == 0 or abs(o) <= 4:
            print(f"  {o:+4d}  {p:.4f} {'#' * int(p * 300):<40s} weakest quarter {q:.4f} {'#' * int(q * 300)}")
    print(f"  mass behind (-40..-1): {prof[OFF < 0].sum():.3f}  ahead (1..20): {prof[OFF > 0].sum():.3f}  "
          f"weakest quarter behind {weak[OFF < 0].sum():.3f} ahead {weak[OFF > 0].sum():.3f}")
