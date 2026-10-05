"""A demo video of one real GoPro lap: the live view, the reference frame the model picks,
and the delta it shows. Usage: PYTHONPATH=. python experiments/real_demo.py RUN LAP"""
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np, torch
from experiments.real_footage import BIN_S, CROP, HZ, ROOT, SPACING_S, VIDEO, frames, pixels, reference_lap, refine_crossings
from train.dataset import _to_chw
from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME_TWO_MODES, _run_estimator, load_model

run, lap_no = sys.argv[1], int(sys.argv[2])
f, t = frames()
starts = refine_crossings(pixels(f), t)
laps = list(zip(starts[:-1], starts[1:]))
(r0, r1), (a, b) = laps[reference_lap(laps)], laps[lap_no - 1]
T = r1 - r0
n = int(round(T / BIN_S))
bin_frame = np.array([int(np.argmin(np.abs(t - (r0 + i * T / n)))) for i in range(n)])
grid = SimpleNamespace(time_s=np.arange(n) * T / n, lap_time_s=T, n_bins=n, pos_m=np.arange(n, dtype=float), track_length_m=float(n))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, payload = load_model(Path("runs") / run / "best.pt", device)
K = int(payload["args"]["clip_len"])
with torch.no_grad():
    desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(f[i:i + 256])).to(device), use_checkpoint=False) for i in range(0, len(f), 256)])
    ref = desc[torch.from_numpy(bin_frame).to(device)]
    ticks = np.arange(a + 11 * SPACING_S, b, 1 / HZ)
    tf = np.array([int(np.argmin(np.abs(t - x))) for x in ticks])
    idx = np.clip(tf[:, None] - np.arange(K - 1, -1, -1)[None, :] * 2, 0, len(t) - 1)
    beliefs = [model.head(torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(idx[c]).to(device)], ref) * model.logit_scale.exp()).softmax(-1).cpu().double()
               for c in np.array_split(np.arange(len(idx)), max(1, len(idx) // 64))]
bins = _run_estimator({"grid": grid, "belief": torch.cat(beliefs).numpy(), "t": ticks}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True).numpy()
cap = cv2.VideoCapture(str(VIDEO)); fps = cap.get(cv2.CAP_PROP_FPS)
x0, x1, y0, y1 = CROP
grab = lambda sec: (cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(sec * fps))), cap.read()[1])[1]
out = ROOT / f"demo_{run}_lap{lap_no}.mp4"
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), HZ, (2 * 520 + 10, 300 + 60))
deltas = []
for i, x in enumerate(ticks):
    ref_t = float(np.interp(bins[i] % n, np.arange(n), grid.time_s))
    delta = (x - a) - ref_t
    deltas.append(delta)
    live = cv2.resize(grab(x)[y0:y1, x0:x1], (520, 281))
    picked = cv2.resize(grab(r0 + ref_t)[y0:y1, x0:x1], (520, 281))
    canvas = np.full((360, 1050, 3), 25, np.uint8)
    canvas[10:291, :520] = live; canvas[10:291, 530:] = picked
    cv2.putText(canvas, f"LIVE lap {lap_no}  {x - a:5.1f} s", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    cv2.putText(canvas, f"REFERENCE lap {reference_lap(laps) + 1} at {ref_t:5.1f} s (model's pick)", (540, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    colour = (80, 220, 80) if delta < 0 else (80, 80, 240)
    cv2.putText(canvas, f"DELTA {delta:+.2f} s", (400, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    writer.write(canvas)
writer.release()
print(out, f"final delta {deltas[-1]:+.2f} s, true lap difference {(b - a) - T:+.2f} s")
