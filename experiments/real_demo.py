"""
A demo video of real laps: on the left the reference lap's frame where PRIMAL places the
car, on the right the live lap; below, PRIMAL's delta beside the true one (the ORB truth,
experiments/real_truth_orb.py) and both drawn over the lap so far. Two-mode reference-
time tracker, runtime clips (12 frames 1/30 s apart), 15 Hz.

Usage: REAL_FOOTAGE=BHGP REAL_CROP=full PYTHONPATH=. python experiments/real_demo.py RUN [LAP ...]
       (no laps: every lap but the reference; writes ROOT/demo_RUN[_crop]_laps....mp4)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from experiments.real_footage import (BIN_S, CROP, HZ, ROOT, SPACING_S, SUFFIX, VIDEO, frames, pixels, reference_lap,
                                      refine_crossings)
from train.dataset import _to_chw
from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME_TWO_MODES, _run_estimator, load_model

PANEL, PLOT_H = (520, 281), 110
GREEN, RED, WHITE, GREY, BLUE = (80, 220, 80), (80, 80, 240), (240, 240, 240), (120, 120, 120), (230, 170, 60)


def text(img, s, xy, colour=WHITE, scale=0.6, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def main() -> None:
    run = sys.argv[1]
    f, t = frames()
    starts = refine_crossings(pixels(f), t)
    laps = list(zip(starts[:-1], starts[1:]))
    r = reference_lap(laps)
    wanted = [int(x) for x in sys.argv[2:]] or [k for k in range(1, len(laps) + 1) if k != r + 1]
    (r0, r1) = laps[r]
    T = r1 - r0
    n = int(round(T / BIN_S))
    bin_frame = np.array([int(np.argmin(np.abs(t - (r0 + i * T / n)))) for i in range(n)])
    grid = SimpleNamespace(time_s=np.arange(n) * T / n, lap_time_s=T, n_bins=n, pos_m=np.arange(n, dtype=float), track_length_m=float(n))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    K = int(payload["args"]["clip_len"])
    stride = max(int(round(SPACING_S / np.median(np.diff(t)))), 1)
    truth = np.load(ROOT / "truth_orb.npz")
    with torch.no_grad():
        desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(f[i:i + 256])).to(device), use_checkpoint=False)
                          for i in range(0, len(f), 256)])
        ref = desc[torch.from_numpy(bin_frame).to(device)]

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    x0, x1, y0, y1 = CROP
    grab = lambda sec: (cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(sec * fps))), cap.read()[1])[1]
    size = (2 * PANEL[0] + 10, PANEL[1] + 70 + PLOT_H)
    out = ROOT / f"demo_{run}{SUFFIX}_laps{''.join(map(str, wanted))}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), HZ, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), HZ, size)
    for lap_no in wanted:
        a, b = laps[lap_no - 1]
        ticks = np.arange(a + (K - 1) * stride / fps, b, 1 / HZ)
        tf = np.array([int(np.argmin(np.abs(t - x))) for x in ticks])
        idx = np.clip(tf[:, None] - np.arange(K - 1, -1, -1)[None, :] * stride, 0, len(t) - 1)
        with torch.no_grad():
            beliefs = [model.head(torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(idx[c]).to(device)], ref)
                                  * model.logit_scale.exp()).softmax(-1).cpu().double()
                       for c in np.array_split(np.arange(len(idx)), max(1, len(idx) // 64))]
        bins = _run_estimator({"grid": grid, "belief": torch.cat(beliefs).numpy(), "t": ticks},
                              EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True).numpy()
        ref_t = np.interp(bins % n, np.arange(n), grid.time_s)
        delta = ((ticks - a) - ref_t + T / 2) % T - T / 2  # reaching the line before the camera is not a lap behind
        tt, te = truth[f"t{lap_no}"], truth[f"e{lap_no}"]
        true = np.where((ticks >= tt[0]) & (ticks <= tt[-1]), (ticks - a) - np.interp(ticks, tt, te), np.nan)
        lim = max(0.5, float(np.nanmax(np.abs(np.r_[delta, true]))) * 1.1)
        err = np.abs(delta - true) * 1000
        for i, x in enumerate(ticks):
            canvas = np.full((size[1], size[0], 3), 25, np.uint8)
            canvas[10:10 + PANEL[1], :PANEL[0]] = cv2.resize(grab(r0 + ref_t[i])[y0:y1, x0:x1], PANEL, interpolation=cv2.INTER_AREA)
            canvas[10:10 + PANEL[1], PANEL[0] + 10:] = cv2.resize(grab(x)[y0:y1, x0:x1], PANEL, interpolation=cv2.INTER_AREA)
            y = 10 + PANEL[1]
            text(canvas, f"REFERENCE lap {r + 1} ({T:.2f} s) at {ref_t[i]:5.1f} s: where PRIMAL places you", (10, y + 22))
            text(canvas, f"LIVE lap {lap_no} of {len(laps)}  {x - a:5.1f} s", (PANEL[0] + 20, y + 22))
            text(canvas, f"PRIMAL {delta[i]:+.2f} s", (10, y + 54), GREEN if delta[i] < 0 else RED, 0.85, 2)
            if np.isfinite(true[i]):
                text(canvas, f"true {true[i]:+.2f} s", (260, y + 54), WHITE, 0.85, 2)
                text(canvas, f"off by {err[i]:3.0f} ms", (480, y + 54), GREEN if err[i] <= 100 else RED, 0.7, 2)
            # the lap so far: PRIMAL (blue) and the truth (white), the delta axis at 0
            top = y + 64
            plot = canvas[top:top + PLOT_H - 6]
            h, w = plot.shape[:2]
            cv2.line(plot, (0, h // 2), (w, h // 2), GREY, 1)
            px = lambda j: int(j / len(ticks) * (w - 1))
            py = lambda v: int(h / 2 - v / lim * (h / 2 - 4))
            for series, colour in ((true, WHITE), (delta, BLUE)):
                pts = [(px(j), py(series[j])) for j in range(0, i + 1) if np.isfinite(series[j])]
                if len(pts) > 1:
                    cv2.polylines(plot, [np.array(pts, np.int32)], False, colour, 2, cv2.LINE_AA)
            text(plot, f"delta over the lap: PRIMAL blue, true white (+-{lim:.1f} s)", (8, 16), GREY, 0.45)
            writer.write(canvas)
        print(f"  lap {lap_no}: {b - a:.2f} s, off by median {np.nanmedian(err):.0f} ms, >100 ms {100 * np.nanmean(err > 100):.0f}%; "
              f"at the line PRIMAL {delta[-1]:+.2f} s, true {(b - a) - T:+.2f} s", flush=True)
    writer.release()
    print(out, f"{out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
