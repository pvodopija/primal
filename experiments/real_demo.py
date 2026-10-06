"""
A demo video of real laps: on the left the reference lap's frame where PRIMAL places the
car, on the right the live lap; below, PRIMAL's delta beside the true one (the ORB truth,
experiments/real_truth_orb.py), or UNAVAILABLE while the confidence (train/live.py's: the
share of the last second's single clips within 0.25 s of the tracker) is under CONF_MIN.
Three graphs over the lap so far: the true delta; PRIMAL's, green within 100 ms of the
truth and red beyond, greyed where it abstains; the confidence and its threshold.
Two-mode reference-time tracker, runtime clips (12 frames 1/30 s apart), 15 Hz.

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
from train.live import AGREE_S
from train.model import soft_argmax_circular

PANEL, PLOT_H, CONF_MIN = (520, 281), 78, 0.5
GREEN, RED, WHITE, GREY, BLUE, SHADE = (80, 220, 80), (80, 80, 240), (240, 240, 240), (120, 120, 120), (230, 170, 60), (55, 55, 55)


def graph(img, top, label, series, i, lo, hi, colours, abstain, line=None):
    """One strip: `series` up to tick i, coloured per tick; abstaining ticks shaded behind."""
    plot = img[top:top + PLOT_H - 6]
    h, w = plot.shape[:2]
    n = len(series)
    px = lambda j: int(j / n * (w - 1))
    py = lambda v: int(h - 3 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (h - 6))
    for j in np.flatnonzero(abstain[: i + 1]):
        cv2.line(plot, (px(j), 0), (px(j), h - 1), SHADE, 2)
    if line is not None:
        cv2.line(plot, (0, py(line)), (w, py(line)), GREY, 1)
    for j in range(1, i + 1):
        if np.isfinite(series[j - 1]) and np.isfinite(series[j]):
            cv2.line(plot, (px(j - 1), py(series[j - 1])), (px(j), py(series[j])), colours[j], 2, cv2.LINE_AA)
    text(plot, label, (8, 14), GREY, 0.45)


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
    size = (2 * PANEL[0] + 10, PANEL[1] + 80 + 3 * PLOT_H)
    out = ROOT / f"demo_{run}{SUFFIX}_confidence_laps{''.join(map(str, wanted))}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), HZ, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), HZ, size)
    every_err, every_conf = [], []
    for lap_no in wanted:
        a, b = laps[lap_no - 1]
        ticks = np.arange(a + (K - 1) * stride / fps, b, 1 / HZ)
        tf = np.array([int(np.argmin(np.abs(t - x))) for x in ticks])
        idx = np.clip(tf[:, None] - np.arange(K - 1, -1, -1)[None, :] * stride, 0, len(t) - 1)
        beliefs, single = [], []
        with torch.no_grad():
            for c in np.array_split(np.arange(len(idx)), max(1, len(idx) // 64)):
                logits = model.head(torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(idx[c]).to(device)], ref) * model.logit_scale.exp())
                beliefs.append(logits.softmax(-1).cpu().double())
                single.append(soft_argmax_circular(logits, window=8).float().cpu())
        bins = _run_estimator({"grid": grid, "belief": torch.cat(beliefs).numpy(), "t": ticks},
                              EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True).numpy()
        ref_t = np.interp(bins % n, np.arange(n), grid.time_s)
        single_t = np.interp(torch.cat(single).numpy() % n, np.arange(n), grid.time_s)
        agree = np.abs((single_t - ref_t + T / 2) % T - T / 2) < AGREE_S
        conf = np.convolve(agree.astype(float), np.ones(int(HZ)))[: len(agree)] / np.minimum(np.arange(1, len(agree) + 1), int(HZ))
        delta = ((ticks - a) - ref_t + T / 2) % T - T / 2  # reaching the line before the camera is not a lap behind
        tt, te = truth[f"t{lap_no}"], truth[f"e{lap_no}"]
        true = np.where((ticks >= tt[0]) & (ticks <= tt[-1]), (ticks - a) - np.interp(ticks, tt, te), np.nan)
        err = np.abs(delta - true) * 1000
        abstain = conf < CONF_MIN
        good = np.where(np.isfinite(err), err <= 100, True)
        colours = [GREEN if g else RED for g in good]
        lim = max(0.5, float(np.nanmax(np.abs(true))) * 1.3)
        for i, x in enumerate(ticks):
            canvas = np.full((size[1], size[0], 3), 25, np.uint8)
            canvas[10:10 + PANEL[1], :PANEL[0]] = cv2.resize(grab(r0 + ref_t[i])[y0:y1, x0:x1], PANEL, interpolation=cv2.INTER_AREA)
            canvas[10:10 + PANEL[1], PANEL[0] + 10:] = cv2.resize(grab(x)[y0:y1, x0:x1], PANEL, interpolation=cv2.INTER_AREA)
            y = 10 + PANEL[1]
            text(canvas, f"REFERENCE lap {r + 1} ({T:.2f} s) at {ref_t[i]:5.1f} s: where PRIMAL places you", (10, y + 22))
            text(canvas, f"LIVE lap {lap_no} of {len(laps)}  {x - a:5.1f} s", (PANEL[0] + 20, y + 22))
            if abstain[i]:
                text(canvas, "PRIMAL  UNAVAILABLE", (10, y + 56), GREY, 0.85, 2)
            else:
                text(canvas, f"PRIMAL {delta[i]:+.2f} s", (10, y + 56), GREEN if delta[i] < 0 else RED, 0.85, 2)
            if np.isfinite(true[i]):
                text(canvas, f"true {true[i]:+.2f} s", (330, y + 56), WHITE, 0.85, 2)
                if not abstain[i]:
                    text(canvas, f"off by {err[i]:3.0f} ms", (560, y + 56), GREEN if good[i] else RED, 0.7, 2)
            text(canvas, f"confidence {conf[i]:.2f}", (820, y + 56), WHITE if not abstain[i] else GREY, 0.7, 2)
            top = y + 74
            graph(canvas, top, f"true delta (+-{lim:.1f} s)", true, i, -lim, lim, [WHITE] * len(ticks), abstain, line=0.0)
            graph(canvas, top + PLOT_H, "PRIMAL delta: green within 100 ms of the truth, red beyond; grey: unavailable",
                  delta, i, -lim, lim, colours, abstain, line=0.0)
            graph(canvas, top + 2 * PLOT_H, f"confidence, threshold {CONF_MIN}", conf, i, 0.0, 1.0, [BLUE] * len(ticks), abstain,
                  line=CONF_MIN)
            writer.write(canvas)
        ok = np.isfinite(err)
        every_err.append(err[ok]); every_conf.append(conf[ok])
        shown = ~abstain[ok]
        print(f"  lap {lap_no}: shown {100 * shown.mean():.0f}% of the time; over 100 ms: all {100 * np.mean(err[ok] > 100):.0f}%, "
              f"shown {100 * np.mean(err[ok][shown] > 100):.0f}%, hidden {100 * np.mean(err[ok][~shown] > 100):.0f}%", flush=True)
    e, c = np.concatenate(every_err), np.concatenate(every_conf)
    print(f"all laps, over 100 ms: {100 * np.mean(e > 100):.1f}% of every tick")
    for th in (0.3, 0.5, 0.7, 0.9):
        shown = c >= th
        print(f"  threshold {th}: shown {100 * shown.mean():.0f}% of the time, of it {100 * np.mean(e[shown] > 100):.1f}% over 100 ms "
              f"(median {np.median(e[shown]):.0f} ms); hidden part {100 * np.mean(e[~shown] > 100):.0f}% over 100 ms")
    writer.release()
    print(out, f"{out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
