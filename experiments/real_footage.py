"""
First test on real footage: a GoPro of a solo session on a small outdoor kart track
(data/real-footage/adventure-solo), with the finish-line crossings noted by hand to the
second.

1. Frames: the centre of the picture above the steering wheel (AC's 1.85:1, no cockpit),
   resized to 148x80, BGR as packed.
2. Lap starts, refined from the video: the hand-noted crossings are only to the second, so
   each one is moved to the frame that looks most like lap 1's start (raw pixels, no
   network), within ±1.5 s.
3. A model-independent "truth": each live lap aligned to the reference lap by raw-pixel
   matching and a monotone path through the whole lap (SeqSLAM-style; it holds up within
   one session in the same light).
4. Every model on laps 2-9 against lap 1 as the reference, at runtime settings (15 Hz,
   clip frames 1/30 s apart, 12 frames): the single-shot readout and the two-mode
   reference-time tracker, scored against that truth in reference milliseconds.
   Control: an AC lap as the reference must fail.

Usage: PYTHONPATH=. python experiments/real_footage.py RUN [RUN ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from train.dataset import LapIndex, _to_chw
from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME_TWO_MODES, _run_estimator, load_model
from train.model import soft_argmax_circular

# Which footage: REAL_FOOTAGE=adventure-solo (default) or BHGP. Crossings hand-noted to the
# second; CROP is x0, x1, y0, y1 of 1280x720 at AC's aspect; TRUTH_CROP what the ORB truth
# sees (the cockpit out: it looks the same in every frame and would match anywhere);
# the reference lap is the first, or the fastest as the product picks it.
FOOTAGE = os.environ.get("REAL_FOOTAGE", "adventure-solo")
_SETS = {
    "adventure-solo": dict(video="adventure-solo/adventure-pov.mp4", crossings=[12, 50, 87, 125, 163, 201, 238, 276, 313, 350],
                           crop=(250, 1030, 0, 422), truth_crop=(250, 1030, 0, 422), reference="first", refine_s=1.5, window_s=1.5),
    # Historic F1 at Brands Hatch GP, a helmet camera at 30 fps; before 3:34 a warm-up lap.
    "BHGP": dict(video="BHGP-POV-super-realistic.mp4", crossings=[214, 305, 392, 476, 561, 645, 728, 813],
                 crop=(224, 1056, 120, 570), truth_crop=(224, 1056, 120, 430), reference="fastest", refine_s=2.0, window_s=3.0, refine="orb",
                 crop_full=(0, 1280, 14, 706)),  # REAL_CROP=full: the whole width, ~80-94°, near the training view
    # A RaceChrono Pro video (car, dash camera, 60 fps) with its GPS delta drawn on: the
    # overlays are greyed out before anything sees a frame. Crossings from RaceChrono's own
    # lap clock (experiments/racechrono_read.py); lap 2 starts from a standstill.
    "race-chrono": dict(video="race-chrono/race-chrono.mp4", crossings=None, crop=(0, 1280, 14, 706), truth_crop=(0, 1280, 285, 490),
                        reference=1, refine_s=1.0, window_s=3.0, refine="orb",
                        masks=[(0, 0, 225, 165), (225, 0, 872, 94), (860, 98, 1280, 282), (438, 492, 668, 720),
                               (678, 558, 832, 718), (0, 696, 160, 720)]),

}
_S = _SETS[FOOTAGE]
ROOT = Path("data/real-footage") / FOOTAGE
ROOT.mkdir(parents=True, exist_ok=True)
VIDEO = Path("data/real-footage") / _S["video"]
CROSSINGS_S = _S["crossings"]
if CROSSINGS_S is None:  # read off the video's own lap clock
    import pandas as _pd
    _ocr = _pd.read_csv(ROOT / "ocr.csv").dropna()
    _laps = [lap for lap, n in _ocr.lap.value_counts().items() if lap >= 2 and n > 100]  # stray reads are single rows
    CROSSINGS_S = [float((_ocr.t - _ocr.clock_s)[_ocr.lap == lap].median()) for lap in sorted(_laps)]
    CROSSINGS_S.append(float(CROSSINGS_S[-1] + _S.get("last_lap_s", 87.6)))


def mask(frame: np.ndarray) -> np.ndarray:
    """Overlays drawn on the video (another product's numbers) greyed out."""
    for x0, y0, x1, y1 in _S.get("masks", []):
        frame[y0:y1, x0:x1] = 128
    return frame
CROP_NAME = os.environ.get("REAL_CROP", "")
CROP, TRUTH_CROP = _S[f"crop_{CROP_NAME}" if CROP_NAME else "crop"], _S["truth_crop"]
SUFFIX = f"_{CROP_NAME}" if CROP_NAME else ""
HZ, SPACING_S, BIN_S = 15.0, 1 / 30, 0.052


def frames() -> tuple[np.ndarray, np.ndarray]:
    cache = ROOT / f"frames_148x80{SUFFIX}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["frames"], z["t"]
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    x0, x1, y0, y1 = CROP
    out, times, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        if CROSSINGS_S[0] - 3 <= t <= CROSSINGS_S[-1] + 3:
            out.append(cv2.resize(mask(f)[y0:y1, x0:x1], (148, 80), interpolation=cv2.INTER_AREA))
            times.append(t)
        i += 1
    frames_, t_ = np.stack(out), np.array(times)
    np.savez(cache, frames=frames_, t=t_)
    return frames_, t_


def pixels(f: np.ndarray) -> np.ndarray:
    """Grey, 37x20, each frame zero-mean unit-norm: the raw-pixel descriptor."""
    g = np.stack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (37, 20), interpolation=cv2.INTER_AREA) for x in f])
    g = g.reshape(len(g), -1).astype(np.float32)
    g -= g.mean(1, keepdims=True)
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)


def reference_lap(laps: list[tuple[float, float]]) -> int:
    """Index of the reference among the laps: the first, or the fastest."""
    if isinstance(_S["reference"], int):
        return _S["reference"]
    return 0 if _S["reference"] == "first" else int(np.argmin([b - a for a, b in laps]))


def refine_crossings(px: np.ndarray, t: np.ndarray) -> list[float]:
    if _S.get("refine") == "orb":  # refined by features: real_truth_orb.orb_crossings writes them
        return json.loads((ROOT / "starts_orb.json").read_text())
    anchor = int(np.argmin(np.abs(t - CROSSINGS_S[0])))
    clip = np.arange(-8, 9, 2)
    refined = [float(t[anchor])]
    for c in CROSSINGS_S[1:]:
        cand = np.flatnonzero(np.abs(t - c) <= _S["refine_s"])
        cand = cand[(cand + clip.min() >= 0) & (cand + clip.max() < len(t))]
        score = [float(np.mean(np.sum(px[k + clip] * px[anchor + clip], 1))) for k in cand]
        refined.append(float(t[cand[int(np.argmax(score))]]))
    return refined


def viterbi(score: np.ndarray, max_step: int) -> np.ndarray:
    """Monotone path through [ticks, bins] log-scores, advancing 0..max_step bins a tick, no wrap."""
    total, back = score[0].copy(), np.zeros(score.shape, np.int16)
    for i in range(1, len(score)):
        options = np.stack([np.r_[np.full(k, -np.inf), total[: len(total) - k]] for k in range(max_step + 1)])
        back[i] = options.argmax(0)
        total = options.max(0) + score[i]
    path = np.empty(len(score), int)
    path[-1] = int(np.argmax(total))
    for i in range(len(score) - 1, 0, -1):
        path[i - 1] = path[i] - back[i, path[i]]
    return path


def main() -> None:
    runs = sys.argv[1:]
    f, t = frames()
    px = pixels(f)
    starts = refine_crossings(px, t)
    laps = list(zip(starts[:-1], starts[1:]))
    print("lap starts (s):", [round(s, 2) for s in starts])
    print("lap times  (s):", [round(b - a, 2) for a, b in laps], "(hand-noted:", [b - a for a, b in zip(CROSSINGS_S[:-1], CROSSINGS_S[1:])], ")")

    ref_start, ref_end = laps[0]
    ref_T = ref_end - ref_start
    n_bins = int(round(ref_T / BIN_S))
    bin_t = ref_start + np.arange(n_bins) * ref_T / n_bins
    bin_frame = np.array([int(np.argmin(np.abs(t - x))) for x in bin_t])
    grid = SimpleNamespace(time_s=np.arange(n_bins) * ref_T / n_bins, lap_time_s=ref_T, n_bins=n_bins,
                           pos_m=np.arange(n_bins, dtype=float), track_length_m=float(n_bins))

    # model-independent truth: raw-pixel scores of each live tick against every reference bin
    truth = {}
    for k, (a, b) in enumerate(laps[1:], start=2):
        ticks_t = np.arange(a + 11 * SPACING_S, b, 1 / HZ)
        tick_frame = np.array([int(np.argmin(np.abs(t - x))) for x in ticks_t])
        clip = np.arange(-4, 1, 2)  # the tick's frame and two before, 1/30 s apart
        sim = np.mean([px[np.clip(tick_frame + c, 0, len(t) - 1)] @ px[np.clip(bin_frame + c, 0, len(t) - 1)].T for c in clip], 0)
        path = viterbi(sim * 30.0, max_step=4)
        truth[k] = (ticks_t, tick_frame, path * ref_T / n_bins)  # reference time at each tick
        print(f"  lap {k}: truth ends at {path[-1] * ref_T / n_bins:5.2f} s of the {ref_T:.2f} s reference (lap {b - a:.2f} s)", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ac_lap = next(l for l in LapIndex.load(Path("data/packed_ac_v3"), split="train").by_track["magione"] if l.s_span > 0.98)
    report = {"starts": starts}
    for run in runs:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        with torch.no_grad():
            desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(f[i:i + 256])).to(device), use_checkpoint=False)
                              for i in range(0, len(f), 256)])
            ac_grid = ac_lap.reference_grid("time")
            ac_ref = model.encode_reference(torch.from_numpy(_to_chw(np.asarray(ac_lap.frames()[ac_grid.frame_idx]))).to(device),
                                            use_checkpoint=False)
        scale = model.logit_scale.exp()
        rows = {}
        for reference_name, ref_desc, g in (("real", desc[torch.from_numpy(bin_frame).to(device)], grid),
                                            ("AC control", ac_ref, SimpleNamespace(time_s=ac_grid.time_s, lap_time_s=ac_grid.lap_time_s,
                                                                                   n_bins=ac_grid.n_bins, pos_m=ac_grid.pos_m,
                                                                                   track_length_m=ac_grid.track_length_m))):
            errs = {"single": [], "tracker": []}
            for k, (ticks_t, tick_frame, true_ref_t) in truth.items():
                idx = np.clip(tick_frame[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * 2, 0, len(t) - 1)
                beliefs, single = [], []
                with torch.no_grad():
                    for chunk in np.array_split(np.arange(len(idx)), max(1, len(idx) // 64)):
                        logits = model.head(torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(idx[chunk]).to(device)], ref_desc) * scale)
                        beliefs.append(logits.softmax(-1).cpu().double())
                        single.append(soft_argmax_circular(logits, window=8).float().cpu())
                stream = {"grid": g, "belief": torch.cat(beliefs).numpy(), "t": ticks_t}
                tracked = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True)
                for kind, bins in (("single", torch.cat(single).numpy()), ("tracker", tracked.numpy())):
                    est = np.interp(np.asarray(bins, float) % g.n_bins, np.arange(g.n_bins), g.time_s)
                    err = np.abs((est - true_ref_t + g.lap_time_s / 2) % g.lap_time_s - g.lap_time_s / 2) * 1000
                    errs[kind].append(err[int(2 * HZ):])
            rows[reference_name] = {kind: dict(median_ms=float(np.median(np.concatenate(v))),
                                               p90_ms=float(np.percentile(np.concatenate(v), 90)),
                                               over_100ms=float(np.mean(np.concatenate(v) > 100))) for kind, v in errs.items()}
        report[run] = rows
        r = rows["real"]
        c = rows["AC control"]
        print(f"{run:22s} single: median {r['single']['median_ms']:5.0f} ms, >100 ms {100 * r['single']['over_100ms']:5.1f}% | "
              f"tracker: median {r['tracker']['median_ms']:5.0f} ms, p90 {r['tracker']['p90_ms']:5.0f} ms, >100 ms {100 * r['tracker']['over_100ms']:5.1f}% "
              f"| AC-reference control, tracker: median {c['tracker']['median_ms']:6.0f} ms", flush=True)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
