"""
A sharper model-independent truth for the real GoPro laps: geometric feature matching.

Raw-pixel similarity is nearly flat on this track (it wandered by up to 3.7 s), so each
live tick is instead matched by ORB features + RANSAC against the reference lap's frames
within ±1.5 s of the constant-pace guess; the reference frame with the most geometrically
consistent matches is where the live frame was taken. Half-resolution crop, no network.

Then every model's single-shot readout and two-mode reference-time tracker at those ticks
are scored against it, and the models are compared with each other.

Usage: PYTHONPATH=. python experiments/real_truth_orb.py RUN [RUN ...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from experiments.real_footage import (_S, BIN_S, CROSSINGS_S, HZ, ROOT, SUFFIX, SPACING_S, TRUTH_CROP, VIDEO, frames, pixels, reference_lap,
                                      refine_crossings)
from train.dataset import _to_chw
from train.estimator import EstimatorConfig
from train.eval import REFERENCE_TIME_TWO_MODES, _run_estimator, load_model
from train.model import soft_argmax_circular
from train.seqslam import SeqSLAM

TRUTH_HZ, WINDOW_S = 5.0, _S["window_s"]


def half_res_crops(times: np.ndarray) -> dict[int, np.ndarray]:
    """Grey half-resolution crops of the video frames nearest the given times."""
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wanted = {int(round(x * fps)) for x in times}
    x0, x1, y0, y1 = TRUTH_CROP
    out, i = {}, 0
    last = max(wanted)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = cv2.resize(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), ((x1 - x0) // 2, (y1 - y0) // 2),
                                interpolation=cv2.INTER_AREA)
        i += 1
    return out, fps


def features(img: np.ndarray):
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
    return orb.detectAndCompute(img, None)


def inliers(a, b) -> int:
    ka, da = a
    kb, db = b
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 12:
        return 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.99)
    return int(mask.sum()) if mask is not None else 0


def orb_crossings() -> list[float]:
    """
    The hand-noted crossings, each moved within ±refine_s to the frame with the most
    geometrically consistent ORB matches to the first crossing's frame. Raw pixels jumped
    up to 2 s at speed, with traffic and a moving head.
    """
    cache = ROOT / "starts_orb.json"
    if cache.exists():
        return json.loads(cache.read_text())
    cands = [np.arange(c - _S["refine_s"], c + _S["refine_s"] + 1e-6, 1 / 30) for c in CROSSINGS_S[1:]]
    crops, fps = half_res_crops(np.concatenate([[CROSSINGS_S[0]]] + cands))
    anchor = features(crops[int(round(CROSSINGS_S[0] * fps))])
    starts = [float(CROSSINGS_S[0])]
    for c in cands:
        scores = [inliers(anchor, features(crops[int(round(x * fps))])) for x in c]
        starts.append(float(c[int(np.argmax(scores))]))
        print(f"  crossing {starts[-1]:.2f} s (noted {c[len(c) // 2]:.0f}), {max(scores)} inliers", flush=True)
    cache.write_text(json.dumps(starts))
    return starts


def match_tick(args):
    live_feat, cand = args
    scores = [inliers(live_feat, c) for c in cand]
    return scores


def main() -> None:
    runs = sys.argv[1:]
    f, t = frames()
    starts = orb_crossings() if _S.get("refine") == "orb" else refine_crossings(pixels(f), t)
    laps = list(zip(starts[:-1], starts[1:]))
    r = reference_lap(laps)
    ref_start, ref_end = laps[r]
    live = [(k, lap) for k, lap in enumerate(laps, start=1) if k - 1 != r]
    print(f"reference: lap {r + 1} ({ref_end - ref_start:.2f} s); lap times", [round(b - a, 2) for a, b in laps])
    T = ref_end - ref_start
    ref_times = np.arange(ref_start, ref_end, 1 / 30)  # reference candidates at 30 Hz
    ticks = {k: np.arange(a + 11 * SPACING_S + 2.0, b, 1 / TRUTH_HZ) for k, (a, b) in live}
    crops, fps = half_res_crops(np.concatenate([ref_times] + list(ticks.values())))
    feat = {i: features(img) for i, img in crops.items()}
    ref_idx = [int(round(x * fps)) for x in ref_times]

    truth = {}
    cache = ROOT / "truth_orb.npz"
    if cache.exists():  # the truth does not depend on any model: compute it once
        z = np.load(cache)
        truth = {int(k[1:]): (z[f"t{k[1:]}"], z[f"e{k[1:]}"], z[f"p{k[1:]}"]) for k in z.files if k.startswith("t")}
        print("truth loaded from", cache)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for k, (a, b) in live:
            if k in truth:
                continue
            jobs, windows = [], []
            for x in ticks[k]:
                guess = ref_start + (x - a) * T / (b - a)
                win = np.flatnonzero(np.abs(ref_times - guess) <= WINDOW_S)
                windows.append(win)
                kp, d = feat[int(round(x * fps))]
                jobs.append(((list(zip([p.pt for p in kp], [0] * len(kp))), d), [(list(zip([p.pt for p in feat[ref_idx[j]][0]], [0] * len(feat[ref_idx[j]][0]))), feat[ref_idx[j]][1]) for j in win]))
            results = list(pool.map(_match_serialised, jobs, chunksize=8))
            est, peak = [], []
            for win, scores in zip(windows, results):
                scores = np.asarray(scores, float)
                j = int(np.argmax(scores))
                # parabolic refinement around the peak, in reference seconds
                if 0 < j < len(scores) - 1 and scores[j] > 0:
                    denom = scores[j - 1] - 2 * scores[j] + scores[j + 1]
                    off = 0.5 * (scores[j - 1] - scores[j + 1]) / denom if denom < 0 else 0.0
                else:
                    off = 0.0
                est.append((ref_times[win[j]] - ref_start) + off / 30)
                peak.append(scores[j])
            truth[k] = (ticks[k], np.array(est), np.array(peak))
            dev = truth[k][1] - (ticks[k] - a) * T / (b - a)
            print(f"  lap {k}: ORB truth - constant pace: median {np.median(np.abs(dev))*1000:4.0f} ms, max {np.abs(dev).max():4.2f} s; "
                  f"median inliers {np.median(peak):.0f}", flush=True)

    if not cache.exists():
        np.savez(cache, **{f"{c}{k}": v for k, tr in truth.items() for c, v in zip("tep", tr)})
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n_bins = int(round(T / BIN_S))
    bin_frame = np.array([int(np.argmin(np.abs(t - (ref_start + i * T / n_bins)))) for i in range(n_bins)])
    grid = SimpleNamespace(time_s=np.arange(n_bins) * T / n_bins, lap_time_s=T, n_bins=n_bins,
                           pos_m=np.arange(n_bins, dtype=float), track_length_m=float(n_bins))
    answers, report = {}, {}
    stride = max(int(round(SPACING_S / np.median(np.diff(t)))), 1)  # clip frames 1/30 s apart at any frame rate
    for run in runs:
        if run == "seqslam":  # the classical baseline through the same clips, tracker and truth (eval lines' settings)
            clip_len = 12
            matcher = SeqSLAM(clip_len, down=(48, 64), temperature=6.0).to(device).eval()
            ref_img = torch.from_numpy(_to_chw(f[bin_frame])).to(device)
            logits_of = lambda rows: matcher(torch.from_numpy(_to_chw(f[rows.reshape(-1)])).to(device)
                                             .view(*rows.shape, *ref_img.shape[1:]), ref_img)[0]
        else:
            model, payload = load_model(Path("runs") / run / "best.pt", device)
            clip_len = int(payload["args"]["clip_len"])
            with torch.no_grad():
                desc = torch.cat([model.encode_reference(torch.from_numpy(_to_chw(f[i:i + 256])).to(device), use_checkpoint=False)
                                  for i in range(0, len(f), 256)])
            ref = desc[torch.from_numpy(bin_frame).to(device)]
            logits_of = lambda rows, desc=desc, ref=ref, model=model: model.head(
                torch.einsum("tkd,nd->tkn", desc[torch.from_numpy(rows).to(device)], ref) * model.logit_scale.exp())
        errs = {"single": [], "tracker": []}
        answers[run] = {}
        for k, (a, b) in live:
            stream_t = np.arange(a + 11 * SPACING_S, b, 1 / HZ)
            tf = np.array([int(np.argmin(np.abs(t - x))) for x in stream_t])
            idx = np.clip(tf[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * stride, 0, len(t) - 1)
            beliefs, single = [], []
            with torch.no_grad():
                for chunk in np.array_split(np.arange(len(idx)), max(1, len(idx) // (2 if run == "seqslam" else 64))):
                    logits = logits_of(idx[chunk])
                    beliefs.append(logits.softmax(-1).cpu().double())
                    single.append(soft_argmax_circular(logits, window=8).float().cpu())
            tracked = _run_estimator({"grid": grid, "belief": torch.cat(beliefs).numpy(), "t": stream_t},
                                     EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True).numpy()
            tt, true_ref, _ = truth[k]
            for kind, bins in (("single", torch.cat(single).numpy()), ("tracker", tracked)):
                est = np.interp(np.interp(tt, stream_t, np.unwrap(bins * 2 * np.pi / n_bins) * n_bins / (2 * np.pi)) % n_bins,
                                np.arange(n_bins), grid.time_s)
                answers[run].setdefault(kind, []).append(est)
                errs[kind].append(np.abs((est - true_ref + T / 2) % T - T / 2) * 1000)
        report[run] = {kind: dict(median_ms=float(np.median(np.concatenate(v))), p90_ms=float(np.percentile(np.concatenate(v), 90)),
                                  over_100ms=float(np.mean(np.concatenate(v) > 100))) for kind, v in errs.items()}
        r = report[run]
        print(f"{run:22s} single: median {r['single']['median_ms']:4.0f} ms, p90 {r['single']['p90_ms']:5.0f}, >100 ms {100*r['single']['over_100ms']:5.1f}% | "
              f"tracker: median {r['tracker']['median_ms']:4.0f} ms, p90 {r['tracker']['p90_ms']:5.0f}, >100 ms {100*r['tracker']['over_100ms']:5.1f}%", flush=True)
    names = list(answers)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = np.abs((np.concatenate(answers[x]["tracker"]) - np.concatenate(answers[y]["tracker"]) + T / 2) % T - T / 2) * 1000
            print(f"  agreement {x} vs {y}: median {np.median(d):4.0f} ms, p90 {np.percentile(d, 90):5.0f} ms")
    out = ROOT / f"results_orb{SUFFIX}.json"
    old = json.loads(out.read_text())["report"] if out.exists() else {}
    out.write_text(json.dumps({"starts": starts, "report": {**old, **report}}, indent=2))


def _match_serialised(job):
    """Rebuild keypoints in the worker (cv2.KeyPoint does not pickle)."""
    (live_pts, live_d), cands = job
    live = ([cv2.KeyPoint(x, y, 1) for (x, y), _ in live_pts], live_d)
    out = []
    for pts, d in cands:
        out.append(inliers(live, ([cv2.KeyPoint(x, y, 1) for (x, y), _ in pts], d)))
    return out


if __name__ == "__main__":
    main()
