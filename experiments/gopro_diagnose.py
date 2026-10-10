"""
Where PRIMAL goes wrong on a run of experiments/gopro_preview.py: each stage against an
independent truth.

Truth: ORB feature matching (ratio test) of sampled live frames against every other
reference frame, at half the preview's size; kept where the best reference frame has
enough matches and clearly more than the rest. Then, at those live frames:
  encoder  the reference bin whose descriptor is most similar (cosine), frame alone;
  head     the run's single-clip read (readings.csv), the model's 12-frame clip alone;
  tracker  the run's reading (readings.csv).
Stops (the picture still for 1.5 s) are found and shown in a contact sheet: live, then the
reference frames each stage and the truth pick.

--sweep also reruns the live engine offline on the run's frames with other tracker settings
(SWEEP), scored against the same truth, into diagnose_sweep.json.

Usage: PYTHONPATH=. python experiments/gopro_diagnose.py data/gopro/<time> [--run-model v3_mobilenet_lr1_s0] [--sweep]
Writes diagnose.json and diagnose_stops.jpg in the run's folder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from experiments.gopro_preview import BIN_S, HEADER, build, read_exact
from experiments.gopro_review import stretches
from train.estimator import EstimatorConfig, ProgressEstimator
from train.eval import load_model
from train.live import TRACKER, LiveDelta, encode_frames

# tracker settings tried by --sweep: the runtime's, then trusting the matcher more and the momentum less
SWEEP = {"runtime": {}, "likelihood 0.3": dict(likelihood_power=0.3), "likelihood 0.6": dict(likelihood_power=0.6),
         "likelihood 1.0": dict(likelihood_power=1.0), "accel 0.05": dict(accel_noise=0.05),
         "likelihood 0.6, accel 0.05": dict(likelihood_power=0.6, accel_noise=0.05),
         "likelihood 1.0, accel 0.1": dict(likelihood_power=1.0, accel_noise=0.1)}


def decode(run: Path, crop: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The run's times, 148x80 model inputs and 424x240 grey frames, from its saved preview."""
    f = pd.read_csv(run / "frames.csv")
    f["stretch"] = stretches(f.stream_pts.to_numpy())
    offset = (f.pts - f.stream_pts).groupby(f.stretch).median()
    probe = subprocess.Popen([str(build("gopro_probe")), "--file", str(run / "stream.ts")],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pts, small, grey = [], [], []
    x0, x1, y0, y1 = crop
    while True:
        head = read_exact(probe.stdout, HEADER.size)
        if len(head) < HEADER.size:
            break
        _, p, _, _, w, h = HEADER.unpack(head)
        img = np.frombuffer(read_exact(probe.stdout, w * h * 4), np.uint8).reshape(h, w, 4)[:, :, :3]
        small.append(cv2.resize(img[y0:y1, x0:x1], (148, 80), interpolation=cv2.INTER_AREA))
        grey.append(cv2.cvtColor(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        pts.append(p)
    raw = np.array(pts)
    return raw + offset.reindex(stretches(raw)).to_numpy(), np.stack(small), np.stack(grey)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--run-model", default="v3_mobilenet_lr1_s0")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    s = json.loads((run / "summary.json").read_text())
    r = pd.read_csv(run / "readings.csv")
    t, small, grey = decode(run, s["crop"])
    r0, T = s["reference_start_pts"], s["reference_lap_s"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(Path("runs") / a.run_model / "best.pt", device)
    raw_desc = torch.cat([encode_frames(model, small[i:i + 256], device) for i in range(0, len(small), 256)])
    D = torch.nn.functional.normalize(raw_desc, dim=1).cpu().numpy()

    nb = int(round(T / BIN_S))
    bin_t = np.arange(nb) * T / nb
    bin_frame = np.array([int(np.argmin(np.abs(t - (r0 + b)))) for b in bin_t])
    ref = np.where((t >= r0) & (t < r0 + T))[0][::2]
    orb = cv2.ORB_create(1000)
    feats = {i: orb.detectAndCompute(grey[i], None)[1] for i in ref}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def orb_truth(i: int) -> tuple[float, int, float] | None:
        _, q = orb.detectAndCompute(grey[i], None)
        if q is None:
            return None
        counts = []
        for j in ref:
            if feats[j] is None or len(feats[j]) < 2:
                counts.append(0); continue
            m = bf.knnMatch(q, feats[j], k=2)
            counts.append(sum(1 for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance))
        c = np.array(counts)
        k = int(np.argmax(c))
        far = np.abs((t[ref] - t[ref[k]] + T / 2) % T - T / 2) > 1.5  # the rest, over 1.5 s away
        second = c[far].max() if far.any() else 0
        return float(t[ref[k]] - r0), int(c[k]), float(c[k] / max(second, 1))

    wrap = lambda x: (x + T / 2) % T - T / 2
    live = np.where(t > r0 + T)[0]
    rt = r.pts.to_numpy()
    rows = []
    for i in live[::10]:
        truth = orb_truth(i)
        k = np.searchsorted(rt, t[i], side="right") - 1
        if truth is None or k < 0:
            continue
        rows.append(dict(t=t[i], truth=truth[0], matches=truth[1], clarity=truth[2],
                         encoder=float(bin_t[np.argmax(D[bin_frame] @ D[i])]),
                         head=float(r.single_ref_time_s.iloc[k]), tracker=float(r.ref_time_s.iloc[k]),
                         confidence=float(r.confidence.iloc[k])))
    x = pd.DataFrame(rows)
    sure = x[(x.matches >= 40) & (x.clarity >= 1.5)]
    report = {"live_frames_scored": len(x), "with_a_clear_truth": len(sure)}
    for stage in ("encoder", "head", "tracker"):
        e = np.abs(wrap(sure[stage] - sure.truth))
        report[stage] = {"median_s": float(np.median(e)), "within_0.5_s": float(np.mean(e < 0.5)), "within_1_s": float(np.mean(e < 1.0))}
    shown = sure[sure.confidence >= 0.5]
    e = np.abs(wrap(shown.tracker - shown.truth))
    report["tracker_when_shown"] = {"share": float(len(shown) / max(len(sure), 1)), "median_s": float(np.median(e)) if len(e) else None,
                                    "within_0.5_s": float(np.mean(e < 0.5)) if len(e) else None}

    # stops: the picture still (mean frame change small) for 1.5 s
    tiny = np.stack([cv2.resize(g, (106, 60), interpolation=cv2.INTER_AREA) for g in grey]).astype(np.float32)
    change = np.r_[np.inf, np.abs(np.diff(tiny, axis=0)).mean((1, 2))]
    still = change < 1.5
    stops, i = [], 0
    while i < len(t):
        j = i
        while j < len(t) and still[j]:
            j += 1
        if j > i and t[j - 1] - t[i] >= 1.5 and t[i] > r0 + T:
            stops.append((i, j - 1))
        i = j + 1
    report["stops"] = []
    sheet = []
    for i, j in stops[:4]:
        m = (i + j) // 2
        truth = orb_truth(m)
        k = np.searchsorted(rt, t[m], side="right") - 1
        picks = {"truth (ORB)": truth[0] if truth else None, "encoder": float(bin_t[np.argmax(D[bin_frame] @ D[m])]),
                 "head": float(r.single_ref_time_s.iloc[k]), "tracker": float(r.ref_time_s.iloc[k])}
        report["stops"].append({"from_s": float(t[i]), "to_s": float(t[j]), "conf": float(r.confidence.iloc[k]), **picks})
        tiles = [cv2.cvtColor(cv2.resize(grey[m], (212, 120)), cv2.COLOR_GRAY2BGR)]
        cv2.putText(tiles[0], f"LIVE {t[m]:.1f} s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        for name, v in picks.items():
            tile = np.zeros((120, 212, 3), np.uint8)
            if v is not None:
                tile = cv2.cvtColor(cv2.resize(grey[int(np.argmin(np.abs(t - (r0 + v))))], (212, 120)), cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, f"{name} {v:.1f}" if v is not None else name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(tile)
        sheet.append(np.hstack(tiles))
    if sheet:
        cv2.imwrite(str(run / "diagnose_stops.jpg"), np.vstack(sheet))
    (run / "diagnose.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if a.sweep:
        sweep = {}
        for name, change in SWEEP.items():
            engine = LiveDelta(model, None, bin_t, T, int(payload["args"]["clip_len"]), device,
                               reference_descriptors=raw_desc[torch.from_numpy(bin_frame).to(raw_desc.device)])
            engine.tracker = ProgressEstimator(engine.times, engine.lap_time, EstimatorConfig(**{**TRACKER, **change}), seed=0)
            engine.start_at(r0 + T, 0.0)
            ticks = [(t[i], x.ref_time_s, x.confidence) for i in live if (x := engine.push_descriptor(raw_desc[i], float(t[i]))) is not None]
            tt = np.array([k[0] for k in ticks])
            k = np.clip(np.searchsorted(tt, sure.t.to_numpy(), side="right") - 1, 0, len(tt) - 1)
            est, conf = np.array([ticks[j][1] for j in k]), np.array([ticks[j][2] for j in k])
            e = np.abs(wrap(est - sure.truth.to_numpy()))
            sweep[name] = {"median_s": float(np.median(e)), "within_0.5_s": float(np.mean(e < 0.5)), "within_1_s": float(np.mean(e < 1.0)),
                           "shown": float(np.mean(conf >= 0.5)), "within_0.5_s_when_shown": float(np.mean(e[conf >= 0.5] < 0.5)) if np.any(conf >= 0.5) else None}
            print(f"{name:<28} median {sweep[name]['median_s']:.2f} s, within 0.5 s {sweep[name]['within_0.5_s']:.0%}, within 1 s "
                  f"{sweep[name]['within_1_s']:.0%}; shown {sweep[name]['shown']:.0%}, within 0.5 s when shown {sweep[name]['within_0.5_s_when_shown']:.0%}", flush=True)
        (run / "diagnose_sweep.json").write_text(json.dumps(sweep, indent=2))


if __name__ == "__main__":
    main()
