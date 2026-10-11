"""
How much does each way the camera can sit differently from the reference cost? A held
head turn (yaw), nod (pitch) or tilt (roll), and a lens wider or narrower than the
training view (zoom), each applied to the whole live lap while the reference stays
straight ahead at the training view: the error of the two-mode reference-time tracker
and of the single clip, and the shift each causes against the level lap (signed:
positive = placed further along the lap).

Held-out tracks, wide laps (rotations are exact re-projections of the 121° renders),
each against another lap of the same session as the reference.

Usage: PYTHONPATH=. python experiments/pose_sensitivity.py RUN [--same-lens]
       (writes runs/RUN/pose_sensitivity[_same_lens].json)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.yaw_bias import signed_error
from experiments.yaw_search import encode, per_offset
from train import camera
from train.dataset import LapIndex
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, load_model
from train.model import soft_argmax_circular
from train.train import default_device

LAPS, STRIDE, AXIS = 3, 2, "time"
T = camera.TRAINING


def zoomed(hfov: float) -> camera.View:
    """The training view's frame and horizon row, through a lens of another width."""
    return camera.View(T.width, T.height, (T.width / 2) / np.tan(np.radians(hfov / 2)), T.cx, T.cy)


CONDITIONS = {"level": dict()}
CONDITIONS |= {f"yaw {a:+g}°": dict(pose=(a, 0, 0)) for a in (-7, -3, 3, 7)}
CONDITIONS |= {f"pitch {a:+g}°": dict(pose=(0, a, 0)) for a in (-7, -3, 3, 7)}
CONDITIONS |= {f"roll {a:+g}°": dict(pose=(0, 0, a)) for a in (-15, -10, -5, 5, 10, 15)}
CONDITIONS |= {f"lens {h:g}°": dict(view=zoomed(h)) for h in (75, 82, 101, 110)}
# --same-lens: the reference through the same lens as the live lap (one camera for both laps,
# as in the product), the lens itself wider or narrower than the training view
SAME_LENS = {"level": dict()} | {f"both {h:g}°": dict(view=zoomed(h)) for h in (60, 75, 82, 101, 110, 120)}
SAME_LENS |= {f"both fisheye {h:g}°": dict(fisheye=h) for h in (100, 120)}


def fisheye(raw: np.ndarray, hfov: float) -> np.ndarray:
    """Frames through an equidistant fisheye (r = f θ) as wide as hfov, from the wide pinhole renders."""
    f = (T.width / 2) / np.radians(hfov / 2)
    x, y = np.meshgrid(np.arange(T.width) + 0.5 - T.cx, np.arange(T.height) + 0.5 - T.cy)
    theta, phi = np.hypot(x, y) / f, np.arctan2(y, x)
    ray = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    W = camera.WIDE
    mx = (W.focal * ray[..., 0] / ray[..., 2] + W.cx - 0.5).astype(np.float32)
    my = (W.focal * ray[..., 1] / ray[..., 2] + W.cy - 0.5).astype(np.float32)
    return np.stack([cv2.remap(fr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for fr in raw])


def render(raw: np.ndarray, t: np.ndarray, rng: np.random.Generator, **kw) -> np.ndarray:
    return fisheye(raw, kw["fisheye"]) if "fisheye" in kw else camera.render_clip(raw, t, rng, **kw)


def main() -> None:
    run, same_lens = sys.argv[1], "--same-lens" in sys.argv
    conditions = SAME_LENS if same_lens else CONDITIONS
    device = torch.device(default_device())
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    clip_len = int(payload["args"]["clip_len"])
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    errs = {c: {"tracker": [], "single": [], "shift": []} for c in conditions}
    for track, group in sorted(wide.by_track.items()):
        whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
        for k, live in enumerate(whole[:LAPS]):
            reference = whole[(k + 1) % len(whole)]
            grid = reference.reference_grid(AXIS)
            rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
            ref_raw = np.asarray(reference.frames()[grid.frame_idx])
            ref = [encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng), device)]
            every = max(int(round(live.fps / STREAM_HZ)), 1)
            ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
            clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
            used = np.unique(clip_idx)
            row = np.full(live.n_frames, -1)
            row[used] = np.arange(used.size)
            raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
            stream = {"grid": grid, "t": t[ticks], "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
            skip = int(ACQUISITION_S * STREAM_HZ)
            level = None
            for name, kw in conditions.items():
                frames = render(raw, t[used], np.random.default_rng(k), **kw)
                ref_c = [encode(model, render(ref_raw, np.zeros(grid.n_bins), np.random.default_rng(0), **kw), device)] if same_lens and kw else ref
                beliefs, _ = per_offset(model, encode(model, frames, device), row[clip_idx], ref_c)
                belief = torch.from_numpy(beliefs[0].astype(np.float64))
                bins = _run_estimator({**stream, "belief": belief.numpy()}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES),
                                      None, None, 0, reference_time=True)
                e_tr = signed_error(stream, bins)[skip:]
                e_si = signed_error(stream, soft_argmax_circular(belief.clamp_min(1e-12).log().float(), window=8))[skip:]
                level = e_tr if level is None else level
                errs[name]["tracker"].append(e_tr)
                errs[name]["single"].append(e_si)
                errs[name]["shift"].append(e_tr - level)
            print(f"  {track[:28]:<28} live {live.lap_id[-5:]} against {reference.lap_id[-5:]}", flush=True)
    report = {}
    print(f"\n{run}: tracker and single clip, |error| (ms) and share over 100 ms; shift against the level lap (ms)")
    for name, e in errs.items():
        tr, si, sh = (np.concatenate(e[x]) for x in ("tracker", "single", "shift"))
        report[name] = r = {"tracker_median_ms": float(np.median(np.abs(tr)) * 1000), "tracker_over_100ms": float(np.mean(np.abs(tr) > 0.1)),
                            "single_median_ms": float(np.median(np.abs(si)) * 1000), "single_over_100ms": float(np.mean(np.abs(si) > 0.1)),
                            "shift_mean_ms": float(np.mean(sh) * 1000), "shift_median_abs_ms": float(np.median(np.abs(sh)) * 1000)}
        print(f"  {name:<12} tracker {r['tracker_median_ms']:5.0f} ms, {100 * r['tracker_over_100ms']:5.1f}% | single {r['single_median_ms']:5.0f} ms, "
              f"{100 * r['single_over_100ms']:5.1f}% | shift mean {r['shift_mean_ms']:+6.0f} ms, median |shift| {r['shift_median_abs_ms']:5.0f} ms")
    (Path("runs") / run / f"pose_sensitivity{'_same_lens' if same_lens else ''}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
