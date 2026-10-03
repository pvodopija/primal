"""
Does a turned head look like progress through the corner?

Turning the camera moves where the road points in the picture; stepping sideways does
not. Through a bend the kart's heading turns, so a head turned by θ toward the bend
shows what the kart would see a little further on, about θ / (turn rate) seconds later,
and a head turned away what it saw a little earlier. If that is why turns hurt, the
error under a held turn should lean: ahead when the head is turned toward the bend,
behind when away, and hardly at all on straights.

Held-out tracks, wide laps against another lap of the same session (straight ahead),
every live lap level and turned by ±7° and ±14°. Per tick, the shift the turn causes
(turned error minus level error, signed: positive = placed further along the lap than
it is) is grouped by the kart's turn rate there (read off the level frames by phase
correlation) and by whether the head points into the bend or away from it. Both the
two-mode reference-time tracker and the single clip (no tracker).

Usage: PYTHONPATH=. python experiments/yaw_bias.py RUN [RUN ...]   (writes runs/RUN/yaw_bias.json)
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.yaw_search import encode, per_offset
from train import camera
from train.dataset import LapIndex
from train.eval import ACQUISITION_S, REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, load_model
from train.model import _interp_circular, soft_argmax_circular
from train.train import default_device

LAPS, STRIDE, AXIS = 4, 2, "time"
TURNS = (7.0, 14.0)
RATE_BANDS = ((0, 5, "straight, under 5°/s"), (5, 15, "gentle bend, 5-15°/s"), (15, 30, "bend, 15-30°/s"),
              (30, 1e9, "tight bend, over 30°/s"))


def heading(level: np.ndarray) -> np.ndarray:
    """The camera's heading per frame, degrees, positive turning right (as yaw_search.corner_look)."""
    grey = [np.float32(cv2.cvtColor(f[: f.shape[0] // 2], cv2.COLOR_BGR2GRAY)) for f in level]
    window = cv2.createHanningWindow(grey[0].shape[::-1], cv2.CV_32F)
    step = np.zeros(len(level))
    for i in range(1, len(level)):
        (dx, _), _ = cv2.phaseCorrelate(grey[i - 1], grey[i], window)
        step[i] = -np.degrees(np.arctan(dx / camera.TRAINING.focal))
    return np.cumsum(step)


def signed_error(stream: dict, bins: torch.Tensor) -> np.ndarray:
    """Estimated minus true reference time, s, wrapped; positive = placed further along."""
    g = stream["grid"]
    times = torch.from_numpy(g.time_s.astype(np.float32))
    est = _interp_circular(times, bins.float(), g.lap_time_s).numpy().astype(np.float64)
    true = _interp_circular(times, stream["target"].float(), g.lap_time_s).numpy().astype(np.float64)
    return (est - true + g.lap_time_s / 2) % g.lap_time_s - g.lap_time_s / 2


def main() -> None:
    device = torch.device(default_device())
    wide = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout")
    for run in sys.argv[1:]:
        model, payload = load_model(Path("runs") / run / "best.pt", device)
        clip_len = int(payload["args"]["clip_len"])
        rows = []  # per tick: rate, then per (turn, sign): tracker shift, single shift, |error|
        checked = False
        for track, group in sorted(wide.by_track.items()):
            whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, AXIS)]
            for k, live in enumerate(whole[:LAPS]):
                reference = whole[(k + 1) % len(whole)]
                grid = reference.reference_grid(AXIS)
                rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
                ref = [encode(model, camera.render_clip(np.asarray(reference.frames()[grid.frame_idx]),
                                                        np.zeros(grid.n_bins), rng), device)]
                every = max(int(round(live.fps / STREAM_HZ)), 1)
                ticks = np.arange((clip_len - 1) * STRIDE, live.n_frames, every)
                clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * STRIDE
                used = np.unique(clip_idx)
                row = np.full(live.n_frames, -1)
                row[used] = np.arange(used.size)
                raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
                stream = {"grid": grid, "t": t[ticks], "target": torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32))}
                level = camera.render_clip(raw, t[used], np.random.default_rng(k))
                if not checked:  # the heading's sign agrees with the camera's yaw: +5° reads as about +5°
                    probe = heading(np.stack([level[0], camera.render(raw[0], camera.WIDE, camera.TRAINING, yaw=5.0)]))[1]
                    print(f"  sign check: a +5° turn reads as {probe:+.1f}°")
                    assert probe > 2.5, probe
                    checked = True
                head = heading(level)
                tu = t[used]
                rate = (np.interp(t[ticks] + 0.25, tu, head) - np.interp(t[ticks] - 0.25, tu, head)) / 0.5
                errors = {}
                for name, yaw in [("level", 0.0)] + [(f"{s * a:+g}", s * a) for a in TURNS for s in (1, -1)]:
                    frames = level if yaw == 0 else camera.render_clip(raw, tu, np.random.default_rng(k), pose=(yaw, 0, 0))
                    beliefs, _ = per_offset(model, encode(model, frames, device), row[clip_idx], ref)
                    belief = torch.from_numpy(beliefs[0].astype(np.float64))
                    bins = _run_estimator({**stream, "belief": belief.numpy()}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES),
                                          None, None, 0, reference_time=True)
                    single = soft_argmax_circular(belief.clamp_min(1e-12).log().float(), window=8)
                    errors[name] = (signed_error(stream, bins), signed_error(stream, single))
                skip = int(ACQUISITION_S * STREAM_HZ)
                for i in range(skip, ticks.size):
                    r = {"rate": float(rate[i]), "level": abs(errors["level"][0][i])}
                    for a in TURNS:
                        for s in (1, -1):
                            name = f"{s * a:+g}"
                            r[name] = (errors[name][0][i] - errors["level"][0][i], errors[name][1][i] - errors["level"][1][i],
                                       abs(errors[name][0][i]))
                    rows.append(r)
                print(f"  {run} {track[:26]:<26} live {live.lap_id[-5:]} reference {reference.lap_id[-5:]}", flush=True)

        rate = np.array([r["rate"] for r in rows])
        report = {"ticks": len(rows), "bands": {}}
        print(f"\n{run}: shift caused by a held turn, ms (positive = placed further along); "
              f"'toward' = head turned into the bend")
        print(f"  {'':<26}{'ticks':>7}{'turn':>6}{'toward: tracker':>17}{'single clip':>13}{'predicted':>11}"
              f"{'away: tracker':>15}{'single clip':>13}{'>100 ms level / turned':>25}")
        for lo, hi, label in RATE_BANDS:
            band = (np.abs(rate) >= lo) & (np.abs(rate) < hi)
            entry = {"ticks": int(band.sum()), "median_rate_deg_s": float(np.median(np.abs(rate[band])))}
            for a in TURNS:
                toward, away, over = [], [], []
                for i in np.flatnonzero(band):
                    for s in (1, -1):
                        shift_tracker, shift_single, err = rows[i][f"{s * a:+g}"]
                        (toward if s * np.sign(rate[i]) > 0 else away).append((shift_tracker, shift_single))
                        over.append(err > 0.1)
                toward, away = np.array(toward) * 1000, np.array(away) * 1000
                level_over = float(np.mean([rows[i]["level"] > 0.1 for i in np.flatnonzero(band)]))
                predicted = float(np.median(np.minimum(a / np.maximum(np.abs(rate[band]), 1e-3), 10.0))) * 1000
                entry[f"{a:g}"] = {"toward_tracker_ms": float(np.median(toward[:, 0])), "toward_single_ms": float(np.median(toward[:, 1])),
                                   "away_tracker_ms": float(np.median(away[:, 0])), "away_single_ms": float(np.median(away[:, 1])),
                                   "predicted_ms": predicted, "over_100ms_level": level_over, "over_100ms_turned": float(np.mean(over))}
                e = entry[f"{a:g}"]
                print(f"  {label:<26}{band.sum():>7}{a:>5g}°{e['toward_tracker_ms']:>+14.0f} ms{e['toward_single_ms']:>+10.0f} ms"
                      f"{('±' + format(min(predicted, 9999), '.0f')):>8} ms{e['away_tracker_ms']:>+12.0f} ms{e['away_single_ms']:>+10.0f} ms"
                      f"{100 * level_over:>14.1f}% / {100 * e['over_100ms_turned']:.1f}%")
            report["bands"][label] = entry
        (Path("runs") / run / "yaw_bias.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
