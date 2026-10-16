"""
How PRIMAL handles a lap driven at another pace than its reference, on real footage with an
exact answer. The reference is one lap at 1x; a live lap (another lap of the same driver, a few
tenths apart) is replayed at a known, changing speed: at playback time tau the screen shows the
original frame at s = w(tau). Where the kart is on the track depends on the frame, not on the
speed it is shown at, so a perfect engine reads at tau the reference time it reads at w(tau)
on the same lap played at 1x. The difference is the error the speed change caused, in seconds
of delta, with no truth source needed. The delta itself moves by tau - w(tau): a lap warped to
take longer finishes that much further behind.

The engine is the app's (train/live.py LiveDelta, the shipped tracker), fed continuously as in
a session: the previous lap's last PREROLL_S at 1x, then the warped lap, a 30 Hz picture. The
source is 60 fps, so slow playback shows real in-between frames; fast playback skips frames as
a camera on a faster kart would (its motion blur is not simulated). Each warp is scored against
the same lap at 1x with the same seed; a second seed at 1x gives the noise floor.

With SPEED_SENSOR=<sigma> the tracker also takes a pace reading every tick: the warp's own rate
(ds/dtau, times the reference's lap over this lap's), trusted to sigma (rate units, 0.1 = 10%), as
a speed sensor read through the reference's speed would give it (gps_speed.py); slightly
optimistic, the laps' own local pace differences unread.

Usage: [SPEED_SENSOR=0.1] PYTHONPATH=. python experiments/speed_warp.py VIDEO START_S RUN [RUN ...]
       (rotax-night: START 75; writes <video dir>/speed_warp/<run>[_sensor<sigma>].json and .png)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.eval import load_model
from train.live import LiveDelta, encode_frames
from train.session import LapCompleted, ReferenceClosed, Session

FPS_OUT, BIN_S, PREROLL_S, LIVE_LAPS, SPAN_LAPS = 30.0, 0.052, 6.0, (2, 3, 4, 5), 7
SENSOR = float(os.environ["SPEED_SENSOR"]) if os.environ.get("SPEED_SENSOR") else None


def speed_profiles(lap_s: float) -> dict:
    """Speed (x real time) as a function of playback time, by name."""
    h = lap_s / 2
    return {
        "1x (baseline, seed 2: noise floor)": lambda tau: np.ones_like(tau),
        "constant 0.5x": lambda tau: np.full_like(tau, 0.5),
        "constant 0.75x": lambda tau: np.full_like(tau, 0.75),
        "constant 0.9x": lambda tau: np.full_like(tau, 0.9),
        "constant 1.1x": lambda tau: np.full_like(tau, 1.1),
        "constant 1.25x": lambda tau: np.full_like(tau, 1.25),
        "constant 1.5x": lambda tau: np.full_like(tau, 1.5),
        "constant 2x": lambda tau: np.full_like(tau, 2.0),
        "half 0.5x, half 1.5x": lambda tau: np.where(tau < h / 0.5, 0.5, 1.5),
        "half 0.75x, half 1.5x (same total)": lambda tau: np.where(tau < h / 0.75, 0.75, 1.5),
        "half 1.5x, half 0.75x (same total)": lambda tau: np.where(tau < h / 1.5, 1.5, 0.75),
        "surges +-20%, 8 s period": lambda tau: 1 + 0.2 * np.sin(2 * np.pi * tau / 8),
        "surges +-40%, 8 s period": lambda tau: 1 + 0.4 * np.sin(2 * np.pi * tau / 8),
        "mistake: 0.3x for 3 s at 20 s": lambda tau: np.where((tau >= 20) & (tau < 23), 0.3, 1.0),
        "one slow sector: 0.85x for 10 s": lambda tau: np.where((tau >= 15) & (tau < 25), 0.85, 1.0),
    }


def frames_60(video: Path, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    cache = video.with_name(video.stem + f"_148x80_60fps_{int(t0)}_{int(t1)}.npz")
    if cache.exists():
        z = np.load(cache)
        return z["frames"], z["t"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ch = int(round(min(h, w * 80 / 148)))
    y0 = int(round((h - ch) / 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    i, out, times = int(t0 * fps), [], []
    while i / fps <= t1:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f[y0:y0 + ch], (148, 80), interpolation=cv2.INTER_AREA))
        times.append(i / fps)
        i += 1
    f_, t_ = np.stack(out), np.array(times)
    np.savez(cache, frames=f_, t=t_)
    return f_, t_


def lines(model, clip_len: int, d: torch.Tensor, t: np.ndarray, start: float) -> list[float]:
    """The session's own line crossings at 1x (30 fps): where every lap starts and ends."""
    every = int(round((1 / FPS_OUT) / np.median(np.diff(t))))
    session = Session(model, clip_len, torch.device("cpu"))
    keep = np.flatnonzero(t >= start)[::every]
    session.start(float(t[keep[0]]))
    out = []
    for k in keep:
        for e in session.push(d[k], float(t[k])):
            if isinstance(e, ReferenceClosed):
                out += [e.line_s - e.lap_s, e.line_s]
            elif isinstance(e, LapCompleted):
                out.append(e.line_s)
    return out


def warp(speed, lap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Playback ticks of the warped lap at 30 Hz and the original time each shows."""
    fine = np.arange(0, 20 * lap_s, 0.001)
    s = np.concatenate([[0.0], np.cumsum(speed(fine[:-1]) * 0.001)])
    end = fine[np.searchsorted(s, lap_s)]
    tau = np.arange(0, end, 1 / FPS_OUT)
    return tau, np.interp(tau, fine, s)


def run_lap(model, clip_len, ref, times, T, d, t, a, b, tau, s, seed: int):
    """The engine over PREROLL_S of the lap before at 1x, then the warped lap; readings in the warped lap."""
    engine = LiveDelta(model, None, times, T, clip_len, torch.device("cpu"), seed=seed, reference_descriptors=ref)
    pre = np.arange(-PREROLL_S, 0, 1 / FPS_OUT)
    if SENSOR is not None:  # the pace at each tick (engine.last_tick is set just before the tracker steps)
        rate_t = np.concatenate([pre, tau])
        rate = np.concatenate([np.ones_like(pre), np.gradient(s, tau)]) * T / (b - a)
        step = engine.tracker.step
        engine.tracker.step = lambda belief, dt: step(belief, dt, speed_obs=float(np.interp(engine.last_tick, rate_t, rate)),
                                                      speed_sigma=SENSOR)
    play = np.concatenate([pre, tau])
    orig = np.concatenate([a + pre, a + s])
    idx = np.clip(np.round(np.interp(orig, t, np.arange(len(t)))).astype(int), 0, len(t) - 1)  # the nearest frame
    out_tau, out_ref, out_conf = [], [], []
    for p, k in zip(play, idx):
        r = engine.push_descriptor(d[k], float(p))
        if r is not None and p >= 0:
            out_tau.append(p)
            out_ref.append(r.ref_time_s)
            out_conf.append(r.confidence)
    return np.array(out_tau), np.array(out_ref), np.array(out_conf)


def circ(x: np.ndarray, T: float) -> np.ndarray:
    return (x + T / 2) % T - T / 2


def main() -> None:
    video, start = Path(sys.argv[1]), float(sys.argv[2])
    out_dir = video.parent / video.stem / "speed_warp"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in sys.argv[3:]:
        model, payload = load_model(Path("runs") / run / "best.pt", torch.device("cpu"))
        clip_len = int(payload["args"]["clip_len"])
        f, t = frames_60(video, start - PREROLL_S - 1, start + 50 * SPAN_LAPS)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        gpu, _ = load_model(Path("runs") / run / "best.pt", device)
        d = torch.cat([encode_frames(gpu, f[i:i + 256], device).cpu() for i in range(0, len(f), 256)])
        c = lines(model, clip_len, d, t, start)
        print(f"{run}: lines at {np.round(c, 2).tolist()}", flush=True)
        a0, b0 = c[0], c[1]
        T = b0 - a0
        n = int(round(T / BIN_S))
        times = np.arange(n) * T / n
        ref = d[np.clip(np.searchsorted(t, a0 + times), 0, len(t) - 1)]
        report, curves, base = {}, {}, {}
        for k in LIVE_LAPS:  # each live lap at 1x, the answer every warp of it is held to
            if k + 1 < len(c):
                L = c[k + 1] - c[k]
                base[k] = run_lap(model, clip_len, ref, times, T, d, t, c[k], c[k + 1], *warp(lambda x: np.ones_like(x), L), seed=0)
        for name, speed in speed_profiles(T).items():
            errs, ends, shown = [], [], []
            for k in LIVE_LAPS:
                if k + 1 >= len(c):
                    continue
                a, b = c[k], c[k + 1]
                L = b - a
                base_tau, base_ref, _ = base[k]
                tau, s = warp(speed, L)
                seed = 2 if name.startswith("1x") else 0
                wt, wr, wc = run_lap(model, clip_len, ref, times, T, d, t, a, b, tau, s, seed=seed)
                s_at = np.interp(wt, tau, s)  # original time on screen at each tick
                expected = np.interp(s_at, base_tau, base_ref)  # what the 1x run read at that frame
                e = circ(wr - expected, T)
                inside = (s_at > 1.0) & (s_at < L - 1.0)
                errs.append(e[inside])
                shown.append(float(np.mean(wc >= 0.5)))
                # at the line: the last ticks' delta against the 1x run's plus the time the warp added
                d_warp = np.median(circ(wt[-8:] - wr[-8:], T))
                d_base = np.median(circ(base_tau[-8:] - base_ref[-8:], T))
                ends.append(float(d_warp - (d_base + (tau[-1] - s[-1]))))
                if k == LIVE_LAPS[0]:
                    curves[name] = (wt, circ(wt - wr, T), circ(wt - expected, T), np.interp(wt, tau, tau - s))
            e = np.abs(np.concatenate(errs))
            report[name] = {"median_ms": 1000 * float(np.median(e)), "p90_ms": 1000 * float(np.percentile(e, 90)),
                            "over_100ms": float(np.mean(e > 0.1)), "at_line_ms": [1000 * x for x in ends],
                            "shown": float(np.mean(shown))}
            print(f"  {name:<38} speed error median {report[name]['median_ms']:5.0f} ms, p90 {report[name]['p90_ms']:5.0f}, "
                  f"{100 * report[name]['over_100ms']:5.1f}% over 100 ms | at the line {np.round(ends, 0).astype(int).tolist()} ms "
                  f"| shown {100 * report[name]['shown']:3.0f}%", flush=True)
        tag = run + (f"_sensor{SENSOR:g}" if SENSOR is not None else "")
        (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
        pick = ["constant 0.75x", "constant 1.5x", "half 0.5x, half 1.5x", "half 0.75x, half 1.5x (same total)",
                "surges +-40%, 8 s period", "mistake: 0.3x for 3 s at 20 s"]
        fig, axes = plt.subplots(len(pick), 1, figsize=(11, 2.3 * len(pick)), sharex=False)
        for ax, name in zip(axes, pick):
            wt, delta, ideal, added = curves[name]
            ax.plot(wt, ideal, color="#222222", lw=2, label="exact (the 1x run's delta + the time the warp added)")
            ax.plot(wt, delta, color="#2c7fb8", lw=1.3, label="PRIMAL on the warped lap")
            ax.set_title(f"{name}: lap {LIVE_LAPS[0] + 1} against lap 1", fontsize=9)
            ax.set_ylabel("delta (s)")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("playback time into the lap (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}.png", dpi=105)


if __name__ == "__main__":
    main()
