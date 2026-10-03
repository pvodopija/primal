"""
The product loop, one frame at a time: what the glasses, or a live overlay on AC, run.

`LiveDelta` holds the reference lap's descriptors and a short history of live frame
descriptors. Push every camera frame with its time (BGR uint8, cropped and resized the
way packing does: 148x80 for the v3 models); at 15 Hz it reads a 12-frame clip spaced
1/30 s, scores it against the reference, and folds the belief into the two-mode
reference-time tracker, exactly as `train.eval stream` does offline. The answer is a
reference time: where in the reference lap the kart is, in the reference's own seconds.
The delta is the live lap's elapsed time minus that.

Replay a recorded lap through it, and compare with the offline evaluation:

    python -m train.live --checkpoint runs/v3_mobilenet_lr1_s0/best.pt \\
        --data data/packed_ac_v3 --reference LAP_ID --live LAP_ID
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train.dataset import Lap, LapIndex, _to_chw
from train.estimator import EstimatorConfig, ProgressEstimator
from train.model import SequenceAligner, soft_argmax_circular

STREAM_HZ = 15.0
SPACING_S = 1 / 30  # stride 2 at 60 fps
# The two-mode reference-time tracker (train/eval.py, REFERENCE_TIME_TWO_MODES).
TRACKER = dict(accel_noise=0.01, likelihood_power=0.15, position_noise=0.01, speed_range=(0.05, 1.6),
               reinject_speed_sd=0.05, cluster_m=0.5, accel_noise_free=0.3, to_free_per_s=0.1, to_follow_per_s=1.0)


@dataclass
class Reading:
    t: float                   # live time of this tick, s
    ref_time_s: float          # the tracker: where in the reference lap the kart is
    confidence: float          # share of the tracker's weight at that place; low = unsure
    single_ref_time_s: float   # the same read off this clip alone, no tracker


class LiveDelta:
    def __init__(self, model: SequenceAligner, reference_frames: np.ndarray, reference_times_s: np.ndarray,
                 lap_time_s: float, clip_len: int, device: torch.device, spacing_s: float = SPACING_S,
                 hz: float = STREAM_HZ, seed: int = 0, window: int = 8) -> None:
        self.model, self.device, self.clip_len = model.eval(), device, clip_len
        self.spacing, self.period, self.window = spacing_s, 1.0 / hz, window
        self.times = np.asarray(reference_times_s, dtype=np.float64)
        self.lap_time = float(lap_time_s)
        with torch.no_grad():
            self.ref = torch.cat([self._encode(reference_frames[i:i + 256]) for i in range(0, len(reference_frames), 256)])
        self.scale = model.logit_scale.exp()
        self.tracker = ProgressEstimator(self.times, self.lap_time, EstimatorConfig(**TRACKER), seed=seed)
        self.history: deque[tuple[float, torch.Tensor]] = deque()
        # Frames between the ones clips use are skipped, when ticks fall on the clip spacing.
        ratio = self.period / self.spacing
        self.skip = abs(ratio - round(ratio)) < 1e-3 and round(ratio) >= 1
        self.next_tick: float | None = None
        self.last_tick: float | None = None

    @classmethod
    def from_lap(cls, model: SequenceAligner, payload: dict, reference: Lap, device: torch.device) -> "LiveDelta":
        grid = reference.reference_grid("time")
        return cls(model, np.asarray(reference.frames()[grid.frame_idx]), grid.time_s, grid.lap_time_s,
                   int(payload["args"]["clip_len"]), device)

    def _encode(self, frames: np.ndarray) -> torch.Tensor:
        return self.model.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames))).to(self.device), use_checkpoint=False)

    def _ref_time(self, bins: float) -> float:
        cycle = np.r_[self.times, self.lap_time]
        return float(np.interp(bins % self.times.size, np.arange(self.times.size + 1), cycle))

    @torch.no_grad()
    def push(self, frame: np.ndarray, t: float) -> Reading | None:
        """
        One camera frame and its time in seconds; a Reading on each 15 Hz tick, else None.
        Clips use frames 1/30 s apart, so at 60 fps every other frame is skipped unencoded.
        """
        if self.skip and self.history and t - self.history[-1][0] < self.spacing - 0.002:
            return None
        self.history.append((t, self._encode(frame[None])[0]))
        span = (self.clip_len - 1) * self.spacing
        while self.history and self.history[0][0] < t - span - 0.25:
            self.history.popleft()
        if self.next_tick is None:
            self.next_tick = self.history[0][0] + span
        if t < self.next_tick - 0.002:  # 2 ms: recorded times are float32
            return None
        self.next_tick = max(self.next_tick + self.period, t + self.period / 2)
        stamps = np.array([s for s, _ in self.history])
        wanted = t - np.arange(self.clip_len - 1, -1, -1) * self.spacing
        pick = np.abs(stamps[None, :] - wanted[:, None]).argmin(1)
        clip = torch.stack([self.history[i][1] for i in pick])
        logits = self.model.head(torch.einsum("kd,nd->kn", clip, self.ref)[None] * self.scale)
        belief = logits.softmax(-1)[0].cpu().double().numpy()  # MPS has no float64
        single = float(soft_argmax_circular(logits, window=self.window)[0])
        dt = self.period if self.last_tick is None else t - self.last_tick
        self.last_tick = t
        estimate = self.tracker.step(belief, dt)
        return Reading(t, float(estimate.position_m) % self.lap_time, float(estimate.confidence), self._ref_time(single))


def main() -> None:
    from train.eval import REFERENCE_TIME_TWO_MODES, _lap_stream, _run_estimator, load_model

    assert TRACKER == REFERENCE_TIME_TWO_MODES, "train/live.py's tracker settings drifted from train/eval.py"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True, help="lap_id")
    parser.add_argument("--live", required=True, help="lap_id")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, payload = load_model(Path(args.checkpoint), device)
    laps = {lap.lap_id: lap for split in ("train", "holdout") for lap in LapIndex.load(Path(args.data), split=split, variants=None).laps}
    reference, live = laps[args.reference], laps[args.live]

    start = time.perf_counter()
    engine = LiveDelta.from_lap(model, payload, reference, device)
    print(f"reference encoded: {engine.times.size} bins in {time.perf_counter() - start:.1f} s")
    frames, t = live.frames(), live.t().astype(np.float64)
    readings, cost = [], []
    for i in range(live.n_frames):
        start = time.perf_counter()
        r = engine.push(np.asarray(frames[i]), float(t[i]))
        cost.append(time.perf_counter() - start)
        if r is not None:
            readings.append(r)
    grid = reference.reference_grid("time")
    true = np.interp(grid.target(np.interp([r.t for r in readings], t, live.s())), np.arange(grid.n_bins + 1),
                     np.r_[grid.time_s, grid.lap_time_s])
    err = np.abs(([r.ref_time_s for r in readings] - true + grid.lap_time_s / 2) % grid.lap_time_s - grid.lap_time_s / 2)[30:] * 1000
    print(f"live replay: {len(readings)} ticks, {1000 * np.mean(cost):.1f} ms a frame on {device} "
          f"(p99 {1000 * np.percentile(cost, 99):.1f}); error median {np.median(err):.0f} ms, >100 ms {100 * np.mean(err > 100):.1f}%")

    stream = _lap_stream(model, int(payload["args"]["clip_len"]), reference, live, "time", 2, device, 8)
    bins = _run_estimator(stream, EstimatorConfig(**REFERENCE_TIME_TWO_MODES), None, None, 0, reference_time=True).numpy()
    offline = np.interp(bins % grid.n_bins, np.arange(grid.n_bins + 1), np.r_[grid.time_s, grid.lap_time_s])
    # the offline evaluation counts frames, the live loop seconds: compare the ticks that fall at the same time
    live_t = np.array([r.t for r in readings])
    nearest = np.abs(stream["t"][:, None] - live_t[None, :]).argmin(0)
    same = np.abs(stream["t"][nearest] - live_t) < 0.005
    gap = np.abs((np.array([r.ref_time_s for r in readings])[same] - offline[nearest[same]] + grid.lap_time_s / 2)
                 % grid.lap_time_s - grid.lap_time_s / 2) * 1000
    print(f"against the offline evaluation, {same.sum()} ticks at the same time: median {np.median(gap):.1f} ms, "
          f"p90 {np.percentile(gap, 90):.1f} ms")


if __name__ == "__main__":
    main()
