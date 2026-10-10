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
AGREE_S = 0.25  # a single clip this close to the tracker agrees with it
# The two-mode reference-time tracker (train/eval.py, REFERENCE_TIME_TWO_MODES).
TRACKER = dict(accel_noise=0.01, likelihood_power=0.15, position_noise=0.01, speed_range=(0.0, 4.0),
               reinject_speed_sd=0.05, cluster_m=0.5, accel_noise_free=0.3, to_free_per_s=0.1, to_follow_per_s=1.0,
               free_speed_floor=0.0)


@dataclass
class Reading:
    t: float                   # live time of this tick, s
    ref_time_s: float          # the tracker: where in the reference lap the kart is
    confidence: float          # share of the last second's single-clip reads that agree with the tracker; low = unsure
    single_ref_time_s: float   # the same read off this clip alone, no tracker


@torch.no_grad()
def encode_frames(model: SequenceAligner, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """Descriptors of (N, H, W, 3) BGR uint8 frames, the same for reference and live frames."""
    return model.encode_reference(torch.from_numpy(_to_chw(np.asarray(frames))).to(device), use_checkpoint=False)


class LiveDelta:
    def __init__(self, model: SequenceAligner, reference_frames: np.ndarray | None, reference_times_s: np.ndarray,
                 lap_time_s: float, clip_len: int, device: torch.device, spacing_s: float = SPACING_S,
                 hz: float = STREAM_HZ, seed: int = 0, window: int = 8,
                 reference_descriptors: torch.Tensor | None = None) -> None:
        """
        The reference is either its frames, encoded here, or `reference_descriptors`
        already encoded by this model (one per bin), as a live loop that kept the
        descriptors of a lap it pushed can swap that lap in without encoding it again.
        """
        self.model, self.device, self.clip_len = model.eval(), device, clip_len
        self.spacing, self.period, self.window = spacing_s, 1.0 / hz, window
        self.times = np.asarray(reference_times_s, dtype=np.float64)
        self.lap_time = float(lap_time_s)
        if reference_descriptors is not None:
            self.ref = reference_descriptors.to(device)
        else:
            with torch.no_grad():
                self.ref = torch.cat([self._encode(reference_frames[i:i + 256])
                                      for i in range(0, len(reference_frames), 256)])
        self.scale = model.logit_scale.exp()
        self.seed = seed
        self.tracker = ProgressEstimator(self.times, self.lap_time, EstimatorConfig(**TRACKER), seed=seed)
        self.history: deque[tuple[float, torch.Tensor]] = deque()
        # Agreement over the last second: whether each tick's single clip lands within
        # AGREE_S of the tracker. On the live logs it picks out ticks more than 300 ms off
        # far better than the tracker's own cluster share (AUROC 0.83-0.97 against 0.54-0.62).
        self.agree: deque[bool] = deque(maxlen=max(int(round(hz)), 1))
        # Frames between the ones clips use are skipped, when ticks fall on the clip spacing.
        ratio = self.period / self.spacing
        self.skip = abs(ratio - round(ratio)) < 1e-3 and round(ratio) >= 1
        self.next_tick: float | None = None
        self.last_tick: float | None = None
        self.hint: tuple[float, float, float] | None = None

    @classmethod
    def from_lap(cls, model: SequenceAligner, payload: dict, reference: Lap, device: torch.device) -> "LiveDelta":
        grid = reference.reference_grid("time")
        return cls(model, np.asarray(reference.frames()[grid.frame_idx]), grid.time_s, grid.lap_time_s,
                   int(payload["args"]["clip_len"]), device)

    def reset(self) -> None:
        """Forget the live frames and the tracker's belief; keep the encoded reference."""
        self.tracker = ProgressEstimator(self.times, self.lap_time, EstimatorConfig(**TRACKER), seed=self.seed)
        self.history.clear()
        self.next_tick = self.last_tick = None
        self.agree.clear()
        self.hint = None

    def start_at(self, t: float, ref_time_s: float = 0.0, sd_s: float = 1.0) -> None:
        """
        Where the kart is known to be: at live time `t` it was at `ref_time_s` of the reference
        (at the line, 0, when a reference lap has just closed). The tracker's first step then
        starts there, moved on at the reference's pace, within `sd_s`, instead of searching
        the whole lap.
        """
        self.hint = (float(t), float(ref_time_s), float(sd_s))

    def _encode(self, frames: np.ndarray) -> torch.Tensor:
        return encode_frames(self.model, frames, self.device)

    def _ref_time(self, bins: float) -> float:
        cycle = np.r_[self.times, self.lap_time]
        return float(np.interp(bins % self.times.size, np.arange(self.times.size + 1), cycle))

    def wants(self, t: float) -> bool:
        """Whether a frame at `t` is one clips use (1/30 s apart); at 60 fps every other one is not."""
        return not (self.skip and self.history and t - self.history[-1][0] < self.spacing - 0.002)

    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> torch.Tensor:
        """One frame's descriptor, as the reference's are made."""
        return self._encode(frame[None])[0]

    @torch.no_grad()
    def push(self, frame: np.ndarray, t: float) -> Reading | None:
        """
        One camera frame and its time in seconds; a Reading on each 15 Hz tick, else None.
        Clips use frames 1/30 s apart, so at 60 fps every other frame is skipped unencoded.
        """
        if not self.wants(t):
            return None
        return self.push_descriptor(self.encode(frame), t)

    @torch.no_grad()
    def push_descriptor(self, descriptor: torch.Tensor, t: float) -> Reading | None:
        """`push`, for a frame the caller encoded (with `encode`) and wants kept."""
        self.history.append((t, descriptor))
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
        if self.hint is not None:
            t0, r0, sd = self.hint
            gap = (self.times - (r0 + t - t0) + self.lap_time / 2) % self.lap_time - self.lap_time / 2
            belief = belief * np.exp(-0.5 * (gap / sd) ** 2)
            belief /= belief.sum()
            self.hint = None
        single = float(soft_argmax_circular(logits, window=self.window)[0])
        dt = self.period if self.last_tick is None else t - self.last_tick
        self.last_tick = t
        estimate = self.tracker.step(belief, dt)
        ref_time, single_time = float(estimate.position_m) % self.lap_time, self._ref_time(single)
        gap = (single_time - ref_time + self.lap_time / 2) % self.lap_time - self.lap_time / 2
        self.agree.append(abs(gap) < AGREE_S)
        return Reading(t, ref_time, float(np.mean(self.agree)), single_time)


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
