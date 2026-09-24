"""
Progress estimator: a particle filter over (track position, speed) that turns
the aligner's per-tick belief into a tracked position.

The aligner answers each tick from scratch. On an unseen circuit it is usually
right and occasionally confidently wrong somewhere that looks alike; a tracker
that knows where the kart was a moment ago, and roughly how fast it was going,
rejects those jumps outright. It also averages the per-tick noise that a single
answer carries.

The belief is consumed exactly as the phone would receive it: one distribution
over reference bins per tick, plus the time since the previous tick. Bins map
to track positions through the reference grid, so either reference axis works.

No learning happens here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EstimatorConfig:
    particles: int = 2000
    # Ticks are correlated and the softmax is overconfident, so each belief is
    # tempered before it multiplies in.
    likelihood_power: float = 0.8
    # Share of each tick's likelihood spread evenly over the lap. A confidently
    # wrong tick then only dents the particles at the true place.
    outlier_floor: float = 0.1
    # Random walk on speed, m/s per sqrt(s): room to brake and accelerate.
    accel_noise: float = 16.0
    # Extra position noise beyond speed x time, m per sqrt(s).
    position_noise: float = 1.0
    # Share of particles redrawn from the current belief every tick, so a wrong
    # lock is abandoned once the evidence consistently points elsewhere.
    reinject: float = 0.02
    # Their starting weight relative to the average: the prior that the filter
    # is lost. Redrawn particles land on the belief's peak, so at full weight a
    # single confidently wrong tick would outweigh the true place.
    reinject_weight: float = 1e-6
    speed_range: tuple[float, float] = (2.0, 90.0)
    # Particles within this distance of the densest place form the estimate.
    cluster_m: float = 15.0


@dataclass
class Estimate:
    position_m: float
    speed_mps: float
    # Share of particle weight in the reported cluster: low means the filter
    # is split between places and the readout should abstain.
    confidence: float
    spread_m: float


class ProgressEstimator:
    def __init__(
        self,
        bin_position_m: np.ndarray,
        track_length_m: float,
        config: EstimatorConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or EstimatorConfig()
        self.length = float(track_length_m)
        n = int(np.asarray(bin_position_m).size)
        self.n_bins = n
        # Bin coordinate <-> position, closed at the lap end for interpolation.
        self._pos = np.concatenate([np.asarray(bin_position_m, dtype=np.float64), [self.length]])
        self._bins = np.arange(n + 1, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.position: np.ndarray | None = None
        self.speed: np.ndarray | None = None
        self.weight: np.ndarray | None = None

    def bin_of(self, position: np.ndarray) -> np.ndarray:
        """Reference bin coordinate of a track position: what a delta lookup needs."""
        return np.interp(np.asarray(position) % self.length, self._pos, self._bins)

    def _to_position(self, bins: np.ndarray) -> np.ndarray:
        return np.interp(bins % self.n_bins, self._bins, self._pos)

    def _likelihood(self, belief: np.ndarray, position: np.ndarray) -> np.ndarray:
        x = self.bin_of(position)
        k0 = np.floor(x).astype(np.int64) % self.n_bins
        frac = x - np.floor(x)
        p = (1.0 - frac) * belief[k0] + frac * belief[(k0 + 1) % self.n_bins]
        floor = self.config.outlier_floor / self.n_bins
        return ((1.0 - self.config.outlier_floor) * p + floor) ** self.config.likelihood_power

    def _sample_from(self, belief: np.ndarray, count: int) -> np.ndarray:
        bins = self.rng.choice(self.n_bins, size=count, p=belief) + self.rng.random(count)
        return self._to_position(bins)

    def step(
        self,
        belief: np.ndarray,
        dt: float,
        speed_obs: float | None = None,
        speed_sigma: float | None = None,
    ) -> Estimate:
        """Fold in one tick. `belief` sums to 1 over the reference bins."""
        c = self.config
        belief = np.asarray(belief, dtype=np.float64)
        belief = belief / belief.sum()
        lo, hi = c.speed_range

        if self.position is None:
            # Acquisition: every place the aligner considers possible, any speed.
            self.position = self._sample_from(belief, c.particles)
            self.speed = self.rng.uniform(lo, hi, c.particles)
            self.weight = np.full(c.particles, 1.0 / c.particles)
        else:
            dt = max(float(dt), 1e-3)
            self.speed = np.clip(
                self.speed + self.rng.normal(0.0, c.accel_noise * np.sqrt(dt), c.particles), lo, hi
            )
            self.position = (
                self.position
                + self.speed * dt
                + self.rng.normal(0.0, c.position_noise * np.sqrt(dt), c.particles)
            ) % self.length
            count = int(round(c.reinject * c.particles))
            if count:
                slot = self.rng.choice(c.particles, size=count, replace=False)
                self.position[slot] = self._sample_from(belief, count)
                mean_v = float(np.average(self.speed, weights=self.weight))
                self.speed[slot] = np.clip(self.rng.normal(mean_v, 8.0, count), lo, hi)
                self.weight[slot] = self.weight.mean() * c.reinject_weight

        self.weight = self.weight * self._likelihood(belief, self.position)
        if speed_obs is not None and speed_sigma:
            self.weight = self.weight * np.exp(-0.5 * ((self.speed - speed_obs) / speed_sigma) ** 2)
        total = self.weight.sum()
        if not np.isfinite(total) or total <= 0.0:
            self.weight = np.full(c.particles, 1.0 / c.particles)
        else:
            self.weight = self.weight / total

        estimate = self._estimate()
        if 1.0 / np.sum(self.weight**2) < c.particles / 2:
            self._resample()
        return estimate

    def _estimate(self) -> Estimate:
        c = self.config
        # Densest place first, then a weighted mean inside it: a plain mean
        # would land between two candidate places when the filter is split.
        width = max(c.cluster_m, 1.0)
        n_cells = max(int(self.length // width), 1)
        cell = (self.position / self.length * n_cells).astype(np.int64) % n_cells
        mass = np.bincount(cell, weights=self.weight, minlength=n_cells)
        mass = mass + np.roll(mass, 1) + np.roll(mass, -1)
        centre = (np.argmax(mass) + 0.5) * self.length / n_cells
        offset = (self.position - centre + self.length / 2) % self.length - self.length / 2
        inside = np.abs(offset) <= width * 1.5
        w = self.weight * inside
        share = float(w.sum())
        if share <= 0.0:
            return Estimate(centre, float(np.average(self.speed, weights=self.weight)), 0.0, width)
        mean_offset = float(np.sum(w * offset) / share)
        spread = float(np.sqrt(np.sum(w * (offset - mean_offset) ** 2) / share))
        return Estimate(
            position_m=(centre + mean_offset) % self.length,
            speed_mps=float(np.sum(w * self.speed) / share),
            confidence=share,
            spread_m=spread,
        )

    def _resample(self) -> None:
        n = self.config.particles
        edges = np.cumsum(self.weight)
        edges[-1] = 1.0
        picks = np.searchsorted(edges, (self.rng.random() + np.arange(n)) / n)
        self.position = self.position[picks]
        self.speed = self.speed[picks]
        self.weight = np.full(n, 1.0 / n)
