"""
The runtime's session logic, as a device app runs it after START: record a reference lap,
close it where the camera sees the start again, then time every lap against it live; a
faster lap that PRIMAL followed well becomes the reference.

This is the reference implementation the apps are checked against: train/release.py writes
SessionConfig into release.json and replays recorded sessions through Session for the
release's golden cases. It holds no device code; frames come in as descriptors with times.

  start(t)              the driver pressed START (at or before the line)
  push(descriptor, t)   every frame's descriptor (the model's encoder output), 1/30 s apart
  close_at(t)           the driver closed the reference by hand (when PRIMAL cannot confirm)
  -> events             Reading, ProposalRejected, ReferenceClosed, LapCompleted
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from train.live import LiveDelta


@dataclass(frozen=True)
class SessionConfig:
    bin_s: float = 0.052            # reference bins on the time axis
    shown_min: float = 0.5          # the delta is shown at or above this confidence
    # The reference's end proposed: the last loop_k frames against its first loop_k (mean cosine
    # of the descriptors) must first fall under loop_away, then rise over loop_back after
    # loop_min_s; the proposal is that rise's peak (it ends when the score falls loop_peak_drop
    # below it, or loop_peak_wait_s after it), the line the first frame of the peak's clip.
    loop_k: int = 12
    loop_away: float = 0.70
    loop_back: float = 0.85
    loop_min_s: float = 5.0
    loop_peak_drop: float = 0.05
    loop_peak_wait_s: float = 0.5
    # The proposal confirmed by PRIMAL: an engine on the reference cut at the proposal reads the
    # frames since check_lead_s before it, with no hint. Each reading at or above check_conf whose
    # reference time is over check_ref_min_s and under (its time since the proposal) +
    # check_ref_margin_s puts the line at its time less its reference time; when check_agree of
    # the last check_window agree within check_mad_s (median absolute deviation), their median is
    # the line. Given up check_verify_s after the proposal.
    check_lead_s: float = 1.0
    check_conf: float = 0.8
    check_ref_min_s: float = 0.2
    check_ref_margin_s: float = 2.0
    check_window: int = 30
    check_agree: int = 15
    check_mad_s: float = 0.15
    # 12 s, not 8: a reference lap that began slowly (RaceChrono's lap 1) is only read confidently
    # ~7 s after the line, which left the second chance below a second of readings at 8 s: Python
    # passed by a hair and the Swift port (other random draws) failed. At 12 s both pass, every seed.
    check_verify_s: float = 12.0
    # At give-up, a second chance: when the reference lap's start ran at another pace (it began
    # slowly, out of the pits), the readings' lines are skewed by that difference, up to
    # check_pace_s before the proposal. If check_agree of the last check_window such readings
    # agree within check_pace_mad_s, PRIMAL is following the reference's start consistently: the
    # closure stands, its line the proposal's (the look-alike's).
    check_pace_s: float = 6.0
    check_pace_mad_s: float = 0.3
    # A line crossing: the tracker's reference time falls from over lap_wrap_high of the lap to
    # under lap_wrap_low; the crossing is that tick's time less its reference time.
    lap_wrap_high: float = 0.75
    lap_wrap_low: float = 0.25
    # ... and only once the lap has run at least this share of the reference lap: just past the
    # line the view is the reference's end as much as its start, so the tracker's first readings
    # can sit either side of the line, which is no crossing.
    lap_min_share: float = 0.5
    # A lap becomes the reference when it is faster, not voided and its delta was shown on at least
    # swap_shown_min of its ticks: 0 since 2026-10-17 (the app's rule: any faster lap), because the
    # share is measured against the current reference, so one with a broken stretch (a spin, traffic)
    # failed every later lap there and was never replaced; the first lap is the likeliest to be messy.
    swap_shown_min: float = 0.0
    # ... and when the lap is at least this share of the session's median lap so far (the reference
    # lap's and every lap's not voided): a "lap" much shorter is a misplaced line, e.g. a pit in-lap
    # read as 34.9 s of ~48.3 (experiments/swap_rule_check.py: 0.9 stops it and no true lap on any
    # session; 0.95 stops three true laps).
    swap_min_of_median: float = 0.9
    # Frames missing for longer than this (s): the lap's time is void and the tracker searches afresh.
    gap_s: float = 10.0


@dataclass
class Reading:
    t: float
    lap: int                 # 0 the first lap after the reference
    ref_time_s: float
    confidence: float
    single_ref_time_s: float
    delta_s: float           # negative: ahead of the reference
    shown: bool


@dataclass
class ProposalRejected:
    t: float
    proposal_s: float


@dataclass
class ReferenceClosed:
    t: float                 # when it was known
    line_s: float            # where the line is, on the session's clock
    lap_s: float             # the reference lap's time
    placed_by: str           # "primal" (its readings), "look-alike" (the second chance) or "hand"


@dataclass
class LapCompleted:
    t: float
    lap: int
    line_s: float            # the crossing that ended it
    lap_s: float
    shown_share: float       # of its ticks, how many showed the delta
    voided: bool             # a stream gap: its time does not count
    became_reference: bool


def build_reference(frames: list[tuple[float, torch.Tensor]], start: float, line: float, bin_s: float
                    ) -> tuple[np.ndarray, torch.Tensor, float]:
    """The reference from `start` to `line`: round(lap / bin_s) bins, each the frame nearest its time."""
    t = np.array([s for s, _ in frames])
    lap = line - start
    n = int(round(lap / bin_s))
    pick = [int(np.argmin(np.abs(t - (start + i * lap / n)))) for i in range(n)]
    return np.arange(n) * lap / n, torch.stack([frames[i][1] for i in pick]), lap


class LoopCloser:
    """Proposes where the reference lap ends: its start view seen again (see SessionConfig)."""

    def __init__(self, config: SessionConfig) -> None:
        self.c = config
        self.start: torch.Tensor | None = None
        self.away, self.peak = False, None  # peak: (score, line time, time seen)

    def reject(self) -> None:
        """A proposal PRIMAL did not confirm: the view must move away again before the next."""
        self.away, self.peak = False, None

    def push(self, frames: list[tuple[float, torch.Tensor]]) -> float | None:
        c, k = self.c, self.c.loop_k
        if len(frames) < 2 * k:
            return None
        if self.start is None:
            self.start = torch.nn.functional.normalize(torch.stack([d for _, d in frames[:k]]), dim=1)
        last = torch.nn.functional.normalize(torch.stack([d for _, d in frames[-k:]]), dim=1)
        score = float((last * self.start).sum(1).mean())
        t, since = frames[-1][0], frames[-1][0] - frames[0][0]
        self.away = self.away or (since > c.loop_min_s / 2 and score < c.loop_away)
        if self.away and since >= c.loop_min_s and score >= c.loop_back and (self.peak is None or score > self.peak[0]):
            self.peak = (score, frames[-k][0], t)
        if self.peak is not None and (score < self.peak[0] - c.loop_peak_drop or t - self.peak[2] > c.loop_peak_wait_s):
            return self.peak[1]
        return None


class ClosureCheck:
    """PRIMAL's check of a proposed end, and where it puts the line (see SessionConfig)."""

    def __init__(self, engine: LiveDelta, proposal: float, frames: list[tuple[float, torch.Tensor]], config: SessionConfig) -> None:
        self.c, self.engine, self.proposal = config, engine, proposal
        self.lines: list[float] = []   # lines from readings that agree with the proposal's pace
        self.paced: list[float] = []   # lines from readings up to check_pace_s earlier (a slow start)
        for s, d in frames:
            if s >= proposal - config.check_lead_s:
                self.push(d, s)

    def push(self, d: torch.Tensor, t: float) -> None:
        c, r = self.c, self.engine.push_descriptor(d, t)
        if r is None or r.confidence < c.check_conf or r.ref_time_s <= c.check_ref_min_s:
            return
        line = t - r.ref_time_s  # where this reading puts the line
        if line > self.proposal - c.check_ref_margin_s:
            self.lines.append(line)
        if line > self.proposal - c.check_ref_margin_s - c.check_pace_s:
            self.paced.append(line)

    @staticmethod
    def _agreed(lines: list[float], window: int, agree: int, mad: float) -> float | None:
        x = np.array(lines[-window:])
        if len(x) >= agree and float(np.median(np.abs(x - np.median(x)))) < mad:
            return float(np.median(x))
        return None

    def line(self) -> float | None:
        """The line, once PRIMAL's readings agree on it."""
        return self._agreed(self.lines, self.c.check_window, self.c.check_agree, self.c.check_mad_s)

    def second_chance(self) -> float | None:
        """At give-up: the proposal's line, if PRIMAL follows the reference's start at a steady offset."""
        steady = self._agreed(self.paced, self.c.check_window, self.c.check_agree, self.c.check_pace_mad_s)
        return None if steady is None else self.proposal


class Session:
    def __init__(self, model, clip_len: int, device: torch.device, config: SessionConfig = SessionConfig(), seed: int = 0) -> None:
        self.model, self.clip_len, self.device, self.c, self.seed = model, clip_len, device, config, seed
        self.state = "ready"
        self.reference_lap_s: float | None = None
        self.laps: list[LapCompleted] = []

    # -- the driver's two buttons ----------------------------------------------------------

    def start(self, t: float) -> None:
        self.state, self.ref_start = "reference", t
        self.frames: list[tuple[float, torch.Tensor]] = []
        self.closer, self.check, self.last_t = LoopCloser(self.c), None, None

    def close_at(self, t: float) -> list:
        """The reference closed by hand at `t` (the driver at the line)."""
        return self._close(t, t, "hand") if self.state == "reference" else []

    # -- every frame -----------------------------------------------------------------------

    def push(self, d: torch.Tensor, t: float) -> list:
        if self.state == "reference":
            return self._record(d, t)
        if self.state == "live":
            return self._live(d, t)
        return []

    def _engine(self, frames, start: float, line: float) -> LiveDelta:
        times, descriptors, lap = build_reference(frames, start, line, self.c.bin_s)
        return LiveDelta(self.model, None, times, lap, self.clip_len, self.device, seed=self.seed, reference_descriptors=descriptors)

    def _record(self, d: torch.Tensor, t: float) -> list:
        self.frames.append((t, d))
        if self.check is None:
            proposal = self.closer.push(self.frames)
            if proposal is not None:
                self.check = ClosureCheck(self._engine(self.frames, self.ref_start, proposal), proposal, self.frames, self.c)
            return []
        self.check.push(d, t)
        line = self.check.line()
        if line is not None:
            return self._close(line, t, "primal")
        if t > self.check.proposal + self.c.check_verify_s:
            line = self.check.second_chance()
            if line is not None:
                return self._close(line, t, "look-alike")
            rejected = ProposalRejected(t, self.check.proposal)
            self.closer.reject()
            self.check = None
            return [rejected]
        return []

    def _close(self, line: float, t: float, placed_by: str) -> list:
        """The reference from START to `line`; the live engine starts at the line, caught up to `t`."""
        self.engine = self._engine(self.frames, self.ref_start, line)
        self.engine.start_at(line, 0.0)
        self.reference_lap_s = line - self.ref_start
        self.lap_times = [self.reference_lap_s]
        events: list = [ReferenceClosed(t, line, self.reference_lap_s, placed_by)]
        self.state, self.lap, self.lap_start = "live", 0, line
        self._new_lap_counters()
        self.last_ref, self.last_t = None, None
        caught_up, self.frames = [(s, d) for s, d in self.frames if s >= line], []
        for s, d in caught_up:
            events += self._live(d, s)
        return events

    def _new_lap_counters(self) -> None:
        self.lap_frames: list[tuple[float, torch.Tensor]] = []
        self.ticks = self.shown_ticks = 0
        self.voided = False

    def _live(self, d: torch.Tensor, t: float) -> list:
        c, events = self.c, []
        if self.last_t is not None and t - self.last_t > c.gap_s:
            self.voided = True  # the lap's time cannot be trusted; the tracker searches afresh
            self.engine.reset()
            self.last_ref = None
        self.last_t = t
        self.lap_frames.append((t, d))
        r = self.engine.push_descriptor(d, t)
        if r is None:
            return events
        lap_len = self.engine.lap_time
        crossing = t - r.ref_time_s
        if (self.last_ref is not None and self.last_ref > c.lap_wrap_high * lap_len and r.ref_time_s < c.lap_wrap_low * lap_len
                and crossing - self.lap_start >= c.lap_min_share * self.reference_lap_s):
            events += self._lap_ended(crossing, t)
        self.last_ref = r.ref_time_s
        delta = (t - self.lap_start) - r.ref_time_s
        shown = r.confidence >= c.shown_min
        self.ticks += 1
        self.shown_ticks += shown
        events.append(Reading(t, self.lap, r.ref_time_s, r.confidence, r.single_ref_time_s, delta, shown))
        return events

    def _lap_ended(self, crossing: float, t: float) -> list:
        lap_s = crossing - self.lap_start
        share = self.shown_ticks / max(self.ticks, 1)
        faster = self.reference_lap_s is not None and lap_s < self.reference_lap_s
        plausible = lap_s >= self.c.swap_min_of_median * float(np.median(self.lap_times))
        became = (not self.voided) and faster and plausible and share >= self.c.swap_shown_min
        if not self.voided:
            self.lap_times.append(lap_s)
        done = LapCompleted(t, self.lap, crossing, lap_s, share, self.voided, became)
        self.laps.append(done)
        if became:
            # the lap, line to line, is the new reference; its engine starts at the crossing
            self.engine = self._engine(self.lap_frames, self.lap_start, crossing)
            self.engine.start_at(crossing, 0.0)
            self.reference_lap_s = lap_s
            for s, d in self.lap_frames:
                if s >= crossing:
                    self.engine.push_descriptor(d, s)  # its clip history; their readings were given already
        self.lap += 1
        self.lap_start = crossing
        carried = [(s, d) for s, d in self.lap_frames if s >= crossing]
        self._new_lap_counters()
        self.lap_frames = carried
        return [done]
