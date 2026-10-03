"""
Drive AC and watch PRIMAL's camera-only delta live, beside the true delta and AC's own.

    python -m capture.live_delta --reference auto     # a packed lap of this track (and car, if there is one)
    python -m capture.live_delta --reference live     # your first clean lap becomes the reference
    python -m capture.live_delta --reference LAP_ID   # a packed lap from --data
    python -m capture.live_delta --reference DIR      # a lap an earlier `live` run saved

    # no AC: replay a recorded session through the same loop
    python -m capture.live_delta --source data/sessions/<session>/video.mp4 --reference LAP_ID --no-window

Frames come from OBS's Virtual Camera (click Start Virtual Camera in OBS). Its scene is
the Game Capture of AC the recordings were made from, so the model sees the same
pixels, without H.264. Each frame is cropped and resized the way `capture.pack` does,
stamped with a monotonic clock, and handed to `train.live.LiveDelta` on a worker thread
capped at two CPU threads, so AC keeps the other cores.

The truth is the timecode barcode in the bottom band: the rig camera's exact track
position on every frame. AC's shared memory adds the lap count, speed and AC's own
delta to its best lap (performance_meter); it also stands in for the barcode, shifted
by the rig's 2.2 m, on frames where the barcode can't be read.

The lap clock starts when the camera crosses the line, which is where the reference's
own time starts (AC's lap timer starts when the car's centre does, ~2.2 m later; it is
used until the camera has crossed once). True delta = lap elapsed - the reference's
time at the camera's position; PRIMAL delta = lap elapsed - Reading.ref_time_s. Both
use one clock, so their difference is PRIMAL's position error alone.

In the overlay window: q quits, r records a new reference from the next lap.
Every tick goes to ticks.csv in D:\\Documents\\Transfer\\live_logs\\<run>\\.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from capture.ac_shm import STATUS_LIVE, STATUS_REPLAY, AcSharedMemory, AcSnapshot
from capture.pack import crop_rows
from capture.timecode import HIGH_REF_CELLS, LOW_REF_CELLS, OverlayGeometry, decode_frame, sample_cells
from train.dataset import Lap, LapIndex, ReferenceGrid, _speed_from_labels, build_reference_grid
from train.live import LiveDelta, Reading

GEOMETRY = OverlayGeometry(x0=458.0, y0=692.0, cell=14.0)  # every recorded session calibrated to this
SOURCE_SIZE = (1280, 720)
HEIGHT = 80
REF_SPACING_M = 2.0  # capture.pack's
RIG_FORWARD_M = 2.2  # the rig camera sits this far ahead of the car's centre (rig.txt forward_m)
STALE_S = 0.5        # frames older than this when the worker gets to them are not encoded
# A recorded reference runs this far past the line at both ends, so the line falls inside
# the data: extrapolating to it from the lap's first or last frame can be badly off when
# the speed there reads near zero (a repeated frame).
LINE_PAD_S = 0.5
LOG_DIR = Path(r"D:\Documents\Transfer\live_logs")
WINDOW = "PRIMAL live delta"


@dataclass
class Frame:
    t: float                  # monotonic seconds at grab
    small: np.ndarray         # (80, 148, 3) BGR, as packed
    cam_s: float              # the camera's track position, NaN if unknown
    s_source: str             # "barcode", "shm" or ""
    snap: AcSnapshot | None


class ObsCamera:
    """OBS's Virtual Camera through DirectShow: OBS's own canvas, uncompressed."""

    live = True

    def __init__(self, index: int) -> None:
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit("OBS Virtual Camera is not running: in OBS, click Start Virtual Camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SOURCE_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SOURCE_SIZE[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)

    def read(self) -> tuple[np.ndarray | None, float]:
        ok, frame = self.cap.read()
        return (frame, time.monotonic()) if ok else (None, 0.0)

    def close(self) -> None:
        self.cap.release()


class VideoFile:
    """A recording played through the loop; `realtime` paces it at its frame rate."""

    def __init__(self, path: Path, realtime: bool) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.live, self.i = realtime, 0
        self.t0 = time.monotonic() if realtime else 0.0

    def read(self) -> tuple[np.ndarray | None, float]:
        ok, frame = self.cap.read()
        if not ok:
            return None, 0.0
        t = self.t0 + self.i / self.fps
        self.i += 1
        if self.live and t > time.monotonic():
            time.sleep(t - time.monotonic())
        return frame, t

    def close(self) -> None:
        self.cap.release()


class Preprocess:
    """capture.pack's crop and resize, and the barcode in the band the crop removes."""

    def __init__(self, frame_size: tuple[int, int]) -> None:
        self.lo, self.hi = crop_rows(SOURCE_SIZE[1], GEOMETRY)
        width = int(round(HEIGHT * SOURCE_SIZE[0] / (self.hi - self.lo) / 2)) * 2
        self.size = (width, HEIGHT)
        if sorted(self.size) != sorted(frame_size):
            raise SystemExit(f"the model expects {frame_size} frames; this crop gives {self.size}")
        self.band = OverlayGeometry(GEOMETRY.x0, GEOMETRY.y0 - self.hi, GEOMETRY.cell)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, int, float, bool]:
        if frame.shape[:2] != SOURCE_SIZE[::-1]:
            raise SystemExit(f"source frames are {frame.shape[1]}x{frame.shape[0]}; set OBS's output to 1280x720")
        small = cv2.resize(frame[self.lo:self.hi], self.size, interpolation=cv2.INTER_AREA)
        counter, spline, ok = decode_frame(cv2.cvtColor(frame[self.hi:], cv2.COLOR_BGR2GRAY), self.band)
        return small, counter, spline, ok

    def levels(self, frame: np.ndarray) -> tuple[float, float]:
        """The barcode's black and white cells: decoded recordings read about 0 and 254."""
        values = sample_cells(cv2.cvtColor(frame[self.hi:], cv2.COLOR_BGR2GRAY), self.band)
        return float(values[list(LOW_REF_CELLS)].mean()), float(values[list(HIGH_REF_CELLS)].mean())


@dataclass
class UiState:
    """What the overlay draws; the threads replace whole values, never mutate them."""

    row: dict | None = None
    preview: np.ndarray | None = None
    message: str = ""
    fps: float = 0.0
    drops: int = 0
    done: bool = False


def capture_loop(source, prep: Preprocess, shm: AcSharedMemory | None, out: queue.Queue, ui: UiState,
                 stop: threading.Event) -> None:
    count, since = 0, time.monotonic()
    levels: list[tuple[float, float]] = []
    track = None
    try:
        while not stop.is_set():
            frame, t = source.read()
            if frame is None:
                break
            small, _, spline, ok = prep(frame)
            snap = shm.snapshot() if shm is not None else None
            if snap is not None:
                key = track_key_of(snap)
                if track is None:
                    track = key
                elif key != track and snap.track:
                    ui.message = f"AC is now on {key}, not {track}: quit (q) and start the tool again"
                    ui.row = None
                    continue
            # Colour levels over a few seconds of driving, not the first frame: AC fades in.
            if ok and len(levels) < 180 and (snap is None or snap.speed_kmh > 5):
                levels.append(prep.levels(frame))
                if len(levels) == 180:
                    black, white = np.median(levels, axis=0)
                    print(f"barcode levels while driving: black {black:.0f}, white {white:.0f} (the recordings: 0 and 254)")
                    if black > 8 or white < 245:
                        ui.message = f"colours differ from the recordings (black {black:.0f}, white {white:.0f})"
            if ok:
                cam_s, s_source = spline, "barcode"
            elif snap is not None and snap.status in (STATUS_LIVE, STATUS_REPLAY) and snap.track_length_m > 0:
                cam_s, s_source = (snap.spline_pos + RIG_FORWARD_M / snap.track_length_m) % 1.0, "shm"
            else:
                cam_s, s_source = float("nan"), ""
            item = Frame(t, small, cam_s, s_source, snap)
            if source.live:
                try:
                    out.put_nowait(item)
                except queue.Full:
                    ui.drops += 1
            else:
                while not stop.is_set():
                    try:
                        out.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        pass
            ui.preview = small
            count += 1
            if t - since >= 1.0 or not source.live and count % 60 == 0:
                ui.fps, count, since = count / max(time.monotonic() - since, 1e-6), 0, time.monotonic()
    finally:
        out.put(None)


@dataclass
class Recording:
    """One lap's frames, line to line plus LINE_PAD_S either side, on its way to being a reference."""

    ends_at: float | None = None
    frames: list = field(default_factory=list)
    t: list = field(default_factory=list)
    s: list = field(default_factory=list)
    barcode: list = field(default_factory=list)
    pit: list = field(default_factory=list)

    def add(self, f: Frame) -> None:
        if np.isfinite(f.cam_s):
            self.frames.append(f.small)
            self.t.append(f.t)
            self.s.append(f.cam_s)
            self.barcode.append(f.s_source == "barcode")
            self.pit.append(bool(f.snap.is_in_pit) if f.snap else False)

    def problem(self, track_length_m: float) -> str | None:
        """Why this lap can't be a reference, or None."""
        if len(self.t) < 600:
            return "too short"
        t, s = np.asarray(self.t), np.asarray(self.s, dtype=np.float64)
        if np.mean(self.barcode) < 0.95:
            return f"barcode read on only {np.mean(self.barcode):.0%} of frames"
        if any(self.pit):
            return "pit lane"
        if np.diff(t).max() > 0.1:
            return f"a {np.diff(t).max():.2f} s hole in the frames"
        x = np.unwrap(s * 2 * np.pi) / (2 * np.pi)
        if x[-1] - x[0] < 1.0:
            return "not a whole lap"
        if (np.diff(x) * track_length_m < -5.0).any():
            return "went backwards"
        if np.diff(np.sort(s)).max() * track_length_m > 6.0:
            return "a gap along the track"
        return None


def wrap(d: float, period: float) -> float:
    return (d + period / 2) % period - period / 2


def lap_delta(elapsed: float, ref_time: float, period: float) -> float:
    """
    Lap elapsed minus a reference time in [0, period). Just past the line the reference
    time can still read the end of the last lap, so a reference time more than half a
    lap ahead of the clock belongs to it. Otherwise no wrapping: a lap may be any amount
    slower than the reference.
    """
    if ref_time - elapsed > period / 2:
        ref_time -= period
    return elapsed - ref_time


class Worker(threading.Thread):
    """Lap clock, truth, LiveDelta and the log, one frame at a time in grab order."""

    def __init__(self, model, clip_len: int, device: torch.device, frames: queue.Queue, ui: UiState,
                 run_dir: Path, track_length_m: float, live_source: bool) -> None:
        super().__init__(daemon=True)
        self.model, self.clip_len, self.device = model, clip_len, device
        self.frames, self.ui, self.run_dir = frames, ui, run_dir
        self.track_length = track_length_m
        self.live_source = live_source
        self.engine: LiveDelta | None = None
        self.grid: ReferenceGrid | None = None
        self.ref_name = ""
        self.want_reference = False
        self.recording: Recording | None = None
        self.recent: deque[Frame] = deque(maxlen=120)
        self.lap_start: float | None = None
        self.crossings = 0
        self.prev: tuple[float, float] | None = None
        self.rows: list[dict] = []
        self.stale = 0
        self.log = open(run_dir / "ticks.csv", "w", newline="")
        self.writer: csv.DictWriter | None = None

    def use(self, engine: LiveDelta, grid: ReferenceGrid, name: str) -> None:
        self.engine, self.grid, self.ref_name = engine, grid, name
        self.ui.message = f"reference: {name} ({grid.lap_time_s:.2f} s)"

    def request_reference(self) -> None:
        self.want_reference = True
        self.ui.message = "reference: recording your next lap from the line - drive it clean"

    def run(self) -> None:
        try:
            while True:
                f = self.frames.get()
                if f is None:
                    break
                self.step(f)
        finally:
            self.log.close()
            self.ui.done = True

    def step(self, f: Frame) -> None:
        crossed = None
        if np.isfinite(f.cam_s):
            if self.prev is not None and self.prev[1] > 0.9 and f.cam_s < 0.1 and f.t - self.prev[0] < 0.5:
                frac = (1.0 - self.prev[1]) / (f.cam_s + 1.0 - self.prev[1])
                crossed = self.prev[0] + frac * (f.t - self.prev[0])
            self.prev = (f.t, f.cam_s)
        if crossed is not None:
            self.lap_start = crossed
            self.crossings += 1
            if self.recording is not None and self.recording.ends_at is None:
                self.recording.ends_at = crossed + LINE_PAD_S
            elif self.want_reference and self.recording is None:
                self.start_recording(crossed)
        self.recent.append(f)
        if self.recording is not None:
            self.recording.add(f)
            if self.recording.ends_at is not None and f.t >= self.recording.ends_at:
                self.finish_recording()
        if self.engine is None:
            return
        if self.live_source and time.monotonic() - f.t > STALE_S:
            self.stale += 1
            return
        reading = self.engine.push(f.small, f.t)
        if reading is not None:
            self.record(reading, f, time.monotonic())

    def start_recording(self, line_t: float) -> None:
        self.recording = Recording()
        for f in self.recent:
            if f.t >= line_t - LINE_PAD_S:
                self.recording.add(f)
        self.ui.message = "reference: recording this lap - drive it clean"

    def finish_recording(self) -> None:
        lap, self.recording = self.recording, None
        problem = lap.problem(self.track_length)
        if problem:
            self.ui.message = f"reference: that lap won't do ({problem}); recording the next one"
            self.start_recording(lap.ends_at - LINE_PAD_S)
            return
        frames, t, s = np.stack(lap.frames), np.asarray(lap.t), np.asarray(lap.s, dtype=np.float64)
        out = self.run_dir / f"reference_{self.crossings:02d}"
        out.mkdir()
        np.save(out / "frames.npy", frames)
        np.save(out / "t.npy", t)
        np.save(out / "s.npy", s)
        (out / "meta.json").write_text(json.dumps({"track_length_m": self.track_length}))
        self.ui.message = "reference: encoding the lap you just drove..."
        engine, grid = engine_from_arrays(self.model, frames, s, t, self.track_length, self.clip_len, self.device)
        self.want_reference = False
        self.use(engine, grid, f"your lap ({out.name})")

    def record(self, r: Reading, f: Frame, now: float) -> None:
        snap, grid = f.snap, self.grid
        period = grid.lap_time_s
        if self.lap_start is not None:
            elapsed, clock = r.t - self.lap_start, "camera"
        elif snap is not None and snap.lap_time_ms > 0:
            elapsed, clock = snap.lap_time_ms / 1000.0, "ac"
        else:
            elapsed, clock = float("nan"), ""
        true_ref = float(np.interp(f.cam_s, grid.s_knots, grid.tau_knots)) if np.isfinite(f.cam_s) else float("nan")
        row = {
            "wall_time": round(time.time(), 3),
            "t": round(r.t, 4),
            "lap": snap.completed_laps if snap else self.crossings,
            "lap_elapsed_s": round(elapsed, 4),
            "clock": clock,
            "ac_lap_time_s": snap.lap_time_ms / 1000.0 if snap else float("nan"),
            "ref_time_s": round(r.ref_time_s, 4),
            "single_ref_time_s": round(r.single_ref_time_s, 4),
            "confidence": round(r.confidence, 4),
            "true_ref_time_s": round(true_ref, 4),
            "primal_delta_s": round(lap_delta(elapsed, r.ref_time_s, period), 4),
            "true_delta_s": round(lap_delta(elapsed, true_ref, period), 4),
            "error_ms": round(1000 * wrap(r.ref_time_s - true_ref, period), 1),
            "performance_meter": snap.performance_meter if snap else float("nan"),
            "spline_pos": round(snap.spline_pos, 6) if snap else float("nan"),
            "cam_s": round(f.cam_s, 6),
            "s_source": f.s_source,
            "speed_kmh": round(snap.speed_kmh, 2) if snap else float("nan"),
            "delay_ms": round(1000 * (now - r.t), 1) if self.live_source else float("nan"),
            "queue": self.frames.qsize(),
            "reference_lap_s": round(period, 3),
            "reference": self.ref_name,
        }
        if self.writer is None:
            self.writer = csv.DictWriter(self.log, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        if len(self.rows) % 15 == 0:
            self.log.flush()
        self.rows.append(row)
        self.ui.row = row


def engine_from_arrays(model, frames: np.ndarray, s: np.ndarray, t: np.ndarray, track_length_m: float,
                       clip_len: int, device: torch.device) -> tuple[LiveDelta, ReferenceGrid]:
    """A reference from a lap recorded here, gridded the way packed laps are."""
    n_bins = max(int(round(track_length_m / REF_SPACING_M)), 8)
    grid = build_reference_grid(s, t, _speed_from_labels(s, t, track_length_m), n_bins, track_length_m, "time")
    engine = LiveDelta(model, frames[grid.frame_idx], grid.time_s, grid.lap_time_s, clip_len, device)
    return engine, grid


def packed_reference(data: Path, spec: str, track_key: str, car: str) -> Lap:
    laps = [lap for split in ("train", "holdout") for lap in LapIndex.load(data, split=split, variants=None).laps]
    if spec != "auto":
        found = [lap for lap in laps if lap.lap_id == spec]
        if not found:
            raise SystemExit(f"no lap {spec} in {data}")
        return found[0]
    here = [lap for lap in laps if lap.track == track_key and lap.usable_as_reference(0.98, "time")]
    if not here:
        raise SystemExit(f"no packed lap of {track_key} in {data}; use --reference live")
    same_car = [lap for lap in here if lap.car_model == car] or here
    # a typical lap: the median lap time among this car's
    times = [lap.reference_grid("time").lap_time_s for lap in same_car]
    return same_car[int(np.argsort(times)[len(times) // 2])]


def track_key_of(snap: AcSnapshot) -> str:
    return f"{snap.track}__{snap.track_config}" if snap.track_config else snap.track


def wait_for_ac(shm: AcSharedMemory) -> AcSnapshot:
    """
    A session that is running now. Shared memory outlives AC while anything holds it
    open (Content Manager does), so a mapping can still describe the last session:
    trust it only once the physics packets advance with the car on track.
    """
    print("waiting for AC to be on track...")
    last = None
    while True:
        snap = shm.snapshot()
        if (snap is not None and last is not None and snap.status == STATUS_LIVE and snap.track
                and snap.track_length_m > 0 and snap.packet_id != last.packet_id):
            time.sleep(1.0)
            again = shm.snapshot()
            if again is not None and track_key_of(again) == track_key_of(snap) and again.packet_id != snap.packet_id:
                return again
        last = snap
        time.sleep(0.5)


def fmt(d: float, unit: str = "") -> str:
    return "--" if not np.isfinite(d) else f"{d:+.2f}{unit}"


def tone(d: float) -> tuple[int, int, int]:
    if not np.isfinite(d):
        return (150, 150, 150)
    return (90, 210, 90) if d < 0 else (80, 80, 235)


def render(ui: UiState) -> np.ndarray:
    w, h = 660, 272
    img = np.full((h, w, 3), 22, np.uint8)
    text = cv2.putText
    font, bold = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
    row = ui.row
    if ui.preview is not None:
        img[12:172, w - 308:w - 12] = cv2.resize(ui.preview, (296, 160), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (w - 309, 11), (w - 12, 172), (90, 90, 90), 1)
    text(img, "PRIMAL delta", (14, 30), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if row is None:
        text(img, "--", (14, 106), bold, 2.4, (150, 150, 150), 4, cv2.LINE_AA)
    else:
        p, tr, ac = row["primal_delta_s"], row["true_delta_s"], row["performance_meter"]
        if row["clock"] != "camera":  # the lap clock starts at the line; AC's out-lap timer is no lap
            p = tr = float("nan")
            text(img, "deltas start when you cross the line", (14, 130), font, 0.5, (90, 200, 230), 1, cv2.LINE_AA)
        text(img, fmt(p), (14, 106), bold, 2.4, tone(p), 4, cv2.LINE_AA)
        text(img, f"true   {fmt(tr)}", (16, 146), font, 0.75, tone(tr), 2, cv2.LINE_AA)
        text(img, f"AC     {fmt(ac)}", (16, 176), font, 0.75, tone(ac), 2, cv2.LINE_AA)
        err = row["error_ms"]
        text(img, f"error {'--' if not np.isfinite(err) else f'{err:+.0f} ms'}   conf {row['confidence']:.2f}",
             (16, 204), font, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        # the reference lap as a bar: where PRIMAL puts you (cyan) and where you are (white)
        x0, x1, y0, y1 = 14, w - 14, 218, 236
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), -1)
        period = row["reference_lap_s"]
        primal_x = int(x0 + (x1 - x0) * (row["ref_time_s"] % period) / period)
        level = int(80 + 175 * min(max(row["confidence"], 0.0), 1.0))
        cv2.rectangle(img, (primal_x - 4, y0 - 3), (primal_x + 4, y1 + 3), (level, level, 0), -1)
        if np.isfinite(row["true_ref_time_s"]):
            true_x = int(x0 + (x1 - x0) * (row["true_ref_time_s"] % period) / period)
            cv2.line(img, (true_x, y0 - 6), (true_x, y1 + 6), (255, 255, 255), 2)
        delay = row["delay_ms"]
        status = f"lap {row['lap']}  {ui.fps:.0f} fps  {'' if not np.isfinite(delay) else f'delay {delay:.0f} ms'}"
        text(img, status, (w - 308, 194), font, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    if ui.drops:
        text(img, f"dropped {ui.drops}", (w - 120, 194), font, 0.45, (80, 80, 235), 1, cv2.LINE_AA)
    text(img, ui.message[:90], (14, 258), font, 0.45, (90, 200, 230), 1, cv2.LINE_AA)
    return img


def summarize(rows: list[dict], worker: Worker, ui: UiState) -> dict:
    out: dict = {"ticks": len(rows), "frames_dropped": ui.drops, "frames_stale": worker.stale}
    if not rows:
        return out
    delay = np.array([r["delay_ms"] for r in rows], dtype=float)
    if np.isfinite(delay).any():
        out["delay_ms"] = {"median": float(np.nanmedian(delay)), "p95": float(np.nanpercentile(delay, 95)),
                           "max": float(np.nanmax(delay))}
    good = [r for r in rows if r["s_source"] == "barcode" and np.isfinite(r["error_ms"])]
    if good:
        err = np.abs([r["error_ms"] for r in good])
        out["error_ms"] = {"ticks": len(good), "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                           "over_100ms_pct": float(100 * np.mean(err > 100)), "over_500ms_pct": float(100 * np.mean(err > 500))}
        laps = sorted({r["lap"] for r in good})
        out["per_lap"] = {str(lap): {"ticks": len(e), "median_ms": float(np.median(e)), "over_100ms_pct": float(100 * np.mean(e > 100))}
                          for lap in laps for e in [np.abs([r["error_ms"] for r in good if r["lap"] == lap])]}
    return out


def main() -> None:
    from train.eval import load_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="auto", help="auto, live, a packed lap_id, or a saved reference folder")
    parser.add_argument("--checkpoint", default="runs/v3_mobilenet_lr1_s0/best.pt")
    parser.add_argument("--data", default="data/packed_ac_v3")
    parser.add_argument("--source", default="obs", help="obs (the Virtual Camera) or a recording's video.mp4")
    parser.add_argument("--camera", type=int, default=0, help="DirectShow index of OBS Virtual Camera")
    parser.add_argument("--realtime", action="store_true", help="with a video source: play it at its frame rate")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for the model; AC needs the rest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--pos", default="1250,770", help="overlay window position x,y")
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    model, payload = load_model(Path(args.checkpoint), device)
    clip_len = int(payload["args"]["clip_len"])
    prep = Preprocess(tuple(payload["frame_size"]))

    if args.source == "obs":
        shm = AcSharedMemory()
        snap = wait_for_ac(shm)
        track_key = track_key_of(snap)
        car, track_length = snap.car_model, snap.track_length_m
        source = ObsCamera(args.camera)
    else:
        shm = None
        meta = json.loads((Path(args.source).parent / "run.json").read_text())
        track_key = f"{meta['track']}__{meta['track_config']}" if meta.get("track_config") else meta["track"]
        car, track_length = meta["car_model"], float(meta["track_length_m"])
        source = VideoFile(Path(args.source), args.realtime)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.log_dir) / f"{stamp}__{track_key}__{args.reference.replace('/', '_').replace(chr(92), '_')[:40]}"
    run_dir.mkdir(parents=True)
    ui = UiState()
    frames: queue.Queue = queue.Queue(maxsize=240)
    worker = Worker(model, clip_len, device, frames, ui, run_dir, track_length, source.live)

    start = time.perf_counter()
    if args.reference == "live":
        worker.request_reference()
        ref_desc = "live"
    elif Path(args.reference).is_dir():
        d = Path(args.reference)
        engine, grid = engine_from_arrays(model, np.load(d / "frames.npy"), np.load(d / "s.npy"), np.load(d / "t.npy"),
                                          json.loads((d / "meta.json").read_text())["track_length_m"], clip_len, device)
        worker.use(engine, grid, d.name)
        ref_desc = str(d)
    else:
        lap = packed_reference(Path(args.data), args.reference, track_key, car)
        if lap.track != track_key:
            raise SystemExit(f"{lap.lap_id} is {lap.track}, but you're on {track_key}")
        worker.use(LiveDelta.from_lap(model, payload, lap, device), lap.reference_grid("time"), lap.lap_id)
        ref_desc = lap.lap_id
        if lap.car_model != car:
            ui.message += f"  (its car: {lap.car_model})"
    print(f"{track_key}, {car}; reference {worker.ref_name or 'from your first clean lap'} "
          f"({time.perf_counter() - start:.1f} s to encode); logging to {run_dir}")
    (run_dir / "meta.json").write_text(json.dumps({
        "started": stamp, "track": track_key, "car": car, "track_length_m": track_length, "reference": ref_desc,
        "checkpoint": args.checkpoint, "source": args.source, "threads": args.threads, "device": args.device,
        "crop_rows": [prep.lo, prep.hi], "frame_size": list(prep.size)}, indent=2))

    stop = threading.Event()
    grabber = threading.Thread(target=capture_loop, args=(source, prep, shm, frames, ui, stop), daemon=True)
    worker.start()
    grabber.start()
    try:
        if args.no_window:
            while not ui.done:
                time.sleep(0.2)
        else:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            x, y = (int(v) for v in args.pos.split(","))
            cv2.moveWindow(WINDOW, x, y)
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
            while not ui.done:
                cv2.imshow(WINDOW, render(ui))
                key = cv2.waitKey(33) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    worker.request_reference()
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        grabber.join(timeout=2)
        try:  # unblock the worker if the grabber couldn't
            frames.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=10)
        source.close()
        if shm is not None:
            shm.close()
    summary = summarize(worker.rows, worker, ui)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
