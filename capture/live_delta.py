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

Two wheel buttons, picked in the overlay the first time a wheel is seen (read through
Windows' joystick API, whichever window has focus; kept in ~/.primal_live_wheel.json):
  RECORD  arms a reference lap, recorded from the line to the line by itself. Pressed up
          to 3 s after the line it starts from that line; around the end of the lap it
          does nothing (the lap stops at the line anyway); mid-lap it cancels.
  CAMERA  cycles the rig's camera presets (CAMERAS): the training view, wider, lower,
          higher, looking into corners, and AC's own cameras (the rig lets go, so AC's
          camera button works again). Each camera keeps its own reference; rig.txt is
          put back as it was on exit.
In the overlay window: q quits, r = RECORD, c = CAMERA, b picks the wheel buttons again.
Every tick goes to ticks.csv in D:\\Documents\\Transfer\\live_logs\\<run>\\.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
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
from capture.install_overlay import find_ac_root
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
WHEEL_FILE = Path.home() / ".primal_live_wheel.json"  # the wheel buttons you picked, per wheel
PRESS_WINDOW_S = 3.0  # RECORD pressed this soon after the line starts the lap from that line

# Camera presets the CAMERA button cycles through, written into the rig's rig.txt (it
# re-reads it twice a second). The first is the view the v3 models were trained on;
# "ac" switches the rig off and hands the camera back to AC, whose own camera button
# then works again. A reference only matches live laps from its own camera.
_TRAINING = dict(enabled=1, replay_only=0, lateral_m=0, wander_m=0, forward_m=2.2, height_m=1.15, fov_deg=60, look=0)
CAMERAS = {
    "training": ("training view", _TRAINING),
    "wide": ("wide, FOV 90", {**_TRAINING, "fov_deg": 90}),
    "low": ("low, 0.6 m", {**_TRAINING, "height_m": 0.6}),
    "high": ("high, 2 m", {**_TRAINING, "height_m": 2.0}),
    "look": ("looks into corners", {**_TRAINING, "look": 1}),
    "ac": ("AC's own cameras", {"enabled": 0}),
}
SETTLE_S = 1.0  # after a camera switch: the rig's re-read, and AC's camera settling
GAP_S = 0.5     # a longer hole in the frames (AC paused) restarts the tracker and the lap
SLOW_KMH = 20.0


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
    camera: str = ""
    paused: bool = False
    lap_flags: str = ""  # why the current lap won't count as clean, if it won't
    rec: str = ""        # "", "armed", "recording" or "encoding"
    prompt: str = ""     # a question that needs the wheel, drawn over the deltas


class Rig:
    """The camera rig's rig.txt: switch presets, and put the file back as it was on exit."""

    def __init__(self) -> None:
        root = find_ac_root()
        self.path = root / "apps" / "lua" / "primal_rig" / "rig.txt" if root else None
        if self.path is None or not self.path.exists():
            raise SystemExit("camera rig not installed (apps/lua/primal_rig/rig.txt)")
        self.original = self.path.read_text()

    def apply(self, values: dict) -> None:
        lines, seen = [], set()
        for line in self.path.read_text().splitlines():
            key = line.split("=")[0].strip()
            if key in values:
                line = f"{key} = {values[key]}"
                seen.add(key)
            lines.append(line)
        lines += [f"{k} = {v}" for k, v in values.items() if k not in seen]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, self.path)

    def restore(self) -> None:
        self.path.write_text(self.original)


class _JoyInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos",
                "dwUpos", "dwVpos", "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1", "dwReserved2")]


class _JoyCaps(ctypes.Structure):
    _fields_ = ([("wMid", ctypes.c_uint16), ("wPid", ctypes.c_uint16), ("szPname", ctypes.c_wchar * 32)]
                + [(f"u{i}", ctypes.c_uint32) for i in range(19)]
                + [("szRegKey", ctypes.c_wchar * 32), ("szOEMVxD", ctypes.c_wchar * 260)])


def button_name(b: int) -> str:
    return {101: "D-pad up", 102: "D-pad right", 103: "D-pad down", 104: "D-pad left"}.get(b, f"button {b}")


class Wheel(threading.Thread):
    """
    Wheel buttons through Windows' own joystick API (winmm; nothing to install), read
    whichever window has focus. D-pad directions count as buttons 101-104. The first
    time a wheel is seen it asks for the two buttons in the overlay, and remembers them.
    """

    ACTIONS = {"record": "RECORD REFERENCE", "camera": "CAMERA"}

    def __init__(self, on_press, ui: UiState) -> None:
        super().__init__(daemon=True)
        self.on_press, self.ui = on_press, ui
        self.winmm = ctypes.windll.winmm
        self.id, self.key = None, ""
        for i in range(16):
            caps = _JoyCaps()
            if self.winmm.joyGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps)) == 0 and self._read(i) is not None:
                self.id, self.key = i, f"{caps.wMid:04x}:{caps.wPid:04x}"
                break
        saved = json.loads(WHEEL_FILE.read_text()) if WHEEL_FILE.exists() else {}
        self.bindings: dict[str, int] = saved.get(self.key, {})
        self.learning: list[str] = []
        if self.id is not None and set(self.bindings) != set(self.ACTIONS):
            self.learn()

    @property
    def found(self) -> bool:
        return self.id is not None

    def _read(self, i: int) -> set[int] | None:
        info = _JoyInfo(dwSize=ctypes.sizeof(_JoyInfo), dwFlags=0xFF)
        if self.winmm.joyGetPosEx(i, ctypes.byref(info)) != 0:
            return None
        pressed = {b + 1 for b in range(32) if info.dwButtons >> b & 1}
        if info.dwPOV != 0xFFFF:
            pressed.add(101 + int(info.dwPOV) // 9000)
        return pressed

    def learn(self) -> None:
        self.bindings, self.learning = {}, list(self.ACTIONS)
        self._ask()

    def _ask(self) -> None:
        self.ui.prompt = f"Press the wheel button for {self.ACTIONS[self.learning[0]]}" if self.learning else ""

    def describe(self) -> str:
        return ", ".join(f"{self.ACTIONS[a]} = {button_name(b)}" for a, b in self.bindings.items())

    def run(self) -> None:
        before: set[int] = set()
        while True:
            now = self._read(self.id) or set()
            for b in sorted(now - before):
                if self.learning:
                    if b in self.bindings.values():
                        continue
                    self.bindings[self.learning.pop(0)] = b
                    self._ask()
                    if not self.learning:
                        saved = json.loads(WHEEL_FILE.read_text()) if WHEEL_FILE.exists() else {}
                        WHEEL_FILE.write_text(json.dumps({**saved, self.key: self.bindings}, indent=2))
                        self.ui.message = f"wheel: {self.describe()}  (b in this window: pick again)"
                else:
                    for action, bound in self.bindings.items():
                        if bound == b:
                            self.on_press(action)
            before = now
            time.sleep(0.015)


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
            if snap is not None and snap.status != STATUS_LIVE:
                ui.preview, ui.paused = small, True
                continue
            ui.paused = False
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
        # A flying lap crosses the line at about the speed it finishes at. One started
        # slowly (out of the pits, after a pause) teaches the tracker a pace no later
        # lap has there, and it lags seconds behind every lap after the line.
        x -= np.floor(np.median(x))
        start, end = (x >= 0) & (x < 0.03), (x >= 0.97) & (x < 1.0)
        if start.sum() > 1 and end.sum() > 1:
            v0 = 0.03 * track_length_m / (t[start].max() - t[start].min()) * 3.6
            v1 = 0.03 * track_length_m / (t[end].max() - t[end].min()) * 3.6
            if v0 < 0.75 * v1:
                return f"not a flying lap: crossed the line at {v0:.0f} km/h, finished at {v1:.0f}"
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
    """
    Lap clock, truth, LiveDelta and the log, one frame at a time in grab order. Each
    camera preset keeps its own reference; button presses arrive as commands and are
    handled between frames, at the time of the frame.
    """

    def __init__(self, model, clip_len: int, device: torch.device, frames: queue.Queue, ui: UiState,
                 run_dir: Path, track_length_m: float, live_source: bool, rig: Rig | None,
                 cameras: list[str] | None = None) -> None:
        super().__init__(daemon=True)
        self.model, self.clip_len, self.device = model, clip_len, device
        self.frames, self.ui, self.run_dir = frames, ui, run_dir
        self.track_length = track_length_m
        self.live_source = live_source
        self.rig = rig
        self.commands: queue.Queue[str] = queue.Queue()
        self.cameras = cameras or list(CAMERAS)
        self.camera = self.cameras[0]
        self.refs: dict[str, tuple[LiveDelta, ReferenceGrid, str]] = {}
        self.engine: LiveDelta | None = None
        self.grid: ReferenceGrid | None = None
        self.ref_name = ""
        self.armed = False
        self.recording: Recording | None = None
        self.recent: deque[Frame] = deque(maxlen=int(60 * (PRESS_WINDOW_S + LINE_PAD_S + 1)))
        self.lap_start: float | None = None
        self.last_line: float | None = None      # the latest line crossing
        self.finished_line: float | None = None  # the line that ended the latest recording
        self.settle_until = 0.0
        self.crossings = 0
        self.last_t: float | None = None
        self.lap_flags: set[str] = {"out lap"}       # reset at each line: what made this lap unclean
        self.primal_line: float | None = None         # the line, as PRIMAL's own position crosses it
        self.prev_reading: Reading | None = None
        self.prev: tuple[float, float] | None = None
        self.rows: list[dict] = []
        self.stale = 0
        self.log = open(run_dir / "ticks.csv", "w", newline="")
        self.writer: csv.DictWriter | None = None
        self.ui.camera = CAMERAS[self.camera][0]

    def flag(self, why: str) -> None:
        self.lap_flags.add(why)
        self.ui.lap_flags = ", ".join(sorted(self.lap_flags))

    def use(self, engine: LiveDelta, grid: ReferenceGrid, name: str) -> None:
        self.primal_line = self.prev_reading = None
        self.refs[self.camera] = (engine, grid, name)
        self.engine, self.grid, self.ref_name = engine, grid, name
        self.ui.message = f"reference: {name} ({grid.lap_time_s:.2f} s)"

    def run(self) -> None:
        try:
            while True:
                f = self.frames.get()
                if f is None:
                    break
                while not self.commands.empty():
                    self.command(self.commands.get_nowait(), f.t)
                self.step(f)
        finally:
            self.log.close()
            self.ui.done = True

    # --- buttons

    def command(self, what: str, t: float) -> None:
        if what == "record":
            self.press_record(t)
        elif what == "camera":
            self.next_camera(t)

    def press_record(self, t: float) -> None:
        """
        One press arms a reference lap; it records from the line to the line by itself.
        Pressed just after the line, it starts from that line. Pressed around the end of
        the lap it changes nothing (the lap ends at the line anyway); mid-lap it cancels.
        """
        rec = self.recording
        s = self.prev[1] if self.prev else float("nan")
        if rec is not None:
            if rec.ends_at is None and 0.1 < s < 0.9:
                self.recording, self.ui.rec = None, ""
                self.ui.message = "reference: recording cancelled"
            return
        if self.finished_line is not None and t - self.finished_line < PRESS_WINDOW_S:
            return  # the press that meant "stop at the line"
        if self.armed:
            self.armed, self.ui.rec = False, ""
            self.ui.message = "reference: cancelled"
        elif self.last_line is not None and t - self.last_line < PRESS_WINDOW_S and self.last_line > self.settle_until:
            self.start_recording(self.last_line)
        else:
            self.armed, self.ui.rec = True, "armed"
            self.ui.message = f"reference ({CAMERAS[self.camera][0]}): recording starts at the line"

    def next_camera(self, t: float) -> None:
        if self.rig is None:
            self.ui.message = "no camera rig with this source"
            return
        names = self.cameras
        self.camera = names[(names.index(self.camera) + 1) % len(names)]
        label, values = CAMERAS[self.camera]
        self.rig.apply(values)
        self.ui.camera = label
        self.recording, self.armed, self.ui.rec = None, False, ""
        # new camera, new picture: no deltas until its reference, and its own line crossing
        self.settle_until, self.lap_start, self.prev, self.ui.row = t + SETTLE_S, None, None, None
        self.flag("camera change")
        self.primal_line = self.prev_reading = None
        if self.camera in self.refs:
            engine, grid, name = self.refs[self.camera]
            engine.reset()
            self.use(engine, grid, name)
        else:
            self.engine = self.grid = None
            self.ref_name = ""
            self.ui.message = f"camera: {label}. No reference yet: press RECORD, then drive a lap from the line"

    # --- frames

    def step(self, f: Frame) -> None:
        if self.last_t is not None and f.t - self.last_t > GAP_S:
            # AC was paused (or frames stopped): nothing since is known, start the lap over
            self.flag("pause")
            self.lap_start = self.prev = self.primal_line = self.prev_reading = None
            if self.engine is not None:
                self.engine.reset()
        self.last_t = f.t
        if f.t < self.settle_until:
            return
        if f.snap is not None:
            if f.snap.tyres_out >= 3:
                self.flag("off track")
            if f.snap.speed_kmh < SLOW_KMH and self.lap_start is not None:
                self.flag("slow")
            if f.snap.is_in_pit:
                self.flag("pits")
        crossed = None
        if np.isfinite(f.cam_s):
            if self.prev is not None and self.prev[1] > 0.9 and f.cam_s < 0.1 and f.t - self.prev[0] < 0.5:
                frac = (1.0 - self.prev[1]) / (f.cam_s + 1.0 - self.prev[1])
                crossed = self.prev[0] + frac * (f.t - self.prev[0])
            self.prev = (f.t, f.cam_s)
        if crossed is not None:
            self.lap_start = self.last_line = crossed
            self.crossings += 1
            self.lap_flags = set()
            self.ui.lap_flags = ""
            if self.recording is not None and self.recording.ends_at is None:
                self.recording.ends_at = crossed + LINE_PAD_S
            elif self.armed and self.recording is None:
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
        self.armed = False
        self.recording = Recording()
        for f in self.recent:
            if f.t >= line_t - LINE_PAD_S:
                self.recording.add(f)
        self.ui.rec = "recording"
        self.ui.message = f"reference ({CAMERAS[self.camera][0]}): recording this lap - drive it clean"

    def finish_recording(self) -> None:
        lap, self.recording = self.recording, None
        self.finished_line = lap.ends_at - LINE_PAD_S
        problem = lap.problem(self.track_length)
        if problem:
            self.start_recording(self.finished_line)
            self.ui.message = f"reference: that lap won't do ({problem}); recording this one instead"
            return
        frames, t, s = np.stack(lap.frames), np.asarray(lap.t), np.asarray(lap.s, dtype=np.float64)
        out = self.run_dir / f"reference_{self.camera}_{self.crossings:02d}"
        out.mkdir()
        np.save(out / "frames.npy", frames)
        np.save(out / "t.npy", t)
        np.save(out / "s.npy", s)
        (out / "meta.json").write_text(json.dumps({"track_length_m": self.track_length, "camera": self.camera,
                                                   "rig": CAMERAS[self.camera][1]}))
        self.ui.rec = "encoding"
        self.ui.message = "reference: encoding the lap you just drove..."
        engine, grid = engine_from_arrays(self.model, frames, s, t, self.track_length, self.clip_len, self.device)
        self.ui.rec = ""
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
        # The camera alone: the lap starts when PRIMAL's own position passes the line.
        last = self.prev_reading
        if last is not None and last.ref_time_s > 0.75 * period and r.ref_time_s < 0.25 * period:
            frac = (period - last.ref_time_s) / (r.ref_time_s + period - last.ref_time_s)
            self.primal_line = last.t + frac * (r.t - last.t)
        self.prev_reading = r
        primal_only = (lap_delta(r.t - self.primal_line, r.ref_time_s, period) if self.primal_line is not None
                       else float("nan"))
        row = {
            "wall_time": round(time.time(), 3),
            "t": round(r.t, 4),
            "lap": self.crossings,
            "lap_flags": ";".join(sorted(self.lap_flags)),
            "ac_completed_laps": snap.completed_laps if snap else -1,
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
            "camera_only_delta_s": round(primal_only, 4),
            "performance_meter": snap.performance_meter if snap else float("nan"),
            "spline_pos": round(snap.spline_pos, 6) if snap else float("nan"),
            "cam_s": round(f.cam_s, 6),
            "s_source": f.s_source,
            "speed_kmh": round(snap.speed_kmh, 2) if snap else float("nan"),
            "tyres_out": snap.tyres_out if snap else -1,
            "delay_ms": round(1000 * (now - r.t), 1) if self.live_source else float("nan"),
            "queue": self.frames.qsize(),
            "camera": self.camera,
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
    if ui.camera:
        text(img, f"cam: {ui.camera}", (150, 30), font, 0.5, (200, 200, 120), 1, cv2.LINE_AA)
    if ui.rec:
        colour = {"armed": (60, 200, 230), "recording": (60, 60, 240), "encoding": (200, 200, 200)}[ui.rec]
        cv2.circle(img, (w - 290, 30), 7, colour, -1)
        text(img, {"armed": "REC at the line", "recording": "REC", "encoding": "encoding"}[ui.rec], (w - 278, 36),
             font, 0.55, colour, 2, cv2.LINE_AA)
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
    if ui.paused:
        text(img, "AC paused: not logging", (w - 308, 214), font, 0.5, (90, 200, 230), 1, cv2.LINE_AA)
    elif ui.lap_flags:
        text(img, f"this lap won't count: {ui.lap_flags}"[:44], (w - 308, 214), font, 0.42, (150, 150, 150), 1,
             cv2.LINE_AA)
    if ui.prompt:
        cv2.rectangle(img, (0, 40), (w - 320, 240), (40, 40, 40), -1)
        text(img, "Wheel setup", (14, 80), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        words = ui.prompt.split(" for ")
        text(img, words[0] + " for", (14, 130), font, 0.65, (90, 200, 230), 2, cv2.LINE_AA)
        text(img, words[-1], (14, 170), bold, 0.9, (90, 230, 255), 2, cv2.LINE_AA)
        text(img, "pick one AC doesn't use; not triangle", (14, 215), font, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
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
        # a lap is clean if nothing flagged it by its end (flags only grow within a lap)
        last_flags = {}
        for r in rows:
            last_flags[r["lap"]] = r["lap_flags"]
        last_flags[max(last_flags)] = ";".join(filter(None, [last_flags[max(last_flags)], "unfinished"]))
        clean = [r for r in good if not last_flags[r["lap"]]]
        if clean:
            e = np.abs([r["error_ms"] for r in clean])
            out["clean_laps"] = {"laps": sorted({r["lap"] for r in clean}), "ticks": len(clean),
                                 "median_ms": float(np.median(e)), "p90_ms": float(np.percentile(e, 90)),
                                 "over_100ms_pct": float(100 * np.mean(e > 100)),
                                 "over_1s_pct": float(100 * np.mean(e > 1000))}
        laps = sorted({r["lap"] for r in good})
        out["lap_flags"] = {str(lap): last_flags[lap] for lap in laps}
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
    parser.add_argument("--cameras", default=",".join(CAMERAS),
                        help=f"the presets CAMERA cycles through, first one at start: {','.join(CAMERAS)}")
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
        rig = Rig()
        rig.apply(CAMERAS[args.cameras.split(",")[0]][1])
    else:
        shm = rig = None
        meta = json.loads((Path(args.source).parent / "run.json").read_text())
        track_key = f"{meta['track']}__{meta['track_config']}" if meta.get("track_config") else meta["track"]
        car, track_length = meta["car_model"], float(meta["track_length_m"])
        source = VideoFile(Path(args.source), args.realtime)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.log_dir) / f"{stamp}__{track_key}__{args.reference.replace('/', '_').replace(chr(92), '_')[:40]}"
    run_dir.mkdir(parents=True)
    ui = UiState()
    frames: queue.Queue = queue.Queue(maxsize=240)
    worker = Worker(model, clip_len, device, frames, ui, run_dir, track_length, source.live, rig, args.cameras.split(","))

    start = time.perf_counter()
    if args.reference == "live":
        worker.armed, ui.rec = True, "armed"
        ui.message = "reference: recording starts at the line"
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

    wheel = Wheel(worker.commands.put, ui) if args.source == "obs" else None
    if wheel is not None and wheel.found:
        wheel.start()
        if wheel.bindings:
            print(f"wheel {wheel.key}: {wheel.describe()}")
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
            said = ""
            while not ui.done:
                if ui.message != said:  # into the console too, so a run can be read back afterwards
                    said = ui.message
                    print(f"{datetime.now():%H:%M:%S} {said}", flush=True)
                cv2.imshow(WINDOW, render(ui))
                key = cv2.waitKey(33) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    worker.commands.put("record")
                elif key == ord("c"):
                    worker.commands.put("camera")
                elif key == ord("b") and wheel is not None and wheel.found:
                    wheel.learn()
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
        if rig is not None:
            rig.restore()
    summary = summarize(worker.rows, worker, ui)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
