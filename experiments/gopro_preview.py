"""
A stand-in for the iPhone app, to retire once the app's live view works: the HERO7's Wi-Fi
preview into PRIMAL on the Mac, received and decoded in hardware by the app repo's GoProKit
(its gopro-probe tool: UDP keep-alive, MPEG-TS, VideoToolbox; GOPROKIT points at the package,
by default ~/code/hotlapp_V1.0/Packages/GoProKit).

The Mac joins the GoPro's Wi-Fi (with internet over a cable, or this runs alone). This
starts the preview over HTTP, crops each frame to the training aspect (148:80), shrinks it
to 148x80 as the real-footage pipeline does and encodes it with the model. The window
shows the frame under a large clock: point the camera at the screen and the clock it
films lags the clock drawn by the whole delay, glass to screen (snapshots every 5 s).

Keys: r at the line starts the reference lap; it closes by itself when the camera sees the
start again (train/session.py: a look-alike proposes, PRIMAL confirms and places the line), or
at a second r. Laps, and a faster lap becoming the reference, as the app does them. From
then on the window shows PRIMAL's live
delta, the tracker starting at the line, a new lap where it wraps past the line. q quits.

Usage: PYTHONPATH=. python experiments/gopro_preview.py [--seconds 600] [--record]
           [--run v3_mobilenet_lr1_s0] [--crop x0,x1,y0,y1] [--fake VIDEO.mp4]
  --record   also start the camera recording 5 s in (does the preview survive it?)
  --auto-reference START[,END]  press r at these stream times (tests; without END it closes by itself)
  --fake     no camera: GoProKit's fake-gopro replays VIDEO as the preview would, or a
             preview this saved (data/gopro/<time>/stream.ts) as it came
  e.g. RaceChrono's lap 3 against its lap 2 (both from experiments/real_footage.py's crossings):
       --fake data/real-footage/race-chrono/race-chrono.mp4 --fake-start 96.39 --auto-reference 2.0,89.28 --seconds 180
       (the replay's stream times run from 1 s at --fake-start; lap 3 should end about +0.32 s)
Writes data/gopro/<time>/: stream.ts, frames.csv, readings.csv, probe.log, snapshots/, summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from train.eval import load_model
from train.live import encode_frames
from train.session import LapCompleted, ProposalRejected, Reading, ReferenceClosed, Session

GOPROKIT = Path(os.environ.get("GOPROKIT", Path.home() / "code/hotlapp_V1.0/Packages/GoProKit"))
CAMERA = "http://10.5.5.9"
HEADER = struct.Struct("<4sdddII")


def build(name: str) -> Path:
    """GoProKit's tool `name` (gopro-probe, fake-gopro), built if it changed."""
    subprocess.run(["swift", "build", "-c", "release", "--package-path", str(GOPROKIT), "--product", name], check=True, capture_output=True)
    return GOPROKIT / ".build/release" / name


def camera(path: str) -> str | None:
    try:
        with urllib.request.urlopen(CAMERA + path, timeout=3) as r:
            return r.read().decode(errors="replace")
    except OSError as e:
        print(f"camera {path}: {e}", flush=True)
        return None


def network_check(out: Path) -> str | None:
    """
    Whether the Mac is on the GoPro's Wi-Fi (10.5.5.x) and the camera's HTTP control answers;
    the camera's info if so. Saved to netcheck.txt. (The camera answers ping with "port
    unreachable", so ping says nothing.)
    """
    ip = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip()
    text = f"Wi-Fi address: {ip or 'none'}\n"
    info = None
    if not ip.startswith("10.5.5."):
        text += ("NOT ON THE GOPRO'S WI-FI (its addresses start 10.5.5.): join its network and run again."
                 + (" 169.254 means the camera has not given the Mac an address: rejoin." if ip.startswith("169.254") else ""))
    else:
        try:
            with urllib.request.urlopen(CAMERA + "/gp/gpControl/info", timeout=3) as r:
                info = r.read().decode(errors="replace")
            text += f"camera: {info}"
        except OSError as e:
            text += f"camera HTTP: {e}\n" + (
                "'No route to host' on the GoPro's own network: macOS blocks this terminal app. System Settings > "
                "Privacy & Security > Local Network, switch it on (or run from Terminal and allow the prompt)."
                if "No route" in str(e) else
                "'Connection refused': the camera's control is off; on the camera, Connections > Connect Device > GoPro App, "
                "then rejoin." if "refused" in str(e) else "The camera did not answer in 3 s.")
    (out / "netcheck.txt").write_text(text)
    print(text, flush=True)
    return info


class StreamClock:
    """
    The stream's frame times, kept steady: the HERO7 restarts its timestamps at 0 now and
    then (twice in the first 2-minute run). Its frames are always whole frame intervals
    apart, so a jump back by over half a second, or forward by over a second, is bridged
    with the number of frame intervals that best fits the time between arrivals (whose
    jitter, 12 ms p90, is under half an interval): exact unless frames were lost in it.
    (Smaller steps back are frames reordered around B-frames, which the GoPro's preview
    does not use but videos replayed by fake_gopro do.)
    """
    def __init__(self) -> None:
        self.offset, self.last, self.last_arrival, self.restarts = 0.0, None, None, 0
        self.step = 1001 / 30000  # the frame interval, learnt from the stream

    def __call__(self, pts: float, arrival: float) -> float:
        t = pts + self.offset
        if self.last is not None:
            dt = t - self.last
            if 0 < dt < 0.1:
                frames = max(round(dt / self.step), 1)
                self.step += 0.05 * (dt / frames - self.step)
            elif not -0.5 < dt < 1.0:
                frames = max(round((arrival - self.last_arrival) / self.step), 1)
                self.offset = self.last + frames * self.step - pts
                t, self.restarts = pts + self.offset, self.restarts + 1
        self.last, self.last_arrival = t if self.last is None else max(t, self.last), arrival
        return t


def read_exact(f, n: int) -> bytes:
    """n bytes from the probe's pipe, which hands them over in pieces; fewer only at its end."""
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        parts.append(b); got += len(b)
    return b"".join(parts)


class LatestFrame:
    """
    The probe's frames read as they come, only the newest kept: when the loop is slower
    than the stream it skips frames instead of falling behind (in the second run a queue
    grew 3 s every 10 s and overflowed).
    """
    def __init__(self, pipe) -> None:
        self.pipe, self.item, self.done, self.skipped = pipe, None, False, 0
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            head = read_exact(self.pipe, HEADER.size)
            if len(head) < HEADER.size:
                break
            _, pts, arrival, decode_ms, w, h = HEADER.unpack(head)
            raw = read_exact(self.pipe, w * h * 4)
            if len(raw) < w * h * 4:
                break
            with self.cond:
                self.skipped += self.item is not None
                self.item = (pts, arrival, decode_ms, w, h, raw)
                self.cond.notify()
        with self.cond:
            self.done = True
            self.cond.notify()

    def next(self):
        with self.cond:
            while self.item is None and not self.done:
                self.cond.wait()
            item, self.item = self.item, None
            return item


def crop_box(w: int, h: int, crop: str | None) -> tuple[int, int, int, int]:
    if crop:
        return tuple(int(v) for v in crop.split(","))
    ch = min(h, int(round(w * 80 / 148)))  # the whole width at the training aspect
    return 0, w, (h - ch) // 2, (h - ch) // 2 + ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--run", default="v3_mobilenet_lr1_s0")
    ap.add_argument("--crop")
    ap.add_argument("--fake")
    ap.add_argument("--fake-start", default="0", help="where in VIDEO the replay starts, s")
    ap.add_argument("--auto-reference", help="START,END: press r at these stream times, s (tests)")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()
    out = Path("data/gopro") / time.strftime("%Y%m%dT%H%M%S")
    (out / "snapshots").mkdir(parents=True)
    if not a.fake:
        summary_info = network_check(out)
        if summary_info is None:
            return
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(Path("runs") / a.run / "best.pt", device)
    clip_len = int(payload["args"]["clip_len"])
    summary = {"run": a.run, "fake": a.fake, "record": a.record}

    probe = subprocess.Popen([str(build("gopro-probe")), "--ts", str(out / "stream.ts"), "--seconds", str(a.seconds + 5)]
                             + (["--camera", "127.0.0.1"] if a.fake else []), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    log = (out / "probe.log").open("w")
    probe_lines: list[str] = []

    def tee() -> None:
        for line in iter(probe.stderr.readline, b""):
            s = line.decode().rstrip()
            print(s, flush=True); log.write(s + "\n"); log.flush(); probe_lines.append(s)
    threading.Thread(target=tee, daemon=True).start()

    if a.fake:
        fake = subprocess.Popen([str(build("fake-gopro")), a.fake, "--seconds", str(a.seconds), "--start", a.fake_start])
    else:
        summary["camera_info"] = summary_info
        camera("/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=restart")

    clock, feed = StreamClock(), LatestFrame(probe.stdout)
    frames, readings = [], []
    session = Session(model, clip_len, device)

    def handle(events: list) -> None:
        for e in events:
            if isinstance(e, Reading):
                readings.append((e.t, e.ref_time_s, e.confidence, e.single_ref_time_s, e.delta_s))
            elif isinstance(e, ReferenceClosed):
                summary.update(reference_start_pts=session.ref_start, reference_lap_s=e.lap_s, reference_placed_by=e.placed_by)
                print(f"reference: {e.lap_s:.2f} s, line at {e.line_s:.2f} s (placed by {e.placed_by}); live delta from here", flush=True)
            elif isinstance(e, ProposalRejected):
                print(f"reference: {e.proposal_s:.2f} s proposed, not confirmed", flush=True)
            elif isinstance(e, LapCompleted):
                print(f"lap {e.lap}: {e.lap_s:.2f} s, delta shown {e.shown_share:.0%}"
                      f"{', the new reference' if e.became_reference else ''}{', void (stream gap)' if e.voided else ''}", flush=True)
    started, recording, last_snapshot = time.time(), False, 0.0
    box = None
    try:
        while time.time() - started < a.seconds:
            item = feed.next()
            if item is None:
                break
            stream_pts, arrival, decode_ms, w, h, raw = item
            pts = clock(stream_pts, arrival)
            read_at = time.time()
            img = np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, :3]
            if box is None:
                box = crop_box(w, h, a.crop)
                summary.update(width=w, height=h, crop=box)
                print(f"preview {w}x{h}; PRIMAL reads crop x {box[0]}-{box[1]}, y {box[2]}-{box[3]} at 148x80", flush=True)
            if a.record and not recording and read_at - started > 5 and not a.fake:
                camera("/gp/gpControl/command/shutter?p=1"); recording = True; summary["record_started_s"] = read_at - started
            x0, x1, y0, y1 = box
            small = cv2.resize(img[y0:y1, x0:x1], (148, 80), interpolation=cv2.INTER_AREA)
            t0 = time.time()
            desc = encode_frames(model, small[None], device)[0]
            desc.cpu()  # wait for the GPU, so the time is the encoder's
            encode_ms = (time.time() - t0) * 1000
            frames.append((pts, stream_pts, arrival, read_at, decode_ms, encode_ms))
            handle(session.push(desc, pts))
            text = ""
            if readings and session.state == "live":
                _, ref_t, conf, _, delta = readings[-1]
                text = f"{delta:+.2f} s" if conf >= session.c.shown_min else "unavailable"
            # the window: the frame under a large clock, then the delta and the timings
            canvas = np.zeros((h + 170, max(w, 900), 3), np.uint8)
            canvas[170:, :w] = img
            now = time.time()
            cv2.putText(canvas, time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)
            n = len(frames)
            fps = 30 / (frames[-1][0] - frames[-31][0]) if n > 31 else 0
            cv2.putText(canvas, f"{w}x{h} {fps:4.1f} fps  decode {decode_ms:4.1f} ms  encode {encode_ms:4.1f} ms  queue {1000 * (read_at - arrival):4.0f} ms",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            state = {"reference": "REFERENCE: recording, closes back at the line",
                     "ready": "r at the line to start the reference"}.get(session.state, "")
            cv2.putText(canvas, text or state, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (80, 200, 80) if text.startswith("-") else (80, 80, 230) if text.startswith("+") else (200, 200, 200), 2)
            if not a.no_window:
                cv2.imshow("PRIMAL GoPro preview", canvas)
            if now - last_snapshot >= 5:
                cv2.imwrite(str(out / "snapshots" / f"{now - started:06.1f}.png"), canvas); last_snapshot = now
            key = -1 if a.no_window else cv2.waitKey(1) & 0xFF
            if a.auto_reference:
                marks = [float(v) for v in a.auto_reference.split(",")]
                if (session.state == "ready" and pts >= marks[0]) or (session.state == "reference" and len(marks) > 1 and pts >= marks[1]):
                    key = ord("r")
            if key == ord("q"):
                break
            if key == ord("r"):
                if session.state == "ready":
                    session.start(pts)
                    print(f"reference: start at {pts:.2f} s", flush=True)
                elif session.state == "reference":
                    handle(session.close_at(pts))
    finally:
        if recording:
            camera("/gp/gpControl/command/shutter?p=0")
        probe.terminate()
        if a.fake:
            fake.terminate()
        if not a.no_window:
            cv2.destroyAllWindows()
        f = pd.DataFrame(frames, columns=["pts", "stream_pts", "arrival", "read_at", "decode_ms", "encode_ms"])
        f.to_csv(out / "frames.csv", index=False)
        pd.DataFrame(readings, columns=["pts", "ref_time_s", "confidence", "single_ref_time_s", "delta_s"]).to_csv(out / "readings.csv", index=False)
        if len(f) > 1:
            gaps = np.diff(f.pts)
            summary.update(frames=len(f), skipped_by_the_loop=feed.skipped, timestamp_restarts=clock.restarts, fps=float(1 / np.median(gaps)), gaps_over_100ms=int(np.sum(gaps > 0.1)),
                           decode_ms_median=float(f.decode_ms.median()), encode_ms_median=float(f.encode_ms[f.encode_ms > 0].median()),
                           queue_ms_median=float(1000 * (f.read_at - f.arrival).median()),
                           arrival_jitter_ms_p90=float(1000 * np.percentile(np.abs((f.arrival - f.pts) - (f.arrival - f.pts).median()), 90)),
                           probe_last=probe_lines[-1] if probe_lines else None)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "camera_info"}, indent=2), "\nwritten to", out, flush=True)


if __name__ == "__main__":
    main()
