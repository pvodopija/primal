"""
Step 0 of the GoPro path: the HERO7's Wi-Fi preview into PRIMAL on the Mac, received and
decoded in hardware by the Swift code the iPhone app will use (ios/PrimalCore through
ios/tools/gopro_probe: UDP keep-alive, MPEG-TS, VideoToolbox).

The Mac joins the GoPro's Wi-Fi (with internet over a cable, or this runs alone). This
starts the preview over HTTP, crops each frame to the training aspect (148:80), shrinks it
to 148x80 as the real-footage pipeline does and encodes it with the model. The window
shows the frame under a large clock: point the camera at the screen and the clock it
films lags the clock drawn by the whole delay, glass to screen (snapshots every 5 s).

Keys: r marks the reference lap's start at the line and again its end (the next crossing);
from then on the window shows PRIMAL's live delta, a new lap starting where the tracker
wraps past the line. q quits.

Usage: PYTHONPATH=. python experiments/gopro_preview.py [--seconds 600] [--record]
           [--run v3_mobilenet_lr1_s0] [--crop x0,x1,y0,y1] [--fake VIDEO.mp4]
  --record   also start the camera recording 5 s in (does the preview survive it?)
  --fake     no camera: ios/tools/.build/fake_gopro replays VIDEO as the preview would
  e.g. RaceChrono's lap 3 against its lap 2 (both from experiments/real_footage.py's crossings):
       --fake data/real-footage/race-chrono/race-chrono.mp4 --fake-start 96.39 --auto-reference 2.0,89.28 --seconds 180
       (the replay's stream times run from 1 s at --fake-start; lap 3 should end about +0.32 s)
Writes data/gopro/<time>/: stream.ts, frames.csv, readings.csv, probe.log, snapshots/, summary.json
"""
from __future__ import annotations

import argparse
import json
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
from train.live import LiveDelta, encode_frames

IOS = Path("ios")
TOOLS = {"gopro_probe": [IOS / "PrimalCore", IOS / "tools/gopro_probe"], "fake_gopro": [IOS / "tools/fake_gopro"]}
CAMERA = "http://10.5.5.9"
BIN_S = 0.052  # reference bin spacing, as experiments/real_footage.py
HEADER = struct.Struct("<4sdddII")


def build(name: str) -> Path:
    """The Swift tool, rebuilt when a source is newer than it."""
    out = IOS / "tools/.build" / name
    sources = sorted(p for d in TOOLS[name] for p in d.glob("*.swift"))
    if not out.exists() or max(p.stat().st_mtime for p in sources) > out.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", "-swift-version", "5", *map(str, sources), "-o", str(out)], check=True)
    return out


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
    then (twice in the first 2-minute run), so a jump back, or forward by over a second,
    is bridged with the time between arrivals.
    """
    def __init__(self) -> None:
        self.offset, self.last, self.last_arrival, self.restarts = 0.0, None, None, 0

    def __call__(self, pts: float, arrival: float) -> float:
        t = pts + self.offset
        if self.last is not None and not (0 < t - self.last < 1.0):
            self.offset = self.last + max(arrival - self.last_arrival, 1 / 60) - pts
            t, self.restarts = pts + self.offset, self.restarts + 1
        self.last, self.last_arrival = t, arrival
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

    probe = subprocess.Popen([str(build("gopro_probe")), "--ts", str(out / "stream.ts"), "--seconds", str(a.seconds + 5)]
                             + (["--camera", "127.0.0.1"] if a.fake else []), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    log = (out / "probe.log").open("w")
    probe_lines: list[str] = []

    def tee() -> None:
        for line in iter(probe.stderr.readline, b""):
            s = line.decode().rstrip()
            print(s, flush=True); log.write(s + "\n"); log.flush(); probe_lines.append(s)
    threading.Thread(target=tee, daemon=True).start()

    if a.fake:
        fake = subprocess.Popen([str(build("fake_gopro")), a.fake, "--seconds", str(a.seconds), "--start", a.fake_start])
    else:
        summary["camera_info"] = summary_info
        camera("/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=restart")

    clock = StreamClock()
    frames, readings, history = [], [], []  # history: (pts, descriptor) while recording the reference
    live: LiveDelta | None = None
    ref_start = lap_start = None
    last_ref = None
    started, recording, last_snapshot = time.time(), False, 0.0
    box = None
    try:
        while time.time() - started < a.seconds:
            head = read_exact(probe.stdout, HEADER.size)
            if len(head) < HEADER.size:
                break
            magic, stream_pts, arrival, decode_ms, w, h = HEADER.unpack(head)
            pts = clock(stream_pts, arrival)
            raw = read_exact(probe.stdout, w * h * 4)
            if len(raw) < w * h * 4:
                break
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
            want = live is None or live.wants(pts)
            t0 = time.time()
            desc = encode_frames(model, small[None], device)[0] if want else None
            if desc is not None:
                desc.cpu()  # wait for the GPU, so the time is the encoder's
            encode_ms = (time.time() - t0) * 1000
            frames.append((pts, stream_pts, arrival, read_at, decode_ms, encode_ms))
            if ref_start is not None and desc is not None:
                history.append((pts, desc))
            text = ""
            if live is not None and desc is not None:
                r = live.push_descriptor(desc, pts)
                if r is not None:
                    if last_ref is not None and last_ref > 0.75 * live.lap_time and r.ref_time_s < 0.25 * live.lap_time:
                        lap_start = pts - r.ref_time_s  # past the line
                    last_ref = r.ref_time_s
                    delta = (pts - lap_start) - r.ref_time_s
                    readings.append((pts, r.ref_time_s, r.confidence, r.single_ref_time_s, delta))
            if readings and live is not None:
                _, ref_t, conf, _, delta = readings[-1]
                text = f"{delta:+.2f} s" if conf >= 0.5 else "unavailable"
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
            state = ("REFERENCE: recording, r at the line again" if ref_start is not None else
                     "r at the line to start the reference" if live is None else "")
            cv2.putText(canvas, text or state, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (80, 200, 80) if text.startswith("-") else (80, 80, 230) if text.startswith("+") else (200, 200, 200), 2)
            if not a.no_window:
                cv2.imshow("PRIMAL GoPro preview", canvas)
            if now - last_snapshot >= 5:
                cv2.imwrite(str(out / "snapshots" / f"{now - started:06.1f}.png"), canvas); last_snapshot = now
            key = -1 if a.no_window else cv2.waitKey(1) & 0xFF
            if a.auto_reference:
                marks = [float(v) for v in a.auto_reference.split(",")]
                if (ref_start is None and live is None and pts >= marks[0]) or (ref_start is not None and pts >= marks[1]):
                    key = ord("r")
            if key == ord("q"):
                break
            if key == ord("r"):
                if ref_start is None:
                    ref_start, history = pts, []
                    print(f"reference: start at {pts:.2f} s", flush=True)
                else:
                    t = np.array([s for s, _ in history]); T = pts - ref_start
                    nb = int(round(T / BIN_S))
                    pick = [int(np.argmin(np.abs(t - (ref_start + i * T / nb)))) for i in range(nb)]
                    ref_desc = torch.stack([history[i][1] for i in pick])
                    live = LiveDelta(model, None, np.arange(nb) * T / nb, T, clip_len, device, reference_descriptors=ref_desc)
                    summary.update(reference_start_pts=ref_start, reference_lap_s=T)
                    lap_start, last_ref, ref_start, history = pts, None, None, []
                    print(f"reference: {T:.2f} s, {nb} bins; live delta from here", flush=True)
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
            summary.update(frames=len(f), timestamp_restarts=clock.restarts, fps=float(1 / np.median(gaps)), gaps_over_100ms=int(np.sum(gaps > 0.1)),
                           decode_ms_median=float(f.decode_ms.median()), encode_ms_median=float(f.encode_ms[f.encode_ms > 0].median()),
                           queue_ms_median=float(1000 * (f.read_at - f.arrival).median()),
                           arrival_jitter_ms_p90=float(1000 * np.percentile(np.abs((f.arrival - f.pts) - (f.arrival - f.pts).median()), 90)),
                           probe_last=probe_lines[-1] if probe_lines else None)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "camera_info"}, indent=2), "\nwritten to", out, flush=True)


if __name__ == "__main__":
    main()
