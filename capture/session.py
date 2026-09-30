"""
Session bookkeeping around an OBS recording.

    python -m capture.session log                       # optional aux telemetry, Ctrl-C to stop
    python -m capture.session import <recording.mp4> --split train --notes "noon, tight line"

`import` creates `data/sessions/<track>__<config>__<stamp>/`, moves the video in
as `video.mp4`, and writes `run.json`. Track name, layout, spline length and car
model are read from AC shared memory, so run it while AC is still open on the
same session; otherwise pass them explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from capture.ac_shm import AcSharedMemory
from capture.install_overlay import find_ac_root

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"
# Short test drives on a new track, checked by `capture.preflight` and never packed.
PREFLIGHT_DIR = SESSIONS_DIR.parent / "preflight"
SCHEMA = 1


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count


def cmd_log(args: argparse.Namespace) -> None:
    shm = AcSharedMemory()
    rows: list[dict] = []
    period = 1.0 / max(args.hz, 1.0)
    print(f"logging at {args.hz} Hz, Ctrl-C to stop")
    try:
        while True:
            snap = shm.snapshot()
            if snap is not None:
                rows.append(
                    {
                        "t": time.perf_counter(),
                        "status": snap.status,
                        "spline_pos": snap.spline_pos,
                        "completed_laps": snap.completed_laps,
                        "speed_kmh": snap.speed_kmh,
                        "heading": snap.heading,
                        "pitch": snap.pitch,
                        "roll": snap.roll,
                        "is_in_pit": snap.is_in_pit,
                        "x": snap.world_pos[0] if snap.world_pos else float("nan"),
                        "y": snap.world_pos[1] if snap.world_pos else float("nan"),
                        "z": snap.world_pos[2] if snap.world_pos else float("nan"),
                    }
                )
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()

    if not rows:
        print("\nnothing logged; AC shared memory was never available")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {len(rows)} rows to {out}")


def attach_rig_log(session: Path) -> str:
    """
    Copy the camera rig's newest log into the session, for recordings rendered from
    a replay through `capture/ac_rig`. The rig starts a new log for every render, so
    the newest one belongs to the recording that just stopped.
    """
    root = find_ac_root()
    logs = sorted((root / "apps" / "lua" / "primal_rig").glob("rig_*.csv"), key=lambda p: p.stat().st_mtime) if root else []
    if not logs:
        raise SystemExit("--rig given but no rig log found under apps/lua/primal_rig")
    if time.time() - logs[-1].stat().st_mtime > 15 * 60:
        raise SystemExit(f"newest rig log {logs[-1].name} is over 15 minutes old; was the rig enabled for this render?")
    shutil.copy2(logs[-1], session / "rig_log.csv")
    return logs[-1].name


def attach_frame_log(session: Path, start: float, end: float) -> str | None:
    """
    Copy the timecode app's per-frame telemetry for a recording into the session.
    The app writes one folder per launch, named by the launch's first logged
    second, in chunks flushed every two seconds; the launch is the latest one
    begun before the recording, and only chunks written between `start` and `end`
    (epoch seconds) are copied, with a minute's margin, since a bot left running
    fills a folder with hours of rows. Rows are matched to frames later by the
    barcode counter, so extra ones are harmless. Returns None when no log covers
    the recording, as with recordings made before the app logged anything.
    """
    root = find_ac_root()
    base = root / "apps" / "lua" / "locamotif_timecode" / "frame_log" if root else None
    if not base or not base.exists():
        return None
    launches = sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) <= start)
    if not launches:
        return None
    launch = base / str(launches[-1])
    chunks = [e for e in os.scandir(launch) if e.name.endswith(".csv") and start - 60 <= e.stat().st_mtime <= end + 60]
    if not chunks:
        print(f"note: frame log {launch.name} has nothing written during the recording; not attached")
        return None
    (session / "frame_log").mkdir()
    for e in chunks:
        shutil.copy2(e.path, session / "frame_log" / e.name)
    return launch.name


def cmd_import(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"{video} not found")

    shm = AcSharedMemory()
    snap = shm.snapshot()
    shm.close()

    track = args.track or (snap.track if snap else "")
    if not track:
        raise SystemExit("AC is not running; pass --track and --track-length")
    config = args.track_config if args.track_config is not None else (snap.track_config if snap else "")
    length = args.track_length or (snap.track_length_m if snap else 0.0)
    if length <= 0.0:
        raise SystemExit("track spline length unknown; pass --track-length")
    car = args.car or (snap.car_model if snap else "")

    fps, width, height, count = _probe_video(video)
    if fps <= 0.0:
        # Every later step (the telemetry window below, packing) places frames in time by fps.
        raise SystemExit(f"{video}: no frame rate in the container; remux it (ffmpeg -c copy) and retry")
    # OBS last writes the file when the recording stops.
    recorded_end = video.stat().st_mtime
    recorded_start = recorded_end - count / fps
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{track}__{config or 'default'}__{stamp}"
    session = (PREFLIGHT_DIR if args.preflight else SESSIONS_DIR) / name
    session.mkdir(parents=True, exist_ok=False)

    destination = session / "video.mp4"
    if args.copy:
        shutil.copy2(video, destination)
    else:
        shutil.move(str(video), destination)

    meta = {
        "schema": SCHEMA,
        "session_id": name,
        "video": destination.name,
        "track": track,
        "track_config": config,
        "track_length_m": float(length),
        "car_model": car,
        "label_offset_m": 0.0,
        "ac_version": snap.ac_version if snap else "",
        "fps": fps,
        "resolution": [width, height],
        "frame_count": count,
        "split": args.split,
        "notes": args.notes,
        "created": stamp,
    }
    if args.rig:
        meta["rig_log"] = attach_rig_log(session)
    frame_log = attach_frame_log(session, recorded_start, recorded_end)
    if frame_log:
        meta["frame_log"] = frame_log
        # The rendering camera's vertical FOV, as the telemetry logged it every frame;
        # `pack` derives the packed image's field of view from it.
        fov = pd.concat(pd.read_csv(c, usecols=["cam_fov_deg"]) for c in (session / "frame_log").glob("*.csv")).cam_fov_deg
        meta["camera_vfov_deg"] = round(float(fov.median()), 2)
        print(f"  camera vertical FOV {meta['camera_vfov_deg']} deg (logged range {fov.min():.1f}-{fov.max():.1f})")
    (session / "run.json").write_text(json.dumps(meta, indent=2))

    print(f"created {session}")
    print(f"  {track}/{config or 'default'}  {length:.1f} m  {car}")
    print(f"  {count} frames at {fps:.2f} fps, {width}x{height}, split={args.split}")
    print("\nnext:")
    print(f"  python -m capture.overlay_decode calibrate {destination}")
    print(f"  python -m capture.overlay_decode decode {destination}")


def cmd_trim(args: argparse.Namespace) -> None:
    session = Path(args.session)
    meta_path = session / "run.json"
    meta = json.loads(meta_path.read_text())
    labels = pd.read_parquet(session / "labels.parquet")
    if args.last_lap:
        good = labels[labels.valid].sort_values("frame_idx")
        s = good.spline_pos.to_numpy()
        wraps = (s[:-1] > 0.9) & (s[1:] < 0.1)
        if not wraps.any():
            raise SystemExit("no finish-line crossing in the decoded labels")
        end_frame = int(good.frame_idx.to_numpy()[1:][wraps][-1])
    else:
        end_frame = args.end_frame
    meta["end_frame"] = end_frame
    meta_path.write_text(json.dumps(meta, indent=2))
    dropped = int((labels.frame_idx >= end_frame).sum())
    print(f"end_frame {end_frame}: packing drops the last {dropped} frames ({dropped / meta['fps']:.1f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    log = sub.add_parser("log", help="log AC shared memory as auxiliary telemetry")
    log.add_argument("--out", default="aux.parquet")
    log.add_argument("--hz", type=float, default=50.0)
    log.set_defaults(func=cmd_log)

    imp = sub.add_parser("import", help="adopt an OBS recording as a session")
    imp.add_argument("video")
    imp.add_argument("--split", default="train", choices=["train", "holdout"])
    imp.add_argument("--notes", default="")
    imp.add_argument("--track", help="override AC shared memory")
    imp.add_argument("--track-config", help="override AC shared memory")
    imp.add_argument("--track-length", type=float, help="spline length in metres")
    imp.add_argument("--car", help="override AC shared memory")
    imp.add_argument("--copy", action="store_true", help="copy instead of moving the video")
    imp.add_argument("--rig", action="store_true", help="recording is a camera-rig render; attach its log")
    imp.add_argument("--preflight", action="store_true",
                     help="a test drive on a new track: import into data/preflight, never packed")
    imp.set_defaults(func=cmd_import)

    trim = sub.add_parser("trim", help="cut footage off the end of a decoded session before packing")
    trim.add_argument("session")
    end = trim.add_mutually_exclusive_group(required=True)
    end.add_argument("--end-frame", type=int, help="first video frame to drop")
    end.add_argument("--last-lap", action="store_true",
                     help="drop everything after the last finish-line crossing: the partial lap and any pause menu")
    trim.set_defaults(func=cmd_trim)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
