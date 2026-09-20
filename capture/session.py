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
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from capture.ac_shm import AcSharedMemory

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{track}__{config or 'default'}__{stamp}"
    session = SESSIONS_DIR / name
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
        "ac_version": snap.ac_version if snap else "",
        "fps": fps,
        "resolution": [width, height],
        "frame_count": count,
        "split": args.split,
        "notes": args.notes,
        "created": stamp,
    }
    (session / "run.json").write_text(json.dumps(meta, indent=2))

    print(f"created {session}")
    print(f"  {track}/{config or 'default'}  {length:.1f} m  {car}")
    print(f"  {count} frames at {fps:.2f} fps, {width}x{height}, split={args.split}")
    print("\nnext:")
    print(f"  python -m capture.overlay_decode calibrate {destination}")
    print(f"  python -m capture.overlay_decode decode {destination}")


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
    imp.set_defaults(func=cmd_import)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
