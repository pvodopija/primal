"""
Checks a packed batch that arrived in USB parts (data/<prefix>_part*/<set>/laps/..., each
part with the set's whole index.json) before anything trains on it: every index entry's lap
present exactly once across the parts; the index copies identical; every lap complete and
readable (frames.npy of n_frames at the index's frame size, s.npy and t.npy of n_frames, t
increasing, s in [0, 1) and spanning what the index says, pose.npz with a row per frame);
frames not blank. Prints a table per set and every problem; exits 1 if there is any.

Usage: PYTHONPATH=. python experiments/check_batch.py data/usb_b1007
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    prefix = Path(sys.argv[1])
    parts = sorted(prefix.parent.glob(prefix.name + "_part*"))
    problems: list[str] = []
    found: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))  # set -> lap path -> dirs
    indexes: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        for s in sorted(p for p in part.iterdir() if p.is_dir()):
            if (s / "index.json").exists():
                indexes[s.name].add(hashlib.md5((s / "index.json").read_bytes()).hexdigest())
            for lap in sorted((s / "laps").iterdir()) if (s / "laps").exists() else []:
                found[s.name][f"laps/{lap.name}"].append(lap)
    print(f"{len(parts)} parts: {', '.join(p.name for p in parts)}")
    for name in sorted(found):
        if len(indexes[name]) != 1:
            problems.append(f"{name}: {len(indexes[name])} different index.json copies")
        entries = json.loads(next((p / name / "index.json") for p in parts if (p / name / "index.json").exists()).read_text())["laps"]
        paths = {e["path"] for e in entries}
        for path, dirs in found[name].items():
            if len(dirs) > 1:
                problems.append(f"{name}: {path} in {len(dirs)} parts")
            if path not in paths:
                problems.append(f"{name}: {path} not in the index")
        frames_total, full, splits, size = 0, 0, defaultdict(int), 0
        for e in entries:
            dirs = found[name].get(e["path"], [])
            if not dirs:
                problems.append(f"{name}: {e['path']} missing")
                continue
            d, n = dirs[0], int(e["n_frames"])
            w, h = e.get("frame_size", (None, None))
            try:
                f = np.load(d / "frames.npy", mmap_mode="r")
                s, t = np.load(d / "s.npy"), np.load(d / "t.npy")
                pose = np.load(d / "pose.npz") if e.get("pose") else None
            except Exception as err:  # truncated or unreadable
                problems.append(f"{name}: {e['path']}: {err}")
                continue
            if f.shape != (n, h, w, 3) or f.dtype != np.uint8:
                problems.append(f"{name}: {e['path']}: frames {f.shape} {f.dtype}, index says {n} x {h}x{w}")
            if len(s) != n or len(t) != n:
                problems.append(f"{name}: {e['path']}: s {len(s)}, t {len(t)}, index {n}")
            if np.any(np.diff(t) <= 0):
                problems.append(f"{name}: {e['path']}: t not increasing")
            if s.min() < 0 or s.max() > 1:  # 1.0 is the line, the same as 0.0
                problems.append(f"{name}: {e['path']}: s outside [0, 1]")
            span = float(np.ptp(np.unwrap(s.astype(np.float64), period=1.0)))
            if abs(span - float(e.get("s_span", span))) > 0.01:
                problems.append(f"{name}: {e['path']}: s spans {span:.3f}, index {e['s_span']:.3f}")
            if pose is not None:
                short = [k for k in pose.files if len(pose[k]) != n]
                if short:
                    problems.append(f"{name}: {e['path']}: pose rows differ from frames in {short}")
            sample = f[np.linspace(0, n - 1, 3).astype(int)]
            if sample.mean() < 5 or sample.std() < 5:
                problems.append(f"{name}: {e['path']}: frames look blank (mean {sample.mean():.0f}, sd {sample.std():.0f})")
            frames_total += n
            full += e.get("s_span", 0) >= 0.98
            splits[e.get("split", "train")] += 1
            size += sum(x.stat().st_size for x in d.iterdir())
        tracks = sorted({e["track"] for e in entries})
        print(f"  {name}: {len(entries)} laps ({full} full), {frames_total:,} frames, {size / 1e9:.1f} GB, "
              f"splits {dict(splits)}, {len(tracks)} tracks")
    print(f"\n{len(problems)} problems")
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
