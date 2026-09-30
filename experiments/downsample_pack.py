"""A lower-resolution copy of a packed set: frames resized with area averaging, labels linked."""
import json, os, sys
from pathlib import Path
import cv2, numpy as np

src, dst, width, height = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
index = json.loads((src / "index.json").read_text())
(dst / "laps").mkdir(parents=True, exist_ok=True)
for n, lap in enumerate(index["laps"]):
    s_dir, d_dir = src / lap["path"], dst / lap["path"]
    d_dir.mkdir(parents=True, exist_ok=True)
    frames = np.load(s_dir / "frames.npy", mmap_mode="r")
    out = np.empty((frames.shape[0], height, width, 3), np.uint8)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(np.asarray(frames[i]), (width, height), interpolation=cv2.INTER_AREA)
    np.save(d_dir / "frames.npy", out)
    for name in ("s.npy", "t.npy"):
        if not (d_dir / name).exists():
            os.symlink((s_dir / name).resolve(), d_dir / name)
    lap["frame_size"] = [width, height]
    print(f"{n + 1}/{len(index['laps'])} {lap['lap_id']}", flush=True)
(dst / "index.json").write_text(json.dumps(index, indent=2))
print("done", dst)
