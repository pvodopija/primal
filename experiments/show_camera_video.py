"""
What training clips look like through the virtual camera, as video: the same six seconds of
a wide training lap level, with the head held turned, with strong shake, and as the sampler
draws them (a new random head pose and shake level for every ~0.4 s clip).

Writes runs/demos/training_view.gif (and .mp4). Usage: PYTHONPATH=. python experiments/show_camera_video.py
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from train.camera import render_clip
from train.dataset import LapIndex

TRACK, START, SECONDS, FPS, SCALE = "ks_laguna_seca", 0.40, 6.0, 20, 3
OUT = Path("runs/demos/training_view")

lap = next(l for l in sorted(LapIndex.load(Path("data/packed_ac_v3_wide"), split="train").by_track[TRACK],
                             key=lambda l: l.lap_id) if l.s_span > 0.98)
step = int(round(lap.fps / FPS))
first = int(lap.n_frames * START)
idx = np.arange(first, first + int(SECONDS * lap.fps), step)
raw, t = np.asarray(lap.frames()[idx]), lap.t()[idx].astype(np.float64)

panels = [
    ("straight ahead: what the model normally sees", render_clip(raw, t, np.random.default_rng(0))),
    ("head held 10 deg right", render_clip(raw, t, np.random.default_rng(0), pose=(10, 0, 0))),
    ("strong shake (level 2): vibration, kerb hits, blur", render_clip(raw, t, np.random.default_rng(4), shake_level=2.0)),
]
# As training draws them: every clip its own pose (yaw +-14, pitch +-4, roll +-8) and shake 0-2.
rng, mixed, captions, per_clip = np.random.default_rng(7), [], [], int(round(0.4 * FPS))
for a in range(0, len(idx), per_clip):
    pose = (rng.uniform(-14, 14), rng.uniform(-4, 4), rng.uniform(-8, 8))
    level = rng.uniform(0, 2)
    clip = render_clip(raw[a:a + per_clip], t[a:a + per_clip], rng, pose=pose, shake_level=level)
    mixed.append(clip)
    captions += [f"yaw {pose[0]:+.0f}  pitch {pose[1]:+.0f}  roll {pose[2]:+.0f}  shake {level:.1f}"] * len(clip)
panels.append(("training clips: new random pose + shake every 0.4 s", np.concatenate(mixed)))


def tile(frame: np.ndarray, title: str, note: str = "") -> np.ndarray:
    img = cv2.resize(frame, (frame.shape[1] * SCALE, frame.shape[0] * SCALE), interpolation=cv2.INTER_LINEAR)
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    if note:
        cv2.putText(img, note, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


frames = []
for k in range(len(idx)):
    tiles = [tile(p[k], title, captions[k] if i == 3 else "") for i, (title, p) in enumerate(panels)]
    gap = np.full((tiles[0].shape[0], 8, 3), 255, np.uint8)
    rows = [np.hstack([tiles[0], gap, tiles[1]]), np.hstack([tiles[2], gap, tiles[3]])]
    frames.append(np.vstack([rows[0], np.full((8, rows[0].shape[1], 3), 255, np.uint8), rows[1]]))

OUT.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(OUT.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), FPS, frames[0].shape[1::-1])
for f in frames:
    writer.write(f)
writer.release()
small = [cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])), interpolation=cv2.INTER_AREA) for f in frames]
gif = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in small]
gif[0].save(OUT.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=int(1000 / FPS), loop=0, optimize=True)
cv2.imwrite(str(OUT.with_suffix(".png")), frames[len(frames) // 2])
print(OUT.with_suffix(".gif"), f"{OUT.with_suffix('.gif').stat().st_size / 1e6:.1f} MB", frames[0].shape)
