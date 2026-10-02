"""What the virtual camera makes of a wide frame: head turns, tilt, shake with blur, Halo's view."""
from pathlib import Path
import cv2, numpy as np
from train.camera import HALO, TRAINING, WIDE, render, render_clip
from train.dataset import LapIndex

lap = next(l for l in sorted(LapIndex.load(Path("data/packed_ac_v3_wide"), split="train").by_track["ks_laguna_seca"], key=lambda l: l.lap_id) if l.s_span > 0.98)
i = int(lap.n_frames * 0.42)
wide = np.asarray(lap.frames()[i])
S = 3
big = lambda img: cv2.resize(img, (img.shape[1] * S, img.shape[0] * S), interpolation=cv2.INTER_NEAREST)
def label(img, text):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 22), (255, 255, 255), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA); return img
pad = lambda img, w: np.hstack([img, np.full((img.shape[0], w - img.shape[1], 3), 255, np.uint8)]) if img.shape[1] < w else img
rows = [label(big(wide), "the wide render: 121 x 87 degrees (268x144)")]
views = [("looking 14 deg left", (-14, 0, 0)), ("straight (the training view)", (0, 0, 0)), ("looking 14 deg right", (14, 0, 0))]
rows.append(np.hstack([label(big(render(wide, WIDE, TRAINING, *p)), t) for t, p in views]))
views = [("looking up 10 deg", (0, 10, 0)), ("head tilted 12 deg", (0, 0, 12)), ("Halo's view: 81 deg, 4:3", None)]
tiles = []
for t, p in views:
    img = render(wide, WIDE, HALO) if p is None else render(wide, WIDE, TRAINING, *p)
    tile = big(img)
    tiles.append(label(cv2.resize(tile, (int(tile.shape[1] * 240 / tile.shape[0]), 240)) if p is None else tile, t))
h = max(x.shape[0] for x in tiles)
rows.append(np.hstack([np.vstack([x, np.full((h - x.shape[0], x.shape[1], 3), 255, np.uint8)]) for x in tiles]))
times = lap.t()[i - 10:i + 2:2].astype(np.float64)
clip = render_clip(np.asarray(lap.frames()[i - 10:i + 2:2]), times, np.random.default_rng(3), shake_level=2.0, exposure_s=0.008)
rows.append(np.hstack([label(big(f), f"strong shake + blur, frame {k + 1}") for k, f in enumerate(clip[:3])]))
w = max(r.shape[1] for r in rows)
sheet = np.vstack([np.vstack([pad(r, w), np.full((8, w, 3), 255, np.uint8)]) for r in rows])
out = Path("runs/demos/virtual_camera.png"); cv2.imwrite(str(out), sheet); print(out, sheet.shape)
