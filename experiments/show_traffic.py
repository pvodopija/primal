"""What the traffic occluders look like: training "blocks" and the test "kart", on real frames."""
from pathlib import Path
import cv2, numpy as np
from train.dataset import LapIndex, draw_traffic

laps = sorted(LapIndex.load(Path("data/packed_ac_v3"), split="train").by_track["ks_laguna_seca"], key=lambda l: l.lap_id)
lap = next(l for l in laps if l.s_span > 0.98)
rows = []
for style, seed in (("blocks", 1), ("blocks", 4), ("kart", 2), ("kart", 7)):
    i = int(lap.n_frames * (0.15 + 0.2 * seed % 1))
    idx = np.arange(i, i + 23, 2)
    frames = np.asarray(lap.frames()[idx])
    out = draw_traffic(frames, np.random.default_rng(seed), lap.t()[idx].astype(np.float64), style=style)
    tile = lambda f: cv2.resize(f, (f.shape[1] * 3, f.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    row = np.hstack([tile(out[0]), np.full((240, 6, 3), 255, np.uint8), tile(out[-1])])
    label = np.full((26, row.shape[1], 3), 255, np.uint8)
    cv2.putText(label, f"{style}: first and last frame of one clip", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    rows += [label, row]
path = Path("runs/demos/traffic_examples.png")
cv2.imwrite(str(path), np.vstack(rows)); print(path)
