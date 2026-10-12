"""
How much fine detail packed frames carry at the model's view: the variance of the
Laplacian of the grey image's upper half (the scenery), median over frames spread along
up to two whole laps per track. Wide packs are first turned into the training view by the
virtual camera, as training sees them. Recorded v3 laps score ~1130; replay re-renders
(packed_ac_v3_wide ~530, b1007 ~440) carry less than half of that detail, which costs
the model precision, so a new set should score near v3 before it trains.

Usage: PYTHONPATH=. python experiments/sharpness.py data/packed_ac_v3 data/packed_ac_new_live [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from train import camera
from train.dataset import LapIndex

FRAMES_PER_LAP, LAPS_PER_TRACK = 40, 2


def detail(img: np.ndarray) -> float:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[: img.shape[0] // 2].astype(np.float32)
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


def main() -> None:
    for path in sys.argv[1:]:
        index = LapIndex.load(Path(path), split=None, variants=None)
        per_track = {}
        for track, laps in sorted(index.by_track.items()):
            values = []
            for lap in sorted(laps, key=lambda x: -x.s_span)[:LAPS_PER_TRACK]:
                for i in np.linspace(0, lap.n_frames - 1, FRAMES_PER_LAP).astype(int):
                    f = np.asarray(lap.frames()[i])
                    values.append(detail(camera.render(f, lap.source_view, camera.TRAINING) if lap.wide else f))
            per_track[track] = float(np.median(values))
        print(f"{path}: detail {np.median(list(per_track.values())):.0f} (median over {len(per_track)} tracks)")
        for track, v in per_track.items():
            print(f"  {track:<40} {v:7.0f}")


if __name__ == "__main__":
    main()
