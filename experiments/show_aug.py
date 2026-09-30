"""What the encoder gets: one packed frame and the head-pose augmentation of the recipe
(roll ±6°, yaw ±4°, pitch ±1°, zoom 1.08, mirroring), with the mirrored-fill areas marked."""
from pathlib import Path
import cv2, numpy as np
from train.dataset import LapIndex, camera_jitter

index = LapIndex.load(Path("data/packed_ac_v3"), split="train")
lap = next(l for l in sorted(index.by_track["ks_laguna_seca"], key=lambda l: l.lap_id) if l.s_span > 0.98)
frame = np.asarray(lap.frames()[int(lap.n_frames * 0.42)])
H, W = frame.shape[:2]
focal = (W / 2) / np.tan(np.radians(91.5 / 2))


def warp(img, roll, yaw, pitch, zoom=1.08, border=cv2.BORDER_REFLECT_101, value=0):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), roll, zoom)
    m[0, 2] += focal * np.tan(np.radians(yaw)); m[1, 2] += focal * np.tan(np.radians(pitch))
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=value)


def tile(img_bgr, title, sub, fill=None, scale=4):
    big = cv2.resize(img_bgr, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    if fill is not None:  # outline the mirrored fill
        m = cv2.resize(fill.astype(np.uint8) * 255, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        overlay = big.copy(); overlay[m > 0] = (255, 0, 255)
        big = cv2.addWeighted(overlay, 0.35, big, 0.65, 0)
        cv2.drawContours(big, contours, -1, (255, 0, 255), 2)
    head = np.full((58, big.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(head, sub, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([head, big])


ones = np.ones((H, W), np.uint8)
cases = [
    ("Head tilted right: roll +6, yaw +4, pitch +1", 6, 4, 1, False),
    ("Head tilted left: roll -6, yaw -4, pitch -1", -6, -4, -1, False),
    ("Mirrored, then roll +3, yaw -2", 3, -2, 0, True),
]
tiles = [tile(frame, "What the encoder gets (148 x 80 pixels)", f"{lap.track}, packed frame, no augmentation")]
for title, r, y, p, mirror in cases:
    src = frame[:, ::-1].copy() if mirror else frame
    out = warp(src, r, y, p)
    fill = warp(ones, r, y, p, border=cv2.BORDER_CONSTANT, value=0) == 0
    tiles.append(tile(out, title, f"zoom 1.08; magenta = mirrored edge fill, {100 * fill.mean():.0f}% of the frame", fill))
rng = np.random.default_rng(7)
typical = camera_jitter(frame[None], rng, zoom=(1.08, 1.08), pitch_deg=1.0, parts=frozenset({"pose"}))[0]
tiles.append(tile(typical, "A typical random draw, as training makes them", "recipe settings, one clip; the reference gets its own draw"))
worst = []
for r in (-6, 6):
    for y in (-4, 4):
        worst.append((warp(ones, r, y, 1, border=cv2.BORDER_CONSTANT, value=0) == 0).mean())
rng = np.random.default_rng(0); shares = []
for _ in range(2000):
    r, y, p = rng.uniform(-6, 6), rng.uniform(-4, 4), rng.uniform(-1, 1)
    shares.append((warp(ones, r, y, p, border=cv2.BORDER_CONSTANT, value=0) == 0).mean())
print(f"mirrored fill: worst corner case {100 * max(worst):.1f}% of the frame; random draws median {100 * np.median(shares):.1f}%, p90 {100 * np.percentile(shares, 90):.1f}%")
gap = np.full((14, tiles[0].shape[1], 3), 255, np.uint8)
left = np.vstack([tiles[0], gap, tiles[1], gap, tiles[2]])
right = np.vstack([tiles[4], gap, tiles[3], gap, np.full_like(tiles[2], 255)])
sheet = np.hstack([left, np.full((left.shape[0], 14, 3), 255, np.uint8), right])
out = "/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/augmentation_example.png"
cv2.imwrite(out, sheet); print(out, sheet.shape)
