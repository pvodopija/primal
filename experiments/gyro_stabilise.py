"""
A GoPro video steadied by its own gyroscope (experiments/gpmf_read.py's GYRO), as Gyroflow
does, without Gyroflow: the camera's orientation integrated from the gyro, smoothed with a
Gaussian (SIGMA_S), and every frame turned from the orientation it had to the smoothed one
through the lens (an equidistant fisheye, the HERO7's wide lens near enough), row by row for
the rolling shutter.

The camera axes, the sync and the lens come from the video itself (`sync`): frame-to-frame
image motion in the scenery (LK flow, a similarity per frame pair) against the gyro
integrated over the same interval. Pavle's night session (HERO7 Black, 1080p60 wide, Auto
Low Light: 1/30 s exposures, each picture written twice): image x follows gyro column 2
(corr 0.94), image y column 1 (0.97), roll column 0 (0.89), the gyro 28 ms behind the frame
times, 890 px a radian at 1920 wide.

`render` writes ROOT/stream_<name>.mp4 at 1280x720, overlays (the set's masks) greyed out
before the warp, so the real-footage scripts read it with REAL_STREAM=<name>; a frame that
repeats the last picture takes the same turn, or the repeat would shake. `preview` writes
the raw and the steadied side by side for a few seconds. `check` measures what is left:
frame-to-frame motion of the steadied video against the raw.

Usage: PYTHONPATH=. python experiments/gyro_stabilise.py render|preview|check REAL_FOOTAGE_SET NAME [SIGMA_S]
       (SIGMA_S 0: the control, the same lens and zoom with no turn)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# the night session's HERO7, measured with `sync`'s method (see the docstring)
AXES = ((1, 1.0), (2, -1.0), (0, -1.0))  # camera x (right), y (down), z (forward) rate: (gyro column, sign)
OFFSET_S = -0.028  # gyro time = frame time + this
FOCAL_1080 = 890.0  # equidistant fisheye, pixels a radian at 1920x1080
READOUT_S = 0.0088  # rolling shutter, top row to bottom (HERO7 1080p)
SIGMA_S = 0.25  # the smoothing: shake and quick nods go, head turns into corners stay
ZOOM = float(os.environ.get("STAB_ZOOM", 1.2))  # the steadied view's focal over the lens's: turned edges stay off the picture
OUT_W, OUT_H = 1280, 720
GRID = 16  # the warp computed every GRID output pixels, interpolated between
REPEAT = 0.6  # mean abs difference (0-255) of 64x36 thumbnails under which a frame repeats the last


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def orientations(gpmf: Path, sigma_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gyro times (on the video's clock), the camera's orientation (world from camera) and its
    smoothed path, as unit quaternions."""
    z = np.load(gpmf)
    g, t = z["GYRO"], z["GYRO_t"] - OFFSET_S
    w = np.stack([s * g[:, c] for c, s in AXES], 1)
    dt = np.diff(t, prepend=t[0] - np.median(np.diff(t)))
    angle = np.linalg.norm(w, axis=1) * dt
    axis = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
    steps = np.concatenate([np.cos(angle / 2)[:, None], axis * np.sin(angle / 2)[:, None]], 1)
    q = np.empty_like(steps)
    q[0] = (1, 0, 0, 0)
    for i in range(1, len(q)):  # body rates: each step turns the camera in its own frame
        q[i] = qmul(q[i - 1], steps[i])
        q[i] /= np.linalg.norm(q[i])
    k = sigma_s / np.median(np.diff(t))
    x = np.arange(-int(4 * k), int(4 * k) + 1)
    kernel = np.exp(-0.5 * (x / k) ** 2)
    kernel /= kernel.sum()
    pad = np.pad(q, ((len(x) // 2, len(x) // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, c], kernel, mode="valid") for c in range(4)], 1)
    return t, q, smooth / np.linalg.norm(smooth, axis=1, keepdims=True)


def at(t: np.ndarray, q: np.ndarray, when: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(t, when) - 1, 0, len(t) - 2)
    a = np.clip((when - t[i]) / (t[i + 1] - t[i]), 0, 1)[..., None]
    out = q[i] * (1 - a) + q[i + 1] * a
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class Warp:
    """Output pixel -> source pixel for a frame: the ray the steadied camera sees through it,
    turned into the camera as it was when that row was read."""

    def __init__(self, src_w: int, src_h: int):
        self.f = FOCAL_1080 * src_w / 1920
        self.c = np.array([src_w / 2, src_h / 2])
        self.src_h = src_h
        gx, gy = np.meshgrid(np.arange(0, OUT_W + GRID, GRID), np.arange(0, OUT_H + GRID, GRID))
        f_out = self.f * ZOOM * OUT_W / src_w
        p = np.stack([(gx - OUT_W / 2) / f_out, (gy - OUT_H / 2) / f_out], -1)
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        self.rays = np.concatenate([np.sin(r) * p / np.maximum(r, 1e-12), np.cos(r)], -1)  # equidistant
        self.row_frac = gy / OUT_H  # the source row is near enough the output row for the readout time


def _grid_maps(warp: Warp, t, q, smooth, when):
    """The grid's maps interpolated to every output pixel (exact at grid points); with no
    smoothed path (SIGMA 0, the control) only the lens's zoom, no turn."""
    rows = when + (warp.row_frac[:, 0] - 0.5) * READOUT_S
    if smooth is None:
        turn = np.broadcast_to(np.eye(3), (len(rows), 3, 3))
    else:
        actual = qmat(at(t, q, rows))
        steady = qmat(at(t, smooth, np.full(len(rows), when)))
        turn = np.einsum("rji,rjk->rik", actual, steady)
    d = np.einsum("rij,rcj->rci", turn, warp.rays)
    theta = np.arctan2(np.linalg.norm(d[..., :2], axis=-1), d[..., 2])
    rho = np.maximum(np.linalg.norm(d[..., :2], axis=-1), 1e-12)
    uv = (warp.c + warp.f * theta[..., None] * d[..., :2] / rho[..., None]).astype(np.float32)
    # grid point (i, j) sits at output pixel (j*GRID, i*GRID): resize maps grid corners to pixel corners
    gh, gw = uv.shape[:2]
    full_w, full_h = (gw - 1) * GRID + 1, (gh - 1) * GRID + 1
    mx = cv2.resize(uv[..., 0], (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    my = cv2.resize(uv[..., 1], (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return mx[:OUT_H, :OUT_W], my[:OUT_H, :OUT_W]


def setting(name: str) -> tuple[Path, dict]:
    os.environ["REAL_FOOTAGE"] = name
    from experiments import real_footage as rf
    return rf.ROOT, rf._S


def frames_with_times(video: Path, start_s: float = 0.0, end_s: float = 1e9):
    """Each frame, its index, and the time of the picture it shows (a repeat keeps the last's)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    last, shown = None, 0.0
    while True:
        ok, f = cap.read()
        if not ok or i / fps > end_s:
            break
        thumb = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if last is None or np.abs(thumb - last).mean() >= REPEAT:
            shown = i / fps
        last = thumb
        yield i, f, shown
        i += 1


def writer(path: Path, w: int, h: int, fps: str = "60000/1001") -> subprocess.Popen:
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps,
                             "-i", "-", "-c:v", "h264_videotoolbox", "-b:v", "14M", "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE)


def masked(frame: np.ndarray, s: dict) -> np.ndarray:
    for x0, y0, x1, y1 in s.get("masks", []):
        frame[y0:y1, x0:x1] = 128
    return frame


def render(root: Path, s: dict, name: str, sigma_s: float, preview: bool) -> None:
    video = root / s["video"].split("/", 1)[1]
    sd = video.with_name(video.stem + "_sd.MP4")  # the file off the SD card, with its telemetry
    t, q, smooth = orientations(root / "gpmf.npz", max(sigma_s, 0.01))
    smooth = smooth if sigma_s > 0 else None
    cap = cv2.VideoCapture(str(sd))
    warp = Warp(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if preview:
        start = float(os.environ.get("PREVIEW_START", s["crossings"][3]))
        out_path = root / f"preview_{name}.mp4"
        pipe = writer(out_path, 2 * 960, 540)
        frames = frames_with_times(sd, start, start + float(os.environ.get("PREVIEW_S", 20)))
    else:
        out_path = root / f"stream_{name}.mp4"
        pipe = writer(out_path, OUT_W, OUT_H)
        frames = frames_with_times(sd)
    turns = []
    for i, f, shown in frames:
        mx, my = _grid_maps(warp, t, q, smooth, shown)
        if preview:
            raw = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA)
            out = cv2.resize(cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderValue=(0, 0, 0)), (960, 540), interpolation=cv2.INTER_AREA)
            cv2.putText(raw, "as recorded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(out, f"gyro-steadied ({sigma_s:.2f} s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = np.hstack([raw, out])
        else:
            frame = cv2.remap(masked(f, s), mx, my, cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        pipe.stdin.write(np.ascontiguousarray(frame).tobytes())
        if smooth is not None:
            rel = qmul(at(t, q, np.array([shown]))[0] * np.array([1, -1, -1, -1]), at(t, smooth, np.array([shown]))[0])
            turns.append(np.degrees(2 * np.arccos(min(abs(rel[0]), 1.0))))
        if i % 3000 == 0:
            print(f"  frame {i}", flush=True)
    pipe.stdin.close()
    pipe.wait()
    turns = np.array(turns) if turns else np.zeros(1)
    print(f"{out_path}: {len(turns)} frames; the turn to the steady path median {np.median(turns):.2f} deg, "
          f"p95 {np.percentile(turns, 95):.2f}, max {turns.max():.2f}")


def motion(video: Path, start: float, end: float, crop: tuple[float, float, float, float]) -> np.ndarray:
    """Frame-to-frame shift (px at 960 wide) and roll between distinct pictures in a crop (fractions)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    prev, rows = None, []
    for _ in range(int((end - start) * fps)):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None and np.abs(cv2.resize(g, (64, 36)).astype(np.float32) - cv2.resize(prev, (64, 36))).mean() < REPEAT:
            continue  # a repeat: no motion to measure
        if prev is not None:
            m = np.zeros_like(g)
            x0, x1, y0, y1 = (int(v * k) for v, k in zip(crop, (960, 960, 540, 540)))
            m[y0:y1, x0:x1] = 255
            p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=m)
            if p0 is not None and len(p0) > 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok_ = st.ravel() == 1
                A, inl = cv2.estimateAffinePartial2D(p0[ok_] - [480, 270], p1[ok_] - [480, 270], method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
                if A is not None and inl.sum() > 10:
                    rows.append((A[0, 2], A[1, 2], np.degrees(np.arctan2(A[1, 0], A[0, 0]))))
        prev = g
    return np.array(rows)


def check(root: Path, s: dict, name: str) -> None:
    video = root / s["video"].split("/", 1)[1]
    start, end = s["crossings"][3], s["crossings"][4]
    report = {}
    # the same scenery in both: the model's crop, less its masked bottom
    for label, path, crop in (("as recorded", video.with_name(video.stem + "_sd.MP4"), (0.18, 0.82, 0.02, 0.5)),
                              (f"steadied {name}", root / f"stream_{name}.mp4", (0.18, 0.82, 0.02, 0.5))):
        m = motion(path, start, end, crop)
        report[label] = {"pairs": len(m), "shift_px_median": float(np.median(np.hypot(m[:, 0], m[:, 1]))),
                         "shift_px_p90": float(np.percentile(np.hypot(m[:, 0], m[:, 1]), 90)),
                         "roll_deg_median": float(np.median(np.abs(m[:, 2]))),
                         "shift_jitter_px": float(np.std(np.diff(m[:, :2], axis=0)))}
        print(f"{label:<28} {len(m)} pairs: shift median {report[label]['shift_px_median']:.2f} px, "
              f"p90 {report[label]['shift_px_p90']:.2f}, roll median {report[label]['roll_deg_median']:.3f} deg, "
              f"jitter {report[label]['shift_jitter_px']:.2f} px (960 wide, lap 4)")
    (root / f"stabilise_{name}.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    mode, footage, name = sys.argv[1:4]
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else SIGMA_S
    root, s = setting(footage)
    if mode == "check":
        check(root, s, name)
    else:
        render(root, s, name, sigma, preview=mode == "preview")


if __name__ == "__main__":
    main()
