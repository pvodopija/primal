"""
Demo video: a turned head, what it costs, and what the head-angle search gives back.

Top: the wide render with the straight-ahead view (grey) and the turned view (yellow), so
the size of the turn is visible. Middle: what the glasses see; the reference frame where
the delta places the kart without the search; the same with the search (turned by the
angle it picked). Bottom: the error of both over the last 15 s, the ±100 ms budget shaded.
Held-out track, a wide lap against another lap of the same session, MobileNet trained on
the wide renders, the two-mode reference-time tracker.

Usage: PYTHONPATH=. python experiments/show_yaw_demo.py held14|corners [RUN]
       (writes runs/demos/yaw_<mode>.mp4 and .gif)
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from experiments.yaw_search import OFFSETS, corner_look, encode, per_offset, render_turning, searched
from train import camera
from train.dataset import LapIndex
from train.eval import REFERENCE_TIME_TWO_MODES, STREAM_HZ, EstimatorConfig, _run_estimator, load_model
from train.model import _interp_circular
from train.train import default_device

TRACK, SECONDS, S = "ks_silverstone__national", 30.0, 3
GREEN, RED, YELLOW, GREY, WHITE, INK = (90, 190, 60), (60, 60, 220), (0, 215, 255), (170, 170, 170), (255, 255, 255), (40, 40, 40)


def outline(yaw: float) -> np.ndarray:
    """The turned training view's border, in wide-render pixels (times S)."""
    h = camera.WIDE.K @ camera.rotation(yaw, 0, 0) @ np.linalg.inv(camera.TRAINING.K)
    w, v = camera.TRAINING.width, camera.TRAINING.height
    edge = [(u, 0) for u in np.linspace(0, w, 20)] + [(w, y) for y in np.linspace(0, v, 10)] + \
           [(u, v) for u in np.linspace(w, 0, 20)] + [(0, y) for y in np.linspace(v, 0, 10)]
    p = h @ np.array([[u, y, 1.0] for u, y in edge]).T
    return np.round((p[:2] / p[2]).T * S).astype(np.int32)


def label(img: np.ndarray, text: str, colour=INK, scale=0.55) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def big(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (img.shape[1] * S, img.shape[0] * S), interpolation=cv2.INTER_LINEAR)


def gauge(yaw: float, size: int = 120) -> np.ndarray:
    """Seen from above: the kart's heading (grey) and where the head points (yellow)."""
    g = np.full((size, size, 3), 255, np.uint8)
    c = (size // 2, size - 20)
    cv2.circle(g, c, 8, INK, -1)
    for angle, colour, thick in ((0.0, GREY, 2), (yaw, YELLOW, 3)):
        a = np.radians(angle)
        cv2.line(g, c, (int(c[0] + 85 * np.sin(a)), int(c[1] - 85 * np.cos(a))), colour, thick, cv2.LINE_AA)
    cv2.putText(g, f"{yaw:+.0f} deg", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return g


def plot(no_search: np.ndarray, search: np.ndarray, i: int, width: int, height: int = 190) -> np.ndarray:
    img = np.full((height, width, 3), 255, np.uint8)
    span, lim = int(15 * STREAM_HZ), 400.0
    y = lambda ms: int(height / 2 - np.clip(ms, -lim, lim) / lim * (height / 2 - 12))
    cv2.rectangle(img, (0, y(100)), (width, y(-100)), (225, 245, 225), -1)
    cv2.line(img, (0, y(0)), (width, y(0)), GREY, 1)
    for ms in (-400, -100, 100, 400):
        cv2.putText(img, f"{ms:+d} ms", (4, y(ms) + (12 if ms > 0 else -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    lo = max(0, i - span)
    for series, colour in ((no_search, RED), (search, GREEN)):
        pts = [(int(width - (i - j) * width / span), y(series[j])) for j in range(lo, i + 1)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, colour, 2, cv2.LINE_AA)
    cv2.putText(img, "delta error, last 15 s:  red = no search,  green = head-angle search,  shaded = within 100 ms",
                (70, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
    return img


def main() -> None:
    mode = sys.argv[1]
    run = sys.argv[2] if len(sys.argv) > 2 else "v3_mobilenet_wide_s0"
    device = torch.device(default_device())
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    clip_len = int(payload["args"]["clip_len"])
    group = LapIndex.load(Path("data/packed_ac_v3_wide"), split="holdout").by_track[TRACK]
    whole = [lap for lap in group if lap.s_span > 0.98 and lap.usable_as_reference(0.98, "time")]
    live, reference = whole[0], whole[1]
    grid = reference.reference_grid("time")
    ref_raw = np.asarray(reference.frames()[grid.frame_idx])
    rng = np.random.default_rng(zlib.crc32(reference.lap_id.encode()))
    refs = [encode(model, camera.render_clip(ref_raw, np.zeros(grid.n_bins), rng, pose=(float(o), 0, 0)), device) for o in OFFSETS]

    every = max(int(round(live.fps / STREAM_HZ)), 1)
    ticks = np.arange((clip_len - 1) * 2, live.n_frames, every)
    clip_idx = ticks[:, None] - np.arange(clip_len - 1, -1, -1)[None, :] * 2
    used = np.unique(clip_idx)
    row = np.full(live.n_frames, -1)
    row[used] = np.arange(used.size)
    raw, t = np.asarray(live.frames()[used]), live.t().astype(np.float64)
    if mode == "held14":
        yaw = np.full(used.size, 14.0)
    else:
        yaw = corner_look(camera.render_clip(raw, t[used], np.random.default_rng(0)), t[used])
    frames = render_turning(raw, yaw)
    beliefs, scores = per_offset(model, encode(model, frames, device), row[clip_idx], refs)
    centre = int(np.argmin(np.abs(OFFSETS)))
    found, pick = searched(beliefs, scores, np.ones(len(OFFSETS), bool), int(10 * STREAM_HZ))
    stream = {"grid": grid, "t": t[ticks]}
    times = torch.from_numpy(grid.time_s.astype(np.float32))
    true = _interp_circular(times, torch.from_numpy(grid.target(live.s()[ticks]).astype(np.float32)), grid.lap_time_s).numpy()
    bins, err = {}, {}
    for name, b in (("no search", beliefs[centre]), ("search", found)):
        bins[name] = _run_estimator({**stream, "belief": b.astype(np.float64)}, EstimatorConfig(**REFERENCE_TIME_TWO_MODES),
                                    None, None, 0, reference_time=True).numpy()
        est = _interp_circular(times, torch.from_numpy(bins[name]).float(), grid.lap_time_s).numpy()
        err[name] = ((est - true + grid.lap_time_s / 2) % grid.lap_time_s - grid.lap_time_s / 2) * 1000
    skip = int(2 * STREAM_HZ)
    whole_lap = {k: 100 * np.mean(np.abs(v[skip:]) > 100) for k, v in err.items()}
    print(f"{mode}: whole lap over 100 ms: no search {whole_lap['no search']:.1f}%, search {whole_lap['search']:.1f}%")

    # the 30 s with the most head movement (corners) or a fixed stretch (held)
    tick_yaw = yaw[row[ticks]]
    n = int(SECONDS * STREAM_HZ)
    start = skip if mode == "held14" else int(np.argmax(np.convolve(np.abs(tick_yaw), np.ones(n), "valid")[skip:]) + skip)
    out = []
    for i in range(start, min(start + n, ticks.size)):
        f = ticks[i]
        wide_img = big(np.asarray(live.frames()[f]))
        cv2.polylines(wide_img, [outline(0.0)], True, GREY, 2, cv2.LINE_AA)
        cv2.polylines(wide_img, [outline(float(tick_yaw[i]))], True, YELLOW, 3, cv2.LINE_AA)
        side = np.full((wide_img.shape[0], 260, 3), 255, np.uint8)
        side[20:140, 70:190] = gauge(float(tick_yaw[i]))
        for k, text in enumerate(["seen from above:", "grey = kart heading", "yellow = head", "",
                                  f"lap time {t[f] - t[0]:5.1f} s"]):
            cv2.putText(side, text, (20, 170 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
        top = label(np.hstack([side, wide_img]),
                    f"the wide picture: grey box = looking straight, yellow box = where the head is turned ({tick_yaw[i]:+.0f} deg)")
        tiles = [label(big(frames[row[f]]), "what the glasses see")]
        for name, offset in (("no search", 0.0), ("search", float(OFFSETS[pick[i]]))):
            b = int(round(bins[name][i])) % grid.n_bins
            view = camera.render(ref_raw[b], camera.WIDE, camera.TRAINING, yaw=offset)
            e = err[name][i]
            title = (f"reference where the delta says you are: {e:+.0f} ms" if name == "no search"
                     else f"with the search (reference turned {offset:+.1f} deg): {e:+.0f} ms")
            tiles.append(label(big(view), title, GREEN if abs(e) <= 100 else RED, 0.5))
        middle = np.hstack([tiles[0], np.full((tiles[0].shape[0], 6, 3), 255, np.uint8), tiles[1],
                            np.full((tiles[0].shape[0], 6, 3), 255, np.uint8), tiles[2]])
        width = max(top.shape[1], middle.shape[1])
        pad = lambda im: np.hstack([im, np.full((im.shape[0], width - im.shape[1], 3), 255, np.uint8)])
        foot = np.full((28, width, 3), 255, np.uint8)
        cv2.putText(foot, f"whole lap, share of moments over 100 ms:  no search {whole_lap['no search']:.1f}%   "
                          f"with the head-angle search {whole_lap['search']:.1f}%", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
        out.append(np.vstack([pad(top), pad(middle), plot(err["no search"], err["search"], i, width), foot]))

    dest = Path(f"runs/demos/yaw_{mode}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), STREAM_HZ, out[0].shape[1::-1])
    for f in out:
        writer.write(f)
    writer.release()
    small = [cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])), interpolation=cv2.INTER_AREA) for f in out[::3]]
    gif = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in small]
    gif[0].save(dest.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=int(3000 / STREAM_HZ), loop=0, optimize=True)
    cv2.imwrite(str(dest.with_suffix(".png")), out[len(out) // 2])
    print(dest.with_suffix(".gif"), f"{dest.with_suffix('.gif').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
