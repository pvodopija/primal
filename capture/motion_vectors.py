"""
Motion vectors from a recorded mp4, as a live video encoder would produce them.

    python -m capture.motion_vectors extract data/sessions/<run>/video.mp4
    python -m capture.motion_vectors extract data/sessions/<run>/video.mp4 --height 360

A video encoder searches, for every block of every frame, where that block came
from in the previous frame. That search is optical flow at full resolution, and
every phone and camera already runs it in hardware. This measures what it
carries: the recording is re-encoded with x264 the way a live encoder runs
(P-frames only, one reference frame, no keyframes after the first, so every
vector spans exactly one frame), decoded with FFmpeg's motion-vector export,
and pooled onto a coarse grid.

OBS's own vectors are not read directly: its x264 uses B-frames, so a vector
can span up to four frames and FFmpeg does not say how many.

Writes `motion_<height>p.npz` next to the video:

    frame   [F]            video frame index, the same count `labels.parquet` uses
    flow    [F, 2, H, W]   mean displacement since the previous frame, in encoded
                           pixels (x right, y down); float16
    cover   [F, H, W]      share of the cell that was inter-coded; 0 where the
                           encoder gave up and intra-coded it; float16
    cell, fps, source_size, encoded_size, encoder

When `labels.parquet` and `run.json` sit next to the video, it also prints a
first check: how well the mean outward (radial) flow follows the car's speed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np

SPEED_WINDOW = 6  # label frames either side for the check's speed estimate


def pool_vectors(
    vectors: np.ndarray, width: int, height: int, cell: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Area-weighted mean displacement since the previous frame per grid cell, and
    the share of each cell the vectors cover.

    `vectors` is FFmpeg's AVMotionVector array. Only past references are kept.
    FFmpeg stores where a block came from (`src = dst + motion / scale`), so the
    displacement of the content is `-motion / scale`.
    """
    grid_w, grid_h = -(-width // cell), -(-height // cell)
    flow = np.zeros((2, grid_h, grid_w), np.float32)
    cover = np.zeros((grid_h, grid_w), np.float32)
    past = vectors[vectors["source"] < 0]
    if past.size == 0:
        return flow, cover

    area = past["w"].astype(np.float64) * past["h"]
    scale = past["motion_scale"].astype(np.float64)
    col = np.clip(past["dst_x"] // cell, 0, grid_w - 1)
    row = np.clip(past["dst_y"] // cell, 0, grid_h - 1)
    idx = (row * grid_w + col).astype(np.int64)
    n = grid_w * grid_h

    weight = np.bincount(idx, area, n)
    fx = np.bincount(idx, area * -past["motion_x"] / scale, n)
    fy = np.bincount(idx, area * -past["motion_y"] / scale, n)
    hit = weight > 0
    flow[0].flat[hit] = fx[hit] / weight[hit]
    flow[1].flat[hit] = fy[hit] / weight[hit]

    cell_w = np.minimum(cell, width - np.arange(grid_w) * cell)
    cell_h = np.minimum(cell, height - np.arange(grid_h) * cell)
    cover[:] = np.clip(weight.reshape(grid_h, grid_w) / np.outer(cell_h, cell_w), 0.0, 1.0)
    return flow, cover


def encoder_motion(
    frames: Iterable,
    width: int,
    height: int,
    fps: float,
    cell: int,
    preset: str = "veryfast",
    crf: int = 18,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Re-encode `frames` (av.VideoFrame, any size and format) P-frames-only and
    yield `(index, flow, cover)` per frame, in order. Frame 0 is the keyframe
    and has no vectors.
    """
    import av
    from av.sidedata.motionvectors import MotionVectors

    rate = Fraction(fps).limit_denominator(1001)
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = width, height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = 1 / rate
    encoder.framerate = rate
    encoder.options = {
        "preset": preset,
        "crf": str(crf),
        "x264-params": "bframes=0:ref=1:keyint=infinite:scenecut=0",
    }
    decoder = av.CodecContext.create("h264", "r")
    decoder.options = {"flags2": "+export_mvs"}

    emitted = 0

    def pooled(decoded) -> tuple[int, np.ndarray, np.ndarray]:
        # P-frames only, so frames come out in the order they went in; the pts
        # check catches an encoder that reorders or drops anyway.
        nonlocal emitted
        if decoded.pts != emitted:
            raise RuntimeError(f"frame {emitted} came back as pts {decoded.pts}")
        emitted += 1
        vectors = next((sd for sd in decoded.side_data if isinstance(sd, MotionVectors)), None)
        if vectors is None:  # the keyframe
            grid = (-(-height // cell), -(-width // cell))
            return int(decoded.pts), np.zeros((2, *grid), np.float32), np.zeros(grid, np.float32)
        return int(decoded.pts), *pool_vectors(vectors.to_ndarray(), width, height, cell)

    def drain(packets) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for packet in packets:
            for decoded in decoder.decode(packet):
                yield pooled(decoded)

    for index, frame in enumerate(frames):
        frame = frame.reformat(width=width, height=height, format="yuv420p")
        # A decoded frame keeps its container's time base; left alone, the
        # encoder would rescale these indices into it.
        frame.pts, frame.time_base = index, encoder.time_base
        yield from drain(encoder.encode(frame))
    yield from drain(encoder.encode(None))
    for decoded in decoder.decode(None):
        yield pooled(decoded)


def radial_flow(
    flow: np.ndarray, cover: np.ndarray, cell: int, size: tuple[int, int], keep_rows: tuple[float, float]
) -> np.ndarray:
    """
    Mean outward flow per frame, in encoded pixels. Driving forward makes the
    scene stream away from the image centre; a turn of the head or the car adds
    a sideways shift that mostly cancels between the left and right halves.
    """
    frames, _, grid_h, grid_w = flow.shape
    cx = (np.arange(grid_w) + 0.5) * cell
    cy = (np.arange(grid_h) + 0.5) * cell
    dx = cx[None, :] - size[0] / 2
    dy = cy[:, None] - size[1] / 2
    radius = np.hypot(dx, dy)
    usable = (radius > 2 * cell) & (cy[:, None] >= keep_rows[0]) & (cy[:, None] < keep_rows[1])
    ux, uy = np.where(usable, dx / np.maximum(radius, 1e-6), 0), np.where(usable, dy / np.maximum(radius, 1e-6), 0)
    outward = flow[:, 0].astype(np.float32) * ux + flow[:, 1].astype(np.float32) * uy
    weight = cover.astype(np.float32) * usable
    return (outward * weight).sum((1, 2)) / np.maximum(weight.sum((1, 2)), 1e-6)


def _speed_check(session: Path, frames: np.ndarray, radial: np.ndarray, fps: float) -> None:
    import pandas as pd

    labels = pd.read_parquet(session / "labels.parquet").sort_values("frame_idx")
    length = float(json.loads((session / "run.json").read_text())["track_length_m"])
    fresh = labels.counter.diff().fillna(1).to_numpy() != 0  # OBS repeated a render: no motion
    good = labels[labels.valid.to_numpy() & fresh]
    idx = good.frame_idx.to_numpy()
    x = np.unwrap(good.spline_pos.to_numpy(np.float64) * 2 * np.pi) / (2 * np.pi) * length
    w = SPEED_WINDOW
    speed = np.full(idx.size, np.nan)
    speed[w:-w] = (x[2 * w :] - x[: -2 * w]) / ((idx[2 * w :] - idx[: -2 * w]) / fps)
    lookup = dict(zip(frames.tolist(), radial.tolist()))
    r = np.array([lookup.get(int(i), np.nan) for i in idx])
    ok = np.isfinite(speed) & np.isfinite(r) & (speed > 2) & (speed < 120) & (r != 0)
    if ok.sum() < 100:
        print(f"check: only {int(ok.sum())} frames joined to labels; nothing to report")
        return
    speed, r = speed[ok], r[ok]
    print(f"check over {ok.sum()} labelled frames: radial flow vs speed, Pearson r = {np.corrcoef(speed, r)[0, 1]:.3f}")
    for lo, hi in [(0, 15), (15, 25), (25, 35), (35, 45), (45, 120)]:
        band = (speed >= lo) & (speed < hi)
        if band.sum() >= 30:
            ratio = r[band] / speed[band]
            print(
                f"  {lo:3d}-{hi:<3d} m/s  {band.sum():6d} frames  radial px/frame median {np.median(r[band]):6.3f}"
                f"  px per (m/s) median {np.median(ratio):.4f}  spread p10-p90 {np.percentile(ratio, 10):.4f}-{np.percentile(ratio, 90):.4f}"
            )


def cmd_extract(args: argparse.Namespace) -> None:
    import av

    video = Path(args.video)
    session = video.parent
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)
        src_w, src_h = stream.codec_context.width, stream.codec_context.height
        height = args.height or src_h
        width = int(round(src_w * height / src_h / 2)) * 2
        total = stream.frames or None
        print(f"{video}: {src_w}x{src_h} at {fps:.2f} fps -> P-only x264 {width}x{height}, cell {args.cell} px")

        def source_frames():
            for i, frame in enumerate(container.decode(stream)):
                if args.limit_frames and i >= args.limit_frames:
                    return
                yield frame

        frames, flows, covers = [], [], []
        started = time.monotonic()
        for index, flow, cover in encoder_motion(
            source_frames(), width, height, fps, args.cell, args.preset, args.crf
        ):
            frames.append(index)
            flows.append(flow.astype(np.float16))
            covers.append(cover.astype(np.float16))
            if index and index % 5000 == 0:
                rate = index / (time.monotonic() - started)
                left = f", ~{(total - index) / rate / 60:.0f} min left" if total else ""
                print(f"  {index} frames, {rate:.0f} frames/s{left}")

    frame_arr = np.asarray(frames, np.int32)
    flow_arr, cover_arr = np.stack(flows), np.stack(covers)
    out = Path(args.out) if args.out else session / f"motion_{height}p.npz"
    np.savez_compressed(
        out,
        frame=frame_arr,
        flow=flow_arr,
        cover=cover_arr,
        cell=args.cell,
        fps=fps,
        source_size=np.array([src_w, src_h]),
        encoded_size=np.array([width, height]),
        encoder=f"libx264 {args.preset} crf {args.crf} bframes=0 ref=1",
    )
    elapsed = time.monotonic() - started
    print(f"{frame_arr.size} frames in {elapsed / 60:.1f} min -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"inter-coded share: median {np.median(cover_arr[1:].astype(np.float32).mean((1, 2))):.2f}")

    if (session / "labels.parquet").exists() and (session / "run.json").exists():
        keep = (0, height)
        overlay = session / "overlay.json"
        if overlay.exists():
            from capture.pack import crop_rows
            from capture.timecode import OverlayGeometry

            lo, hi = crop_rows(src_h, OverlayGeometry.from_dict(json.loads(overlay.read_text())))
            keep = (lo * height / src_h, hi * height / src_h)
        _speed_check(session, frame_arr, radial_flow(flow_arr, cover_arr, args.cell, (width, height), keep), fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ext = sub.add_parser("extract", help="re-encode P-only and write pooled motion vectors")
    ext.add_argument("video")
    ext.add_argument("--height", type=int, default=None, help="encode at this height (default: the recording's)")
    ext.add_argument("--cell", type=int, default=32, help="grid cell size in encoded pixels")
    ext.add_argument("--preset", default="veryfast")
    ext.add_argument("--crf", type=int, default=18)
    ext.add_argument("--limit-frames", type=int, default=0, help="stop after this many frames (a quick trial)")
    ext.add_argument("--out", default=None)
    ext.set_defaults(func=cmd_extract)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
