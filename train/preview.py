"""
Look at packed laps: what actually reaches the network.

    python -m train.preview list  --data data/packed_synth
    python -m train.preview video --data data/packed_synth --lap synth00__lap00
    python -m train.preview pair  --data data/packed_synth --track synth00
    python -m train.preview sheet --data data/packed_synth --lap synth00__lap00
    python -m train.preview clip  --data data/packed_synth
    python -m train.preview infer --data data/packed_synth --checkpoint runs/<name>/best.pt

`video` plays one lap. `pair` plays a live lap beside the reference frame at the
same track position, which is the comparison the model is asked to make.
`sheet` is a contact sheet sampled evenly along the lap. `clip` shows one
training sample exactly as the sampler produces it, photometric jitter and
stride included. `infer` runs a trained checkpoint along a lap and plots the
belief over the reference beside the frame it picked.

Upscaling uses nearest-neighbour on purpose. The frames really are 160x96 or
smaller, and a smooth interpolation would flatter them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from train.dataset import AlignmentBatches, Lap, LapIndex, SampleConfig, _to_chw

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _nearest_frame(s: np.ndarray, target_s: float) -> int:
    """Index of the frame whose track position is nearest `target_s`, around the loop."""
    gap = np.abs(np.asarray(s, dtype=np.float64) - target_s) % 1.0
    return int(np.argmin(np.minimum(gap, 1.0 - gap)))


def _upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return np.ascontiguousarray(frame)
    return cv2.resize(
        frame,
        (frame.shape[1] * scale, frame.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )


def _label(
    canvas: np.ndarray,
    lines: list[str],
    x: int = 8,
    y: int = 20,
    font_scale: float = 0.45,
    line_height: int = 18,
) -> None:
    for i, text in enumerate(lines):
        position = (x, y + i * line_height)
        stroke = 2 if font_scale >= 0.55 else 1
        cv2.putText(
            canvas, text, position, FONT, font_scale, (0, 0, 0), stroke + 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, text, position, FONT, font_scale, (255, 255, 255), stroke, cv2.LINE_AA
        )


def _progress_bar(canvas: np.ndarray, fraction: float) -> None:
    height, width = canvas.shape[:2]
    thickness = 6
    cv2.rectangle(canvas, (0, height - thickness), (width, height), (40, 40, 40), -1)
    cv2.rectangle(
        canvas, (0, height - thickness), (int(width * fraction), height), (60, 200, 255), -1
    )


def _belief_plot(
    canvas: np.ndarray,
    belief: np.ndarray,
    single_frame: np.ndarray,
    true_bin: float,
    predicted_bin: float,
    tracks: list[tuple[str, float, tuple[int, int, int]]] = (),
) -> None:
    """
    The model's distribution over reference bins, truth and prediction marked,
    plus each tracker's estimate as (name, bin, colour).

    Both curves are scaled to their own maximum, so this shows shape rather than
    absolute confidence: one sharp peak means the place is resolved, two peaks
    mean the track is aliased there and the model is honestly reporting both.
    """
    height, width = canvas.shape[:2]
    top, bottom = 30, height - 24
    canvas[:] = (26, 24, 22)
    cv2.rectangle(canvas, (0, top), (width - 1, bottom), (52, 48, 44), 1)

    def x_of(bin_index: float) -> int:
        return int(round(bin_index / belief.size * (width - 1)))

    def curve(values: np.ndarray, colour: tuple[int, int, int], thickness: int) -> None:
        scaled = values / max(float(values.max()), 1e-9)
        xs = np.linspace(0, width - 1, values.size)
        ys = bottom - scaled * (bottom - top)
        points = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [points], False, colour, thickness, cv2.LINE_AA)

    curve(single_frame, (110, 105, 100), 1)
    curve(belief, (60, 210, 255), 2)
    cv2.line(canvas, (x_of(true_bin), top), (x_of(true_bin), bottom), (90, 230, 90), 2)
    cv2.line(canvas, (x_of(predicted_bin), top), (x_of(predicted_bin), bottom), (230, 90, 230), 1)

    def marker(bin_index: float, colour: tuple[int, int, int], y: int) -> None:
        # A tick above the plot, so an answer stays visible when its line
        # sits on top of another.
        x = x_of(bin_index)
        cv2.fillPoly(canvas, [np.array([[x - 6, y - 9], [x + 6, y - 9], [x, y]], np.int32)], colour)

    marker(predicted_bin, (230, 90, 230), top - 2)
    for row, (_, tracked_bin, colour) in enumerate(tracks):
        cv2.line(canvas, (x_of(tracked_bin), top), (x_of(tracked_bin), bottom), colour, 1)
        marker(tracked_bin, colour, top + 12 + 14 * row)

    _label(canvas, ["belief over the reference lap"], x=8, y=20)
    legend = [
        ("12-frame belief", (60, 210, 255)),
        ("newest frame alone", (110, 105, 100)),
        ("true", (90, 230, 90)),
        ("single-shot", (230, 90, 230)),
    ] + [(name, colour) for name, _, colour in tracks]
    for i, (text, colour) in enumerate(legend):
        x = width - 140 * len(legend) + i * 140
        cv2.line(canvas, (x, 15), (x + 18, 15), colour, 2)
        cv2.putText(canvas, text, (x + 24, 19), FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def _writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"could not open a writer for {path}")
    return writer


def _find_lap(index: LapIndex, lap_id: str | None) -> Lap:
    if lap_id is None:
        return index.laps[0]
    for lap in index.laps:
        if lap.lap_id == lap_id:
            return lap
    raise SystemExit(f"no lap {lap_id}; run `list` to see what is packed")


def _load_index(data: str, split: str | None) -> LapIndex:
    index = LapIndex.load(data, split=split)
    if not index.laps:
        raise SystemExit(f"no laps in {data} for split={split}")
    return index


def _pick_pair(
    index: LapIndex, track: str, reference_id: str | None, live_id: str | None
) -> tuple[Lap, Lap]:
    """(reference, live) for one track, defaulting the reference to full coverage."""
    if track not in index.by_track:
        raise SystemExit(f"no track {track}; have {sorted(index.by_track)}")
    group = sorted(index.by_track[track], key=lambda lap: lap.lap_id)
    if len(group) < 2:
        raise SystemExit(f"track {track} has only one lap; a pair needs two")
    if reference_id:
        reference = next(lap for lap in group if lap.lap_id == reference_id)
    else:
        reference = next((lap for lap in group if lap.s_span >= 0.9), group[0])
    if live_id:
        live = next(lap for lap in group if lap.lap_id == live_id)
    else:
        live = next(lap for lap in group if lap.lap_id != reference.lap_id)
    return reference, live


def cmd_list(args: argparse.Namespace) -> None:
    index = _load_index(args.data, None)
    print(f"{'lap':<40} {'track':<12} {'split':<8} {'frames':>7} {'span':>6} {'bins':>6} {'m/bin':>6}")
    for lap in sorted(index.laps, key=lambda lap: lap.lap_id):
        print(
            f"{lap.lap_id:<40} {lap.track:<12} {lap.split:<8} {lap.n_frames:>7} "
            f"{lap.s_span:>6.3f} {lap.ref_bins:>6} {lap.ref_spacing_m:>6.2f}"
        )


def cmd_video(args: argparse.Namespace) -> None:
    index = _load_index(args.data, None)
    lap = _find_lap(index, args.lap)
    frames = lap.frames()
    s = lap.s()

    out = Path(args.out) if args.out else Path(args.data) / f"preview_{lap.lap_id}.mp4"
    height, width = frames.shape[1] * args.scale, frames.shape[2] * args.scale
    writer = _writer(out, (width, height), args.fps)
    try:
        for i in range(lap.n_frames):
            canvas = _upscale(np.asarray(frames[i]), args.scale)
            if not args.raw:
                _label(
                    canvas,
                    [
                        f"{lap.lap_id}",
                        f"s = {s[i]:.4f}   {s[i] * lap.track_length_m:7.1f} m",
                        f"frame {i + 1}/{lap.n_frames}   t = {lap.t()[i]:5.2f} s",
                    ],
                )
                _progress_bar(canvas, float(s[i]))
            writer.write(canvas)
    finally:
        writer.release()
    print(f"wrote {out}  ({lap.n_frames} frames, {width}x{height}, {args.fps:g} fps)")


def cmd_pair(args: argparse.Namespace) -> None:
    """
    Live lap on the left, the reference frame at the same track position on the
    right. Two different laps, so the two halves never match pixel for pixel;
    what they share is the place.
    """
    index = _load_index(args.data, None)
    track = args.track or sorted(index.by_track)[0]
    reference_lap, live_lap = _pick_pair(index, track, args.reference, args.live)

    reference_frames = reference_lap.frames()
    reference_s = reference_lap.s()
    live_frames = live_lap.frames()
    live_s = live_lap.s()

    out = Path(args.out) if args.out else Path(args.data) / f"preview_pair_{track}.mp4"
    tile_h, tile_w = live_frames.shape[1] * args.scale, live_frames.shape[2] * args.scale
    gap = 8
    writer = _writer(out, (tile_w * 2 + gap, tile_h), args.fps)
    try:
        for i in range(live_lap.n_frames):
            frame_index = _nearest_frame(reference_s, float(live_s[i]))

            left = _upscale(np.asarray(live_frames[i]), args.scale)
            right = _upscale(np.asarray(reference_frames[frame_index]), args.scale)
            _label(
                left,
                [
                    "LIVE   " + live_lap.lap_id,
                    f"s = {live_s[i]:.4f}   {live_s[i] * live_lap.track_length_m:7.1f} m",
                    f"line {live_lap.line_mean_m:+.2f} m",
                ],
            )
            _label(
                right,
                [
                    "REFERENCE   " + reference_lap.lap_id,
                    f"bin {bin_index}/{n_bins}   s = {reference_s[frame_index]:.4f}",
                    f"line {reference_lap.line_mean_m:+.2f} m",
                ],
            )
            _progress_bar(left, float(live_s[i]))
            _progress_bar(right, bin_index / n_bins)

            canvas = np.zeros((tile_h, tile_w * 2 + gap, 3), np.uint8)
            canvas[:, :tile_w] = left
            canvas[:, tile_w + gap :] = right
            writer.write(canvas)
    finally:
        writer.release()
    print(f"wrote {out}")
    print(f"  live      {live_lap.lap_id} ({live_lap.n_frames} frames, car {live_lap.car_model})")
    print(f"  reference {reference_lap.lap_id} ({n_bins} bins at {reference_lap.ref_spacing_m:.2f} m)")


def cmd_sheet(args: argparse.Namespace) -> None:
    index = _load_index(args.data, None)
    lap = _find_lap(index, args.lap)
    frames = lap.frames()
    s = lap.s()

    picks = np.linspace(0, lap.n_frames - 1, args.count).round().astype(int)
    columns = args.columns
    rows = int(np.ceil(len(picks) / columns))
    tile_h, tile_w = frames.shape[1] * args.scale, frames.shape[2] * args.scale
    gap = 4
    canvas = np.full(
        (rows * tile_h + (rows - 1) * gap, columns * tile_w + (columns - 1) * gap, 3), 25, np.uint8
    )
    for position, frame_index in enumerate(picks):
        row, column = divmod(position, columns)
        tile = _upscale(np.asarray(frames[frame_index]), args.scale)
        _label(tile, [f"{s[frame_index] * lap.track_length_m:.0f} m"], x=6, y=16)
        y, x = row * (tile_h + gap), column * (tile_w + gap)
        canvas[y : y + tile_h, x : x + tile_w] = tile

    out = Path(args.out) if args.out else Path(args.data) / f"preview_{lap.lap_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}  ({len(picks)} frames from {lap.lap_id}, {canvas.shape[1]}x{canvas.shape[0]})")


def cmd_clip(args: argparse.Namespace) -> None:
    """One training sample as the sampler builds it, jitter and stride included."""
    index = _load_index(args.data, args.split)
    config = SampleConfig(
        batch_size=1,
        clip_len=args.clip_len,
        roll_reference=not args.no_roll,
        jitter=not args.no_jitter,
    )
    batch = AlignmentBatches(index, config, steps=args.step + 1, seed=args.seed)[args.step]

    live = batch["live"][0].numpy()
    reference = batch["reference"].numpy()
    n_bins = reference.shape[0]
    target = float(batch["target"][0])
    bin_index = int(round(target)) % n_bins
    spacing = float(batch["ref_spacing_m"])

    def to_bgr(chw: np.ndarray) -> np.ndarray:
        return (np.moveaxis(chw, 0, -1) * 255.0).clip(0, 255).astype(np.uint8)

    tiles = []
    for k in range(live.shape[0]):
        tile = _upscale(to_bgr(live[k]), args.scale)
        _label(tile, [f"live t-{live.shape[0] - 1 - k}"], x=6, y=16)
        tiles.append(tile)

    reference_row = []
    for offset in (-args.context, 0, args.context):
        tile = _upscale(to_bgr(reference[(bin_index + offset) % n_bins]), args.scale)
        if offset == 0:
            _label(tile, ["ANSWER", f"bin {bin_index}/{n_bins}"], x=6, y=16)
        else:
            _label(tile, [f"ref {offset:+d} bins", f"{offset * spacing:+.0f} m"], x=6, y=16)
        reference_row.append(tile)

    rows = [
        np.concatenate(tiles[i : i + args.columns], axis=1)
        for i in range(0, len(tiles), args.columns)
    ]
    rows.append(np.concatenate(reference_row, axis=1))

    gap = 8
    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + gap * (len(rows) - 1)
    canvas = np.full((height, width, 3), 25, np.uint8)
    y = 0
    for row in rows:
        x = (width - row.shape[1]) // 2
        canvas[y : y + row.shape[0], x : x + row.shape[1]] = row
        y += row.shape[0] + gap

    out = Path(args.out) if args.out else Path(args.data) / f"preview_clip_{args.step}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}")
    print(f"  track {batch['track']}   reference {batch['reference_lap']}   live {batch['live_laps'][0]}")
    print(f"  roll {batch['roll']}   target bin {target:.2f} of {n_bins}   {batch['ref_spacing_m']:.2f} m/bin")
    print("  top row: the K live frames, newest on the right. bottom row: the reference")
    print("  frame the model should pick, and its neighbours for scale.")


def cmd_infer(args: argparse.Namespace) -> None:
    """
    Run a checkpoint along a live lap against a reference lap and film it.

    Live frames are encoded once and the descriptors reused across ticks. That is
    the sliding-window caching the runtime wants: the encoder is strictly
    per-frame and uses GroupNorm, so a cached descriptor is bit-identical to
    recomputing it.
    """
    import torch

    from train.estimator import EstimatorConfig, ProgressEstimator
    from train.eval import SPEED_FED, _instantaneous_speed, load_model, reference_axis_of
    from train.model import compute_metrics, soft_argmax_circular

    index = _load_index(args.data, None)
    track = args.track or sorted(index.by_track)[0]
    reference_lap, live_lap = _pick_pair(index, track, args.reference, args.live)

    device = torch.device(args.device)
    model, payload = load_model(Path(args.checkpoint), device)
    clip_len = int(payload["args"]["clip_len"])

    grid = reference_lap.reference_grid(reference_axis_of(payload))
    ref_raw = np.asarray(reference_lap.frames()[grid.frame_idx])
    n_bins = grid.n_bins
    spacing = reference_lap.ref_spacing_m
    grid_pos = torch.from_numpy(grid.pos_m.astype(np.float32))
    grid_time = torch.from_numpy(grid.time_s.astype(np.float32))
    # The tracker the phone would run on top of the aligner, fed the same ticks.
    estimator = ProgressEstimator(grid.pos_m, grid.track_length_m)
    # Optionally a second tracker given the labelled speed, as `eval stream` prices
    # a speed sensor: what the same lap looks like once speed is known.
    speed_estimator = (
        ProgressEstimator(grid.pos_m, grid.track_length_m, EstimatorConfig(**SPEED_FED))
        if args.speed_sigma
        else None
    )
    live_t = live_lap.t()
    previous_t: float | None = None
    filter_m: list[float] = []
    filter_ms: list[float] = []
    fed_m: list[float] = []
    fed_ms: list[float] = []
    tick = 0
    reference_frames, reference_s = reference_lap.frames(), reference_lap.s()
    live_raw = np.asarray(live_lap.frames())
    live_s = live_lap.s()
    speed = live_lap.speed_mps()
    true_speed = _instantaneous_speed(
        live_s, live_t, np.arange(live_lap.n_frames), live_lap.track_length_m
    )

    with torch.no_grad():
        reference_descriptors = model.encode_reference(
            torch.from_numpy(_to_chw(ref_raw)).to(device), use_checkpoint=False
        )
        live_descriptors = model.encode_reference(
            torch.from_numpy(_to_chw(live_raw)).to(device), use_checkpoint=False
        )
    logit_scale = model.logit_scale.exp()

    tile_h, tile_w = live_raw.shape[1] * args.scale, live_raw.shape[2] * args.scale
    gap = 8
    canvas_w = tile_w * 3 + gap * 2
    plot_h = args.plot_height
    out = Path(args.out) if args.out else Path(args.data) / f"infer_{live_lap.lap_id}.mp4"
    writer = _writer(out, (canvas_w, tile_h + plot_h + gap), args.fps)

    errors_m: list[float] = []
    errors_ms: list[float] = []
    try:
        for i in range(0, live_lap.n_frames, args.every):
            indices = np.clip(i - np.arange(clip_len - 1, -1, -1) * args.stride, 0, None)
            with torch.no_grad():
                clip = live_descriptors[torch.from_numpy(indices).to(device)][None]
                correlation = torch.einsum("bkd,nd->bkn", clip, reference_descriptors) * logit_scale
                logits = model.head(correlation)
                predicted = float(soft_argmax_circular(logits, window=args.window)[0])
                belief = logits.softmax(-1)[0].float().cpu().numpy()
                single = correlation[0, -1].softmax(-1).float().cpu().numpy()

            true_bin = float(grid.target(float(live_s[i])))
            m, ms = compute_metrics(
                torch.tensor([predicted]), torch.tensor([true_bin]),
                grid_pos, grid_time, grid.track_length_m, grid.lap_time_s,
            )
            error_m, error_ms = float(m[0]), float(ms[0])
            dt = 1.0 / 15.0 if previous_t is None else float(live_t[i]) - previous_t
            previous_t = float(live_t[i])
            tracked = float(estimator.bin_of(estimator.step(belief.astype(np.float64), dt).position_m))
            fm, fms = compute_metrics(
                torch.tensor([tracked]), torch.tensor([true_bin]),
                grid_pos, grid_time, grid.track_length_m, grid.lap_time_s,
            )
            filter_m.append(float(fm[0]))
            filter_ms.append(float(fms[0]))
            tracks = [("filter", tracked, (245, 245, 245))]
            if speed_estimator is not None:
                fed = tick % args.speed_every == 0
                fed_bin = float(speed_estimator.bin_of(speed_estimator.step(
                    belief.astype(np.float64), dt,
                    speed_obs=float(true_speed[i]) if fed else None,
                    speed_sigma=args.speed_sigma if fed else None,
                ).position_m))
                sm, sms = compute_metrics(
                    torch.tensor([fed_bin]), torch.tensor([true_bin]),
                    grid_pos, grid_time, grid.track_length_m, grid.lap_time_s,
                )
                fed_m.append(float(sm[0]))
                fed_ms.append(float(sms[0]))
                tracks.append(("filter + speed", fed_bin, (40, 170, 255)))
            tick += 1
            errors_m.append(error_m)
            errors_ms.append(error_ms)

            live_tile = _upscale(live_raw[i], args.scale)
            picked = _upscale(ref_raw[int(round(predicted)) % n_bins], args.scale)
            # The reference frame at the live frame's exact position, not the one
            # filed under the nearest bin: that can sit half a bin away and make
            # a correct answer look wrong.
            exact = _nearest_frame(reference_s, float(live_s[i]))
            truth = _upscale(np.asarray(reference_frames[exact]), args.scale)
            _label(
                live_tile,
                [
                    f"LIVE  {live_lap.lap_id}",
                    f"{float(live_s[i]) * live_lap.track_length_m:7.1f} m   {speed[i]:4.1f} m/s",
                    f"line {live_lap.line_mean_m:+.2f} m",
                ],
                font_scale=0.6,
                line_height=24,
            )
            _label(
                picked,
                [
                    "SINGLE-SHOT PICK",
                    f"bin {int(round(predicted)) % n_bins}/{n_bins}",
                    f"err {error_m:5.2f} m / {error_ms:5.0f} ms",
                    f"filter err {filter_m[-1]:5.2f} m / {filter_ms[-1]:5.0f} ms",
                ] + ([f"+ speed err {fed_m[-1]:5.2f} m / {fed_ms[-1]:5.0f} ms"] if fed_m else []),
                font_scale=0.6,
                line_height=24,
            )
            _label(
                truth,
                [
                    f"TRUE  {reference_lap.lap_id}",
                    f"bin {true_bin:.1f}/{n_bins}   at {float(reference_s[exact]) * grid.track_length_m:.1f} m",
                    f"line {reference_lap.line_mean_m:+.2f} m",
                ],
                font_scale=0.6,
                line_height=24,
            )
            _progress_bar(live_tile, float(live_s[i]))

            canvas = np.full((tile_h + plot_h + gap, canvas_w, 3), 25, np.uint8)
            for column, tile in enumerate((live_tile, picked, truth)):
                x = column * (tile_w + gap)
                canvas[:tile_h, x : x + tile_w] = tile
            _belief_plot(
                canvas[tile_h + gap :], belief, single, true_bin, predicted, tracks
            )
            writer.write(canvas)
    finally:
        writer.release()

    metres = np.array(errors_m)
    print(f"wrote {out}  ({metres.size} ticks, {canvas_w}x{tile_h + plot_h + gap})")
    print(f"  live      {live_lap.lap_id}  line {live_lap.line_mean_m:+.2f} m")
    print(f"  reference {reference_lap.lap_id}  line {reference_lap.line_mean_m:+.2f} m, {n_bins} bins at {spacing:.2f} m")
    print(
        f"  error     median {np.median(metres):.2f} m / {np.median(errors_ms):.0f} ms   "
        f"p90 {np.percentile(metres, 90):.2f} m   worst {metres.max():.2f} m   "
        f"within 1 bin {float((metres <= spacing).mean()) * 100:.1f}%"
    )
    tracked = np.array(filter_m)
    print(
        f"  filter    median {np.median(tracked):.2f} m / {np.median(filter_ms):.0f} ms   "
        f"p90 {np.percentile(tracked, 90):.2f} m   worst {tracked.max():.2f} m   "
        f"over 100 ms {np.mean(np.array(filter_ms) > 100) * 100:.1f}%  (single-shot "
        f"{np.mean(np.array(errors_ms) > 100) * 100:.1f}%)"
    )
    if fed_m:
        fed = np.array(fed_m)
        print(
            f"  + speed   median {np.median(fed):.2f} m / {np.median(fed_ms):.0f} ms   "
            f"p90 {np.percentile(fed, 90):.2f} m   worst {fed.max():.2f} m   "
            f"over 100 ms {np.mean(np.array(fed_ms) > 100) * 100:.1f}%  "
            f"(labelled speed every {args.speed_every} ticks, stated +-{args.speed_sigma} m/s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True)
    common.add_argument("--out")
    common.add_argument("--scale", type=int, default=4)

    listing = sub.add_parser("list", help="what is packed")
    listing.add_argument("--data", required=True)
    listing.set_defaults(func=cmd_list)

    video = sub.add_parser("video", parents=[common], help="play one lap")
    video.add_argument("--lap")
    video.add_argument("--fps", type=float, default=30.0)
    video.add_argument("--raw", action="store_true", help="no labels or progress bar")
    video.set_defaults(func=cmd_video)

    pair = sub.add_parser("pair", parents=[common], help="live lap beside the reference")
    pair.add_argument("--track")
    pair.add_argument("--reference", help="lap id to use as the reference")
    pair.add_argument("--live", help="lap id to use as the live lap")
    pair.add_argument("--fps", type=float, default=30.0)
    pair.set_defaults(func=cmd_pair)

    sheet = sub.add_parser("sheet", parents=[common], help="contact sheet along the lap")
    sheet.add_argument("--lap")
    sheet.add_argument("--count", type=int, default=24)
    sheet.add_argument("--columns", type=int, default=6)
    sheet.set_defaults(func=cmd_sheet)

    clip = sub.add_parser("clip", parents=[common], help="one training sample as sampled")
    clip.add_argument("--split", default="train")
    clip.add_argument("--clip-len", type=int, default=12)
    clip.add_argument("--step", type=int, default=0)
    clip.add_argument("--seed", type=int, default=0)
    clip.add_argument("--context", type=int, default=6, help="neighbour bins to show")
    clip.add_argument("--columns", type=int, default=6)
    clip.add_argument("--no-jitter", action="store_true")
    clip.add_argument("--no-roll", action="store_true")
    clip.set_defaults(func=cmd_clip)

    infer = sub.add_parser("infer", parents=[common], help="film a checkpoint localizing a lap")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--track")
    infer.add_argument("--reference", help="lap id to use as the reference")
    infer.add_argument("--live", help="lap id to use as the live lap")
    infer.add_argument("--stride", type=int, default=1, help="frames between clip samples")
    infer.add_argument("--every", type=int, default=1, help="localize every Nth live frame")
    infer.add_argument("--window", type=int, default=8)
    infer.add_argument("--plot-height", type=int, default=240)
    infer.add_argument("--fps", type=float, default=30.0)
    infer.add_argument("--device", default=None)
    infer.add_argument(
        "--speed-sigma",
        type=float,
        default=None,
        help="also run a tracker given the labelled speed, stated to this many m/s",
    )
    infer.add_argument("--speed-every", type=int, default=1, help="ticks between speed readings")
    infer.set_defaults(func=cmd_infer)

    args = parser.parse_args()
    if getattr(args, "device", None) is None and args.cmd == "infer":
        from train.train import default_device

        args.device = default_device()
    args.func(args)


if __name__ == "__main__":
    main()
