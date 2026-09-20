"""
Gates and negative controls.

    python -m train.eval gates    --data data/packed_synth --checkpoint runs/<name>/best.pt
    python -m train.eval leakage  --data data/packed_synth

`gates` measures a trained aligner on seen laps, held-out laps and held-out
tracks, and runs the wrong-reference control. `leakage` trains the frame-only
control model from scratch, which is a different question and so a separate
command.

The controls matter as much as the gates. A good number on the held-out track is
only meaningful if the model is genuinely reading the reference, and if nothing
on screen is handing over the answer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.dataset import AlignmentBatches, LapIndex, SampleConfig, circular_soft_target
from train.model import (
    AlignmentMetrics,
    FrameOnlyRegressor,
    SequenceAligner,
    compute_metrics,
    soft_argmax_circular,
    summarise,
)
from train.seqslam import SeqSLAM
from train.train import default_device, holdout_live_laps

ML_ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint: Path, device: torch.device) -> tuple[SequenceAligner, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = payload["args"]
    model = SequenceAligner(
        clip_len=saved["clip_len"],
        dim=saved["dim"],
        width=saved["width"],
        hidden=saved["hidden"],
        frame_size=tuple(payload["frame_size"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def build_matcher(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, int]:
    """
    The trained aligner or the SeqSLAM baseline behind one interface, so the
    measurement path below cannot differ between them.
    """
    if getattr(args, "matcher", "primal") == "seqslam":
        model = SeqSLAM(
            clip_len=args.clip_len,
            down=tuple(args.sq_down),
            patch=args.sq_patch,
            temperature=args.sq_temperature,
            norm_window=args.sq_norm_window,
        ).to(device)
        model.eval()
        return model, args.clip_len
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --matcher seqslam")
    model, payload = load_model(Path(args.checkpoint), device)
    return model, payload["args"]["clip_len"]


@torch.no_grad()
def measure(
    model: SequenceAligner,
    dataset: AlignmentBatches,
    device: torch.device,
    window: int,
    swap_reference_from: AlignmentBatches | None = None,
) -> tuple[AlignmentMetrics, dict[str, AlignmentMetrics], float]:
    """
    Run a dataset through the model.

    `swap_reference_from` replaces each batch's reference with one drawn from a
    different track, keeping the live clips and targets. That is the control:
    the reference is the only route by which track identity can reach the
    output, so cutting it must destroy the prediction.
    """
    metres: dict[str, list[np.ndarray]] = {}
    milliseconds: dict[str, list[np.ndarray]] = {}
    spacing: dict[str, float] = {}
    entropies: list[float] = []

    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    wrong = iter(DataLoader(swap_reference_from, batch_size=None, num_workers=0)) if swap_reference_from else None

    for batch in loader:
        reference = batch["reference"]
        target = batch["target"]
        speed = batch["ref_speed_mps"]
        spacing_m = batch["ref_spacing_m"]
        if wrong is not None:
            try:
                other = next(wrong)
            except StopIteration:
                break
            if other["track"] == batch["track"]:
                continue
            reference, speed = other["reference"], other["ref_speed_mps"]
            spacing_m = other["ref_spacing_m"]
            bins = reference.shape[0]
            # Keep the target on the same axis length so the error stays comparable.
            if bins != batch["reference"].shape[0]:
                target = target * (bins / batch["reference"].shape[0])

        logits, _ = model(batch["live"].to(device), reference.to(device), use_checkpoint=False)
        probabilities = logits.softmax(dim=-1)
        entropies.append(
            float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
        )
        predicted = soft_argmax_circular(logits, window=window)
        m, ms = compute_metrics(
            predicted, target.to(device), logits.shape[-1], spacing_m, speed.to(device)
        )
        key = batch["track"]
        metres.setdefault(key, []).append(m)
        milliseconds.setdefault(key, []).append(ms)
        spacing[key] = spacing_m

    per_track = {
        track: summarise(metres[track], milliseconds[track], spacing[track]) for track in metres
    }
    overall = summarise(
        [a for v in metres.values() for a in v],
        [a for v in milliseconds.values() for a in v],
        float(np.mean(list(spacing.values()))) if spacing else 1.0,
    )
    return overall, per_track, float(np.mean(entropies)) if entropies else float("nan")


def cmd_gates(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, payload = load_model(Path(args.checkpoint), device)
    clip_len = payload["args"]["clip_len"]
    data = Path(args.data)

    train_index = LapIndex.load(data, split="train")
    holdout_index = LapIndex.load(data, split="holdout")
    reserved = holdout_live_laps(train_index, args.holdout_laps)
    trainable = frozenset(lap.lap_id for lap in train_index.laps) - reserved

    def make(index: LapIndex, reference_only, live_only, seed: int) -> AlignmentBatches:
        config = SampleConfig(
            batch_size=args.batch_size,
            clip_len=clip_len,
            roll_reference=True,
            jitter=False,
            reference_only=reference_only,
            live_only=live_only,
        )
        return AlignmentBatches(index, config, steps=args.steps, seed=seed)

    report: dict = {"checkpoint": str(args.checkpoint), "gates": {}, "controls": {}}

    print("=" * 100)
    print("GATES")
    print("=" * 100)

    cases = [("seen laps, seen tracks", train_index, trainable, trainable, 11)]
    if reserved:
        cases.append(("G1  held-out laps, seen tracks", train_index, trainable, reserved, 22))
    if holdout_index.laps:
        cases.append(("G2  held-out tracks", holdout_index, None, None, 33))

    for label, index, reference_only, live_only, seed in cases:
        dataset = make(index, reference_only, live_only, seed)
        overall, per_track, entropy = measure(model, dataset, device, args.window)
        print(f"\n{label}")
        print(f"  {overall.format()}")
        print(f"  mean predictive entropy {entropy:.3f} nats")
        for track in sorted(per_track):
            print(f"    {track:<24} {per_track[track].format()}")
        report["gates"][label] = {
            "overall": asdict(overall),
            "entropy": entropy,
            "per_track": {k: asdict(v) for k, v in per_track.items()},
        }

    print("\n" + "=" * 100)
    print("NEGATIVE CONTROLS")
    print("=" * 100)

    matched = make(train_index, trainable, trainable, 44)
    mismatched = make(train_index, trainable, trainable, 55)
    n_bins = train_index.laps[0].ref_bins
    spacing_m = train_index.laps[0].ref_spacing_m
    chance_m = n_bins / 4.0 * spacing_m

    baseline, _, baseline_entropy = measure(model, matched, device, args.window)
    control, _, control_entropy = measure(
        model, matched, device, args.window, swap_reference_from=mismatched
    )
    uniform_entropy = float(np.log(n_bins))

    print("\nwrong-reference control: live clips paired with another track's reference")
    print(f"  matched reference   median {baseline.median_m:7.2f} m   entropy {baseline_entropy:.3f} nats")
    print(f"  wrong reference     median {control.median_m:7.2f} m   entropy {control_entropy:.3f} nats")
    print(f"  chance for {n_bins} bins at {spacing_m:.2f} m: {chance_m:.1f} m; uniform entropy {uniform_entropy:.3f} nats")
    verdict = "PASS" if control.median_m > 10 * max(baseline.median_m, 1e-6) else "FAIL"
    print(f"  {verdict}: error must collapse to chance when the reference is wrong")
    report["controls"]["wrong_reference"] = {
        "matched_median_m": baseline.median_m,
        "wrong_median_m": control.median_m,
        "matched_entropy": baseline_entropy,
        "wrong_entropy": control_entropy,
        "chance_median_m": chance_m,
        "uniform_entropy": uniform_entropy,
        "verdict": verdict,
    }

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "gates.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


# What `lines` can sweep error against. Same machinery, different viewpoint
# axis: how far the live lap sits from the reference laterally, or how far it
# is turned. Each carries its own bin edges because degrees and metres are not
# comparable, and its own "near"/"far" thresholds for the summary ratio.
SEPARATION_AXES = {
    "line": {
        "attribute": "line_mean_m",
        "unit": "m",
        "title": "CROSS-LINE: error vs how far the live line sits from the reference line",
        "label": "line separation",
        "edges": [0.0, 0.5, 1.0, 2.0, 3.0, float("inf")],
        "near": 0.5,
        "far": 2.0,
        "near_name": "same line",
        "far_name": "opposite side",
    },
    "yaw": {
        "attribute": "yaw_mean_deg",
        "unit": "deg",
        "title": "CROSS-YAW: error vs how far the live camera is turned from the reference",
        "label": "yaw separation",
        "edges": [0.0, 2.0, 5.0, 10.0, 20.0, float("inf")],
        "near": 2.0,
        "far": 10.0,
        "near_name": "same heading",
        "far_name": "turned away",
    },
}


def cmd_lines(args: argparse.Namespace) -> None:
    """
    Sweep every ordered lap pair and report error against how far apart the two
    driving lines are.

    This is the question a single aggregate number hides: an aligner can look
    healthy on average while failing whenever the live lap runs a different line
    from the reference, because same-line pairs dominate the average. Plotting
    error against line separation separates "matches places" from "matches
    viewpoints".
    """
    axis = SEPARATION_AXES[args.axis]
    device = torch.device(args.device)
    model, clip_len = build_matcher(args, device)
    index = LapIndex.load(Path(args.data), split=args.split)
    if not index.laps:
        raise SystemExit(f"no laps in {args.data} for split={args.split}")

    rows: list[dict] = []
    for track, group in sorted(index.by_track.items()):
        for reference in group:
            if reference.s_span < 0.9:
                continue
            for live in group:
                if live.lap_id == reference.lap_id:
                    continue
                config = SampleConfig(
                    batch_size=args.batch_size,
                    clip_len=clip_len,
                    roll_reference=True,
                    jitter=False,
                    reference_only=frozenset({reference.lap_id}),
                    live_only=frozenset({live.lap_id}),
                )
                dataset = AlignmentBatches(
                    LapIndex(list(group)), config, steps=args.steps, seed=hash(live.lap_id) % 9973
                )
                overall, _, entropy = measure(model, dataset, device, args.window)
                rows.append(
                    {
                        "track": track,
                        "reference": reference.lap_id,
                        "live": live.lap_id,
                        "axis": args.axis,
                        "separation": abs(
                            getattr(live, axis["attribute"])
                            - getattr(reference, axis["attribute"])
                        ),
                        "median_m": overall.median_m,
                        "p90_m": overall.p90_m,
                        "within_1_bin": overall.within_1_bin,
                        "entropy": entropy,
                    }
                )

    if not rows:
        raise SystemExit("no usable lap pairs; each track needs a full-coverage reference lap")

    unit = axis["unit"]
    print("=" * 100)
    print(axis["title"])
    print("=" * 100)
    separation = np.array([row["separation"] for row in rows])
    median = np.array([row["median_m"] for row in rows])
    edges = axis["edges"]
    print(f"\n  {axis['label']:<22} {'pairs':>6} {'median err':>12} {'worst pair':>12}")
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (separation >= low) & (separation < high)
        if not mask.any():
            continue
        label = (
            f"{low:.1f} to {high:.1f} {unit}"
            if np.isfinite(high)
            else f"{low:.1f} {unit} and up"
        )
        print(
            f"  {label:<22} {int(mask.sum()):>6} "
            f"{np.median(median[mask]):>10.2f} m {median[mask].max():>10.2f} m"
        )

    print(f"\n  {'pair':<46} {'sep':>7} {'median':>9} {'p90':>9} {'within1':>8}")
    for row in sorted(rows, key=lambda r: -r["separation"])[: args.show]:
        pair = f"{row['live']} vs {row['reference']}"
        print(
            f"  {pair:<46} {row['separation']:>5.2f} {unit} {row['median_m']:>7.2f} m "
            f"{row['p90_m']:>7.2f} m {row['within_1_bin'] * 100:>7.1f}%"
        )

    near = median[separation < axis["near"]]
    far = median[separation >= axis["far"]]
    if near.size and far.size:
        ratio = float(np.median(far) / max(np.median(near), 1e-6))
        print(
            f"\n  {axis['near_name']} {np.median(near):.2f} m, "
            f"{axis['far_name']} {np.median(far):.2f} m, ratio {ratio:.2f}x"
        )
        print("  A large ratio means the model is matching viewpoint rather than place.")

    default_name = "lines.json" if args.axis == "line" else f"lines_{args.axis}.json"
    if args.out:
        out = Path(args.out)
    elif args.checkpoint:
        out = Path(args.checkpoint).parent / default_name
    else:
        out = Path(args.data) / default_name
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


class FrameSampler(torch.utils.data.Dataset):
    """Random single frames with soft `s` bin targets, no reference lap."""

    def __init__(self, index: LapIndex, bins: int, batch_size: int, steps: int, seed: int):
        self.laps = index.laps
        self.bins = bins
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, step: int) -> dict:
        rng = np.random.default_rng((self.seed, step))
        images, targets = [], []
        for _ in range(self.batch_size):
            lap = self.laps[int(rng.integers(0, len(self.laps)))]
            frame_i = int(rng.integers(0, lap.n_frames))
            frame = np.asarray(lap.frames()[frame_i])
            images.append(np.moveaxis(frame, -1, 0).astype(np.float32) / 255.0)
            targets.append(float(lap.s()[frame_i]) * self.bins)
        target = np.array(targets)
        return {
            "image": torch.from_numpy(np.stack(images)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "soft_target": torch.from_numpy(circular_soft_target(target, self.bins, 2.0)),
        }


def cmd_leakage(args: argparse.Namespace) -> None:
    """
    Train a single-frame, no-reference regressor and compare seen to held-out
    tracks.

    On seen tracks this is *expected* to work; memorising one circuit from
    pixels is easy, which is precisely why an absolute regressor is the wrong
    factorisation rather than a bug. The diagnostic is the held-out track:
    absolute position on a circuit the weights have never seen is not a
    learnable function of appearance, so a good score there means something on
    screen is giving the answer away. A track map widget in the HUD does exactly
    that.
    """
    device = torch.device(args.device)
    data = Path(args.data)
    train_index = LapIndex.load(data, split="train")
    holdout_index = LapIndex.load(data, split="holdout")
    if not holdout_index.laps:
        raise SystemExit("no holdout split; the leakage control needs a held-out track")

    shape = np.load(train_index.laps[0].directory / "frames.npy", mmap_mode="r").shape
    frame_size = (int(shape[1]), int(shape[2]))
    model = FrameOnlyRegressor(bins=args.bins, frame_size=frame_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_set = FrameSampler(train_index, args.bins, args.batch_size, args.steps, seed=7)
    loader = DataLoader(train_set, batch_size=None, num_workers=0)
    print(f"training the frame-only control for {args.steps} steps on {sorted(train_index.by_track)}")
    for step, batch in enumerate(loader, start=1):
        logits = model(batch["image"].to(device))
        loss = -(batch["soft_target"].to(device) * logits.log_softmax(-1)).sum(-1).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  loss {loss.item():.4f}")

    @torch.no_grad()
    def score(index: LapIndex, seed: int) -> dict:
        model.eval()
        sampler = FrameSampler(index, args.bins, args.batch_size, 40, seed=seed)
        errors = []
        for batch in DataLoader(sampler, batch_size=None, num_workers=0):
            logits = model(batch["image"].to(device))
            predicted = soft_argmax_circular(logits, window=6)
            target = batch["target"].to(device)
            offset = (predicted - target + args.bins / 2) % args.bins - args.bins / 2
            errors.append(offset.abs().cpu().numpy())
        model.train()
        stacked = np.concatenate(errors)
        length = index.laps[0].track_length_m
        return {
            "median_bins": float(np.median(stacked)),
            "median_m": float(np.median(stacked) / args.bins * length),
            "within_5_bins": float((stacked <= 5).mean()),
        }

    seen = score(train_index, 101)
    unseen = score(holdout_index, 202)
    chance_bins = args.bins / 4.0

    print("\n" + "=" * 100)
    print("LEAKAGE CONTROL: single frame, no reference lap")
    print("=" * 100)
    print(f"  chance level: {chance_bins:.1f} bins")
    print(f"  seen tracks       median {seen['median_bins']:7.2f} bins ({seen['median_m']:7.1f} m)  within5 {seen['within_5_bins'] * 100:5.1f}%")
    print(f"  held-out tracks   median {unseen['median_bins']:7.2f} bins ({unseen['median_m']:7.1f} m)  within5 {unseen['within_5_bins'] * 100:5.1f}%")
    verdict = "PASS" if unseen["median_bins"] > 0.5 * chance_bins else "FAIL"
    print(f"\n  {verdict}: held-out-track score must sit near chance.")
    if verdict == "FAIL":
        print("  Something on screen encodes position. Check for a track map, delta bar,")
        print("  or lap-time widget left enabled during capture.")

    report = {"chance_bins": chance_bins, "seen": seen, "held_out": unseen, "verdict": verdict}
    out = Path(args.out) if args.out else ML_ROOT / "runs" / "leakage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gates = sub.add_parser("gates", help="measure a trained aligner and run the reference control")
    gates.add_argument("--data", required=True)
    gates.add_argument("--checkpoint", required=True)
    gates.add_argument("--batch-size", type=int, default=8)
    gates.add_argument("--steps", type=int, default=40, help="batches per case")
    gates.add_argument("--holdout-laps", type=int, default=1)
    gates.add_argument("--window", type=int, default=8)
    gates.add_argument("--out")
    gates.add_argument("--device", default=default_device())
    gates.set_defaults(func=cmd_gates)

    lines = sub.add_parser("lines", help="error as a function of live/reference line separation")
    lines.add_argument("--data", required=True)
    lines.add_argument("--checkpoint")
    lines.add_argument("--matcher", default="primal", choices=["primal", "seqslam"])
    lines.add_argument("--clip-len", type=int, default=12, help="only for --matcher seqslam")
    lines.add_argument("--sq-down", type=int, nargs=2, default=(48, 64), metavar=("H", "W"))
    lines.add_argument("--sq-patch", type=int, default=4)
    lines.add_argument("--sq-temperature", type=float, default=6.0)
    lines.add_argument("--sq-norm-window", type=int, default=0)
    lines.add_argument("--split", default="train")
    lines.add_argument(
        "--axis",
        default="line",
        choices=sorted(SEPARATION_AXES),
        help="viewpoint axis to sweep error against",
    )
    lines.add_argument("--batch-size", type=int, default=8)
    lines.add_argument("--steps", type=int, default=8, help="batches per lap pair")
    lines.add_argument("--window", type=int, default=8)
    lines.add_argument("--show", type=int, default=12, help="worst-separated pairs to list")
    lines.add_argument("--out")
    lines.add_argument("--device", default=default_device())
    lines.set_defaults(func=cmd_lines)

    leak = sub.add_parser("leakage", help="train the frame-only control from scratch")
    leak.add_argument("--data", required=True)
    leak.add_argument("--bins", type=int, default=256)
    leak.add_argument("--batch-size", type=int, default=16)
    leak.add_argument("--steps", type=int, default=600)
    leak.add_argument("--lr", type=float, default=6e-4)
    leak.add_argument("--out")
    leak.add_argument("--device", default=default_device())
    leak.set_defaults(func=cmd_leakage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
