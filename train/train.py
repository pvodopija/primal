"""
Train the sequence aligner.

    # G0: overfit one (live lap, reference lap) pair to sub-bin error
    python -m train.train --data data/packed_synth --gate g0

    # G1: held-out laps of seen tracks
    python -m train.train --data data/packed_synth --gate g1

Checkpoints and a metrics log land in `runs/<name>/`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.dataset import AlignmentBatches, LapIndex, SampleConfig, same_drive
from train.model import (
    MOBILENET_V3_LARGE_IMAGENET,
    MOBILENET_V3_SMALL_IMAGENET,
    RESNET18_IMAGENET,
    SequenceAligner,
    alignment_loss,
    compute_metrics,
    reference_grad_bins,
    soft_argmax_circular,
    summarise,
)

ML_ROOT = Path(__file__).resolve().parents[1]


def default_device() -> str:
    """CUDA, then Apple Silicon, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GateConfig:
    """Presets for the staged gates, so a run is one flag rather than eight."""

    name: str
    description: str
    tracks: int | None
    laps_per_track: int | None
    roll_reference: bool
    jitter: bool
    steps: int
    eval_split: str

    @staticmethod
    def get(name: str) -> "GateConfig":
        gates = {
            "g0": GateConfig(
                name="g0",
                description="overfit one lap pair; no augmentation, no rolling",
                tracks=1,
                laps_per_track=2,
                roll_reference=False,
                jitter=False,
                steps=400,
                eval_split="train",
            ),
            "g1": GateConfig(
                name="g1",
                description="held-out laps of seen tracks; full augmentation",
                tracks=None,
                laps_per_track=None,
                roll_reference=True,
                jitter=True,
                steps=3000,
                eval_split="train",
            ),
            "g2": GateConfig(
                name="g2",
                description="held-out tracks; full augmentation",
                tracks=None,
                laps_per_track=None,
                roll_reference=True,
                jitter=True,
                steps=6000,
                eval_split="holdout",
            ),
        }
        if name not in gates:
            raise SystemExit(f"unknown gate {name}; pick one of {sorted(gates)}")
        return gates[name]


def holdout_live_laps(index: LapIndex, per_track: int) -> frozenset[str]:
    """
    One or more laps per track reserved as live-only.

    Held out at lap level, not frame level: a frame-level split leaks the map
    through neighbouring frames and reports a fantasy number. Tracks with too
    few laps to spare one keep all of theirs, since training needs at least two.
    """
    reserved: set[str] = set()
    for group in index.by_track.values():
        # Only recorded drives: every run holds out the same laps whether or not
        # it also trains on re-renders.
        ordered = sorted((lap for lap in group if lap.variant == "live"), key=lambda lap: lap.lap_id)
        spare = len(ordered) - 2
        if spare <= 0:
            continue
        # The last whole laps: a partial one would leave its track out of the test.
        whole = [lap for lap in ordered if lap.s_span > 0.98] or ordered
        reserved.update(lap.lap_id for lap in whole[-min(per_track, spare) :])
    return frozenset(reserved)


def build_index(data: Path, gate: GateConfig, split: str, tracks: list[str] | None = None,
                variants: tuple[str, ...] = ("live",)) -> LapIndex:
    index = LapIndex.load(data, split=split, tracks=tracks, variants=variants)
    if gate.tracks is not None:
        keep = sorted(index.by_track)[: gate.tracks]
        index = LapIndex.load(data, split=split, tracks=keep, variants=variants)
    if gate.laps_per_track is not None:
        index = LapIndex.load(
            data,
            split=split,
            tracks=sorted(index.by_track),
            max_laps_per_track=gate.laps_per_track,
            variants=variants,
        )
    return index


@torch.no_grad()
def evaluate(
    model: SequenceAligner, loader: DataLoader, device: torch.device, window: int
) -> tuple[dict, dict]:
    model.eval()
    metres: dict[str, list[np.ndarray]] = {}
    milliseconds: dict[str, list[np.ndarray]] = {}
    spacing: dict[str, float] = {}
    for batch in loader:
        live = batch["live"].to(device)
        reference = batch["reference"].to(device)
        target = batch["target"].to(device)
        logits, _ = model(live, reference, use_checkpoint=False)
        predicted = soft_argmax_circular(logits, window=window)
        m, ms = compute_metrics(
            predicted,
            target,
            batch["ref_pos_m"],
            batch["ref_time_s"],
            batch["track_length_m"],
            batch["lap_time_s"],
        )
        track = batch["track"]
        metres.setdefault(track, []).append(m)
        milliseconds.setdefault(track, []).append(ms)
        spacing[track] = batch["ref_spacing_m"]
    model.train()

    per_track = {
        track: summarise(metres[track], milliseconds[track], spacing[track]) for track in metres
    }
    overall = summarise(
        [a for v in metres.values() for a in v],
        [a for v in milliseconds.values() for a in v],
        float(np.mean(list(spacing.values()))) if spacing else 1.0,
    )
    return {"overall": overall}, per_track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--gate", default="g1", choices=["g0", "g1", "g2"])
    parser.add_argument(
        "--tracks",
        default=None,
        help="comma-separated tracks to train on (default: every track in the split); "
        "the rest of the train split then serves as extra unseen tracks at evaluation",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--steps", type=int, default=None, help="override the gate preset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=12)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-bins", type=float, default=2.0)
    parser.add_argument(
        "--reference-axis",
        default="time",
        choices=["distance", "time"],
        help="space reference bins evenly in reference lap time (one bin = the same slice of "
        "delta everywhere) or in track distance",
    )
    parser.add_argument(
        "--aux-all-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="supervise every clip frame's correlation row against its own position, not just the last",
    )
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--refine-weight", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument(
        "--holdout-laps", type=int, default=1, help="live-only laps per track for the g1 gate"
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=24)
    # Preparing each step's reference on the CPU, not the GPU, bounds a step;
    # workers overlap it. Every step seeds its own sampling, so the worker count
    # does not change what is drawn.
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true", help="disable reference recompute")
    parser.add_argument(
        "--ref-grad-random",
        type=float,
        default=None,
        help="partial reference gradients: only bins near the clips' targets plus this random share "
        "of the rest carry gradient (default: every bin, the original recipe)",
    )
    parser.add_argument("--ref-grad-window", type=int, default=8, help="bins either side of each target")
    parser.add_argument("--camera-aug", action="store_true", help="head pose, blur, occluder, vignette and JPEG jitter")
    parser.add_argument("--mirror-p", type=float, default=0.0, help="share of steps mirrored, reference and clips together")
    parser.add_argument("--aug-zoom", type=float, nargs=2, default=(1.03, 1.12),
                        help="zoom range for --camera-aug; equal values fix it for live and reference alike")
    parser.add_argument("--aug-pitch", type=float, default=3.0, help="pitch amplitude in degrees for --camera-aug")
    parser.add_argument("--aug-parts", default="pose,blur,occlude,vignette,jpeg",
                        help="which --camera-aug effects to apply, comma-separated")
    parser.add_argument("--init", default=None, help="start from this checkpoint's weights (e.g. a clean fine-tune)")
    parser.add_argument("--teacher", default=None,
                        help="checkpoint of a stronger model to learn from as well (distillation), e.g. a resnet18 run")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--encoder", choices=("small", "resnet18", "mobilenet", "mobilenet_large"), default="small",
                        help="the 0.67M from-scratch encoder, ImageNet ResNet-18 up to layer3, or ImageNet "
                             "MobileNetV3-Small / -Large cut to Halo's size")
    parser.add_argument("--encoder-weights", default=None,
                        help="pretrained weights for a pretrained encoder (default: torchvision's ImageNet file in the "
                             "torch cache); 'none' trains the same architecture from scratch")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="keep a pretrained encoder's backbone fixed; only its projection and the head learn")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.3,
                        help="learning rate of a pretrained backbone, as a share of --lr")
    parser.add_argument("--norm", choices=("group", "none"), default="group",
                        help="GroupNorm in encoder and head, or none (Halo's NPU cannot run GroupNorm)")
    parser.add_argument("--aug-yaw", type=float, default=4.0, help="yaw amplitude in degrees for --camera-aug")
    parser.add_argument("--traffic", type=float, default=0.0,
                        help="share of live clips with a kart ahead drawn in, as dark blocks in perspective")
    parser.add_argument("--p-fold", type=float, default=0.0,
                        help="share of clips that turn back at a random frame and retrace it, [a b c d c b]")
    parser.add_argument("--with-look", action="store_true",
                        help="also train on look-into-the-corner re-renders (index variant 'look')")
    parser.add_argument("--target-at", choices=("last", "middle"), default="last",
                        help="localise the clip's last frame (live tracking) or its middle (placing finished laps)")
    parser.add_argument("--strides", default="1,2,3,4",
                        help="frames between clip frames, drawn uniformly per clip; runtime uses 4, so centre them there")
    parser.add_argument("--wide", default=None,
                        help="also train on this set of wide renders, seen through the virtual camera (train/camera.py)")
    parser.add_argument("--head", type=float, nargs=3, default=(14.0, 4.0, 8.0), metavar=("YAW", "PITCH", "ROLL"),
                        help="head pose amplitudes in degrees for live clips from --wide laps")
    parser.add_argument("--shake", type=float, default=0.0,
                        help="camera shake on every live clip, level drawn from [0, this]; 1 = a typical kart")
    args = parser.parse_args()

    gate = GateConfig.get(args.gate)
    steps = args.steps if args.steps is not None else gate.steps
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    data = Path(args.data)
    tracks = args.tracks.split(",") if args.tracks else None
    train_index = build_index(data, gate, split="train", tracks=tracks,
                              variants=("live", "look") if args.with_look else ("live",))
    if tracks and sorted(train_index.by_track) != sorted(tracks):
        raise SystemExit(f"--tracks names not in the train split: {sorted(set(tracks) - set(train_index.by_track))}")
    eval_index = build_index(data, gate, split=gate.eval_split, tracks=tracks if gate.eval_split == "train" else None)

    # G1 asks about unseen laps of seen tracks, so some laps are live-only. G0
    # deliberately trains and evaluates on the same pair, and G2 evaluates on a
    # different split entirely, so neither needs a lap-level split.
    reserved: frozenset[str] = frozenset()
    if gate.name == "g1" and args.holdout_laps > 0:
        reserved = holdout_live_laps(train_index, args.holdout_laps)
    if args.wide:
        # Wide renders re-render recorded drives, some of them the held-out ones: any wide
        # lap that could be a held-out drive stays out, or G1 would test on a seen drive.
        wide = LapIndex.load(Path(args.wide), split="train", tracks=sorted(train_index.by_track))
        held = [lap for lap in train_index.laps if lap.lap_id in reserved]
        kept = [w for w in wide.laps if not any(same_drive(w, h) for h in held)]
        print(f"wide: {len(kept)} of {len(wide.laps)} laps ({len(wide.laps) - len(kept)} could be held-out drives)")
        train_index = LapIndex(train_index.laps + kept)
    trainable = frozenset(lap.lap_id for lap in train_index.laps) - reserved

    train_config = SampleConfig(
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        roll_reference=gate.roll_reference,
        jitter=gate.jitter,
        sigma_bins=args.sigma_bins,
        reference_axis=args.reference_axis,
        reference_only=trainable if reserved else None,
        live_only=trainable if reserved else None,
        camera_aug=args.camera_aug,
        mirror_p=args.mirror_p,
        aug_zoom=tuple(args.aug_zoom),
        aug_pitch_deg=args.aug_pitch,
        aug_yaw_deg=args.aug_yaw,
        p_fold=args.p_fold,
        p_traffic=args.traffic,
        aug_parts=frozenset(args.aug_parts.split(",")),
        strides=tuple(int(s) for s in args.strides.split(",")),
        target_at=args.target_at,
        head_yaw_deg=args.head[0],
        head_pitch_deg=args.head[1],
        head_roll_deg=args.head[2],
        shake_max=args.shake,
    )
    train_set = AlignmentBatches(train_index, train_config, steps=steps, seed=args.seed)

    eval_config = SampleConfig(
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        roll_reference=gate.roll_reference,
        jitter=False,
        sigma_bins=args.sigma_bins,
        reference_axis=args.reference_axis,
        reference_only=trainable if reserved else None,
        live_only=reserved if reserved else None,
        strides=tuple(int(s) for s in args.strides.split(",")),
        target_at=args.target_at,
    )
    # A distinct seed stream, so evaluation clips are not the training clips.
    eval_set = AlignmentBatches(eval_index, eval_config, steps=args.eval_steps, seed=args.seed + 9973)

    sample_shape = np.load(next(l for l in train_index.laps if not l.wide).directory / "frames.npy", mmap_mode="r").shape
    frame_size = (int(sample_shape[1]), int(sample_shape[2]))
    model = SequenceAligner(
        clip_len=args.clip_len,
        dim=args.dim,
        width=args.width,
        hidden=args.hidden,
        frame_size=frame_size,
        norm=args.norm,
        encoder=args.encoder,
        encoder_weights=None if args.init or args.encoder_weights == "none" else (
            Path(args.encoder_weights) if args.encoder_weights else
            {"resnet18": RESNET18_IMAGENET, "mobilenet": MOBILENET_V3_SMALL_IMAGENET,
             "mobilenet_large": MOBILENET_V3_LARGE_IMAGENET}.get(args.encoder)),
    ).to(device)
    if args.freeze_backbone:
        for p in model.encoder.backbone_parameters():
            p.requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
        print(f"started from {args.init}")

    name = args.name or f"{gate.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run = ML_ROOT / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(
        json.dumps({"args": vars(args), "gate": asdict(gate), "parameters": parameters}, indent=2)
    )

    print(f"gate {gate.name}: {gate.description}")
    print(f"train tracks {sorted(train_index.by_track)} ({len(trainable)} laps usable)")
    print(f"eval  tracks {sorted(eval_index.by_track)}, split={gate.eval_split}")
    if reserved:
        print(f"held out as live-only: {sorted(reserved)}")
    print(f"frame {frame_size[1]}x{frame_size[0]}, reference bins {train_index.laps[0].ref_bins}")
    print(f"model {parameters / 1e6:.2f}M params on {device}\n")

    loader = DataLoader(train_set, batch_size=None, num_workers=args.workers)
    eval_loader = DataLoader(eval_set, batch_size=None, num_workers=0)
    teacher = None
    if args.teacher:
        payload = torch.load(args.teacher, map_location=device, weights_only=False)
        saved = payload["args"]
        teacher = SequenceAligner(
            clip_len=saved["clip_len"], dim=saved["dim"], width=saved["width"], hidden=saved["hidden"],
            frame_size=tuple(payload["frame_size"]), norm=saved.get("norm", "group"),
            encoder=saved.get("encoder", "small"),
        ).to(device)
        teacher.load_state_dict(payload["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"distilling from {args.teacher} (weight {args.distill_weight}, temperature {args.distill_temp})")
    if hasattr(model.encoder, "backbone_parameters"):
        # A pretrained backbone learns gentler than the layers trained from scratch.
        backbone = {id(p) for p in model.encoder.backbone_parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) in backbone and p.requires_grad],
                   "lr": args.lr * args.backbone_lr_scale},
                  {"params": [p for p in model.parameters() if id(p) not in backbone], "lr": args.lr}]
        max_lr = [args.lr * args.backbone_lr_scale, args.lr]
    else:
        groups, max_lr = model.parameters(), args.lr
    optimiser = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=max_lr, total_steps=steps, pct_start=0.15
    )

    log: list[dict] = []
    best = float("inf")
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        live = batch["live"].to(device)
        reference = batch["reference"].to(device)
        target = batch["target"].to(device)
        soft_target = batch["soft_target"].to(device)

        grad_index = None
        if args.ref_grad_random is not None:
            grad_index = reference_grad_bins(
                batch["frame_targets"], reference.shape[0], args.ref_grad_window, args.ref_grad_random, generator
            ).to(device)
        logits, correlation = model(
            live, reference, use_checkpoint=not args.no_checkpoint, ref_grad_index=grad_index
        )
        parts = alignment_loss(
            logits,
            correlation,
            soft_target,
            target,
            aux_weight=args.aux_weight,
            refine_weight=args.refine_weight,
            window=args.window,
            frame_target=batch["frame_targets"].to(device) if args.aux_all_frames else None,
            sigma_bins=args.sigma_bins,
        )
        total = parts.total
        if teacher is not None:
            # Learn from the teacher too: its belief over the reference (where the clip is)
            # and its per-frame similarity rows (which places look alike), both softened.
            with torch.no_grad():
                teacher_logits, teacher_correlation = teacher(live, reference, use_checkpoint=False)
            temp = args.distill_temp
            head_kl = torch.nn.functional.kl_div(
                torch.log_softmax(logits / temp, -1), torch.softmax(teacher_logits / temp, -1), reduction="batchmean"
            ) * temp * temp
            rows_kl = torch.nn.functional.kl_div(
                torch.log_softmax(correlation.flatten(0, 1) / temp, -1),
                torch.softmax(teacher_correlation.flatten(0, 1) / temp, -1), reduction="batchmean",
            ) * temp * temp
            total = total + args.distill_weight * (head_kl + 0.5 * rows_kl)
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                predicted = soft_argmax_circular(logits, window=args.window)
                metres, _ = compute_metrics(
                    predicted,
                    target,
                    batch["ref_pos_m"],
                    batch["ref_time_s"],
                    batch["track_length_m"],
                    batch["lap_time_s"],
                )
            print(
                f"step {step:5d}/{steps}  loss {parts.total.item():7.4f} "
                f"(ce {parts.cross_entropy.item():6.3f} aux {parts.auxiliary.item():6.3f} "
                f"ref {parts.refine.item():6.4f})  train median {np.median(metres):7.2f} m  "
                f"{(time.time() - started) / step:5.2f} s/step"
            )

        if step % args.eval_every == 0 or step == steps:
            overall, per_track = evaluate(model, eval_loader, device, args.window)
            summary = overall["overall"]
            print(f"  eval  {summary.format()}")
            for track in sorted(per_track):
                print(f"        {track:<24} {per_track[track].format()}")
            log.append({"step": step, "overall": asdict(summary),
                        "per_track": {k: asdict(v) for k, v in per_track.items()}})
            (run / "metrics.json").write_text(json.dumps(log, indent=2))
            if summary.median_m < best:
                best = summary.median_m
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "frame_size": list(frame_size),
                        "step": step,
                        "median_m": best,
                    },
                    run / "best.pt",
                )

    print(f"\nbest eval median {best:.2f} m; artifacts in {run}")


if __name__ == "__main__":
    main()
