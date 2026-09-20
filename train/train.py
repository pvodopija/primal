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

from train.dataset import AlignmentBatches, LapIndex, SampleConfig
from train.model import (
    SequenceAligner,
    alignment_loss,
    compute_metrics,
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
        ordered = sorted(group, key=lambda lap: lap.lap_id)
        spare = len(ordered) - 2
        if spare <= 0:
            continue
        reserved.update(lap.lap_id for lap in ordered[-min(per_track, spare) :])
    return frozenset(reserved)


def build_index(data: Path, gate: GateConfig, split: str) -> LapIndex:
    index = LapIndex.load(data, split=split)
    if gate.tracks is not None:
        keep = sorted(index.by_track)[: gate.tracks]
        index = LapIndex.load(data, split=split, tracks=keep)
    if gate.laps_per_track is not None:
        index = LapIndex.load(
            data,
            split=split,
            tracks=sorted(index.by_track),
            max_laps_per_track=gate.laps_per_track,
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
            logits.shape[-1],
            batch["ref_spacing_m"],
            batch["ref_speed_mps"].to(device),
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
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--refine-weight", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument(
        "--holdout-laps", type=int, default=1, help="live-only laps per track for the g1 gate"
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true", help="disable reference recompute")
    args = parser.parse_args()

    gate = GateConfig.get(args.gate)
    steps = args.steps if args.steps is not None else gate.steps
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    data = Path(args.data)
    train_index = build_index(data, gate, split="train")
    eval_index = build_index(data, gate, split=gate.eval_split)

    # G1 asks about unseen laps of seen tracks, so some laps are live-only. G0
    # deliberately trains and evaluates on the same pair, and G2 evaluates on a
    # different split entirely, so neither needs a lap-level split.
    reserved: frozenset[str] = frozenset()
    if gate.name == "g1" and args.holdout_laps > 0:
        reserved = holdout_live_laps(train_index, args.holdout_laps)
    trainable = frozenset(lap.lap_id for lap in train_index.laps) - reserved

    train_config = SampleConfig(
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        roll_reference=gate.roll_reference,
        jitter=gate.jitter,
        sigma_bins=args.sigma_bins,
        reference_only=trainable if reserved else None,
        live_only=trainable if reserved else None,
    )
    train_set = AlignmentBatches(train_index, train_config, steps=steps, seed=args.seed)

    eval_config = SampleConfig(
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        roll_reference=gate.roll_reference,
        jitter=False,
        sigma_bins=args.sigma_bins,
        reference_only=trainable if reserved else None,
        live_only=reserved if reserved else None,
    )
    # A distinct seed stream, so evaluation clips are not the training clips.
    eval_set = AlignmentBatches(eval_index, eval_config, steps=args.eval_steps, seed=args.seed + 9973)

    sample_shape = np.load(train_index.laps[0].directory / "frames.npy", mmap_mode="r").shape
    frame_size = (int(sample_shape[1]), int(sample_shape[2]))
    model = SequenceAligner(
        clip_len=args.clip_len,
        dim=args.dim,
        width=args.width,
        hidden=args.hidden,
        frame_size=frame_size,
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters())

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
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.15
    )

    log: list[dict] = []
    best = float("inf")
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        live = batch["live"].to(device)
        reference = batch["reference"].to(device)
        target = batch["target"].to(device)
        soft_target = batch["soft_target"].to(device)

        logits, correlation = model(live, reference, use_checkpoint=not args.no_checkpoint)
        parts = alignment_loss(
            logits,
            correlation,
            soft_target,
            target,
            aux_weight=args.aux_weight,
            refine_weight=args.refine_weight,
            window=args.window,
        )
        optimiser.zero_grad(set_to_none=True)
        parts.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                predicted = soft_argmax_circular(logits, window=args.window)
                metres, _ = compute_metrics(
                    predicted,
                    target,
                    logits.shape[-1],
                    batch["ref_spacing_m"],
                    batch["ref_speed_mps"].to(device),
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
