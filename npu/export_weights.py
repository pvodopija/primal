"""
Step 1 of the NPU estimate, in the training environment: dump a checkpoint's weights,
real calibration data, and PyTorch's own outputs for checking the port.

Writes one .npz with
- every parameter under its PyTorch name;
- encoder calibration frames (training tracks) and check frames (held-out tracks), uint8
  BGR as packed, with PyTorch's fp32 embeddings of the check frames;
- head inputs (correlation grids, [K, N] times the learned scale) for each reference
  length asked for, calibration and check sets, with PyTorch's head logits.

Usage: python -m npu.export_weights CHECKPOINT DATA OUT.npz [--bins-tracks track=... ]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train.dataset import LapIndex, _to_chw
from train.eval import load_model, reference_axis_of


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--tracks", default="ks_silverstone__national,ks_brands_hatch__indy",
                        help="reference tracks for the head: one long, one kart-length")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    model, payload = load_model(Path(args.checkpoint), torch.device("cpu"))
    model.eval()
    clip_len, axis = int(payload["args"]["clip_len"]), reference_axis_of(payload)
    out: dict[str, np.ndarray] = {f"param/{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    out["clip_len"] = np.array(clip_len)

    def frames_from(split: str, n: int) -> np.ndarray:
        laps = LapIndex.load(Path(args.data), split=split).laps
        picks = []
        for _ in range(n):
            lap = laps[int(rng.integers(len(laps)))]
            picks.append(np.asarray(lap.frames()[int(rng.integers(lap.n_frames))]))
        return np.stack(picks)

    with torch.no_grad():
        out["enc_calib"] = frames_from("train", args.frames)
        out["enc_check"] = frames_from("holdout", 100)
        out["enc_check_emb"] = model.encoder(torch.from_numpy(_to_chw(out["enc_check"]))).numpy()

        scale = float(model.logit_scale.exp())
        for track in args.tracks.split(","):
            laps = sorted(LapIndex.load(Path(args.data)).by_track[track], key=lambda l: l.lap_id)
            ref = next(l for l in laps if l.usable_as_reference(0.98, axis))
            grid = ref.reference_grid(axis)
            ref_emb = model.encoder(torch.from_numpy(_to_chw(np.asarray(ref.frames()[grid.frame_idx]))))
            lives = [l for l in laps if l.session_id != ref.session_id and l.s_span > 0.98][:3]
            grids = []
            for live in lives:
                emb = model.encoder(torch.from_numpy(_to_chw(np.asarray(live.frames()))))
                for _ in range(80):
                    t = int(rng.integers((clip_len - 1) * args.stride, live.n_frames))
                    idx = t - np.arange(clip_len - 1, -1, -1) * args.stride
                    grids.append((emb[idx] @ ref_emb.T).numpy() * scale)
            grids = np.stack(grids).astype(np.float32)  # [S, K, N]
            n = grid.n_bins
            if f"head{n}_calib" in out:  # the arrays are keyed by bin count
                raise SystemExit(f"{track} has {n} bins like an earlier track; pick tracks of different lengths")
            out[f"head{n}_calib"] = grids[:160]
            out[f"head{n}_check"] = grids[160:]
            out[f"head{n}_check_logits"] = model.head(torch.from_numpy(grids[160:])).numpy()
            print(f"{track}: {n} bins, {len(grids)} correlation grids")
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, {k: v.shape for k, v in out.items() if not k.startswith("param/")})


if __name__ == "__main__":
    main()
