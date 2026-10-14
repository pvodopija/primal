"""
Step 1 of the MobileNet port to Halo's NPU, in the training environment: a run's encoder
(train/model.py MobileNetEncoder) as a list of operations with BatchNorm folded into the
convolutions, the input normalisation, the projection reordered for a channels-last flatten,
calibration frames (AC training tracks) and test frames (AC held-out tracks, and two real
videos never trained on: RaceChrono's car, the indoor kart GoPro) with PyTorch's float32
descriptors for them. npu/build_mobilenet.py rebuilds it in TensorFlow from this.

Usage: PYTHONPATH=. python -m npu.export_mobilenet RUN OUT.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train.dataset import LapIndex, _to_chw
from train.eval import load_model
from train.model import POOL_GRID

CALIBRATION, PER_SET = 256, 240


def _fold(seq: nn.Sequential) -> tuple[np.ndarray, np.ndarray, dict]:
    """A conv + BatchNorm (+ activation) as one conv with a bias."""
    conv, bn = seq[0], seq[1]
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    w = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    b = (bn.bias - bn.running_mean * scale).detach()
    act = type(seq[2]).__name__.lower() if len(seq) > 2 else None
    return w.numpy(), b.numpy(), {"stride": conv.stride[0], "pad": conv.padding[0], "groups": conv.groups, "act": act}


def ops_of(encoder: nn.Module) -> tuple[list[dict], dict[str, np.ndarray]]:
    ops, arrays = [], {}

    def conv(seq: nn.Sequential) -> None:
        w, b, meta = _fold(seq)
        key = f"w{len(arrays)}"
        arrays[key], arrays[key + "b"] = w, b
        ops.append({"type": "conv", "w": key, "b": key + "b", **meta})

    for k, part in enumerate(encoder.features):
        if k == 0:
            conv(part)
            continue
        ops.append({"type": "block_start"})
        for layer in part.block:
            if isinstance(layer, nn.Sequential):
                conv(layer)
            else:  # squeeze-excite
                key = f"se{len(arrays)}"
                for name in ("fc1", "fc2"):
                    fc = getattr(layer, name)
                    arrays[f"{key}{name}"], arrays[f"{key}{name}b"] = fc.weight.detach().numpy(), fc.bias.detach().numpy()
                ops.append({"type": "se", "key": key})
        ops.append({"type": "block_end", "residual": bool(part.residual)})
    return ops, arrays


def frames(split: str, n: int, seed: int) -> np.ndarray:
    laps = LapIndex.load(Path("data/packed_ac_v3"), split=split).laps
    rng = np.random.default_rng(seed)
    return np.stack([np.asarray(laps[i].frames()[int(rng.integers(0, laps[i].n_frames))]) for i in rng.integers(0, len(laps), n)])


def main() -> None:
    run, out = sys.argv[1], Path(sys.argv[2])
    model, _ = load_model(Path("runs") / run / "best.pt", torch.device("cpu"))
    e = model.encoder
    ops, arrays = ops_of(e)
    # the projection, its 1440 inputs reordered from PyTorch's (C, H, W) flatten to (H, W, C)
    rows, cols = POOL_GRID
    w = e.project.weight.detach().numpy()  # (128, C*H*W)
    channels = w.shape[1] // (rows * cols)
    order = np.array([c * rows * cols + h * cols + x for h in range(rows) for x in range(cols) for c in range(channels)])
    arrays["proj_w"], arrays["proj_b"] = w[:, order], e.project.bias.detach().numpy()
    real = lambda path: (lambda f: f[np.linspace(0, len(f) - 1, PER_SET).astype(int)])(np.load(path)["frames"])
    sets = {"AC held-out tracks": frames("holdout", PER_SET, 0),
            "real: RaceChrono": real("data/real-footage/race-chrono/frames_148x80.npz"),
            "real: indoor kart": real("data/real-footage/indoor-pov/frames_148x80_c80.npz")}
    test = np.concatenate(list(sets.values()))
    with torch.no_grad():
        descriptors = e(torch.from_numpy(_to_chw(test))).numpy()
    np.savez(out, ops=json.dumps(ops), set_names=json.dumps(list(sets)), per_set=PER_SET,
             mean=e.mean.flatten().numpy(), std=e.std.flatten().numpy(),
             calibration=frames("train", CALIBRATION, 1), test=test, descriptors=descriptors, **arrays)
    print(f"{run}: {sum(o['type'] == 'conv' for o in ops)} convolutions, {sum(o['type'] == 'se' for o in ops)} squeeze-excites -> {out}")


if __name__ == "__main__":
    main()
