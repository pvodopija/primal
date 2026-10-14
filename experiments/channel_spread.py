"""
How unevenly a model's channels are scaled, the thing int8 cannot represent: for every
BatchNorm and activation output of the encoder, its loudest channel's largest magnitude over
its median channel's, on 128 AC training frames. int8 gives a tensor one scale, so a spread of
30x leaves the quiet channels a few of its 256 steps. Writes runs/RUN/channel_spread.json.

Usage: PYTHONPATH=. python experiments/channel_spread.py RUN [RUN ...]
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


def main() -> None:
    laps = LapIndex.load(Path("data/packed_ac_v3"), split="train").laps
    rng = np.random.default_rng(0)
    x = torch.from_numpy(_to_chw(np.stack([np.asarray(laps[i].frames()[int(rng.integers(0, laps[i].n_frames))])
                                           for i in rng.integers(0, len(laps), 128)])))
    for run in sys.argv[1:]:
        model, _ = load_model(Path("runs") / run / "best.pt", torch.device("cpu"))
        seen = []
        hooks = [m.register_forward_hook(lambda m, i, o, n=n: seen.append((n, o.detach().abs().amax(dim=(0, 2, 3)))))
                 for n, m in model.encoder.named_modules() if isinstance(m, (nn.BatchNorm2d, nn.Hardswish, nn.ReLU))]
        with torch.no_grad():
            model.encoder(x)
        for h in hooks:
            h.remove()
        ratio = {n: float(r.max() / r.median().clamp_min(1e-6)) for n, r in seen}
        worst = sorted(ratio.items(), key=lambda kv: -kv[1])[:5]
        print(f"{run}: loudest/median channel, median over {len(ratio)} tensors {np.median(list(ratio.values())):.1f}x; worst: "
              + ", ".join(f"{n} {v:.1f}x" for n, v in worst))
        (Path("runs") / run / "channel_spread.json").write_text(json.dumps(ratio, indent=2))


if __name__ == "__main__":
    main()
