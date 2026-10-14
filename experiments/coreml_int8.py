"""
How a model's encoder survives Core ML at the Neural Engine's precisions, measured as the
cosine between Core ML's descriptors and PyTorch's float32 ones on 240 frames each from the
held-out AC tracks and from two real videos never trained on (RaceChrono's car, which is
also the release's golden set, and the indoor kart GoPro): float16 (the iOS release's default), and int8 weights (per channel) with
int8 activations (calibrated on 128 AC training frames), as the Neural Engine runs an A8W8
model and the nearest stand-in on this Mac for Halo's int8 NPU (whose compiler, Vela, is not
here). The release's tolerance is a cosine of 0.995 on every frame.

Usage: PYTHONPATH=. python experiments/coreml_int8.py RUN [RUN ...]   (writes runs/RUN/coreml_int8.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_activations,
                                         linear_quantize_weights)
from PIL import Image

from train.dataset import LapIndex, _to_chw
from train.eval import load_model
from train.release import Encoder

FRAMES, CALIBRATION = 240, 128


def frames(split: str, n: int, seed: int) -> np.ndarray:
    laps = LapIndex.load(Path("data/packed_ac_v3"), split=split).laps
    rng = np.random.default_rng(seed)
    return np.stack([np.asarray(laps[i].frames()[int(rng.integers(0, laps[i].n_frames))]) for i in rng.integers(0, len(laps), n)])


def main() -> None:
    calibration = frames("train", CALIBRATION, 1)
    real = lambda path: (lambda f: f[np.linspace(0, len(f) - 1, FRAMES).astype(int)])(np.load(path)["frames"])
    sets = {"AC held-out tracks": frames("holdout", FRAMES, 0),
            "real: RaceChrono": real("data/real-footage/race-chrono/frames_148x80.npz"),
            "real: indoor kart": real("data/real-footage/indoor-pov/frames_148x80_c80.npz")}
    test = np.concatenate(list(sets.values()))
    image = lambda f: Image.fromarray(np.ascontiguousarray(f[..., ::-1]))  # BGR as stored -> the RGB PIL wants
    for run in sys.argv[1:]:
        model, _ = load_model(Path("runs") / run / "best.pt", torch.device("cpu"))
        enc = Encoder(model.encoder).eval()
        with torch.no_grad():
            reference = model.encoder(torch.from_numpy(_to_chw(test))).numpy()  # the wrapper takes one image
            traced = torch.jit.trace(enc, torch.from_numpy(np.moveaxis(test[:1], -1, 1).astype(np.float32) / 255))
        fp16 = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=(1, 3, 80, 148), color_layout=ct.colorlayout.BGR, scale=1 / 255)],
                          outputs=[ct.TensorType(name="descriptor")], minimum_deployment_target=ct.target.iOS17,
                          compute_precision=ct.precision.FLOAT16)
        a8 = linear_quantize_activations(fp16, OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric")),
                                         sample_data=[{"image": image(f)} for f in calibration])
        w8a8 = linear_quantize_weights(a8, OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel")))
        report = {}
        for name, m in (("float16", fp16), ("int8 weights + int8 activations", w8a8)):
            d = np.stack([m.predict({"image": image(f)})["descriptor"].reshape(-1) for f in test])
            cos = np.sum(d * reference, 1) / np.linalg.norm(d, axis=1) / np.linalg.norm(reference, axis=1)
            for k, (set_name, part) in enumerate(sets.items()):
                c = cos[k * FRAMES:(k + 1) * FRAMES]
                report[f"{name}, {set_name}"] = {"cosine_min": float(c.min()), "cosine_median": float(np.median(c)),
                                                 "below_0.995": int(np.sum(c < 0.995))}
                print(f"{run:<24} {name:<32} {set_name:<20} cosine min {c.min():.4f}, median {np.median(c):.4f}, "
                      f"{np.sum(c < 0.995):3d} of {len(c)} below 0.995", flush=True)
        (Path("runs") / run / "coreml_int8.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
