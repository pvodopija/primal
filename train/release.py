"""
A model release: what any device needs to run PRIMAL, and nothing about training or about
any particular device or app.

  releases/<name>/
    encoder.mlpackage  Core ML: a 148x80 BGR image (pixel values / 255) -> the 128-d unit descriptor
    head.mlpackage     Core ML: a clip's 12 descriptors and the reference's N (any N) -> N logits
                       (the correlation scaled by the model's temperature, then the alignment head,
                       its circular padding written out so the converter needs no circular mode)
    release.json       the runtime contract: preprocessing, clip timing, reference bins, the
                       single-clip read, the tracker, the confidence, the shown threshold
    golden/            test cases: frames (PNG) -> descriptors; clip + reference -> logits; each
                       .f32 raw little-endian float32, shapes in golden/index.json

The models are checked against PyTorch here, on the golden frames (Core ML on this Mac, all
compute units), and the result goes in release.json.

Usage: PYTHONPATH=. python -m train.release RUN [--name primal-v3-mobilenet-20261011]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from train.eval import load_model
from train.live import AGREE_S, SPACING_S, STREAM_HZ, TRACKER, encode_frames

GOLDEN_FRAMES = Path("data/real-footage/race-chrono/frames_148x80.npz")  # real frames, never trained on
BIN_S = 0.052  # reference bin spacing on the time axis, as experiments/real_footage.py
CONF_MIN = 0.5  # the delta is shown at or above this confidence


class Encoder(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)[0]


def _circular(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """`conv` with its circular padding written out as a wrap-around concatenation."""
    p = conv.dilation[0] * (conv.kernel_size[0] // 2)
    if p:
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
    return F.conv1d(x, conv.weight, conv.bias, dilation=conv.dilation)


class Head(nn.Module):
    """The model's matching: clip (K, D) against reference (N, D), scaled, through the head."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.head = model.head
        self.register_buffer("scale", model.logit_scale.exp().detach().clone())

    def forward(self, clip: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        h = self.head
        x = _circular(h.entry, (clip @ reference.T)[None] * self.scale)
        for block in h.blocks:
            x = x + _circular(block[2], block[1](block[0](x)))
        return h.exit(x)[0, 0]


def write_f32(path: Path, a: np.ndarray) -> list[int]:
    np.ascontiguousarray(a, dtype="<f4").tofile(path)
    return list(a.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--name")
    a = ap.parse_args()
    name = a.name or f"primal-{a.run.replace('_', '-')}-{time.strftime('%Y%m%d')}"
    out = Path("releases") / name
    if out.exists():
        shutil.rmtree(out)
    (out / "golden").mkdir(parents=True)

    device = torch.device("cpu")
    model, payload = load_model(Path("runs") / a.run / "best.pt", device)
    model.eval()
    clip_len = int(payload["args"]["clip_len"])

    # golden frames: real ones the model never saw, a lap's worth spaced out
    z = np.load(GOLDEN_FRAMES)
    frames = z["frames"]
    take = np.linspace(0, len(frames) - 1, 240).astype(int)
    golden = frames[take]  # (240, 80, 148, 3) BGR uint8
    with torch.no_grad():
        descriptors = encode_frames(model, golden, device).numpy()
        enc = Encoder(model.encoder).eval()
        x = torch.from_numpy(np.moveaxis(golden[:1], -1, 1).astype(np.float32) / 255)
        assert np.allclose(enc(x).numpy(), descriptors[0], atol=1e-5), "the encoder wrapper drifted from encode_frames"
        traced_enc = torch.jit.trace(enc, x)
        clip, reference = torch.from_numpy(descriptors[:clip_len]), torch.from_numpy(descriptors)
        head = Head(model).eval()
        reference_logits = model.head(torch.einsum("kd,nd->kn", clip, reference)[None] * model.logit_scale.exp())[0].numpy()
        assert np.allclose(head(clip, reference).numpy(), reference_logits, atol=1e-4), "the head wrapper drifted from the model"
        traced_head = torch.jit.trace(head, (clip, reference))

    encoder_ml = ct.convert(traced_enc, inputs=[ct.ImageType(name="image", shape=(1, 3, 80, 148), color_layout=ct.colorlayout.BGR, scale=1 / 255)],
                            outputs=[ct.TensorType(name="descriptor")], minimum_deployment_target=ct.target.iOS17,
                            compute_precision=ct.precision.FLOAT16)
    head_ml = ct.convert(traced_head, inputs=[ct.TensorType(name="clip", shape=(clip_len, 128)),
                                              ct.TensorType(name="reference", shape=(ct.RangeDim(32, 20000, default=240), 128))],
                         outputs=[ct.TensorType(name="logits")], minimum_deployment_target=ct.target.iOS17,
                         compute_precision=ct.precision.FLOAT16)
    for m, short in ((encoder_ml, "encoder"), (head_ml, "head")):
        m.short_description = f"PRIMAL {short}, run {a.run} ({name})"
        m.save(str(out / f"{short}.mlpackage"))

    # Core ML against PyTorch on the golden cases (this Mac: CPU, GPU and Neural Engine as Core ML picks)
    ml_desc = np.stack([encoder_ml.predict({"image": Image.fromarray(f[..., ::-1])})["descriptor"].reshape(-1) for f in golden])
    cosine = np.sum(ml_desc * descriptors, 1) / np.linalg.norm(ml_desc, axis=1)
    ml_logits = head_ml.predict({"clip": clip.numpy(), "reference": reference.numpy()})["logits"].reshape(-1)
    check = {"encoder_cosine_min": float(cosine.min()), "encoder_cosine_median": float(np.median(cosine)),
             "head_logits_max_abs_diff": float(np.abs(ml_logits - reference_logits).max()),
             "head_argmax_same": bool(np.argmax(ml_logits) == np.argmax(reference_logits))}
    print(json.dumps(check, indent=2))

    # golden cases for the device's own tests
    for i, f in enumerate(golden):
        cv2.imwrite(str(out / "golden" / f"frame_{i:03d}.png"), f)
    index = {"frames": "frame_###.png, 148x80 BGR as stored (PNG)", "count": int(len(golden)),
             "descriptors": write_f32(out / "golden" / "descriptors.f32", descriptors),
             "clip": "the first clip_len descriptors", "reference": "all the descriptors",
             "logits": write_f32(out / "golden" / "logits.f32", reference_logits),
             "tolerance": {"descriptor_cosine_min": 0.995, "logits_argmax": "same bin"}}
    (out / "golden" / "index.json").write_text(json.dumps(index, indent=2))

    contract = {
        "name": name, "run": a.run, "made": time.strftime("%Y-%m-%d %H:%M"),
        "input": {"size": [148, 80], "channels": "BGR", "range": "pixel / 255",
                  "crop": "the camera's view cut to the training view, ~91.5 deg wide, at the 148:80 aspect; "
                          "a lens much wider is cropped toward it, never stretched",
                  "resize": "area averaging (as OpenCV INTER_AREA)"},
        "descriptor": {"dim": 128, "unit_length": True},
        "clip": {"length": clip_len, "spacing_s": SPACING_S, "frames": "the clip_len frames nearest to t - k * spacing_s"},
        "ticks_hz": STREAM_HZ,
        "reference": {"bin_s": BIN_S, "bins": "round(lap_s / bin_s) bins at i * lap_s / bins, each the frame nearest that time",
                      "circular": True},
        "single_read": {"soft_argmax_window_bins": 8, "circular": True},
        "tracker": {"settings": TRACKER, "space": "reference time, circular over the lap",
                    "reference_python": "train/estimator.py ProgressEstimator with train/live.py LiveDelta"},
        "confidence": {"agree_s": AGREE_S, "window_s": 1.0,
                       "rule": "share of the last second's single reads within agree_s of the tracker"},
        "shown_at_or_above": CONF_MIN,
        "check": check,
    }
    (out / "release.json").write_text(json.dumps(contract, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()
