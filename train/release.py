"""
A model release: what any device needs to run PRIMAL, and nothing about training or about
any particular device or app.

  releases/<name>/
    encoder.mlpackage  Core ML: a 148x80 BGR image (pixel values / 255) -> the 128-d unit descriptor
    head.mlpackage     Core ML: a clip's 12 descriptors and the reference's N (any N) -> N logits
                       (the correlation scaled by the model's temperature, then the alignment head,
                       its circular padding written out so the converter needs no circular mode)
    release.json       the runtime contract: preprocessing, clip timing, reference bins, the
                       single-clip read, the tracker (every setting), the live engine's constants,
                       the confidence, the shown threshold, the session (train/session.py's
                       SessionConfig)
    golden/            test cases: frames (PNG) -> descriptors; clip + reference -> logits; camera
                       frames (PNG) -> model inputs (the crop and the area resize); a whole session
                       (RaceChrono's laps as descriptors) -> Python's results, three seeds; each
                       .f32 / .f64 raw little-endian, shapes and expectations in golden/index.json

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

from dataclasses import asdict

from train.estimator import EstimatorConfig
from train.eval import load_model
from train.live import AGREE_S, SPACING_S, STREAM_HZ, TRACKER, encode_frames
from train.session import LapCompleted, Reading, ReferenceClosed, Session, SessionConfig

GOLDEN_FRAMES = Path("data/real-footage/race-chrono/frames_148x80.npz")  # real frames, never trained on
# camera frames for the preprocessing cases: (video, frame indices); a GoPro preview session
# (848x480) and RaceChrono's video (1280x720)
PREPROCESS_SOURCES = [(Path("data/gopro/20261010T200332/video.mp4"), [300, 1500, 3000]),
                      (Path("data/real-footage/race-chrono/race-chrono.mp4"), [1000, 9000])]
REPLAY_CROSSINGS = Path("data/real-footage/race-chrono/starts_orb.json")  # RaceChrono's own lap crossings
REPLAY_SEEDS = (0, 1, 2)
MAX_FPS = 30  # what the app passes on
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


def write_f64(path: Path, a: np.ndarray) -> list[int]:
    np.ascontiguousarray(a, dtype="<f8").tofile(path)
    return list(a.shape)


def model_input(image: np.ndarray) -> np.ndarray:
    """The contract's preprocessing: the whole width at the 148:80 aspect, centred, area-resized to 148x80."""
    h, w = image.shape[:2]
    ch = min(h, int(round(w * 80 / 148)))
    y0 = (h - ch) // 2
    return cv2.resize(image[y0:y0 + ch], (148, 80), interpolation=cv2.INTER_AREA)


def at_most(t: np.ndarray, fps: float) -> np.ndarray:
    """Indices of the frames an app passes on at `fps` at most (by the frames' own times)."""
    keep, last = [], -np.inf
    for i, s in enumerate(t):
        if s - last >= 1 / fps - 0.004:
            keep.append(i)
            last = s
    return np.array(keep)


def session_summary(events: list) -> dict:
    closed = next(e for e in events if isinstance(e, ReferenceClosed))
    laps = [e for e in events if isinstance(e, LapCompleted)]
    readings = [e for e in events if isinstance(e, Reading)]
    return {"reference": {"line_s": closed.line_s, "lap_s": closed.lap_s, "placed_by": closed.placed_by},
            "laps": [{"lap_s": e.lap_s, "line_s": e.line_s, "shown_share": e.shown_share, "became_reference": e.became_reference}
                     for e in laps],
            "shown": float(np.mean([r.shown for r in readings]))}


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

    # preprocessing: camera frames and the model inputs the contract makes of them
    index["preprocess"] = []
    for video, frames_at in PREPROCESS_SOURCES:
        cap = cv2.VideoCapture(str(video))
        for k in frames_at:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, image = cap.read()
            assert ok, f"no frame {k} in {video}"
            i = len(index["preprocess"])
            cv2.imwrite(str(out / "golden" / f"source_{i:03d}.png"), image)
            cv2.imwrite(str(out / "golden" / f"input_{i:03d}.png"), model_input(image))
            index["preprocess"].append({"source": f"source_{i:03d}.png", "input": f"input_{i:03d}.png",
                                        "width": int(image.shape[1]), "height": int(image.shape[0])})
    index["preprocess_tolerance"] = {"max_abs_pixel_difference": 1}

    # a whole session: RaceChrono's laps, START at lap 1's crossing (a lap that began slowly)
    crossings = json.loads(REPLAY_CROSSINGS.read_text())
    zr = np.load(GOLDEN_FRAMES)
    t_all, f_all = zr["t"], zr["frames"]
    use = at_most(t_all, MAX_FPS)
    use = use[(t_all[use] >= crossings[0] - 0.5) & (t_all[use] <= crossings[-1] + 6)]
    replay_t = t_all[use].astype(np.float64)
    with torch.no_grad():
        replay_d = torch.cat([encode_frames(model, f_all[use][i:i + 256], device) for i in range(0, len(use), 256)])
    runs = []
    for seed in REPLAY_SEEDS:
        session, events = Session(model, clip_len, device, seed=seed), []
        for i, s_ in enumerate(replay_t):
            if session.state == "ready" and s_ >= crossings[0]:
                session.start(float(s_))
            events += session.push(replay_d[i], float(s_))
        runs.append(session_summary(events))
        print(f"replay seed {seed}: reference {runs[-1]['reference']['lap_s']:.2f} s, laps "
              f"{[round(l['lap_s'], 2) for l in runs[-1]['laps']]}, became {[l['became_reference'] for l in runs[-1]['laps']]}")
    index["session_replay"] = {
        "descriptors": write_f32(out / "golden" / "replay_descriptors.f32", replay_d.numpy()),
        "times": write_f64(out / "golden" / "replay_times.f64", replay_t),
        "start_s": float(crossings[0]), "true_crossings_s": crossings, "python": runs,
        "tolerance": {"time_s": 0.15, "rule": "each line and lap time within time_s of the Python runs' range; "
                      "the same laps becoming the reference; shown within 0.05 of the Python runs' lowest"}}
    (out / "golden" / "index.json").write_text(json.dumps(index, indent=2))

    contract = {
        "name": name, "run": a.run, "made": time.strftime("%Y-%m-%d %H:%M"),
        "input": {"size": [148, 80], "channels": "BGR", "range": "pixel / 255", "aspect": [148, 80],
                  "crop": "the camera's view cut to the training view, ~91.5 deg wide, at the 148:80 aspect; "
                          "a lens much wider is cropped toward it, never stretched",
                  "resize": "area averaging (as OpenCV INTER_AREA)"},
        "descriptor": {"dim": 128, "unit_length": True},
        "clip": {"length": clip_len, "spacing_s": SPACING_S, "frames": "the clip_len frames nearest to t - k * spacing_s"},
        "ticks_hz": STREAM_HZ,
        "reference": {"bin_s": BIN_S, "bins": "round(lap_s / bin_s) bins at i * lap_s / bins, each the frame nearest that time",
                      "circular": True},
        "single_read": {"soft_argmax_window_bins": 8, "circular": True},
        "tracker": {"settings": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(EstimatorConfig(**TRACKER)).items()},
                    "space": "reference time, circular over the lap",
                    "reference_python": "train/estimator.py ProgressEstimator with train/live.py LiveDelta"},
        "confidence": {"agree_s": AGREE_S, "window_s": 1.0,
                       "rule": "share of the last second's single reads within agree_s of the tracker"},
        "live": {"history_keep_s": 0.25, "tick_slack_s": 0.002, "start_hint_sd_s": 1.0,
                 "agree_window_ticks": int(round(STREAM_HZ)),
                 "reference_python": "train/live.py LiveDelta.push_descriptor"},
        "shown_at_or_above": CONF_MIN,
        "session": {**asdict(SessionConfig()), "reference_python": "train/session.py Session"},
        "check": check,
    }
    (out / "release.json").write_text(json.dumps(contract, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()
