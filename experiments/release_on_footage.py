"""
A release's Core ML encoder on footage, against its PyTorch run: the cosine of every frame's
descriptor, on the processor alone (as the iOS Simulator runs it) and with every compute unit
(the Mac's nearest to the phone), then the session (train/session.py) replayed on the Core ML
descriptors, START at a moment, no hand close. A release passes its golden frames (AC, RaceChrono,
indoor) and can still fail footage unlike them: this checks it on the footage itself.

Usage: PYTHONPATH=. python experiments/release_on_footage.py RELEASE_DIR FRAMES.npz START_S [EVERY]
       (FRAMES.npz as experiments/closure_check.py caches them; EVERY: every n-th frame for the cosines)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from PIL import Image

from train.dataset import _to_chw
from train.eval import load_model
from train.session import LapCompleted, ReferenceClosed, Session


def coreml_descriptors(package: Path, frames: np.ndarray, units) -> np.ndarray:
    m = ct.models.MLModel(str(package), compute_units=units)
    image = lambda f: Image.fromarray(np.ascontiguousarray(f[..., ::-1]))  # BGR as stored -> the RGB PIL wants
    d = np.stack([m.predict({"image": image(f)})["descriptor"].reshape(-1) for f in frames]).astype(np.float32)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def replay(model, clip_len: int, d: torch.Tensor, t: np.ndarray) -> str:
    session = Session(model, clip_len, torch.device("cpu"))
    session.start(float(t[0]))
    closed, laps = None, []
    for k in range(len(t)):
        for e in session.push(d[k], float(t[k])):
            closed = e if isinstance(e, ReferenceClosed) else closed
            laps += [e] if isinstance(e, LapCompleted) else []
    head = f"closed at {closed.line_s:.2f} s, lap {closed.lap_s:.2f} s" if closed else "NEVER closed"
    return head + (f"; laps {[round(l.lap_s, 2) for l in laps][:8]}" if laps else "")


def main() -> None:
    release, cache, start_s = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    run = json.loads((release / "release.json").read_text())["run"]
    z = np.load(cache)
    keep = z["t"] >= start_s
    f, t = z["frames"][keep], z["t"][keep]
    model, payload = load_model(Path("runs") / run / "best.pt", torch.device("cpu"))
    with torch.no_grad():
        ref = torch.cat([model.encoder(torch.from_numpy(_to_chw(f[i:i + 256]))) for i in range(0, len(f), 256)]).numpy()
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    print(f"{release.name} on {cache.name} from {start_s} s ({len(f)} frames)")
    for label, units in (("processor only (Simulator)", ct.ComputeUnit.CPU_ONLY), ("all compute units", ct.ComputeUnit.ALL)):
        d = coreml_descriptors(release / "encoder.mlpackage", f[::every], units)
        cos = np.sum(d * ref[::every], 1)
        print(f"  {label:<28} cosine to PyTorch: min {cos.min():.4f}, median {np.median(cos):.4f}, "
              f"{np.mean(cos < 0.995) * 100:.1f}% of frames below 0.995", flush=True)
    d = coreml_descriptors(release / "encoder.mlpackage", f, ct.ComputeUnit.CPU_ONLY)
    print(f"  session on PyTorch's descriptors:      {replay(model, int(payload['args']['clip_len']), torch.from_numpy(ref), t)}")
    print(f"  session on Core ML's (processor only): {replay(model, int(payload['args']['clip_len']), torch.from_numpy(d), t)}")


if __name__ == "__main__":
    main()
