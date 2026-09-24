"""
End-to-end check of the capture path without Assetto Corsa.

Renders a short synthetic "recording" with the timecode grid burned in, then
runs the same calibrate / decode / pack chain a real session goes through and
asserts the recovered labels match the labels that were rendered in.

Run: python -m tests.test_pipeline_e2e
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from capture.pack import pack_session
from capture.timecode import COLS, SPLINE_MAX, OverlayGeometry

ML_ROOT = Path(__file__).resolve().parents[1]
WORK = ML_ROOT / "data" / "_e2e"


def _run(*args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", *args], cwd=ML_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")


def test_capture_pipeline_roundtrip() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    session = WORK / "session"

    _run(
        "train.synthetic",
        "recording",
        "--out",
        str(session),
        "--laps",
        "2",
        "--length-m",
        "220",
        "--fps",
        "30",
        "--width",
        "640",
        "--height",
        "384",
    )

    with np.load(session / "truth.npz") as data:
        truth = {"counter": data["counter"], "s": data["s"]}
    rendered = OverlayGeometry.from_dict(json.loads((session / "overlay_truth.json").read_text()))

    # Seed the locator with a deliberately sloppy box, as a human would drag it.
    sloppy = f"{int(rendered.x0) - 6},{int(rendered.y0) + 5},{int(rendered.cell * COLS) + 9},60"
    _run(
        "capture.overlay_decode",
        "calibrate",
        str(session / "video.mp4"),
        "--roi",
        sloppy,
        "--frames",
        "12",
    )
    found = json.loads((session / "overlay.json").read_text())
    assert found["pass_rate"] == 1.0, f"locator scored {found['pass_rate']}"

    _run("capture.overlay_decode", "decode", str(session / "video.mp4"))
    labels = pd.read_parquet(session / "labels.parquet")

    assert bool(labels.valid.all()), f"{int((~labels.valid).sum())} frames failed the checksum"
    assert len(labels) == truth["counter"].size
    assert np.array_equal(labels.counter.to_numpy(), truth["counter"])
    # Only the 20-bit quantisation should separate decoded from rendered labels.
    assert np.abs(labels.spline_pos.to_numpy() - truth["s"]).max() <= 2.0 / SPLINE_MAX

    entries = pack_session(
        session,
        WORK / "packed",
        size=(160, 96),
        ref_spacing_m=1.0,
        min_frames=60,
    )
    assert len(entries) == 2, f"expected 2 laps, packed {len(entries)}"

    meta = json.loads((session / "run.json").read_text())
    for entry in entries:
        lap = WORK / "packed" / entry["path"]
        frames = np.load(lap / "frames.npy", mmap_mode="r")
        s = np.load(lap / "s.npy")
        assert frames.shape == (entry["n_frames"], 96, 160, 3)
        assert s.shape == (entry["n_frames"],)
        assert entry["ref_bins"] == round(meta["track_length_m"] / 1.0)
        # The timecode band must be gone: no pure-white/pure-black cell rows left.
        assert float(np.asarray(frames[::10]).std()) > 5.0
        del frames

    _run("capture.pack", str(session), "--out", str(WORK / "packed_cli"), "--min-frames", "60")
    index = json.loads((WORK / "packed_cli" / "index.json").read_text())
    assert len(index["laps"]) == 2

    shutil.rmtree(WORK)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline tests passed")
