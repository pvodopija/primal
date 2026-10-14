"""
Every frame of a real-footage frame cache through an int8 TFLite encoder on TFLite's
reference kernels (the integer arithmetic the Ethos-U55 follows): the descriptors Halo would
compute, unit length, for experiments/real_truth_orb.py to score (PRIMAL_DESCRIPTORS).

Usage: .venv-npu/bin/python npu/encode_tflite.py ENCODER.tflite FRAMES.npz OUT.npy
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf


def main() -> None:
    model, frames_path, out = sys.argv[1:4]
    frames = np.load(frames_path)["frames"]
    it = tf.lite.Interpreter(model_path=model, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i["quantization"]
    so, zo = o["quantization"]
    d = np.empty((len(frames), o["shape"][-1]), np.float32)
    for k, f in enumerate(frames):
        v = f[None].astype(np.float32) / 255.0  # BGR in [0, 1], as the model's input
        it.set_tensor(i["index"], np.clip(np.round(v / si + zi), -128, 127).astype(np.int8) if i["dtype"] == np.int8 else v)
        it.invoke()
        r = it.get_tensor(o["index"])[0]
        d[k] = (r.astype(np.float32) - zo) * so if o["dtype"] == np.int8 else r
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    np.save(out, d)
    print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
