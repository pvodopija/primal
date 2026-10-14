"""
Step 2 of the MobileNet port, in .venv-npu (TensorFlow + Vela): rebuild the exported encoder
(npu/export_mobilenet.py) in Keras with PyTorch's padding (explicit, symmetric) and order of
operations, check it against PyTorch in float32, convert it to full-integer int8 TFLite with
real calibration frames, run the int8 model (the integer arithmetic the Ethos-U55 executes)
on the test frames and compare with PyTorch's float32 descriptors, and compile it with Vela
for Halo (Ethos-U55-128, 160 MHz, npu/vela.ini): which operators run on the NPU, time, memory.
The descriptor's final unit-length scaling stays outside the model (128 numbers, on the CPU);
cosine similarity ignores it.

Usage: .venv-npu/bin/python npu/build_mobilenet.py EXPORT.npz OUT_DIR [CLIP_PERCENTILE, e.g. 99.99]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu.build_tflite import to_int8, vela  # noqa: E402


def run_tflite(blob: bytes, x: np.ndarray) -> np.ndarray:
    """The int8 model on TFLite's reference kernels (the arithmetic the Ethos-U follows), one frame at a time."""
    interpreter = tf.lite.Interpreter(model_content=blob,
                                      experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    i, o = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    out = []
    for sample in x:
        v = sample[None].astype(np.float32)
        if i["dtype"] == np.int8:
            s, z = i["quantization"]
            v = np.clip(np.round(v / s + z), -128, 127).astype(np.int8)
        interpreter.set_tensor(i["index"], v)
        interpreter.invoke()
        r = interpreter.get_tensor(o["index"])
        if o["dtype"] == np.int8:
            s, z = o["quantization"]
            r = (r.astype(np.float32) - z) * s
        out.append(r[0])
    return np.stack(out)


# "explicit": hard-sigmoid as a clamp, so the converter keeps MUL / MINIMUM / MAXIMUM instead of
# fusing HARD_SWISH, which has no 16-bit-activation kernel in TFLite
HARD = os.environ.get("NPU_HARD", "fused")


def hard_sigmoid(t):
    if HARD == "explicit":
        return tf.minimum(tf.maximum(t * (1.0 / 6.0) + 0.5, 0.0), 1.0)
    return tf.nn.relu6(t + 3.0) / 6.0


def build(p: dict, clips: list[tuple[float, float]] | None = None) -> keras.Model:
    """The encoder in Keras. With `clips`, every activation point is clamped to its range
    (as quantization-aware training clipped it), so int8 calibration sees the trained ranges
    instead of rare outliers; without, the points are recorded (model.points) to measure."""
    points = []

    def point(t):
        if clips is None:
            points.append(t)
            return t
        lo, hi = clips[len(points)]
        points.append(t)
        return keras.layers.Lambda(lambda v, lo=lo, hi=hi: tf.clip_by_value(v, lo, hi))(t)

    inp = keras.Input((80, 148, 3), batch_size=1, name="image")  # BGR in [0, 1]; one frame, static shapes for the NPU
    # BGR -> RGB and ImageNet's normalisation, as one fixed 1x1 convolution
    norm = keras.layers.Conv2D(3, 1, name="normalise")
    x = norm(inp)
    w = np.zeros((1, 1, 3, 3), np.float32)
    for c, src in enumerate((2, 1, 0)):
        w[0, 0, src, c] = 1.0 / p["std"][c]
    norm.set_weights([w, (-p["mean"] / p["std"]).astype(np.float32)])
    block_in, block_outputs = None, []
    for i, op in enumerate(json.loads(str(p["ops"]))):
        if op["type"] == "block_start":
            block_in = x
        elif op["type"] == "block_end":
            if op["residual"]:
                x = point(keras.layers.Add()([block_in, x]))
            block_outputs.append(x)
        elif op["type"] == "se":
            k = op["key"]
            s = keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
            fc1 = keras.layers.Conv2D(p[k + "fc1"].shape[0], 1, activation="relu")
            s = fc1(s)
            fc1.set_weights([p[k + "fc1"].transpose(2, 3, 1, 0), p[k + "fc1b"]])
            fc2 = keras.layers.Conv2D(p[k + "fc2"].shape[0], 1)
            s = fc2(s)
            fc2.set_weights([p[k + "fc2"].transpose(2, 3, 1, 0), p[k + "fc2b"]])
            x = point(keras.layers.Multiply()([x, keras.layers.Lambda(hard_sigmoid)(s)]))
        else:  # conv (BatchNorm folded)
            w, b = p[op["w"]], p[op["b"]]
            if op["pad"]:
                x = keras.layers.ZeroPadding2D(op["pad"])(x)
            if op["groups"] > 1:
                layer = keras.layers.DepthwiseConv2D(w.shape[-1], strides=op["stride"])
                x = layer(x)
                layer.set_weights([w.transpose(2, 3, 0, 1), b])
            else:
                layer = keras.layers.Conv2D(w.shape[0], w.shape[-1], strides=op["stride"])
                x = layer(x)
                layer.set_weights([w.transpose(2, 3, 1, 0), b])
            if op["act"]:  # the folded conv's own output is an int8 tensor too, before its activation
                x = point(x)
            if op["act"] == "relu":
                x = keras.layers.ReLU()(x)
            elif op["act"] == "hardswish":
                x = keras.layers.Lambda(lambda t: t * hard_sigmoid(t))(x)
            x = point(x)
    x = keras.layers.Reshape((int(np.prod(x.shape[1:])),))(x)
    proj = keras.layers.Dense(p["proj_w"].shape[0], name="project")
    x = proj(x)
    proj.set_weights([p["proj_w"].T, p["proj_b"]])
    model = keras.Model(inp, x, name="mobilenet_encoder")
    model.block_outputs = block_outputs  # each block's output, for locating int8 losses
    model.points = points
    return model


def clip_ranges(p: dict, calib: np.ndarray, percentile: float) -> list[tuple[float, float]]:
    """Each activation point's range over the calibration frames: the percentile and its mirror."""
    model = build(p)
    probe = keras.Model(model.input, model.points)
    values = [[] for _ in model.points]
    for k in range(0, len(calib), 1):
        for i, v in enumerate(probe(calib[k:k + 1], training=False)):
            flat = v.numpy().ravel()
            values[i].append(flat[np.random.default_rng(k).integers(0, flat.size, min(flat.size, 4096))])
    out = []
    for v in values:
        v = np.concatenate(v)
        out.append((float(np.percentile(v, 100 - percentile)), float(np.percentile(v, percentile))))
    return out


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def main() -> None:
    export, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    p = dict(np.load(export))
    to_float = lambda f: f.astype(np.float32) / 255.0  # BGR uint8 -> [0, 1]
    test, calib = to_float(p["test"]), to_float(p["calibration"])
    names, per = json.loads(str(p["set_names"])), int(p["per_set"])
    percentile = float(sys.argv[3]) if len(sys.argv) > 3 else None
    model = build(p, clip_ranges(p, calib, percentile) if percentile else None)
    report = {"clip_percentile": percentile}
    fp32 = np.concatenate([model(test[k:k + 1], training=False).numpy() for k in range(len(test))])
    c = cosines(fp32, p["descriptors"])
    report["keras_float32_cosine_min"] = float(c.min())
    print(f"Keras float32 against PyTorch: cosine min {c.min():.6f} (the port is exact if ~1)")
    blob, ops = to_int8(model, calib, out_dir / "encoder_int8.tflite")
    print("int8 operators:", ", ".join(ops))
    q = run_tflite(blob, test)
    c = cosines(q, p["descriptors"])
    for k, name in enumerate(names):
        part = c[k * per:(k + 1) * per]
        report[f"int8, {name}"] = {"cosine_min": float(part.min()), "cosine_median": float(np.median(part)),
                                   "below_0.995": int(np.sum(part < 0.995))}
        print(f"  int8 TFLite, {name:<20} cosine min {part.min():.4f}, median {np.median(part):.4f}, "
              f"{np.sum(part < 0.995):3d} of {len(part)} below 0.995")
    for mode in ("Sram_Only", "Shared_Sram"):
        v = vela(out_dir / "encoder_int8.tflite", out_dir, mode)
        report[f"vela {mode}"] = {k: val for k, val in v.items() if k != "raw"}
        print(f"  Vela {mode:<11} NPU ops {v['npu_ops']}, CPU ops {v['cpu_ops']} {v['cpu_detail']}, {v['ms']:.2f} ms, "
              f"SRAM {v['sram_kib']:.0f} KiB, flash {v['flash_kib']:.0f} KiB, utilisation {100 * v['utilisation']:.0f}%")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
