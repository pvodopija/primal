"""
Step 2 of the NPU estimate, in .venv-npu (TensorFlow + Vela, no PyTorch): rebuild the
encoder and head in Keras from the exported weights, check them against PyTorch, convert
to int8 TFLite with real calibration data, and compile with Vela for Halo's NPU
(Ethos-U55, 128 MACs/cycle, 160 MHz; `npu/vela.ini`).

Two variants:
- "as trained": the exact architecture (GroupNorm, GELU), weights copied. Checked against
  PyTorch in fp32 and int8, then compiled: which operators run on the NPU, and the cost.
- "no norm, GELU" and "no norm, ReLU": the candidate replacements (GroupNorm removed or
  folded away; GELU runs on the NPU, GroupNorm does not), same shapes. Their weights are
  not retrained, so only their cycles and memory mean anything.

Usage: .venv-npu/bin/python npu/build_tflite.py EXPORT.npz OUT_DIR
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras

HERE = Path(__file__).resolve().parent
GROUPS, EPS = 8, 1e-5


VARIANTS = {"trained": "as trained", "gelu": "no norm, GELU", "relu": "no norm, ReLU"}


def gn(name: str, variant: str):
    return keras.layers.GroupNormalization(GROUPS, epsilon=EPS, name=name) if variant == "trained" else (lambda x: x)


def act(variant: str):
    return keras.layers.ReLU() if variant == "relu" else keras.layers.Activation(lambda x: tf.nn.gelu(x, approximate=False))


def encoder(p: dict, variant: str) -> keras.Model:
    x = inp = keras.Input((80, 148, 3), batch_size=1, name="frame")  # fixed batch: no shape arithmetic
    for i in range(5):
        w = p[f"param/encoder.stem.{3 * i}.weight"]
        conv = keras.layers.Conv2D(w.shape[0], 3, strides=2, padding="valid", name=f"conv{i}")
        x = keras.layers.ZeroPadding2D(1)(x)  # PyTorch's symmetric padding=1, not Keras "same"
        x = conv(x)
        conv.set_weights([w.transpose(2, 3, 1, 0), p[f"param/encoder.stem.{3 * i}.bias"]])
        norm = gn(f"norm{i}", variant)
        x = norm(x)
        if variant == "trained":
            norm.set_weights([p[f"param/encoder.stem.{3 * i + 1}.weight"], p[f"param/encoder.stem.{3 * i + 1}.bias"]])
        x = act(variant)(x)
    x = keras.layers.Reshape((3 * 5 * 128,))(x)  # NHWC order: (h, w, c)
    w = p["param/encoder.project.weight"]  # [128, C*H*W] in PyTorch's (c, h, w) order
    c, h, wd = 128, 3, 5
    w = w.reshape(w.shape[0], c, h, wd).transpose(2, 3, 1, 0).reshape(h * wd * c, w.shape[0])
    dense = keras.layers.Dense(w.shape[1], name="project")
    x = dense(x)
    dense.set_weights([w, p["param/encoder.project.bias"]])
    # Scaling the 128 numbers to unit length stays on the CPU: trivial there, and the
    # NPU has no int8 kernel for it.
    return keras.Model(inp, x, name=f"encoder_{variant}")


def circular(x, pad: int):
    return keras.layers.Concatenate(axis=1)([x[:, -pad:], x, x[:, :pad]])


def head(p: dict, n_bins: int, clip_len: int, variant: str) -> keras.Model:
    x = inp = keras.Input((n_bins, clip_len), batch_size=1, name="correlation")
    w = p["param/head.entry.weight"]
    conv = keras.layers.Conv1D(w.shape[0], w.shape[2], padding="valid", name="entry")
    x = conv(circular(x, w.shape[2] // 2))
    conv.set_weights([w.transpose(2, 1, 0), p["param/head.entry.bias"]])
    for i in range(4):
        d = 2 ** i
        norm = gn(f"block{i}_norm", variant)
        y = norm(x)
        if variant == "trained":
            norm.set_weights([p[f"param/head.blocks.{i}.0.weight"], p[f"param/head.blocks.{i}.0.bias"]])
        y = act(variant)(y)
        w = p[f"param/head.blocks.{i}.2.weight"]
        conv = keras.layers.Conv1D(w.shape[0], w.shape[2], dilation_rate=d, padding="valid", name=f"block{i}_conv")
        y = conv(circular(y, d * (w.shape[2] // 2)))
        conv.set_weights([w.transpose(2, 1, 0), p[f"param/head.blocks.{i}.2.bias"]])
        x = keras.layers.Add()([x, y])
    norm = gn("exit_norm", variant)
    x = norm(x)
    if variant == "trained":
        norm.set_weights([p["param/head.exit.0.weight"], p["param/head.exit.0.bias"]])
    x = act(variant)(x)
    w = p["param/head.exit.2.weight"]
    conv = keras.layers.Conv1D(1, 1, name="exit")
    x = conv(x)
    conv.set_weights([w.transpose(2, 1, 0), p["param/head.exit.2.bias"]])
    x = keras.layers.Reshape((n_bins,))(x)
    return keras.Model(inp, x, name=f"head{n_bins}_{variant}")


def to_int8(model: keras.Model, calib: np.ndarray, path: Path) -> tuple[bytes, list[str]]:
    def representative():
        for sample in calib:
            yield [sample[None].astype(np.float32)]

    for strict in (True, False):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        if strict:  # everything in int8, as the NPU wants
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        try:
            blob = converter.convert()
            break
        except Exception as error:  # an op with no int8 kernel: keep it in float, and say so
            print(f"  {model.name}: strict int8 conversion failed ({str(error).splitlines()[0][:120]}); allowing float ops")
    path.write_bytes(blob)
    return blob, _ops(blob)


def _ops(blob: bytes) -> list[str]:
    interpreter = tf.lite.Interpreter(model_content=blob)
    return sorted({d["op_name"] for d in interpreter._get_ops_details()})


def run_tflite(blob: bytes, x: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_content=blob)
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


def soft_argmax(logits: np.ndarray, window: int = 8) -> np.ndarray:
    n = logits.shape[-1]
    peak = logits.argmax(-1)
    offsets = np.arange(-window, window + 1)
    idx = (peak[:, None] + offsets[None]) % n
    w = np.exp(np.take_along_axis(logits, idx, 1) - logits.max(-1, keepdims=True))
    return peak + (w * offsets).sum(1) / w.sum(1)


def vela(tflite: Path, out_dir: Path, mode: str) -> dict:
    cmd = [str(Path(sys.executable).with_name("vela")), str(tflite), "--accelerator-config", "ethos-u55-128",
           "--config", str(HERE / "vela.ini"), "--system-config", "Alif_B1", "--memory-mode", mode,
           "--output-dir", str(out_dir / mode)]
    text = subprocess.run(cmd, capture_output=True, text=True).stdout
    grab = lambda pattern: (m.group(1) if (m := re.search(pattern, text)) else None)
    summary = next((out_dir / mode).glob(f"{tflite.stem}_summary_*.csv"))
    row = next(csv.DictReader(summary.open()))
    return {
        "npu_ops": grab(r"NPU operators = (\d+)"), "cpu_ops": grab(r"CPU operators = (\d+)"),
        "sram_kib": float(row["sram_memory_used"]), "flash_kib": float(row["off_chip_flash_memory_used"]),
        "macs": int(float(row["nn_macs"])), "ms": 1000 * float(row["inference_time"]),
        "utilisation": float(row["nn_macs"]) / max(float(row["cycles_total"]), 1) / 128,
        "cpu_detail": sorted({m for m in re.findall(r"operator: (\w+), ofm", text)}),
        "raw": text,
    }


def main() -> None:
    export, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    p = dict(np.load(export))
    clip_len = int(p["clip_len"])
    frames = lambda k: p[k].astype(np.float32) / 255.0

    print("== encoder, as trained")
    enc = encoder(p, "trained")
    fp32 = enc.predict(frames("enc_check"), batch_size=1, verbose=0)
    fp32 /= np.linalg.norm(fp32, axis=1, keepdims=True)
    print(f"  fp32 Keras vs PyTorch: max |diff| {np.abs(fp32 - p['enc_check_emb']).max():.2e}")
    blob, ops = to_int8(enc, frames("enc_calib"), out_dir / "encoder_trained.tflite")
    q = run_tflite(blob, frames("enc_check"))
    cos = (q * p["enc_check_emb"]).sum(1) / np.linalg.norm(q, axis=1) / np.linalg.norm(p["enc_check_emb"], axis=1)
    print(f"  int8 vs PyTorch: cosine mean {cos.mean():.4f}, min {cos.min():.4f}; ops {ops}")
    results = {}
    for v, label in VARIANTS.items():
        if v != "trained":
            to_int8(encoder(p, v), frames("enc_calib"), out_dir / f"encoder_{v}.tflite")
        results[f"encoder, {label}"] = {m: vela(out_dir / f"encoder_{v}.tflite", out_dir, m) for m in ("Sram_Only", "Shared_Sram")}

    for n in sorted(int(k[4:-6]) for k in p if k.startswith("head") and k.endswith("_calib")):
        print(f"== head, {n} bins")
        calib = p[f"head{n}_calib"].transpose(0, 2, 1)
        check = p[f"head{n}_check"].transpose(0, 2, 1)
        hd = head(p, n, clip_len, "trained")
        fp32 = hd.predict(check, batch_size=1, verbose=0)
        print(f"  fp32 Keras vs PyTorch: max |diff| {np.abs(fp32 - p[f'head{n}_check_logits']).max():.2e}")
        blob, ops = to_int8(hd, calib, out_dir / f"head{n}_trained.tflite")
        q = run_tflite(blob, check)
        ref_pick, q_pick = soft_argmax(p[f"head{n}_check_logits"]), soft_argmax(q)
        gap = np.abs((q_pick - ref_pick + n / 2) % n - n / 2)
        print(f"  int8 vs PyTorch readout: median {np.median(gap):.2f} bins, p90 {np.percentile(gap, 90):.2f}, "
              f"same peak within 1 bin {100 * np.mean(gap <= 1):.0f}%; ops {ops}")
        for v, label in VARIANTS.items():
            if v != "trained":
                to_int8(head(p, n, clip_len, v), calib, out_dir / f"head{n}_{v}.tflite")
            results[f"head {n} bins, {label}"] = {m: vela(out_dir / f"head{n}_{v}.tflite", out_dir, m) for m in ("Sram_Only", "Shared_Sram")}

    print("\n== Vela on Halo's Ethos-U55-128 at 160 MHz: per inference, weights in SRAM | weights in MRAM")
    for name, modes in results.items():
        a, b = modes["Sram_Only"], modes["Shared_Sram"]
        rate = 30.0 if name.startswith("encoder") else 15.0
        print(f"  {name:30s} NPU ops {a['npu_ops']:>3}, CPU ops {a['cpu_ops']:>2} | {a['ms']:6.2f} ms | {b['ms']:6.2f} ms "
              f"(NPU busy at {rate:.0f}/s: {a['ms'] * rate / 10:4.1f}% | {b['ms'] * rate / 10:4.1f}%) | "
              f"SRAM {a['sram_kib']:.0f} KiB | {b['sram_kib']:.0f} KiB + MRAM {b['flash_kib']:.0f} KiB | "
              f"{a['macs'] / 1e6:.1f} M MACs, {100 * a['utilisation']:.0f}% of peak")
        if a["cpu_detail"]:
            print(f"      not on the NPU: {', '.join(a['cpu_detail'])}")
    (out_dir / "vela_raw.txt").write_text("\n\n".join(f"### {k} {m}\n{v['raw']}" for k, ms in results.items() for m, v in ms.items()))


if __name__ == "__main__":
    main()
