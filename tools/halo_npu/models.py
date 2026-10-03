"""
Int8 TFLite stand-ins for PRIMAL's networks, sized for the Halo NPU.

Vela's cycle and memory estimate depends on topology and tensor shapes, not on
trained values, so the weights here are random. Mirrors `train/model.py`
with two deployability changes: GroupNorm is dropped (no Ethos-U lowering; a
deployed trunk would fold BatchNorm into the conv), and Conv1d is written as a
height-1 Conv2D with 'same' padding in place of circular padding.
"""

import os
import sys

import numpy as np
import tensorflow as tf

L = tf.keras.layers


def frame_encoder(h, w, c=3, width=32, dim=128, gelu=False, reduce=None):
    """FrameEncoder: 5 x (3x3 stride-2 conv + act) -> flatten -> linear.

    `reduce` inserts a 1x1 conv down to that many channels before the flatten,
    shrinking the projection's weights while keeping the spatial layout.
    """
    x = inp = tf.keras.Input((h, w, c), batch_size=1)
    for out in [width, width * 2, width * 3, width * 4, width * 4]:
        x = L.Conv2D(out, 3, strides=2, padding="same")(x)
        x = L.Activation(tf.nn.gelu)(x) if gelu else L.ReLU()(x)
    if reduce:
        x = L.ReLU()(L.Conv2D(reduce, 1)(x))
    x = L.Dense(dim)(L.Flatten()(x))
    return tf.keras.Model(inp, x)


def bayer_encoder(h, w, pool):
    """FrameEncoder fed the raw BGGR8 CPI buffer.

    A 2x2 stride-2 conv on the mosaic is a 1x1 conv on the (B, G, G, R)
    space-to-depth tensor, so the network learns its own demosaic and the CPU
    does no debayer.
    """
    x = inp = tf.keras.Input((h, w, 1), batch_size=1)
    x = L.ReLU()(L.Conv2D(8, 2, strides=2)(x))
    if pool:
        x = L.AveragePooling2D(2)(x)
    return tf.keras.Model(inp, frame_encoder(x.shape[1], x.shape[2], 8)(x))


def alignment_head(n_ref, clip_len=8, hidden=64, blocks=4, kernel=5):
    """AlignmentHead on the 1 x n_ref x clip_len correlation signal."""
    x = inp = tf.keras.Input((1, n_ref, clip_len), batch_size=1)
    x = L.Conv2D(hidden, (1, kernel), padding="same")(x)
    for i in range(blocks):
        y = L.Conv2D(hidden, (1, kernel), dilation_rate=(1, 2**i), padding="same")(L.ReLU()(x))
        x = L.Add()([x, y])
    return tf.keras.Model(inp, L.Conv2D(1, 1)(L.ReLU()(x)))


def mobilenet_v2(h, w, alpha):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(h, w, 3), alpha=alpha, include_top=False, weights=None, pooling="avg"
    )
    x = inp = tf.keras.Input((h, w, 3), batch_size=1)
    return tf.keras.Model(inp, L.Dense(128)(base(x)))


MODELS = {
    "trunk_96x160_rgb": lambda: frame_encoder(96, 160),
    "trunk_96x160_rgb_gelu": lambda: frame_encoder(96, 160, gelu=True),
    "trunk_96x160_rgb_reduce32": lambda: frame_encoder(96, 160, reduce=32),
    "trunk_96x160_rgb_w24_reduce32": lambda: frame_encoder(96, 160, width=24, reduce=32),
    "trunk_120x160_rgb": lambda: frame_encoder(120, 160),
    "trunk_120x160_gray": lambda: frame_encoder(120, 160, c=1),
    "trunk_120x160_rgb_w48": lambda: frame_encoder(120, 160, width=48),
    "trunk_240x320_rgb": lambda: frame_encoder(240, 320),
    "bayer_qvga_240x320": lambda: bayer_encoder(240, 320, pool=False),
    "bayer_vga_480x640": lambda: bayer_encoder(480, 640, pool=True),
    "head_n1800_k8": lambda: alignment_head(1800),
    "head_n256_k8": lambda: alignment_head(256),
    # Reference points: MobileNetV2-1.0/224 has published Alif timings.
    "mobilenetv2_035_96x160": lambda: mobilenet_v2(96, 160, 0.35),
    "mobilenetv2_100_224x224": lambda: mobilenet_v2(224, 224, 1.0),
}


def to_int8(model, path):
    shape = [1, *model.input_shape[1:]]

    def representative():
        rng = np.random.default_rng(0)
        for _ in range(16):
            yield [rng.uniform(-1, 1, shape).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with open(path, "wb") as f:
        f.write(converter.convert())


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(out_dir, exist_ok=True)
    for name, build in MODELS.items():
        to_int8(build(), os.path.join(out_dir, name + ".tflite"))
        print(name)
