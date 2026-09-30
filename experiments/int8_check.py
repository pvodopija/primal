"""How much does 8-bit arithmetic cost? Simulated on the Mac for a trained checkpoint:
the 128-number embedding sent over Bluetooth, the encoder's weights, and its activations."""
import copy, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from train.dataset import AlignmentBatches, LapIndex, SampleConfig
from train.eval import load_model, measure, reference_axis_of
from train.train import holdout_live_laps

ckpt, data = Path(sys.argv[1]), Path(sys.argv[2])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
base, payload = load_model(ckpt, device)
axis, clip_len = reference_axis_of(payload), payload["args"]["clip_len"]
train_index, holdout_index = LapIndex.load(data, split="train"), LapIndex.load(data, split="holdout")
reserved = holdout_live_laps(train_index, 1)
trainable = frozenset(l.lap_id for l in train_index.laps) - reserved

def cases(steps=40):
    make = lambda idx, ro, lo, seed: AlignmentBatches(idx, SampleConfig(batch_size=8, clip_len=clip_len, roll_reference=True,
        jitter=False, reference_axis=axis, reference_only=ro, live_only=lo), steps=steps, seed=seed)
    return {"G1 held-out laps": make(train_index, trainable, reserved, 22), "G2 Silverstone": make(holdout_index, None, None, 33)}

def quantise_symmetric(x, dim=None):
    scale = (x.abs().amax(dim=dim, keepdim=True) if dim is not None else x.abs().max()).clamp_min(1e-12) / 127.0
    return torch.round(x / scale).clamp(-127, 127) * scale

class Int8Embedding(nn.Module):
    def __init__(self, encoder): super().__init__(); self.encoder = encoder
    def forward(self, x):
        d = self.encoder(x)
        d = quantise_symmetric(d, dim=-1)
        return torch.nn.functional.normalize(d, dim=-1)

def weights_int8(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for mod in m.encoder.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.copy_(quantise_symmetric(mod.weight, dim=tuple(range(1, mod.weight.dim()))))
    return m

def activations_int8(model, calibration):
    m = copy.deepcopy(model)
    ranges, hooks = {}, []
    layers = [mod for mod in m.encoder.modules() if isinstance(mod, (nn.Conv2d, nn.GroupNorm, nn.GELU, nn.Linear))]
    def observe(mod, inp, out):
        lo, hi = ranges.get(mod, (np.inf, -np.inf))
        ranges[mod] = (min(lo, out.min().item()), max(hi, out.max().item()))
    hooks = [l.register_forward_hook(observe) for l in layers]
    with torch.no_grad():
        for i, batch in enumerate(torch.utils.data.DataLoader(calibration, batch_size=None)):
            m(batch["live"].to(device), batch["reference"].to(device), use_checkpoint=False)
            if i >= 5: break
    for h in hooks: h.remove()
    def fake(mod, inp, out):
        lo, hi = ranges[mod]
        scale = max(hi - lo, 1e-8) / 255.0
        zero = round(-lo / scale)
        return (torch.clamp(torch.round(out / scale) + zero, 0, 255) - zero) * scale
    for l in layers: l.register_forward_hook(fake)
    return m

variants = {"float (as trained)": base}
emb = copy.deepcopy(base); emb.encoder = Int8Embedding(emb.encoder); variants["embedding 8-bit"] = emb
variants["+ weights 8-bit"] = weights_int8(emb)
calib = AlignmentBatches(train_index, SampleConfig(batch_size=8, clip_len=clip_len, jitter=False, reference_axis=axis,
                          reference_only=trainable, live_only=trainable), steps=6, seed=99)
full = weights_int8(base); full = activations_int8(full, calib); full.encoder = Int8Embedding(full.encoder)
variants["+ activations 8-bit (all)"] = full

print(f"{ckpt}: medians over 40 batches per case (same clips for every variant)")
for name, model in variants.items():
    model.eval()
    line = f"  {name:28s}"
    for label, ds in cases().items():
        overall, _, _ = measure(model, ds, device, 8)
        line += f"  {label}: {overall.median_m:5.2f} m (p90 {overall.p90_m:6.2f}, within5 {100*overall.within_5_bin:3.0f}%)"
    print(line, flush=True)
