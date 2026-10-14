"""
Simulated low-precision arithmetic, for training a model that survives its target hardware
(quantization-aware training) and for measuring what that hardware costs it.

The iPhone's Neural Engine computes in float16; Halo's NPU (Arm Ethos class) in int8. A model
can be fine in float32 and lose a third of its golden frames' agreement in float16 when a few
blocks amplify rounding (2026-10-15: newtracks_pitch_s0's blocks 5-7 turn a 1e-3 nudge into a
100% change on its worst frame). `install(module, mode)` makes every Conv2d and Linear in
`module` compute with rounded weights and every leaf's output rounded:

  "fp16": weights and activations rounded to float16;
  "int8": weights to int8 per output channel (symmetric), activations to int8 per tensor
          (asymmetric) over a range that follows the 99.99th percentile of what the layer
          has seen (an exponential average, frozen with `freeze_ranges`), clipping outliers
          as a deployed int8 model does.

Rounding passes gradients straight through (x + (q(x) - x).detach()), so a model trains
under it. Weights are rounded by a parametrization: `float_state_dict` gives the state dict
under the usual names (the float weights), so checkpoints load as before.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize

MODES = ("fp16", "int8")
PERCENTILE = 0.9999
MOMENTUM = 0.01  # how fast an activation range follows new batches
SAMPLE = 65_536  # activations sampled per layer for its range
EVERY = 10  # a layer's range is refreshed every this many calls


def _ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return x + (q - x).detach()


def round_fp16(x: torch.Tensor) -> torch.Tensor:
    return _ste(x, x.half().float())


def round_int8_per_channel(w: torch.Tensor) -> torch.Tensor:
    """Symmetric int8, one scale per output channel (dim 0)."""
    scale = w.detach().abs().flatten(1).amax(1).clamp_min(1e-8) / 127.0
    shape = (-1,) + (1,) * (w.dim() - 1)
    return _ste(w, torch.round(w / scale.view(shape)).clamp(-127, 127) * scale.view(shape))


class _Weight(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return round_fp16(w) if self.mode == "fp16" else round_int8_per_channel(w)


class _ActivationRange:
    """An int8 activation range per layer, averaged over what it has seen: the 99.99th
    percentile ("percentile"), or the plain minimum and maximum ("minmax"), which is how
    TFLite calibrates. Kept on the layer's device and refreshed every EVERY calls, so
    training never waits on it."""

    def __init__(self, ranges: str = "percentile"):
        self.ranges = ranges
        self.lo: torch.Tensor | None = None
        self.hi: torch.Tensor | None = None
        self.frozen = False
        self.calls = 0

    def _update(self, out: torch.Tensor) -> None:
        flat = out.detach().flatten().float()
        if self.ranges == "minmax":
            hi, lo = flat.max(), flat.min()
        else:
            if flat.numel() > SAMPLE:  # a random sample gives the percentile as well, far faster
                flat = flat[torch.randint(0, flat.numel(), (SAMPLE,), device=flat.device)]
            k = max(int(round((1 - PERCENTILE) * flat.numel())), 1)
            hi, lo = flat.topk(k).values[-1], flat.topk(k, largest=False).values[-1]
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo = self.lo + MOMENTUM * EVERY * (lo - self.lo)
            self.hi = self.hi + MOMENTUM * EVERY * (hi - self.hi)

    def __call__(self, module: nn.Module, inputs, out: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(out) or not out.is_floating_point():
            return out
        if self.lo is None or (not self.frozen and self.calls % EVERY == 0):
            self._update(out)
        self.calls += 1
        lo, hi = self.lo.clamp(max=0.0), self.hi.clamp(min=0.0)
        scale = (hi - lo).clamp_min(1e-8) / 255.0
        zero = torch.round(-lo / scale)
        q = (torch.clamp(torch.round(out / scale) + zero, 0, 255) - zero) * scale
        return _ste(out, q)


class Handle:
    """What `install` did, to undo it or freeze its ranges."""

    def __init__(self):
        self.hooks: list = []
        self.ranges: list[_ActivationRange] = []
        self.weighted: list[nn.Module] = []

    def freeze_ranges(self, frozen: bool = True) -> None:
        for r in self.ranges:
            r.frozen = frozen

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        for m in self.weighted:
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)
        self.hooks, self.ranges, self.weighted = [], [], []


def install(module: nn.Module, mode: str, ranges: str = "percentile") -> Handle:
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}, not one of {MODES}")
    handle = Handle()
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and not parametrize.is_parametrized(m, "weight"):
            parametrize.register_parametrization(m, "weight", _Weight(mode))
            handle.weighted.append(m)
        if len(list(m.children())) == 0 and not isinstance(m, (nn.Dropout, nn.Identity)):
            if mode == "fp16":
                handle.hooks.append(m.register_forward_hook(lambda mod, i, o: round_fp16(o) if torch.is_tensor(o) and o.is_floating_point() else o))
            else:
                r = _ActivationRange(ranges)
                handle.ranges.append(r)
                handle.hooks.append(m.register_forward_hook(r))
    return handle


def float_state_dict(model: nn.Module) -> dict:
    """The state dict under the usual names, whether or not weights are parametrized."""
    return {k.replace(".parametrizations.weight.original", ".weight"): v for k, v in model.state_dict().items()}


BALANCE_FRAMES = 16


class RangeBalance:
    """Channel balancing for int8: int8 gives a whole tensor one scale, so a channel far
    louder than the rest leaves the quiet ones a few steps. For every BatchNorm and activation
    output it measures each channel's largest magnitude in the batch and, when the loudest
    channel exceeds the median one by more than `target` times, adds the log of the excess:
    trained with it, the network spreads its signal evenly (2026-10-15: the stem's channels
    spanned 28-34x). Only on tensors that train (gradients on);
    `clear` before a step's forward, `penalty` after it."""

    def __init__(self, module: nn.Module, target: float = 4.0):
        self.target = float(target)
        self.terms: list[torch.Tensor] = []
        self.seen: set[int] = set()  # layers measured this step: once each is plenty, and fast
        self.hooks = [m.register_forward_hook(self._hook) for m in module.modules()
                      if isinstance(m, (nn.BatchNorm2d, nn.Hardswish, nn.ReLU, nn.ReLU6))]

    def _hook(self, module: nn.Module, inputs, out: torch.Tensor) -> None:
        # gradients on, not module.training: the encoder's BatchNorms stay in eval mode (ImageNet's statistics)
        if (id(module) in self.seen or not (torch.is_grad_enabled() and torch.is_tensor(out)
                                            and out.requires_grad and out.dim() == 4)):
            return
        self.seen.add(id(module))
        # each channel's loudest value over a few frames: a copy (so in-place activations after
        # this layer cannot touch what the gradient needs) small enough that a step's hundreds of
        # frames do not multiply the activation memory
        r = out[:BALANCE_FRAMES].clone().abs().amax(dim=(0, 2, 3)) + 1e-6
        excess = torch.log(r.max()) - torch.log(r.median()) - torch.log(torch.tensor(self.target, device=r.device))
        self.terms.append(torch.relu(excess))

    def clear(self) -> None:
        self.terms, self.seen = [], set()

    def penalty(self) -> torch.Tensor | float:
        return torch.stack(self.terms).mean() if self.terms else 0.0

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
