"""
Correlation-based sequence alignment.

    live clip  [B, K, 3, H, W] --\
                                  encoder (shared)  ->  L [B, K, D]
    reference  [N, 3, H, W] ----/                       R [N, D]

    C = L R^T                                          [B, K, N]
    dilated 1-D conv head over the reference axis       logits [B, N]

Track identity reaches the output only through the dot product, so the weights
cannot encode a circuit: swap the reference and the answer changes with it. The
head is convolutional along the reference axis with circular padding, which
makes it translation-equivariant on a loop and independent of lap length, so one
set of weights serves tracks of any size.

The output is a distribution over reference bins read out by soft-argmax, not a
scalar. Under perceptual aliasing the truthful answer is multi-modal; a scalar
head trained with L2 is obliged to emit the conditional mean, which on a track
where turn 2 resembles turn 6 is a piece of tarmac the car has never visited.
Sub-bin precision comes from the windowed expectation, as in stereo disparity
and integral keypoint regression.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _norm(channels: int, kind: str = "group") -> nn.Module:
    # GroupNorm, not BatchNorm: live and reference pass through the same encoder
    # with wildly different batch sizes, so batch statistics would not match.
    # "none" leaves it out: Halo's NPU cannot run GroupNorm (docs/ml-pivot.md,
    # *Measured for Halo's NPU with Vela*).
    return nn.GroupNorm(min(8, channels), channels) if kind == "group" else nn.Identity()


# Spatial grid the stem is pooled to before projection. Not 1x1: where a
# landmark sits in frame is real evidence for precise alignment, and pooling it
# away costs sub-bin precision. At the default 160x96 the stem already emits
# exactly this grid, so the pool is an identity there and only does work when
# the input resolution differs.
POOL_GRID = (3, 5)


def _pool_to_grid(features: Tensor, grid: tuple[int, int]) -> Tensor:
    """
    Resample a feature map to a fixed grid, on any device.

    `adaptive_avg_pool2d` would say this in one line, but on MPS it is only
    implemented when the input divides the output, and the stem emits 3x4 at
    128x80 against a 3x5 grid. So: take the exact integer average where one
    exists, then resample whatever fraction is left over. When the stem already
    emits the grid -- which it does at the default 160x96 -- both steps are
    skipped and this is an identity.
    """
    height, width = features.shape[-2:]
    rows, columns = grid
    if (height, width) == grid:
        return features
    if height % rows == 0 and width % columns == 0:
        return F.adaptive_avg_pool2d(features, grid)
    kernel = (max(1, height // rows), max(1, width // columns))
    if kernel != (1, 1):
        features = F.avg_pool2d(features, kernel_size=kernel, stride=kernel)
    if features.shape[-2:] == grid:
        return features
    return F.interpolate(features, size=grid, mode="bilinear", align_corners=False)


class FrameEncoder(nn.Module):
    """
    Small from-scratch CNN producing one L2-normalised descriptor per frame.

    Resolution-agnostic by construction: the stem is adaptively pooled to a
    fixed grid, so the weight shapes do not depend on the input size and one
    checkpoint runs on 128x80 synthetic frames and 160x96 AC frames alike.
    Flattening the stem directly would weld the training resolution into
    `project`, and a checkpoint would then refuse to load at any other size.

    Note that this makes the weights *loadable* across resolutions, not
    *invariant* to them. A different field of view still moves the scene across
    the frame; that is a packing-time concern, handled by cropping to a
    canonical FOV before resizing.
    """

    def __init__(self, dim: int = 128, width: int = 32, frame_size: tuple[int, int] | None = None,
                 norm: str = "group"):
        super().__init__()
        # frame_size is accepted and ignored: checkpoints record it, and it is
        # still the right thing to report, but it no longer shapes any weight.
        del frame_size
        channels = [3, width, width * 2, width * 3, width * 4, width * 4]
        layers: list[nn.Module] = []
        for inp, out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(inp, out, 3, stride=2, padding=1), _norm(out, norm), nn.GELU()]
        self.stem = nn.Sequential(*layers)

        self.project = nn.Linear(channels[-1] * POOL_GRID[0] * POOL_GRID[1], dim)
        self.dim = dim

    def forward(self, images: Tensor) -> Tensor:
        features = _pool_to_grid(self.stem(images), POOL_GRID).flatten(1)
        return F.normalize(self.project(features), dim=-1)


RESNET18_IMAGENET = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"


class _BasicBlock(nn.Module):
    """ResNet's basic block, with torchvision's attribute names so its weights load as-is."""

    def __init__(self, inp: int, out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = (
            nn.Sequential(nn.Conv2d(inp, out, 1, stride=stride, bias=False), nn.BatchNorm2d(out))
            if stride != 1 or inp != out else None
        )

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNetEncoder(nn.Module):
    """
    A pretrained "how to see" encoder: ImageNet ResNet-18 up to layer3 (stride 16), pooled
    to the same 3x5 grid and projected to the same unit-length descriptor as FrameEncoder.

    BatchNorm keeps its ImageNet statistics (eval mode even while training), so live
    clips and reference laps, which pass through in very different batches, are
    normalised identically; on an NPU it folds into the convolutions. Frames arrive as
    packed (BGR in [0, 1]) and are converted to ImageNet's RGB normalisation here.
    """

    def __init__(self, dim: int = 128, weights: Path | None = RESNET18_IMAGENET):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(_BasicBlock(64, 64, 1), _BasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, 2), _BasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(_BasicBlock(128, 256, 2), _BasicBlock(256, 256, 1))
        self.project = nn.Linear(256 * POOL_GRID[0] * POOL_GRID[1], dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.dim = dim
        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            own = self.state_dict()
            self.load_state_dict({k: v for k, v in state.items() if k in own and v.shape == own[k].shape}, strict=False)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith("project")]

    def train(self, mode: bool = True) -> "ResNetEncoder":
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        x = (images[:, [2, 1, 0]] - self.mean) / self.std
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, stride=2, padding=1)
        x = self.layer3(self.layer2(self.layer1(x)))
        features = _pool_to_grid(x, POOL_GRID).flatten(1)
        return F.normalize(self.project(features), dim=-1)


MOBILENET_V3_SMALL_IMAGENET = Path.home() / ".cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"


def _divisible(v: float, divisor: int = 8) -> int:
    new = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new + divisor if new < 0.9 * v else new


def _conv_bn(inp: int, out: int, kernel: int, stride: int = 1, groups: int = 1, act: type | None = nn.Hardswish):
    layers = [nn.Conv2d(inp, out, kernel, stride, (kernel - 1) // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out, eps=0.001, momentum=0.01)]
    return nn.Sequential(*layers, *([act(inplace=True)] if act else []))


class _SqueezeExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        squeeze = _divisible(channels // 4)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        return x * F.hardsigmoid(self.fc2(F.relu(self.fc1(s))))


class _InvertedResidual(nn.Module):
    """MobileNetV3's block, with torchvision's layout so its weights load as-is."""

    def __init__(self, inp: int, kernel: int, expanded: int, out: int, se: bool, act: type, stride: int):
        super().__init__()
        layers = [] if expanded == inp else [_conv_bn(inp, expanded, 1, act=act)]
        layers.append(_conv_bn(expanded, expanded, kernel, stride, groups=expanded, act=act))
        if se:
            layers.append(_SqueezeExcite(expanded))
        layers.append(_conv_bn(expanded, out, 1, act=None))
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and inp == out

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        return x + y if self.residual else y


class MobileNetEncoder(nn.Module):
    """
    A pretrained encoder small enough for Halo: ImageNet MobileNetV3-Small up to its last
    inverted-residual block (96 channels at stride 32, which at 148x80 is exactly the 3x5
    grid), projected to the same unit-length descriptor. BatchNorm keeps ImageNet's
    statistics, as in ResNetEncoder, and folds into the convolutions on an NPU.
    """

    SETTINGS = [  # input, kernel, expanded, output, squeeze-excite, activation, stride
        (16, 3, 16, 16, True, nn.ReLU, 2), (16, 3, 72, 24, False, nn.ReLU, 2), (24, 3, 88, 24, False, nn.ReLU, 1),
        (24, 5, 96, 40, True, nn.Hardswish, 2), (40, 5, 240, 40, True, nn.Hardswish, 1),
        (40, 5, 240, 40, True, nn.Hardswish, 1), (40, 5, 120, 48, True, nn.Hardswish, 1),
        (48, 5, 144, 48, True, nn.Hardswish, 1), (48, 5, 288, 96, True, nn.Hardswish, 2),
        (96, 5, 576, 96, True, nn.Hardswish, 1), (96, 5, 576, 96, True, nn.Hardswish, 1),
    ]

    def __init__(self, dim: int = 128, weights: Path | None = MOBILENET_V3_SMALL_IMAGENET):
        super().__init__()
        self.features = nn.Sequential(_conv_bn(3, 16, 3, 2), *(_InvertedResidual(*s) for s in self.SETTINGS))
        self.project = nn.Linear(96 * POOL_GRID[0] * POOL_GRID[1], dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.dim = dim
        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            own = self.state_dict()
            matched = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
            expected = [k for k in own if k.startswith("features.")]
            missing = [k for k in expected if k not in matched]
            if missing:
                raise ValueError(f"MobileNetV3 weights do not fit: {missing[:5]}")
            self.load_state_dict(matched, strict=False)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith("project")]

    def train(self, mode: bool = True) -> "MobileNetEncoder":
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        x = self.features((images[:, [2, 1, 0]] - self.mean) / self.std)
        features = _pool_to_grid(x, POOL_GRID).flatten(1)
        return F.normalize(self.project(features), dim=-1)


class AlignmentHead(nn.Module):
    """
    Reads the K x N correlation matrix as a 1-D signal along the reference axis
    with K channels, and emits one logit per reference bin.
    """

    def __init__(self, clip_len: int, hidden: int = 64, blocks: int = 4, kernel: int = 5, norm: str = "group"):
        super().__init__()
        self.entry = nn.Conv1d(
            clip_len, hidden, kernel, padding=kernel // 2, padding_mode="circular"
        )
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            dilation = 2**i
            self.blocks.append(
                nn.Sequential(
                    _norm(hidden, norm),
                    nn.GELU(),
                    nn.Conv1d(
                        hidden,
                        hidden,
                        kernel,
                        dilation=dilation,
                        padding=dilation * (kernel // 2),
                        padding_mode="circular",
                    ),
                )
            )
        self.exit = nn.Sequential(_norm(hidden, norm), nn.GELU(), nn.Conv1d(hidden, 1, 1))

    def forward(self, correlation: Tensor) -> Tensor:
        x = self.entry(correlation)
        for block in self.blocks:
            x = x + block(x)
        return self.exit(x).squeeze(1)


class SequenceAligner(nn.Module):
    def __init__(
        self,
        clip_len: int,
        dim: int = 128,
        width: int = 32,
        hidden: int = 64,
        blocks: int = 4,
        frame_size: tuple[int, int] = (96, 160),
        norm: str = "group",
        encoder: str = "small",
        encoder_weights: Path | None = None,
    ):
        super().__init__()
        if encoder == "resnet18":
            self.encoder = ResNetEncoder(dim=dim, weights=encoder_weights)
        elif encoder == "mobilenet":
            self.encoder = MobileNetEncoder(dim=dim, weights=encoder_weights)
        else:
            self.encoder = FrameEncoder(dim=dim, width=width, frame_size=frame_size, norm=norm)
        self.head = AlignmentHead(clip_len=clip_len, hidden=hidden, blocks=blocks, norm=norm)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_reference(
        self,
        reference: Tensor,
        chunk: int = 64,
        use_checkpoint: bool = True,
        grad_index: Tensor | None = None,
    ) -> Tensor:
        """
        Encode the reference lap in chunks.

        Gradients must flow through the reference as well as the live clip, but a
        1000-frame reference does not fit in memory as one autograd graph, so the
        chunks are recomputed in the backward pass.

        With `grad_index`, only those bins carry gradient: every bin is encoded
        once without a graph, and the chosen ones again with one. The reference is
        most of each step's frames, and this is where a step's time goes; the bins
        near the answers and a random sample elsewhere keep the useful gradient.
        """
        if grad_index is not None and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                frozen = torch.cat(
                    [self.encoder(reference[s : s + chunk]) for s in range(0, reference.shape[0], chunk)]
                )
            picked = reference[grad_index]
            fresh = torch.cat(
                [self.encoder(picked[s : s + chunk]) for s in range(0, picked.shape[0], chunk)]
            )
            return frozen.index_put((grad_index,), fresh)
        outputs = []
        for start in range(0, reference.shape[0], chunk):
            block = reference[start : start + chunk]
            if use_checkpoint and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self.encoder, block, use_reentrant=False))
            else:
                outputs.append(self.encoder(block))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        live: Tensor,
        reference: Tensor,
        use_checkpoint: bool = True,
        ref_grad_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, clip_len = live.shape[:2]
        live_descriptors = self.encoder(live.flatten(0, 1)).view(batch, clip_len, -1)
        reference_descriptors = self.encode_reference(
            reference, use_checkpoint=use_checkpoint, grad_index=ref_grad_index
        )
        correlation = torch.einsum("bkd,nd->bkn", live_descriptors, reference_descriptors)
        correlation = correlation * self.logit_scale.exp()
        return self.head(correlation), correlation


class FrameOnlyRegressor(nn.Module):
    """
    Control model: one frame in, a distribution over normalised `s` out, with no
    reference lap at all. The track can only live in its weights.

    On tracks it was trained on this is *expected* to work; that is exactly why
    an absolute regressor is the wrong factorisation, not evidence of a bug. The
    diagnostic value is on held-out tracks, where absolute position is not a
    learnable function of appearance. If it scores well there, something on
    screen is giving the answer away.
    """

    def __init__(self, bins: int = 256, width: int = 32, frame_size: tuple[int, int] = (96, 160)):
        super().__init__()
        self.encoder = FrameEncoder(dim=256, width=width, frame_size=frame_size)
        self.classifier = nn.Linear(256, bins)
        self.bins = bins

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.encoder(images))


def reference_grad_bins(
    frame_targets: Tensor, n_bins: int, window: int, random_fraction: float, generator: torch.Generator | None = None
) -> Tensor:
    """
    Reference bins that keep gradient in a partial-gradient step: every bin within
    `window` of any clip frame's target, where the loss is decided, plus a random
    `random_fraction` of the rest, so distant look-alikes still get pushed apart.
    """
    offsets = torch.arange(-window, window + 1)
    near = (frame_targets.detach().round().long().reshape(-1, 1).cpu() + offsets).remainder(n_bins)
    count = int(round(random_fraction * n_bins))
    sample = torch.randperm(n_bins, generator=generator)[:count]
    return torch.unique(torch.cat([near.reshape(-1), sample]))


def soft_argmax_circular(logits: Tensor, window: int = 8) -> Tensor:
    """
    Continuous bin index from a bin distribution.

    Expectation over a window around the dominant peak rather than the whole
    axis: a global expectation would be dragged toward secondary modes, which is
    the same averaging failure a scalar regressor has.
    """
    probabilities = logits.softmax(dim=-1)
    n_bins = probabilities.shape[-1]
    peak = probabilities.argmax(dim=-1)
    offsets = torch.arange(-window, window + 1, device=logits.device)
    indices = (peak[:, None] + offsets[None, :]) % n_bins
    weights = probabilities.gather(1, indices)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (peak.float() + (weights * offsets.float()).sum(dim=1)) % n_bins


def circular_offset(predicted: Tensor, target: Tensor, n_bins: int) -> Tensor:
    """Signed bin offset, taking the shorter way round the loop."""
    return (predicted - target + n_bins / 2.0) % n_bins - n_bins / 2.0


def circular_gaussian(target: Tensor, n_bins: int, sigma: float) -> Tensor:
    """Normalised Gaussian over bins around continuous targets, wrapped at the lap end."""
    bins = torch.arange(n_bins, device=target.device, dtype=target.dtype)
    offset = torch.remainder(bins - target[..., None] + n_bins / 2, n_bins) - n_bins / 2
    weight = torch.exp(-0.5 * (offset / sigma) ** 2)
    return weight / weight.sum(dim=-1, keepdim=True)


@dataclass
class LossParts:
    total: Tensor
    cross_entropy: Tensor
    auxiliary: Tensor
    refine: Tensor


def alignment_loss(
    logits: Tensor,
    correlation: Tensor,
    soft_target: Tensor,
    target: Tensor,
    aux_weight: float = 0.5,
    refine_weight: float = 0.5,
    window: int = 8,
    frame_target: Tensor | None = None,
    sigma_bins: float = 2.0,
) -> LossParts:
    """
    Soft cross-entropy on the head, plus two supporting terms.

    `auxiliary` supervises the raw correlation row of the frame being localised
    against the same soft target. That makes the encoder a usable place
    descriptor on its own, which is what gets early training off the ground
    before the head has learned anything.

    `refine` is an L1 on the soft-argmax, gated to samples whose peak is already
    within `window` bins. Ungated it would sharpen a confidently wrong peak.

    `frame_target` [B, K], when given, supervises every clip frame's row against
    its own position instead of only the last frame's. Every frame then has to
    localise on its own, which sharpens what the head combines.
    """
    n_bins = logits.shape[-1]
    cross_entropy = -(soft_target * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

    if frame_target is None:
        current_row = correlation[:, -1, :]
        auxiliary = -(soft_target * current_row.log_softmax(dim=-1)).sum(dim=-1).mean()
    else:
        rows = circular_gaussian(frame_target, n_bins, sigma_bins)
        auxiliary = -(rows * correlation.log_softmax(dim=-1)).sum(dim=-1).mean()

    predicted = soft_argmax_circular(logits, window=window)
    offset = circular_offset(predicted, target, n_bins).abs()
    near = (offset.detach() <= window).float()
    refine = (offset * near).sum() / near.sum().clamp_min(1.0) / n_bins

    total = cross_entropy + aux_weight * auxiliary + refine_weight * refine
    return LossParts(total=total, cross_entropy=cross_entropy, auxiliary=auxiliary, refine=refine)


@dataclass
class AlignmentMetrics:
    count: int
    median_m: float
    p90_m: float
    worst_m: float
    median_ms: float
    p90_ms: float
    worst_ms: float
    within_1_bin: float
    within_5_bin: float

    def format(self) -> str:
        return (
            f"n={self.count}  "
            f"median {self.median_m:6.2f} m / {self.median_ms:6.1f} ms  "
            f"p90 {self.p90_m:7.2f} m / {self.p90_ms:7.1f} ms  "
            f"worst {self.worst_m:8.2f} m  "
            f"within1 {self.within_1_bin * 100:5.1f}%  within5 {self.within_5_bin * 100:5.1f}%"
        )


def _interp_circular(values: Tensor, x: Tensor, period: float) -> Tensor:
    """A per-bin quantity that wraps at `period`, interpolated at continuous bins."""
    values = values.to(device=x.device, dtype=x.dtype)
    n = values.shape[0]
    base = torch.floor(x)
    k0 = base.long() % n
    k1 = (k0 + 1) % n
    step = torch.remainder(values[k1] - values[k0] + period / 2, period) - period / 2
    return torch.remainder(values[k0] + (x - base) * step, period)


def _circular_distance(a: Tensor, b: Tensor, period: float) -> Tensor:
    return (torch.remainder(a - b + period / 2, period) - period / 2).abs()


def compute_metrics(
    predicted: Tensor,
    target: Tensor,
    pos_m: Tensor,
    time_s: Tensor,
    track_length_m: float,
    lap_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-sample (metres, milliseconds) error, read off the reference grid.

    Milliseconds are the difference in reference lap time between the predicted
    and the true place, which is exactly the error the delta readout would show.
    Dividing metres by local speed only approximates that, and the approximation
    is worst in slow corners where it matters most.
    """
    length, lap = float(track_length_m), float(lap_time_s)
    metres = _circular_distance(
        _interp_circular(pos_m, predicted, length), _interp_circular(pos_m, target, length), length
    )
    seconds = _circular_distance(
        _interp_circular(time_s, predicted, lap), _interp_circular(time_s, target, lap), lap
    )
    return metres.detach().cpu().numpy(), (seconds * 1000.0).detach().cpu().numpy()


def summarise(
    metres_list: list[np.ndarray], milliseconds_list: list[np.ndarray], spacing_m: float
) -> AlignmentMetrics:
    metres = np.concatenate(metres_list) if len(metres_list) else np.zeros(0)
    milliseconds = np.concatenate(milliseconds_list) if len(milliseconds_list) else np.zeros(0)
    if metres.size == 0:
        return AlignmentMetrics(0, *([float("nan")] * 6), 0.0, 0.0)
    # Nominal bins, i.e. metres over the distance-axis spacing, so "within one
    # bin" means the same thing on either reference axis.
    bins = metres / max(spacing_m, 1e-9)
    return AlignmentMetrics(
        count=int(metres.size),
        median_m=float(np.median(metres)),
        p90_m=float(np.percentile(metres, 90)),
        worst_m=float(metres.max()),
        median_ms=float(np.median(milliseconds)),
        p90_ms=float(np.percentile(milliseconds, 90)),
        worst_ms=float(milliseconds.max()),
        within_1_bin=float((bins <= 1.0).mean()),
        within_5_bin=float((bins <= 5.0).mean()),
    )
