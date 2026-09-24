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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _norm(channels: int) -> nn.Module:
    # GroupNorm, not BatchNorm: live and reference pass through the same encoder
    # with wildly different batch sizes, so batch statistics would not match.
    return nn.GroupNorm(min(8, channels), channels)


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

    def __init__(self, dim: int = 128, width: int = 32, frame_size: tuple[int, int] | None = None):
        super().__init__()
        # frame_size is accepted and ignored: checkpoints record it, and it is
        # still the right thing to report, but it no longer shapes any weight.
        del frame_size
        channels = [3, width, width * 2, width * 3, width * 4, width * 4]
        layers: list[nn.Module] = []
        for inp, out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(inp, out, 3, stride=2, padding=1), _norm(out), nn.GELU()]
        self.stem = nn.Sequential(*layers)

        self.project = nn.Linear(channels[-1] * POOL_GRID[0] * POOL_GRID[1], dim)
        self.dim = dim

    def forward(self, images: Tensor) -> Tensor:
        features = _pool_to_grid(self.stem(images), POOL_GRID).flatten(1)
        return F.normalize(self.project(features), dim=-1)


class AlignmentHead(nn.Module):
    """
    Reads the K x N correlation matrix as a 1-D signal along the reference axis
    with K channels, and emits one logit per reference bin.
    """

    def __init__(self, clip_len: int, hidden: int = 64, blocks: int = 4, kernel: int = 5):
        super().__init__()
        self.entry = nn.Conv1d(
            clip_len, hidden, kernel, padding=kernel // 2, padding_mode="circular"
        )
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            dilation = 2**i
            self.blocks.append(
                nn.Sequential(
                    _norm(hidden),
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
        self.exit = nn.Sequential(_norm(hidden), nn.GELU(), nn.Conv1d(hidden, 1, 1))

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
    ):
        super().__init__()
        self.encoder = FrameEncoder(dim=dim, width=width, frame_size=frame_size)
        self.head = AlignmentHead(clip_len=clip_len, hidden=hidden, blocks=blocks)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_reference(self, reference: Tensor, chunk: int = 64, use_checkpoint: bool = True) -> Tensor:
        """
        Encode the reference lap in chunks.

        Gradients must flow through the reference as well as the live clip, but a
        1000-frame reference does not fit in memory as one autograd graph, so the
        chunks are recomputed in the backward pass.
        """
        outputs = []
        for start in range(0, reference.shape[0], chunk):
            block = reference[start : start + chunk]
            if use_checkpoint and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self.encoder, block, use_reentrant=False))
            else:
                outputs.append(self.encoder(block))
        return torch.cat(outputs, dim=0)

    def forward(
        self, live: Tensor, reference: Tensor, use_checkpoint: bool = True
    ) -> tuple[Tensor, Tensor]:
        batch, clip_len = live.shape[:2]
        live_descriptors = self.encoder(live.flatten(0, 1)).view(batch, clip_len, -1)
        reference_descriptors = self.encode_reference(reference, use_checkpoint=use_checkpoint)
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
