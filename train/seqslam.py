"""
Classical SeqSLAM as a baseline matcher.

Milford & Wyeth, "SeqSLAM: Visual route-based navigation for sunny summer days
and stormy winter nights", ICRA 2012.

Deliberately the same interface as `train.model.SequenceAligner`, so `train.eval`
measures it through an identical path: the same lap pairs, the same rolled
reference, the same soft-argmax readout, the same metrics. Only the production
of the K x N score matrix differs -- learned descriptors and a dilated conv head
against patch-normalised pixel difference and a straight-line search.

Giving it the soft-argmax readout is deliberate. SeqSLAM natively matches to an
integer reference index, so without it the comparison would measure readout
resolution rather than descriptor quality, which is not the question.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train.model import _pool_to_grid


def patch_normalise(images: Tensor, down: tuple[int, int], patch: int) -> Tensor:
    """
    Greyscale, downsample, then zero-mean unit-variance within each patch.

    Patch normalisation rather than a global one is what buys SeqSLAM its
    illumination invariance, and it is the whole of its appearance model.
    """
    if down[0] % patch or down[1] % patch:
        raise ValueError(f"down {down} must divide by patch {patch}")
    grey = images.mean(dim=1, keepdim=True)
    # Not interpolate(mode="area"): that is adaptive_avg_pool2d, which MPS only
    # implements when the input divides the output. 80x128 down to 48x64 does not.
    small = _pool_to_grid(grey, down)
    mean = F.avg_pool2d(small, patch, stride=patch)
    mean_sq = F.avg_pool2d(small * small, patch, stride=patch)
    std = (mean_sq - mean * mean).clamp_min(1e-8).sqrt()
    up = dict(size=small.shape[-2:], mode="nearest")
    return ((small - F.interpolate(mean, **up)) / F.interpolate(std, **up)).flatten(1)


def local_normalise(scores: Tensor, window: int) -> Tensor:
    """
    SeqSLAM's contrast enhancement, over a circular reference axis.

    `window <= 1` standardises globally, which is the default because a true
    match sits in a broad low basin: a window narrow enough to be local
    subtracts the basin along with the peak and moves the argmin off the answer.
    """
    if window <= 1:
        centred = scores - scores.mean(dim=-1, keepdim=True)
        return centred / scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
    window += 1 - window % 2
    padded = F.pad(scores.unsqueeze(1), (window // 2, window // 2), mode="circular")
    mean = F.avg_pool1d(padded, window, stride=1)
    mean_sq = F.avg_pool1d(padded * padded, window, stride=1)
    std = (mean_sq - mean * mean).clamp_min(1e-8).sqrt()
    return ((scores.unsqueeze(1) - mean) / std).squeeze(1)


class SeqSLAM(nn.Module):
    def __init__(
        self,
        clip_len: int,
        down: tuple[int, int] = (24, 32),
        patch: int = 8,
        max_velocity: float = 4.0,
        n_velocities: int = 41,
        temperature: float = 6.0,
        norm_window: int = 0,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.down = down
        self.patch = patch
        self.temperature = temperature
        self.norm_window = norm_window
        self.register_buffer(
            "velocities", torch.linspace(0.0, max_velocity, n_velocities), persistent=False
        )

    def forward(
        self, live: Tensor, reference: Tensor, use_checkpoint: bool = False
    ) -> tuple[Tensor, Tensor]:
        batch, clip_len = live.shape[:2]
        n_bins = reference.shape[0]
        device = live.device

        live_vec = patch_normalise(live.flatten(0, 1), self.down, self.patch)
        ref_vec = patch_normalise(reference, self.down, self.patch)
        difference = (
            torch.cdist(live_vec, ref_vec, p=1).view(batch, clip_len, n_bins) / live_vec.shape[1]
        )

        # The final live frame is the one being localised, so every candidate
        # trajectory ends at bin n and runs backwards at a constant velocity.
        back = (clip_len - 1 - torch.arange(clip_len, device=device)).float()
        bins = torch.arange(n_bins, device=device).float()
        best = None
        for velocity in self.velocities.tolist():
            index = ((bins[None, :] - back[:, None] * velocity).round().long() % n_bins).clamp(
                0, n_bins - 1
            )
            walked = difference.gather(2, index.unsqueeze(0).expand(batch, clip_len, n_bins))
            score = walked.mean(dim=1)
            best = score if best is None else torch.minimum(best, score)

        logits = -local_normalise(best, self.norm_window) * self.temperature
        return logits, -difference
