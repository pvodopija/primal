import torch

from train.model import SequenceAligner, reference_grad_bins


def _model() -> SequenceAligner:
    torch.manual_seed(0)
    return SequenceAligner(clip_len=4, dim=16, width=8, hidden=8).train()


def test_partial_gradients_leave_descriptors_unchanged():
    model = _model()
    reference = torch.rand(40, 3, 40, 64)
    full = model.encode_reference(reference, use_checkpoint=False)
    picked = torch.tensor([0, 5, 6, 39])
    partial = model.encode_reference(reference, use_checkpoint=False, grad_index=picked)
    assert torch.allclose(full, partial, atol=1e-5)


def test_only_the_chosen_bins_carry_gradient():
    model = _model()
    reference = torch.rand(40, 3, 40, 64)
    picked = torch.tensor([3, 4, 5])
    descriptors = model.encode_reference(reference, use_checkpoint=False, grad_index=picked)

    others = torch.ones(40, dtype=torch.bool)
    others[picked] = False
    descriptors[others].sum().backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in model.encoder.parameters())

    model.zero_grad()
    descriptors = model.encode_reference(reference, use_checkpoint=False, grad_index=picked)
    descriptors[picked].sum().backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.encoder.parameters())


def test_grad_bins_cover_every_target_and_wrap_at_the_line():
    targets = torch.tensor([[0.2, 99.6], [50.0, 50.4]])
    bins = reference_grad_bins(targets, n_bins=100, window=2, random_fraction=0.0)
    assert set(bins.tolist()) == {98, 99, 0, 1, 2, 48, 49, 50, 51, 52}
    with_random = reference_grad_bins(targets, 100, 2, 0.2, torch.Generator().manual_seed(0))
    assert set(bins.tolist()) <= set(with_random.tolist())
    assert len(with_random) <= 10 + 20
