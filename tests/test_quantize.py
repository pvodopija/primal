"""Simulated float16 / int8 arithmetic (train/quantize.py) for quantization-aware training."""
import torch
from torch import nn

from train import quantize


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Hardswish(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))


def test_install_and_remove_leave_the_model_as_it_was():
    net, x = _net(), torch.randn(2, 3, 16, 16)
    before = net(x)
    for mode in quantize.MODES:
        handle = quantize.install(net, mode)
        assert not torch.equal(net(x), before) or mode == "fp16"
        handle.remove()
        assert torch.equal(net(x), before)


def test_the_float_state_dict_loads_under_the_usual_names():
    net = _net()
    quantize.install(net, "int8")
    fresh = _net()
    fresh.load_state_dict(quantize.float_state_dict(net))
    assert set(fresh.state_dict()) == set(quantize.float_state_dict(net))


def test_gradients_pass_through_the_rounding():
    net = _net()
    quantize.install(net, "int8")
    net(torch.randn(4, 3, 16, 16)).pow(2).sum().backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_int8_weights_sit_on_a_per_channel_grid():
    w = torch.randn(5, 3, 3, 3)
    q = quantize.round_int8_per_channel(w)
    scale = w.abs().flatten(1).amax(1) / 127
    steps = q / scale.view(-1, 1, 1, 1)
    assert torch.allclose(steps, steps.round(), atol=1e-4)
    assert (q - w).abs().max() <= scale.max() / 2 + 1e-6


def test_range_balance_penalises_a_loud_channel_and_its_gradient_evens_it_out():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    with torch.no_grad():
        net[0].weight[0] *= 50  # one channel far louder than the rest
    net.train()
    net[1].eval()  # frozen statistics, as the encoder's BatchNorms
    balance = quantize.RangeBalance(net, target=4.0)
    x = torch.randn(16, 3, 16, 16)
    optimiser = torch.optim.SGD(net.parameters(), lr=0.05)
    net.eval()
    with torch.no_grad():
        before = net[1](net[0](x)).abs().amax(dim=(0, 2, 3))
    net.train()
    net[1].eval()
    first = None
    for _ in range(60):
        balance.clear()
        net(x)
        p = balance.penalty()
        first = float(p) if first is None else first
        optimiser.zero_grad()
        p.backward()
        optimiser.step()
    net.eval()
    with torch.no_grad():
        after = net[1](net[0](x)).abs().amax(dim=(0, 2, 3))
    assert first > 0
    assert float(after.max() / after.median()) < float(before.max() / before.median())
