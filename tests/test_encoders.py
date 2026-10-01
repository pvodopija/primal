import pytest
import torch

from train.model import MOBILENET_V3_SMALL_IMAGENET, RESNET18_IMAGENET, MobileNetEncoder, ResNetEncoder, SequenceAligner


@pytest.mark.parametrize("cls, path", [(MobileNetEncoder, MOBILENET_V3_SMALL_IMAGENET), (ResNetEncoder, RESNET18_IMAGENET)])
def test_pretrained_backbones_load_exactly_and_keep_their_statistics(cls, path):
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded")
    encoder = cls(weights=path).train()
    state = torch.load(path, map_location="cpu", weights_only=True)
    shared = [k for k in encoder.state_dict() if k in state]
    assert len(shared) > 50 and all(torch.equal(encoder.state_dict()[k], state[k]) for k in shared)
    assert all(not m.training for m in encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    out = encoder(torch.rand(2, 3, 80, 148))
    assert out.shape == (2, 128) and torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_aligner_builds_every_encoder_without_weights():
    for encoder in ("small", "resnet18", "mobilenet"):
        model = SequenceAligner(clip_len=4, encoder=encoder)
        logits, _ = model(torch.rand(1, 4, 3, 80, 148), torch.rand(40, 3, 80, 148), use_checkpoint=False)
        assert logits.shape == (1, 40)
