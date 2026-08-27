import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import get_model


def test_get_model_unsupported_architecture():
    try:
        get_model("vgg16", num_classes=10)
        assert False, "expected ValueError for unsupported architecture"
    except ValueError:
        pass


def test_get_model_output_shape():
    model = get_model("resnet18", num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
