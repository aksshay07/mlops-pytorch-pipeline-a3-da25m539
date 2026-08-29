import torch
from torch import nn
from torchvision.models import resnet18


def get_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


if __name__ == "__main__":
    model = get_model("resnet18", 10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print("output shape:", out.shape)
