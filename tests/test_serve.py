import io
import os
import sys
from pathlib import Path

import torch
from PIL import Image

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC_DIR)


def _make_fake_checkpoint(path: Path) -> None:
    from model import get_model

    model = get_model("resnet18", num_classes=10)
    torch.save({"model_state_dict": model.state_dict()}, path)


def test_health_and_predict(tmp_path):
    checkpoint_path = tmp_path / "fake_checkpoint.pt"
    _make_fake_checkpoint(checkpoint_path)
    os.environ["CHECKPOINT_PATH"] = str(checkpoint_path)

    sys.modules.pop("serve", None)
    import serve
    from fastapi.testclient import TestClient

    with TestClient(serve.app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        image = Image.new("RGB", (32, 32), color=(128, 64, 32))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        predict_response = client.post(
            "/predict",
            files={"image": ("test.png", buffer, "image/png")},
        )
        assert predict_response.status_code == 200
        predictions = predict_response.json()["predictions"]
        assert len(predictions) == 10
        assert abs(sum(predictions.values()) - 1.0) < 0.01
