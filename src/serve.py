import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from dataset import get_transforms
from model import get_model

CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"))
ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "resnet18")
NUM_CLASSES = int(os.environ.get("MODEL_NUM_CLASSES", "10"))
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: torch.nn.Module | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    net = get_model(architecture=ARCHITECTURE, num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()
    model = net
    yield

app = FastAPI(title="CIFAR-10 Classifier app", lifespan=lifespan)

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):  # noqa: B008
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    return {
        "predictions": {
            CIFAR10_CLASSES[i]: round(p, 4) for i, p in enumerate(probabilities)
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
