from fastapi import FastAPI, UploadFile, File
import shutil
import torch
import torch.nn.functional as F
import cv2
import os

app = FastAPI()

IMAGE_SIZE = 32
THRESHOLD = 0.5


# ================= MODEL =================

class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1,4,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(4,8,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(8,16,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16*4*4,32)
        )

    def forward(self,x):
        return self.fc(self.conv(x))


class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingNet()

    def forward(self,x1,x2):
        return self.embedding(x1), self.embedding(x2)


# ================= LOAD MODEL =================

model = SiameseNetwork()
model.load_state_dict(
    torch.load("../models/siamese_fingerprint_zkml.pth", map_location="cpu")
)
model.eval()


# ================= IMAGE =================

def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    return torch.tensor(img).reshape(1,1,IMAGE_SIZE,IMAGE_SIZE)


# ================= API =================

@app.post("/verify")
async def verify(
    enrolled: UploadFile = File(...),
    test: UploadFile = File(...)
):
    os.makedirs("temp", exist_ok=True)

    enrolled_path = f"temp/{enrolled.filename}"
    test_path = f"temp/{test.filename}"

    with open(enrolled_path, "wb") as f:
        shutil.copyfileobj(enrolled.file, f)

    with open(test_path, "wb") as f:
        shutil.copyfileobj(test.file, f)

    img1 = process_image(enrolled_path)
    img2 = process_image(test_path)

    with torch.no_grad():
        e1, e2 = model(img1, img2)
        distance = F.pairwise_distance(e1, e2).item()

    result = "Genuine" if distance < THRESHOLD else "Altered"

    return {
        "distance": distance,
        "result": result
    }
