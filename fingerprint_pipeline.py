
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import onnx
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_PATH = r"C:\Users\priya\OneDrive\Desktop\fzkml\dataset\SOCOFing"  # Change this path
os.makedirs("models", exist_ok=True)

MODEL_SAVE_PATH = "models/siamese_fingerprint_zkml.pth"
ONNX_SAVE_PATH = "models/siamese_fingerprint_zkml.onnx"
INPUT_JSON_PATH = "models/input.json"

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
IMAGE_SIZE = 96
MARGIN = 1.0
THRESHOLD = 0.5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)


def build_subject_split(root):
    real = os.path.join(root, "Real")
    subjects = defaultdict(list)

    for f in os.listdir(real):
        if f.endswith(".BMP"):
            sid = f.split("__")[0]
            subjects[sid].append(os.path.join(real, f))

    ids = list(subjects.keys())
    random.shuffle(ids)

    split = int(0.8 * len(ids))
    train_ids = set(ids[:split])
    val_ids = set(ids[split:])

    train_imgs = []
    val_imgs = []

    for sid in train_ids:
        train_imgs.extend([(p, sid) for p in subjects[sid]])

    for sid in val_ids:
        val_imgs.extend([(p, sid) for p in subjects[sid]])

    return train_imgs, val_imgs


class SiameseDataset(Dataset):
    def __init__(self, images, fixed_pairs=False):
        self.images = images
        self.fixed_pairs = fixed_pairs
        self.by_label = defaultdict(list)

        for p, l in images:
            self.by_label[l].append(p)

        self.labels = list(self.by_label.keys())

        if fixed_pairs:
            self.pairs = []
            rng = random.Random(SEED)
            for p, l in images:
                if rng.random() > 0.5 and len(self.by_label[l]) > 1:
                    cands = [x for x in self.by_label[l] if x != p]
                    p2 = rng.choice(cands)
                    lab = 1
                else:
                    other = rng.choice([x for x in self.labels if x != l])
                    p2 = rng.choice(self.by_label[other])
                    lab = 0
                self.pairs.append((p, p2, lab))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.fixed_pairs:
            p1, p2, lab = self.pairs[idx]
        else:
            p1, l1 = self.images[idx]
            if random.random() > 0.5 and len(self.by_label[l1]) > 1:
                cands = [x for x in self.by_label[l1] if x != p1]
                p2 = random.choice(cands)
                lab = 1
            else:
                other = random.choice([x for x in self.labels if x != l1])
                p2 = random.choice(self.by_label[other])
                lab = 0

        return load_image(p1), load_image(p2), torch.tensor(lab, dtype=torch.float32)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*12*12,64)
        )
    def forward(self,x):
        return self.fc(self.conv(x))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingNet()
    def forward(self,x1,x2):
        return self.embedding(x1), self.embedding(x2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin
    def forward(self,e1,e2,label):
        d = F.pairwise_distance(e1,e2)
        return (label*d*d + (1-label)*torch.clamp(self.margin-d,min=0)**2).mean()


def main():
    train_imgs, val_imgs = build_subject_split(DATASET_PATH)

    train_loader = DataLoader(SiameseDataset(train_imgs), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SiameseDataset(val_imgs, fixed_pairs=True), batch_size=BATCH_SIZE)

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running = 0
        for i1,i2,l in train_loader:
            i1,i2,l = i1.to(device),i2.to(device),l.to(device)
            optimizer.zero_grad()
            e1,e2 = model(i1,i2)
            loss = criterion(e1,e2,l)
            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss={running:.4f}")

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for i1,i2,l in val_loader:
                e1,e2 = model(i1.to(device), i2.to(device))
                d = F.pairwise_distance(e1,e2)
                pred = (d < THRESHOLD).float().cpu().numpy()
                yp.extend(pred)
                yt.extend(l.numpy())

        acc = accuracy_score(yt, yp)
        print("Validation Accuracy:", round(acc,4))

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved:", MODEL_SAVE_PATH)

    cm = confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm.ravel()

    print("\nAccuracy :", accuracy_score(yt, yp))
    print("Precision:", precision_score(yt, yp))
    print("Recall   :", recall_score(yt, yp))
    print("F1 Score :", f1_score(yt, yp))
    print("\nConfusion Matrix\n", cm)
    print("FAR:", fp/(fp+tn))
    print("FRR:", fn/(fn+tp))

    # --------------------------------------
    # Generate Calibration JSON
# --------------------------------------
    sample_img1, sample_img2, _ = val_loader.dataset[0]

    sample_img1 = sample_img1.unsqueeze(0).to(device)
    sample_img2 = sample_img2.unsqueeze(0).to(device)

    with torch.no_grad():
        emb1, emb2 = model(sample_img1, sample_img2)

    data_json = {
    "input_data": [
        sample_img1.cpu().numpy().flatten().tolist(),
        sample_img2.cpu().numpy().flatten().tolist()
    ]
}

    with open(INPUT_JSON_PATH, "w") as f:
        json.dump(data_json, f, indent=2)

    print("Calibration data saved to", INPUT_JSON_PATH)


    # --------------------------------------
    # EXPORT TO ONNX (EZKL Compatible)
    # --------------------------------------
    model.eval()
    
    dummy_img1 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32).to(device)
    dummy_img2 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32).to(device)
    
    torch.onnx.export(
        model,
        (dummy_img1, dummy_img2),
        ONNX_SAVE_PATH,
        export_params=True,
        input_names=["input1", "input2"],
        output_names=["embedding1", "embedding2"],
        dynamic_axes={
            "input1": {0: "batch_size"},
            "input2": {0: "batch_size"},
            "embedding1": {0: "batch_size"},
            "embedding2": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    # Downgrade IR version for EZKL compatibility
    onnx_model = onnx.load(ONNX_SAVE_PATH)
    onnx_model.ir_version = 8
    onnx.save(onnx_model, ONNX_SAVE_PATH)
    
    print("ONNX exported and IR version adjusted for EZKL.")

if __name__ == "__main__":
    main()
