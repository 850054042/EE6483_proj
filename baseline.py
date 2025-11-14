import os
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.glob("*")
                             if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        img = Image.open(fp).convert("RGB")
        img = self.transform(img)
        return img, fp.stem

def plot_curve(train_vals, val_vals, title, save_path, ylabel):
    plt.figure(figsize=(6,4))
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@torch.no_grad()
def predict(model, loader, device, out_csv):
    model.eval()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            for p, name in zip(preds, fnames):
                w.writerow([name, int(p)])


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    out_root = Path("outputs") / "baseline_resnet18"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_root / "best.pt"
    submission_path = out_root / "submission.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder("data/train", transform=transform)
    val_set = datasets.ImageFolder("data/val", transform=transform)
    test_set = TestFolder("data/test", transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    num_classes = 2

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0
    epochs = 10

    print("Starting baseline (ResNet18, no augment)...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        total_correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_set)
        train_acc = total_correct / len(train_set)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / len(val_set)
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}/{epochs} "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} "
              f"| Time {time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model to {model_path}")

    print(f"Training done. Best Val Acc = {best_acc:.4f}")

    plot_curve(train_losses, val_losses, "Loss Curve", fig_dir / "loss_curve.png", ylabel="Loss")
    plot_curve(train_accs, val_accs, "Accuracy Curve", fig_dir / "acc_curve.png", ylabel="Accuracy")
    print(f"Saved curves to {fig_dir}")

    print("Generating submission.csv...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    predict(model, test_loader, device, submission_path)
    print(f"Saved {submission_path}")


if __name__ == "__main__":
    main()
