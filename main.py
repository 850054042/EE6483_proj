#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, datasets, models
from torchvision.datasets import CIFAR10
from torch import amp

class TestFolder(Dataset):
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
        if self.transform:
            img = self.transform(img)
        return img, fp.stem


class CIFAR10TestLike(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, str(i)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, target):
        logpt = -self.ce(logits, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean()

def build_model(backbone: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    else:
        raise ValueError("Unsupported backbone. Choose resnet18/resnet50")

    if freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return model

def accuracy(logits, y):
    pred = logits.argmax(1)
    return (pred == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs

    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs
    return running_loss / n, running_acc / n


@torch.no_grad()
def predict_to_csv(model, loader, device, out_csv, label_map=None):
    model.eval()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for x, img_id in loader:
            x = x.to(device)
            with amp.autocast(device_type=device.type, enabled=False):
                logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            for i, p in enumerate(pred):
                lab = int(label_map[p]) if label_map is not None else int(p)
                w.writerow([img_id[i], lab])

def build_transforms(img_size=224, dataset="dogs"):
    if dataset == "dogs":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        train_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return train_tf, test_tf


def prepare_dataloaders(args, device):
    """
    Returns: train_loader, val_loader, test_loader, num_classes, label_map
    """
    img_size = args.img_size
    train_tf, test_tf = build_transforms(img_size, "dogs" if args.dataset == "dogs" else "cifar10")

    num_workers = max(0, int(args.workers))
    pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0

    if args.dataset == "dogs":
        train_dir = Path(args.data_root) / "train"
        val_dir = Path(args.data_root) / "val"
        test_dir = Path(args.data_root) / "test"

        train_set = datasets.ImageFolder(train_dir, transform=train_tf)
        val_set = datasets.ImageFolder(val_dir, transform=test_tf)
        test_set = TestFolder(test_dir, transform=test_tf)

        sampler = None
        if args.use_weighted_sampler:
            counts = np.bincount(train_set.targets)
            class_weights = 1.0 / np.maximum(counts, 1)
            sample_weights = class_weights[train_set.targets]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                            num_samples=len(sample_weights),
                                            replacement=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=(sampler is None), sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_mem,
                                  persistent_workers=persistent)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_mem,
                                persistent_workers=persistent)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_mem,
                                 persistent_workers=persistent)

        print(f"[DATA] train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        print(f"[DATA] classes={train_set.classes}")

        num_classes = 2
        label_map = None
        return train_loader, val_loader, test_loader, num_classes, label_map

    else:
        train_set = CIFAR10(args.data_root, train=True, download=True, transform=train_tf)
        val_set = CIFAR10(args.data_root, train=False, download=True, transform=test_tf)

        test_like = CIFAR10TestLike(val_set)

        sampler = None
        if args.use_weighted_sampler:
            targets = np.array(train_set.targets)
            counts = np.bincount(targets, minlength=10)
            class_weights = 1.0 / np.maximum(counts, 1)
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                            num_samples=len(sample_weights),
                                            replacement=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=(sampler is None), sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_mem,
                                  persistent_workers=persistent)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_mem,
                                persistent_workers=persistent)
        test_loader = DataLoader(test_like, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_mem,
                                 persistent_workers=persistent)

        print(f"[DATA] CIFAR-10 train={len(train_set)}, val(test)={len(val_set)}")

        num_classes = 10
        label_map = None
        return train_loader, val_loader, test_loader, num_classes, label_map


def plot_lr_curve(lrs, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(lrs)
    plt.title("Learning Rate Curve")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_acc_curve(train_accs, val_accs, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def save_error_samples(model, loader, device, save_path, num_samples=12):
    model.eval()
    wrong_images = []
    wrong_preds = []
    wrong_labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu()
        y = y.cpu()

        wrong_mask = preds != y
        if wrong_mask.sum() > 0:
            for img, pred, lbl, wrong in zip(x.cpu(), preds, y, wrong_mask):
                if wrong:
                    wrong_images.append(img)
                    wrong_preds.append(int(pred))
                    wrong_labels.append(int(lbl))
                if len(wrong_images) >= num_samples:
                    break
        if len(wrong_images) >= num_samples:
            break

    if len(wrong_images) == 0:
        print("[WARN] No wrong predictions found for visualization.")
        return

    cols = 4
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(12, 3 * rows))

    for i in range(num_samples):
        img = wrong_images[i]
        img = img.permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Pred: {wrong_preds[i]} | GT: {wrong_labels[i]}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@torch.no_grad()
def plot_confusion_matrix(model, loader, device, class_names, save_path, normalize=True):
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Confusion Matrix")
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i,
                format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dogs", choices=["dogs", "cifar10"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", help="freeze all but final FC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_focal", action="store_true", help="use FocalLoss instead of CE")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--predict", action="store_true", help="skip training, only predict with outputs/best.pt")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)

    args = parser.parse_args()

    exp_name_parts = [args.backbone]
    if args.freeze_backbone:
        exp_name_parts.append("freeze_backbone")
    if args.use_focal:
        exp_name_parts.append("focal_loss")

    exp_name = "_".join(exp_name_parts)
    output_dir = Path("outputs") / args.dataset / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best.pt"
    submission_path = output_dir / "submission.csv"
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.txt"

    args.model_path = str(model_path)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lrs = []

    set_seed(args.seed)
    device = get_device()
    Path("outputs").mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, num_classes, label_map = prepare_dataloaders(args, device)
    model = build_model(args.backbone, num_classes, pretrained=True,
                        freeze_backbone=args.freeze_backbone).to(device)

    if args.dataset == "dogs":
        class_names = train_loader.dataset.classes
    else:
        class_names = train_loader.dataset.classes

    if args.predict:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[Predict] Loaded {args.model_path}")
        predict_to_csv(model, test_loader, device, submission_path, label_map)
        print(f"Saved {submission_path}")
        return

    if args.use_focal:
        criterion = FocalLoss(gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = amp.GradScaler() if (args.use_amp and device.type == "cuda") else None

    best_acc, patience, best_epoch = 0.0, 5, -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        for param_group in optimizer.param_groups:
            lrs.append(param_group["lr"])

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"time={time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), args.model_path)
            print(f"  * New best acc {best_acc:.4f} at epoch {epoch}, saved to {args.model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  * Early stopping triggered.")
                break

    print(f"[Summary] Best val acc={best_acc:.4f} at epoch={best_epoch}")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i, (tl, ta, vl, va) in enumerate(zip(train_losses, train_accs, val_losses, val_accs), start=1):
            writer.writerow([i, tl, ta, vl, va])

    with open(summary_path, "w") as f:
        f.write(f"Best validation accuracy: {best_acc:.6f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Freeze backbone: {args.freeze_backbone}\n")
        f.write(f"Use focal loss: {args.use_focal}\n")
        f.write(f"Use weighted sampler: {args.use_weighted_sampler}\n")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    predict_to_csv(model, test_loader, device, submission_path, label_map)
    print(f"Saved {submission_path}")

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_lr_curve(lrs, fig_dir / "lr_curve.png")
    plot_loss_curve(train_losses, val_losses, fig_dir / "loss_curve.png")
    plot_acc_curve(train_accs, val_accs, fig_dir / "acc_curve.png")
    save_error_samples(model, val_loader, device, fig_dir / "errors.png")
    plot_confusion_matrix(model, val_loader, device, class_names,
                          fig_dir / "confusion_matrix.png", normalize=True)

    print(f"[INFO] Metrics saved to: {metrics_path}")
    print(f"[INFO] Summary saved to: {summary_path}")
    print(f"[INFO] Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
