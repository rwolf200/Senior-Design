#!/usr/bin/env python3
"""
pokemon_train_fixed.py - Stable EfficientNet-B0 Pokémon classifier
MTG-style pipeline with oversized/corrupt image protection.
"""

import os, argparse
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

# ---------------- config ----------------
DATA_ROOT = "dataset"

CLASSES = [
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "dragon",
    "colorless", "trainer", "energy"
]

NUM_CLASSES = len(CLASSES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_EPOCHS = 150
DEFAULT_BATCH = 256
DEFAULT_LR = 6e-4
DEFAULT_NUM_WORKERS = 10          # FIXED: prevent AMD worker deadlocks
SAVE_PATH = "pokemon_best_b0.pth"

torch.backends.cudnn.benchmark = True


# ---------------- image safety checks ----------------
def safe_open(path):
    """Open image safely, skip corrupt or oversized images."""
    try:
        img = Image.open(path)
        img.verify()  # check corruption
        img = Image.open(path).convert("RGB")

        # Skip absurdly large images (EfficientNet hates these)
        if img.width > 3000 or img.height > 3000:
            print(f"[SKIP] Oversized image: {path} ({img.width}x{img.height})")
            return None

        return img
    except Exception as e:
        print(f"[SKIP] Corrupt image: {path} ({e})")
        return None


# ---------------- dataset builder ----------------
def build_dataset(root, split="train"):
    entries = []
    split_root = os.path.join(root, split)

    for cls in CLASSES:
        folder = os.path.join(split_root, cls)
        if not os.path.isdir(folder):
            continue

        for fn in os.listdir(folder):
            if fn.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp",".tiff",".tif")):
                entries.append((os.path.join(folder, fn), CLASSES.index(cls)))

    return entries



# ---------------- dataset ----------------
class PokemonDataset(Dataset):
    def __init__(self, entries, transform=None): 
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]

        # Try to load safely
        try:
            img = Image.open(path).convert("RGB")

            # Skip absurdly large images
            if img.width > 3000 or img.height > 3000:
                print(f"[SKIP] Oversized: {path}")
                img = Image.new("RGB", (224,224), (0,0,0))

        except Exception as e:
            print(f"[SKIP] Corrupt: {path} ({e})")
            img = Image.new("RGB", (224,224), (0,0,0))

        if self.transform:
            img = self.transform(img)

        return img, label



# ---------------- transforms (MTG-style) ----------------
def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((256,256)),   # FIXED: force small before aug
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.15, 0.15, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ---------------- model factory ----------------
def create_model():
    model = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(DEVICE).float()


# ---------------- train ----------------
def train(args):
    entries = build_dataset(args.data_root, "train")
    if not entries:
        print("No images found.")
        return
    train_e, val_e = train_test_split(entries, test_size=0.1, random_state=42)
    print(f"Train {len(train_e)} | Val {len(val_e)}")

    train_tf, val_tf = get_transforms()
    train_ds = PokemonDataset(train_e, train_tf)
    val_ds = PokemonDataset(val_e, val_tf)

    pin_memory = args.device.startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,      # FIXED: prevent AMD deadlocks
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, len(train_loader))

        # validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                with torch.amp.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                    logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(y.cpu().numpy())

        acc = accuracy_score(labels_all, preds_all)
        print(f"Epoch {epoch} | TrainLoss {avg_train_loss:.4f} | ValAcc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            print("Saved best model ✔")

        scheduler.step()

    print(f"\nBest accuracy: {best_acc:.4f}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--save_path", default=SAVE_PATH)

    args = parser.parse_args()
    train(args)
