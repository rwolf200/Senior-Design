import os, re, argparse, random, shutil, json
from io import BytesIO
from PIL import Image
import requests
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights, MobileNet_V3_Small_Weights

# ---------------- config ----------------
DATA_ROOT = "dataset"
TYPES = ["creature", "planeswalker", "artifact", "enchantment", "battle", "instant", "sorcery", "land"]
NUM_CLASSES = len(TYPES)
DEFAULT_EPOCHS = 500
DEFAULT_BATCH = 256
DEFAULT_LR = 8.04e-4
DEFAULT_NUM_WORKERS = 8
SAVE_PATH = "mtg_best_f1_types.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_THRESHOLD = 0.6
EARLY_STOPPING_PATIENCE = 50
_invalid_fn_re = re.compile(r'[\/:"*?<>|]+')


# ---------------- utilities ----------------
def sanitize_filename(name: str):
    """
    Sanitize a filename by replacing invalid filesystem characters and collapsing
    double slashes.

    Args:
        name (str): Original filename string.

    Returns:
        str: Cleaned filename safe for saving on disk.
    """
    name = name.replace("//", "_")
    name = _invalid_fn_re.sub("_", name)
    return name.strip()


def build_labels_for_split(root: str, split: str = "train"):
    """
    Build a list of (image_path, type_list) entries for a given split.

    The expected directory layout is:
        root/split/<type>/*.ext

    Args:
        root (str): Dataset root directory.
        split (str): Split name, e.g. "train" or "val".

    Returns:
        list[tuple[str, list[str]]]: List of (path, [types]) pairs.
    """
    split_root = os.path.join(root, split)
    fname_to_types = {}

    # Collect filenames and associated types
    for t in TYPES:
        folder = os.path.join(split_root, t)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")):
                continue
            fname_to_types.setdefault(fn, set()).add(t)

    # Resolve each filename to a concrete path and sorted type list
    entries = []
    for fname, type_set in fname_to_types.items():
        chosen_path = None
        # Multi- or single-typed: pick the first existing path
        for t in type_set:
            candidate = os.path.join(split_root, t, fname)
            if os.path.isfile(candidate):
                chosen_path = candidate
                break

        if not chosen_path:
            continue
        entries.append((chosen_path, sorted(list(type_set))))

    return entries


# ---------------- dataset ----------------
class MTGColorDataset(Dataset):
    """
    Dataset for MTG type classification.

    Each sample returns an image tensor and a multi-hot label vector of length
    NUM_CLASSES.
    """

    def __init__(self, entries, transform=None):
        """
        Args:
            entries (list): List of (path, type_list) pairs.
            transform (callable, optional): Transform to apply to each image.
        """
        self.entries = entries
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Load an image and its multi-label target.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (image, label_vector)
        """
        path, type_list = self.entries[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Multi-hot label vector
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        for t in type_list:
            if t in TYPES:
                label[TYPES.index(t)] = 1.0

        return img, label


# ---------------- transforms ----------------
def get_transforms():
    """
    Create training and validation transforms.

    Returns:
        tuple: (train_transform, val_transform)
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ---------------- model factory ----------------
def create_model(backbone: str, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Create a classification model with the specified backbone.

    Args:
        backbone (str): One of "resnet18", "efficientnet_b0", "mobilenet_v3".
        num_classes (int): Number of output classes.
        device (str): Device string, e.g. "cuda" or "cpu".

    Returns:
        torch.nn.Module: Model moved to the specified device.
    """
    b = backbone.lower()
    if b == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif b == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif b == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model.to(device).float()


# ---------------- threshold utils ----------------
def find_best_thresholds(all_trues, all_probs, step=0.01):
    """
    Find per-class thresholds that maximize F1 for card types.

    Args:
        all_trues (array-like): Ground truth multi-hot labels.
        all_probs (array-like): Predicted probabilities.
        step (float): Threshold step size for search.

    Returns:
        list[float]: Thresholds for each class.
    """
    all_trues = np.asarray(all_trues)
    all_probs = np.asarray(all_probs)
    best_thresholds = []

    # Tune thresholds for each card type
    for c in range(len(TYPES)):
        best_t, best_f = 0.5, -1.0
        y_true = all_trues[:, c]
        p = all_probs[:, c]
        for t in np.arange(0.1, 1.0, step):
            y_pred = (p >= t).astype(int)
            if y_pred.sum() == 0 and y_true.sum() > 0:
                continue
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        best_thresholds.append(best_t)

    return best_thresholds


def types_from_logits_with_thresholds(logits, thresholds):
    """
    Convert logits to card types using per-class thresholds.

    Args:
        logits (torch.Tensor or np.ndarray): Raw logits for each class.
        thresholds (list[float]): Thresholds for each class.

    Returns:
        list[str]: List of predicted card types.
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = np.array(logits)

    out = []
    for i, t in enumerate(TYPES):
        if probs[i] >= thresholds[i]:
            out.append(t)

    return out


# ---------------- train ----------------
def train(args):
    """
    Train the multi-label MTG type classifier.

    This function handles dataset creation, dataloaders, model setup,
    training loop, validation, threshold tuning, checkpoint saving,
    and early stopping based on macro F1.
    """
    entries = build_labels_for_split(args.data_root, "train")

    if not entries:
        print("No images found in dataset/train")
        return

    # Train/validation split
    train_e, val_e = train_test_split(entries, test_size=args.val_split, random_state=42)
    print(f"Train {len(train_e)} | Val {len(val_e)}")

    train_tf, val_tf = get_transforms()
    train_ds = MTGColorDataset(train_e, train_tf)
    val_ds = MTGColorDataset(val_e, val_tf)

    pin_memory = args.device.startswith("cuda")
    persistent_workers = args.num_workers > 0 and pin_memory

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    model = create_model(args.backbone, NUM_CLASSES, args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    best_f1 = 0.0
    epochs_no_improve = 0
    thresholds = [DEFAULT_THRESHOLD] * NUM_CLASSES

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        # Training loop
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=args.device.startswith("cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, len(train_loader))

        # Validation loop
        model.eval()
        all_probs, all_trues = [], []
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                with torch.amp.autocast(device_type='cuda', enabled=args.device.startswith("cuda")):
                    logits = model(x)
                    val_loss += float(criterion(logits, y).item())
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                    all_trues.append(y.cpu().numpy())

        if all_probs:
            all_probs = np.concatenate(all_probs, 0)
            all_trues = np.concatenate(all_trues, 0)
            thresholds = find_best_thresholds(all_trues, all_probs)
            preds = (all_probs >= np.array(thresholds)).astype(int)
            f1 = f1_score(all_trues,preds,average='macro',zero_division=0)
        else:
            f1 = 0.0

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch} | TrainLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | F1 {f1:.4f}")
        print(" Per-class thresholds:", {TYPES[i]: thresholds[i] for i in range(NUM_CLASSES)})

        # Check for improvement and save best model
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.save_path)
            thr_path = args.save_path.replace(".pth", "_thresholds.json")
            with open(thr_path, 'w') as fh:
                json.dump({TYPES[i]: thresholds[i] for i in range(NUM_CLASSES)}, fh)
            print("Saved best model & thresholds")

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Step LR scheduler each epoch
        scheduler.step()


# ---------------- test helpers ----------------
def evaluate(model, dataloader, device, thresholds=None):
    """
    Evaluate a trained model on a dataloader and print metrics.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for evaluation.
        device (str): Device string.
        thresholds (list[float], optional): Per-class thresholds. If None,
            DEFAULT_THRESHOLD is used for all classes.

    Returns:
        float: Micro-averaged F1 score.
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            if thresholds is None:
                preds = (probs > DEFAULT_THRESHOLD).float()
            else:
                t_arr = torch.tensor(thresholds, device=device).float()
                preds = (probs > t_arr).float()
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()

    print("\n=== TEST METRICS ===")
    label_sums = all_labels.sum(axis=0)
    print("Label positives:", label_sums.tolist())
    pred_sums = all_preds.sum(axis=0)
    print("Predicted positives:", pred_sums.tolist())

    # Per-class metrics
    for i, c in enumerate(TYPES):
        p = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        r = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        print(f"{c}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

    micro = f1_score(all_labels, all_preds, average="micro")
    macro = f1_score(all_labels, all_preds, average="macro")
    print("Micro F1:", micro, "| Macro F1:", macro)
    return micro


# ---------------- test modes ----------------
def test_local(args):
    """
    Test the model on a random local image from a split.

    Args:
        args: Parsed CLI arguments with data_root, split, backbone, model_path, device.
    """
    entries = build_labels_for_split(args.data_root, args.split)
    if not entries:
        print("No entries found.")
        return

    path, actual_types = random.choice(entries)
    print("Selected:", path, "Actual:", actual_types)

    train_tf, val_tf = get_transforms()
    img = Image.open(path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(args.device)

    model = create_model(args.backbone, NUM_CLASSES, args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    pred = types_from_logits_with_thresholds(probs[0],[DEFAULT_THRESHOLD] * NUM_CLASSES)
    print("Predicted:", pred)


def test_file(args):
    """
    Test the model on a specific image file path.

    Args:
        args: Parsed CLI arguments with file, backbone, model_path, device.
    """
    if not os.path.isfile(args.file):
        print("File not found:", args.file)
        return

    train_tf, val_tf = get_transforms()
    img = Image.open(args.file).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(args.device)

    model = create_model(args.backbone, NUM_CLASSES, args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    pred = types_from_logits_with_thresholds(probs[0],[DEFAULT_THRESHOLD] * NUM_CLASSES)
    print("Predicted:", pred)


# ---------------- test_scryfall ----------------
def test_scryfall(args, n_tests=5000):
    """
    Test the model on random cards from the Scryfall API.

    This repeatedly samples random cards, filters out tokens/emblems,
    downloads the image, runs the model, and compares predicted types
    to Scryfall's type_line.

    Args:
        args: Parsed CLI arguments with backbone, model_path, device.
        n_tests (int): Number of random cards to test.
    """
    # Load model
    model = create_model(args.backbone, num_classes=NUM_CLASSES, device=args.device)
    if not os.path.exists(args.model_path):
        print("Model file not found:", args.model_path)
        return
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # Load thresholds from training
    thr_path = args.model_path.replace(".pth", "_thresholds.json")
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            loaded_thresholds = json.load(f)
        thresholds = [loaded_thresholds.get(c, DEFAULT_THRESHOLD)for c in TYPES]
    else:
        thresholds = [DEFAULT_THRESHOLD] * NUM_CLASSES

    correct = 0
    wrong = []
    total = 0

    for i in range(n_tests):

        # Sample a random non-token, non-emblem card
        while True:
            data = requests.get("https://api.scryfall.com/cards/random").json()
            layout = data.get("layout", "")
            type_line = data.get("type_line", "").lower()
            if layout not in ["token", "emblem"] and "token" not in type_line:
                break

        name = data.get("name", "unknown")
        type_line = data.get("type_line") or data.get("card_faces", [{}])[0].get("type_line", "")
        true_types = []
        left = type_line.split("—")[0].strip().lower()
        parts = left.split()

        for t in TYPES:
            if t in parts:
                true_types.append(t)

        # Get image URL (front face if double-faced)
        url = data.get("image_uris", {}).get("normal")
        if not url and "card_faces" in data and data["card_faces"]:
            url = data["card_faces"][0].get("image_uris", {}).get("normal")
        if not url:
            continue

        try:
            img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
        except Exception as e:
            print(f"Failed to open image for {name}: {e}")
            continue

        _, val_tf = get_transforms()
        x = val_tf(img).unsqueeze(0).to(args.device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=args.device.startswith("cuda")):
            logits = model(x)

        # Use thresholds loaded from training
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        pred_types = [TYPES[i] for i in range(NUM_CLASSES) if probs[i] >= thresholds[i]
        ]

        total += 1
        if sorted(pred_types) == sorted(true_types):
            correct += 1
        else:
            wrong.append({
                "name": name,
                "scryfall": true_types,
                "predicted": pred_types,
                "image": url
            })

        if (i + 1) % 50 == 0:
            print(f"Tested {i+1}/{n_tests} cards… wrong {len(wrong)}/{total} ({len(wrong)/max(1,total)*100:.2f}%)")

    print("\n===== Scryfall Test Summary =====")
    print(f"Total tested:  {total}")
    print(f"Correct:       {correct}")
    print(f"Incorrect:     {len(wrong)}")
    print(f"Accuracy:      {correct/total:.4f}" if total else "Accuracy: N/A")

    if wrong:
        print("\nSample wrong predictions:")
        for w in wrong[:10]:
            print(w)


# ---------------- LR Finder ----------------
try:
    from torch_lr_finder import LRFinder
    HAS_LR_FINDER = True
except ImportError:
    HAS_LR_FINDER = False


def lr_find(args):
    """
    Run a learning rate range test to help choose a good LR.

    Uses torch_lr_finder if available, otherwise falls back to a simple
    exponential LR sweep and saves a plot.
    """
    print("Running LR range test...")

    # Build dataset/loaders just like in training
    entries = build_labels_for_split(args.data_root, "train")
    train_e, val_e = train_test_split(entries,test_size=args.val_split,random_state=42)
    train_tf, val_tf = get_transforms()
    train_ds = MTGColorDataset(train_e, train_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = create_model(args.backbone, NUM_CLASSES, args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),lr=1e-7,weight_decay=1e-4)  # start tiny

    if HAS_LR_FINDER:
        print("Using torch_lr_finder implementation.")
        lr_finder = LRFinder(model,optimizer,criterion,device=args.device)

        lr_finder.range_test(
            train_loader,
            end_lr=1,
            num_iter=200,
            step_mode="exp"
        )

        # Plot learning rate vs loss
        lr_finder.plot()
        lr_finder.reset()

    else:
        # Minimal built-in LR finder (fallback)
        print("torch_lr_finder not installed. Using built-in LR finder.")
        lrs = []
        losses = []

        lr = 1e-7
        max_lr = 1
        mult = (max_lr / lr) ** (1 / 200)

        for i, (x, y) in enumerate(train_loader):
            if i > 200:
                break

            # Update LR for this iteration
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.train()
            optimizer.zero_grad()

            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            lrs.append(lr)
            losses.append(loss.item())

            lr *= mult

        # Save LR curve
        import matplotlib.pyplot as plt
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("LR Range Test")
        out = "lr_find.png"
        plt.savefig(out)
        print(f"Saved LR finder plot to {out}")

    print("LR Finder completed.")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test_local", "test_file", "test_scryfall", "lr_find"]
    )

    parser.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=["resnet18", "efficientnet_b0", "mobilenet_v3"]
    )

    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--save_path", default=SAVE_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--early_stopping_patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--file", type=str, help="File to test")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model_path", default=SAVE_PATH)
    args = parser.parse_args()

    if args.mode == "train":train(args)
    elif args.mode == "test_local":test_local(args)
    elif args.mode == "test_file":test_file(args)
    elif args.mode == "test_scryfall":test_scryfall(args)
    elif args.mode == "lr_find":lr_find(args)
