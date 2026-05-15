#!/usr/bin/env python3
import argparse
import random
import requests
from io import BytesIO
from PIL import Image
import time

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def create_model():
    # Load EfficientNet-B0 with ImageNet weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier to match your number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))

    return model.to(DEVICE)


def load_model(path):
    model = create_model()
    state = torch.load(path, map_location=DEVICE)

    # Load safely (EfficientNet is stable, so strict=True is fine)
    model.load_state_dict(state, strict=True)

    model.eval()
    return model



# ---------------- config ----------------
CLASSES = [
    "grass", "fire", "water", "lightning", "psychic",
    "fighting", "darkness", "metal", "dragon",
    "colorless", "trainer", "energy"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

eff_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
eff_tf = eff_weights.transforms()

val_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=eff_tf.mean,
        std=eff_tf.std
    ),
])



# 🔥 Fast persistent HTTP session
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# 🔥 Global cache of cards with images
CARD_CACHE = []


# ---------------- FAST data fetch ----------------
CARD_CACHE = []

def refill_cache(max_retries=5):
    """Fetch 250 cards with images, with retry + backoff."""
    global CARD_CACHE

    url = (
        "https://api.pokemontcg.io/v2/cards"
        "?pageSize=250"
        "&q=images.large:*"
    )

    for attempt in range(max_retries):
        try:
            # 🔥 longer timeout, API is slow sometimes
            res = session.get(url, timeout=15)
            data = res.json().get("data", [])

            CARD_CACHE = [
                card for card in data
                if "images" in card and "large" in card["images"]
            ]

            random.shuffle(CARD_CACHE)
            return

        except Exception as e:
            print(f"[refill_cache] Retry {attempt+1}/{max_retries} after error: {e}")
            time.sleep(1 + attempt * 0.5)  # 🔥 exponential backoff

    raise RuntimeError("Failed to refill cache after multiple retries")


def get_sample():
    """Return a random card with an image, instantly."""
    global CARD_CACHE

    if not CARD_CACHE:
        refill_cache()

    card = CARD_CACHE.pop()

    name = card.get("name", "unknown")
    supertype = card.get("supertype", "")
    types = card.get("types", [])
    img_url = card["images"]["large"]

    # Determine label
    if supertype == "Trainer":
        label = "trainer"
    elif supertype == "Energy":
        label = "energy"
    else:
        label_map = {
            "Grass": "grass",
            "Fire": "fire",
            "Water": "water",
            "Lightning": "lightning",
            "Psychic": "psychic",
            "Fighting": "fighting",
            "Darkness": "darkness",
            "Metal": "metal",
            "Dragon": "dragon",
        }
        mapped = [label_map[t] for t in types if t in label_map]
        label = mapped[0] if mapped else "colorless"

    # 🔥 image fetch also gets retry protection
    for _ in range(3):
        try:
            img_res = session.get(img_url, timeout=10)
            img = Image.open(BytesIO(img_res.content)).convert("RGB")
            return img, name, label, img_url
        except:
            time.sleep(0.2)

    # If image fails repeatedly, just skip and get another
    return get_sample()


# ---------------- inference ----------------
def predict(model, img):
    x = val_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return CLASSES[pred]


# ---------------- test loop ----------------
def test(model, n=200):
    correct = 0
    total = 0

    for i in range(n):
        img, name, label, img_url = get_sample()
        pred = predict(model, img)

        total += 1
        match = pred == label
        correct += int(match)

        print(f"\n[{total}] {name}")
        print("True:", label)
        print("Pred:", pred)
        print("Match:", match)
        print("Image:", img_url)

    print("\n===== FINAL =====")
    print(f"Accuracy: {correct}/{total} = {correct/max(1,total):.4f}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pokemon_best_b0.pth")
    parser.add_argument("--n_tests", type=int, default=200)

    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_path)

    print("Testing...")
    test(model, args.n_tests)
