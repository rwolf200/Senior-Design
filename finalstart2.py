import os, json, time
import easyocr
import numpy as np
import cv2
import RPi.GPIO as GPIO
from time import sleep
from picamera2 import Picamera2
import tkinter as tk
import threading
from PIL import Image, ImageTk

import torch
import torch.nn.functional as F
from torchvision import transforms, models

# Global state shared between sorter thread and GUI
processed_photo = None          # Latest processed image for GUI display
stop_event = threading.Event()  # Event to signal sorter thread to stop
ip = []                         # Model or OCR output (e.g., name, type, color)
fb = 1                          # Selected bin index
t = 0                           # Time taken for last card processing

# Model paths
MODEL_PATH_C = "best.pth"           # MTG color model
MODEL_PATH_T = "best_types.pth"     # MTG type model
MODEL_PATH_P = "pokemon_best.pth"   # Pokémon type model
DEFAULT_THRESHOLD = 0.5

# Color mapping for MTG (not directly used in logic but kept for reference)
COLORS = ["white", "blue", "black", "red", "green"]
LETTERS = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}

# GPIO pin configuration for servos
pins = [12, 18, 13, 19, 24, 23]
i = 0
GPIO.setmode(GPIO.BCM)
for p in pins:
    GPIO.setup(p, GPIO.OUT)

# Servo PWM configuration
f = 50               # PWM frequency in Hz
duty_cycle = 7.5     # Neutral duty cycle for standard servo

GPIO.setwarnings(False)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (3280, 2464)},
    sensor={"output_size": (3280, 2464)}
)
picam2.configure(config)
picam2.start()


def load_model_thresh(m):
    """
    Load an EfficientNet model and its per-class thresholds from disk.

    Parameters:
        m (str): Path to the .pth model file. A corresponding JSON file with
                 thresholds is expected at the same path with '_thresholds.json'
                 appended before the extension.

    Returns:
        tuple:
            model (torch.nn.Module): Loaded EfficientNet model in eval mode.
            thresholds (list[float]): List of thresholds for each class.
    """
    thr_path = m.replace(".pth", "_thresholds.json")

    # Load thresholds from JSON file
    with open(thr_path, "r") as f:
        loaded_thresholds = json.load(f)

    class_names = list(loaded_thresholds.keys())
    num_classes = len(class_names)

    # Build EfficientNet with correct output dimension
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Load model weights
    model.load_state_dict(torch.load(m, map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    # Extract thresholds in class order
    thresholds = [loaded_thresholds[c] for c in class_names]
    return model, thresholds


def load_model():
    """
    Load the Pokémon type EfficientNet model.

    Returns:
        torch.nn.Module: Loaded EfficientNet model in eval mode with 12 outputs.
    """
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 12)
    model.load_state_dict(torch.load(MODEL_PATH_P, map_location="cpu"))
    model = model.to("cpu")
    model.eval()
    return model


WUBRG_ORDER = "WUBRG"


def sort(m, ip, b=5, g="MTG", l1=1, l2=26):
    """
    Map the interpreted card information (name/type/color) to a bin index.

    Parameters:
        m (str): Sorting method. One of "name", "type", "color".
        ip:      Interpreted property from OCR or model:
                 - For "name": string (card name).
                 - For "type": string or list of strings.
                 - For "color": list of color codes (e.g., ['r'], ['g'], etc.).
        b (int): Number of bins available.
        g (str): Game type. "MTG" or "Pokemon".
        l1 (int): Lower bound of alphabet range (1 = 'a').
        l2 (int): Upper bound of alphabet range (26 = 'z').

    Returns:
        int: Bin index (1-based). Defaults to 1 or b in out-of-range cases.
    """
    # MTG sorting rules
    if g == "MTG":
        # Sort by name into alphabet bins
        if m == "name":
            n = ord(ip[0][0].lower()) - ord('a') + 1

            # Check if letter is within configured range
            if l1 < n and l2 >= n:
                # Compute base bin size and distribute remainder
                dis = (l2 - (l1 - 1)) // b
                ext = (l2 - (l1 - 1)) % b
                bins = [dis] * b

                i = 0
                while ext > 0:
                    bins[i] += 1
                    i += 1
                    ext -= 1

                # Find which bin the letter falls into
                i = 0
                while bins[i] < n:
                    n -= bins[i]
                    i += 1
                return i + 1
            else:
                print("out of range")
                return 1

        # Sort by MTG type
        elif m == "type":
            # ip may be a list; handle common cases
            if isinstance(ip, list):
                ip_lower = [x.lower() for x in ip]
            else:
                ip_lower = [str(ip).lower()]

            if "land" in ip_lower:
                return 1
            elif "creature" in ip_lower:
                return 2

            if b > 3:
                ty = ["instant", "sorcery", "artifact", "enchantment", "planeswalker", "battle"]
                # If ip is a single string, use it directly
                if isinstance(ip, str):
                    ip_val = ip.lower()
                else:
                    # Fallback: use first element if list
                    ip_val = ip_lower[0]

                if ip_val in ty:
                    if b > ty.index(ip_val) + 2:
                        return ty.index(ip_val) + 2
                    else:
                        return b
                else:
                    return b
            else:
                return 3

        # Sort by MTG color
        elif m == "color":
            # ip is expected to be a list of color codes like ['r'], ['g'], etc.
            match ip:
                case ['r']:
                    return 1
                case ['g']:
                    return 2

            if b > 3:
                color = ['b', 'u', 'w']
                if len(ip) == 1:
                    if ip[0] in color and b >= color.index(ip[0]) + 2:
                        return color.index(ip[0]) + 3
                elif len(ip) > 1:
                    # Multicolor card
                    if b > 5:
                        return 6
                    else:
                        return b
                else:
                    # Colorless or unknown
                    if b > 6:
                        return 7
                    else:
                        return b
            else:
                return 3

    # Pokémon sorting rules
    elif g == "Pokemon":
        # Sort by name into alphabet bins
        if m == "name":
            n = ord(ip[0][0].lower()) - ord('a') + 1
            if l1 < n and l2 >= n:
                dis = (l2 - (l1 - 1)) // b
                ext = (l2 - (l1 - 1)) % b
                bins = [dis] * b

                i = 0
                while ext > 0:
                    bins[i] += 1
                    i += 1
                    ext -= 1

                i = 0
                while i < len(bins) and bins[i] < n:
                    n -= bins[i]
                    i += 1
                if i >= len(bins):
                    return b
                return i + 1
            else:
                print("out of range")
                return 1

        # Sort by Pokémon type
        elif m == "type":
            ip_val = ip.lower()

            match ip_val:
                case "energy":
                    return 1
                case "trainer":
                    return 2

            if b > 3:
                ty = [
                    "grass", "fire", "water", "lightning", "psychic",
                    "fighting", "darkness", "metal", "dragon", "colorless"
                ]
                if ip_val in ty:
                    if b > ty.index(ip_val) + 2:
                        return ty.index(ip_val) + 2
                    else:
                        return b
                else:
                    return b
            else:
                return 3

    else:
        print("not currently implemented")
        return


# Initialize EasyOCR reader (CPU only)
reader = easyocr.Reader(['en'], gpu=False)


def crop_top_bottom(cropped, top_pct=0.5, bottom_pct=0.2, debug=False):
    """
    Crop the top and bottom regions of an image and stack them vertically.

    This is used to remove the art portion of a card while keeping the
    name and type text regions.

    Parameters:
        cropped (np.ndarray): Input image.
        top_pct (float): Fraction of height from the bottom to keep.
        bottom_pct (float): Fraction of height from the top to keep.
        debug (bool): If True, prints debug information.

    Returns:
        np.ndarray: Cropped image containing top and bottom regions.
    """
    h, w = cropped.shape[:2]

    # Bottom region (near card name/type area)
    top_end = int(h * (1 - top_pct))
    top_region = cropped[top_end:h, :]

    # Top region (top border area)
    bottom_start = int(h * bottom_pct)
    bottom_region = cropped[0:bottom_start, :]

    # Stack top and bottom vertically
    final_crop = np.vstack([bottom_region, top_region])

    if debug:
        print(f"Top+Bottom crop: top {top_region.shape[0]} px, bottom {bottom_region.shape[0]} px, width {w}")

    return final_crop


def rotate_crop_top_ocr(img, pad=10, max_dim=800, debug=False, double_check=False, crop_art=True):
    """
    Normalize card orientation and optionally crop out the art region.

    Steps:
        1. Apply a perspective transform to approximate a flat card.
        2. Rotate the card to a consistent orientation.
        3. Use OCR on the top region to detect if the card is upside down.
        4. Optionally crop out the art using crop_top_bottom().

    Parameters:
        img (np.ndarray): Raw camera frame.
        pad (int): Unused, reserved for future padding logic.
        max_dim (int): Unused here, reserved for future scaling logic.
        debug (bool): If True, prints debug information.
        double_check (bool): Unused, reserved for future orientation checks.
        crop_art (bool): If True, removes art region using crop_top_bottom().

    Returns:
        np.ndarray: Rotated and optionally cropped image.
    """
    h, w = img.shape[:2]

    # Destination points for perspective transform (full image)
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Source points approximating card corners (tuned empirically)
    src_pts = np.float32([[0 + 400, 0 + 500], [w - 400, 500], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, matrix, (w, h))

    # Rotate card to upright orientation
    cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Extract top region for orientation check
    crop_h, crop_w = cropped.shape[:2]
    top_region = cropped[0:int(crop_h * 0.20), :]

    # Blur and resize for OCR
    top_blur = cv2.GaussianBlur(top_region, (7, 7), 0)
    small_h, small_w = top_region.shape[:2]
    scale_top = 150 / max(small_h, small_w)

    if scale_top < 1:
        top_small = cv2.resize(
            top_blur,
            (int(small_w * scale_top), int(small_h * scale_top * 1.5))
        )
    else:
        top_small = top_blur

    # Run OCR on top region to detect text
    ocr_results = reader.readtext(top_small)
    text_found = [text for _, text, prob in ocr_results if prob > 0.05]

    if debug:
        print(f"Top 20% OCR found {len(ocr_results)} text regions: {text_found}")

    # If no text is found, assume card is upside down and rotate 180 degrees
    if len(text_found) == 0:
        cropped = cv2.rotate(cropped, cv2.ROTATE_180)
        if debug:
            print("Flipped 180 because top region had no text.")

    # Optionally crop out art region
    if crop_art:
        cropped = crop_top_bottom(cropped)

    return cropped


def preprocess_for_ocr(region, try_invert=False):
    """
    Preprocess an image region to improve OCR performance.

    Steps:
        1. Convert to grayscale.
        2. Apply CLAHE for contrast enhancement.
        3. Optionally invert the image.
        4. Apply Otsu thresholding.
        5. Resize to a target width for better OCR.

    Parameters:
        region (np.ndarray): BGR image region to preprocess.
        try_invert (bool): If True, invert the grayscale image.

    Returns:
        np.ndarray: Preprocessed binary image suitable for OCR.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    gray = clahe.apply(gray)

    if try_invert:
        gray = cv2.bitwise_not(gray)

    # Otsu thresholding to binarize
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Resize to a fixed width for consistent OCR performance
    h, w = gray.shape
    target_width = 450
    scale = target_width / w
    if scale < 1:
        gray = cv2.resize(
            gray,
            (target_width, int(h * scale * 2.5)),
            interpolation=cv2.INTER_AREA
        )

    return gray


def clean_text(text):
    """
    Clean OCR text by normalizing characters and removing noise.

    Parameters:
        text (str): Raw OCR text.

    Returns:
        str: Cleaned, uppercased text.
    """
    text = text.upper()
    text = text.replace("0", "O").replace("1", "I")
    text = text.replace("|", "I")
    text = text.replace("(", "")
    return text.strip()


def OCR(image, game="MTG", debug=False):
    """
    Perform OCR on the top portion of a card image to extract the card name.

    For MTG:
        - Returns the first high-confidence text string.

    For Pokémon:
        - Filters out known non-name words (e.g., BASIC, STAGE, TRAINER).
        - Returns the longest remaining string as the name.

    Parameters:
        image (np.ndarray): Input card image (BGR).
        game (str): "MTG" or "Pokemon".
        debug (bool): If True, prints debug information.

    Returns:
        str or None: Detected card name, or None if no suitable text is found.
    """
    h, w = image.shape[:2]

    # Crop top 25% where the card name is typically located
    region = image[0:int(h * 0.25), :]
    processed = preprocess_for_ocr(region)
    cv2.imwrite("pre.jpg", processed)

    results = reader.readtext(processed)

    # Extract text with sufficient confidence and length
    texts = [
        clean_text(text)
        for (_, text, prob) in results
        if prob > 0.2 and len(text.strip()) > 2
    ]

    if debug:
        print("OCR raw results:", results)
        print("Filtered texts:", texts)

    if not texts:
        return None

    if game == "MTG":
        # For MTG, assume first text is the card name
        return texts[0]

    # Pokémon name extraction
    raw_texts = texts
    ignore_words = [
        "BASIC", "STAGE", "EVOLVES", "LEVEL", "HP",
        "TRAINER", "SUPPORTER", "ITEM", "STADIUM"
    ]

    filtered = []
    for t in raw_texts:
        if any(t.startswith(w) for w in ignore_words):
            continue
        filtered.append(t)

    if not filtered:
        # Fallback: choose the longest string if all were filtered
        return max(raw_texts, key=len)

    # Choose the longest remaining string as the name
    return max(filtered, key=len)


def color(img, model, thresholds):
    """
    Predict MTG card color(s) using a multi-label EfficientNet model.

    Parameters:
        img (np.ndarray): Input card image (BGR or RGB).
        model (torch.nn.Module): Loaded EfficientNet model.
        thresholds (list[float]): Per-class thresholds for color prediction.

    Returns:
        list[str]: List of predicted color codes (e.g., ['w', 'u']).
    """
    colors = ['w', 'u', 'b', 'r', 'g']

    # Convert BGR to RGB if needed
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing pipeline for EfficientNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = preprocess(img).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0][:5]  # ignore colorless

    print(probs)

    # Apply thresholds to determine which colors are present
    predicted = [colors[i] for i, p in enumerate(probs) if p >= thresholds[i]]
    return predicted


def TypeMTG(img, model, thresholds):
    """
    Predict MTG card type(s) using a multi-label EfficientNet model.

    Parameters:
        img (np.ndarray): Input card image (BGR or RGB).
        model (torch.nn.Module): Loaded EfficientNet model.
        thresholds (list[float]): Per-class thresholds for type prediction.

    Returns:
        list[str]: List of predicted MTG types.
    """
    Types = [
        "creature", "planeswalker", "artifact", "enchantment",
        "battle", "instant", "sorcery", "land"
    ]

    # Convert BGR to RGB if needed
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing pipeline for EfficientNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = preprocess(img).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Select all types above threshold
    predicted = [Types[i] for i, p in enumerate(probs) if p >= thresholds[i]]

    # If nothing meets threshold, choose the highest probability type
    if not predicted:
        predicted = [Types[int(np.argmax(probs))]]

    return predicted


def TypeP(img, model):
    """
    Predict Pokémon card type using a single-label EfficientNet model.

    Parameters:
        img (np.ndarray): Input card image (BGR or RGB).
        model (torch.nn.Module): Loaded EfficientNet model.

    Returns:
        str: Predicted Pokémon type (e.g., "fire", "water", "trainer").
    """
    Types = [
        "grass", "fire", "water", "lightning", "psychic",
        "fighting", "darkness", "metal", "dragon",
        "colorless", "trainer", "energy"
    ]

    # Convert BGR to RGB if needed
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing pipeline for EfficientNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = preprocess(img).unsqueeze(0)

    # Forward pass with softmax
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    # Return the type with highest probability
    return Types[int(np.argmax(probs))]


def Servo(servo, direction=""):
    """
    Control a single servo motor connected to a given GPIO pin.

    For pins [12, 13, 18, 19], standard PWM is used.
    For other pins, manual pulse generation is used.

    Parameters:
        servo (int): GPIO pin number.
        direction (str): "L" for left, "R" for right, "" for default/center.
    """
    # PWM-based servos
    if servo in [12, 13, 18, 19]:
        pwm = GPIO.PWM(servo, f)

        # Adjust duty cycle based on direction
        if direction == "R":
            pwm.start(duty_cycle - 1.2)
        elif direction == "L":
            pwm.start(duty_cycle + 1.2)
        else:
            pwm.start(duty_cycle - 5)

        sleep(0.4)
        pwm.start(duty_cycle)  # Return to neutral
        sleep(0.1)
        pwm.stop()

    # Pulse-based servos
    else:
        if direction == "R":
            for _ in range(50):
                GPIO.output(servo, GPIO.HIGH)
                sleep(0.0005)
                GPIO.output(servo, GPIO.LOW)
                sleep(0.0195)
        elif direction == "L":
            for _ in range(50):
                GPIO.output(servo, GPIO.HIGH)
                sleep(0.0025)
                GPIO.output(servo, GPIO.LOW)
                sleep(0.0175)
        else:
            for _ in range(50):
                GPIO.output(servo, GPIO.HIGH)
                sleep(0.0005)
                GPIO.output(servo, GPIO.LOW)
                sleep(0.0195)

        # Return to center position
        for _ in range(50):
            GPIO.output(servo, GPIO.HIGH)
            sleep(0.0015)
            GPIO.output(servo, GPIO.LOW)
            sleep(0.0185)


def move_card(b, bins=7):
    """
    Move a card to the specified bin by actuating the appropriate servos.

    The logic steps through servo pairs and decides whether to move left,
    right, or straight based on the target bin.

    Parameters:
        b (int): Target bin index (1-based).
        bins (int): Total number of bins.
    """
    i = 1
    while i <= b and i < bins:
        if b == i:
            # Move card left at this servo
            Servo(pins[i - 1], "L")
        elif b == i + 1:
            # Move card right at this servo
            Servo(pins[i - 1], "R")
        else:
            # Pass card straight through
            Servo(pins[i])
        i += 2


def Sorter(m, b, g):
    """
    Main sorting loop that runs in a background thread.

    Steps:
        1. Load the appropriate model(s) based on method and game.
        2. Continuously capture images from the camera.
        3. Normalize orientation and optionally crop art.
        4. Run OCR or classification model.
        5. Map result to a bin using sort().
        6. Move the card using move_card().
        7. Update global variables for GUI display.

    Parameters:
        m (str): Sorting method ("name", "type", "color").
        b (int): Number of bins.
        g (str): Game type ("MTG" or "Pokemon").
    """
    global ip, fb, t
    art = True  # Whether to crop art region

    # Load appropriate model(s) depending on method and game
    if m == "color":
        model, thresholds = load_model_thresh(MODEL_PATH_C)
        art = False

    if m == "type" and g == "MTG":
        model, thresholds = load_model_thresh(MODEL_PATH_T)
        art = False

    if m == "type" and g == "Pokemon":
        model = load_model()
        thresholds = None
        art = False

    count = 0  # Count of consecutive failures (e.g., OCR None)

    while not stop_event.is_set():
        global processed_photo
        start = time.time()

        # Capture image from camera
        img = picam2.capture_array()

        # Normalize orientation and optionally crop art
        crop = rotate_crop_top_ocr(img, crop_art=art)

        # Prepare image for GUI display
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        show_img = cv2.resize(img_rgb, (int(0.15 * h), int(0.15 * w)))
        show_img = Image.fromarray(show_img)
        processed_photo = show_img

        # Run selected method
        if m == "color":
            ip = color(crop, model, thresholds)

        elif m == "type" and g == "MTG":
            ip = TypeMTG(crop, model, thresholds)

        elif m == "type" and g == "Pokemon":
            ip = TypeP(crop, model)

        else:
            # Name-based sorting using OCR
            ip = OCR(crop, g)

        # If OCR/model failed, increment failure count and optionally move to default bin
        if ip is None:
            count += 1
            if count >= 4:
                # After several failures, move card to default bin
                move_card(b)
            continue

        # Reset failure count on success
        count = 0

        # Compute bin index based on interpreted property
        fb = sort(m, ip, b, g)

        # Move card to computed bin
        move_card(fb)

        # Record processing time
        t = time.time() - start


def update_gui():
    """
    Periodically update the GUI with the latest processed image and text output.

    Uses global variables:
        processed_photo: latest image from Sorter()
        ip: model/OCR output
        fb: selected bin
        t: processing time
    """
    global processed_photo, img_display, txt_display, viewer, ip, fb, t

    if stop_event.is_set():
        return

    if processed_photo is not None:
        # Convert PIL image to Tkinter PhotoImage
        tk_photo = ImageTk.PhotoImage(processed_photo)
        img_display.config(text="Cards", image=tk_photo)
        img_display.image = tk_photo

        # Update text display with current output, bin, and time
        txt_display.config(text=f"Output: {ip}\nBin: {fb}\nTime: {t:.2f} s")

    # Schedule next update
    root.after(1000, update_gui)


def handle_button_click():
    """
    Handle the 'Confirm Selection' button click in the main GUI.

    Steps:
        1. Hide the main window.
        2. Open a new viewer window.
        3. Start the Sorter thread with selected method and game.
        4. Start periodic GUI updates.
    """
    global img_display, txt_display, viewer
    root.withdraw()
    viewer = tk.Toplevel(root)
    viewer.geometry("800x480")

    def on_close():
        """
        Handle closing of the viewer window:
            - Signal sorter thread to stop.
            - Destroy root window.
        """
        stop_event.set()
        root.destroy()

    viewer.protocol("WM_DELETE_WINDOW", on_close)

    # Image display label
    img_display = tk.Label(viewer, text="Cards")
    img_display.pack(anchor="w", padx=10)

    # Text display label
    txt_display = tk.Label(viewer, text="Processing")
    txt_display.pack(anchor="ne")

    # Read selected method and game from radio buttons
    method = method_var.get()
    mode = game_var.get()

    # Start sorter in a background thread
    threading.Thread(
        target=Sorter,
        args=(method, 7, mode),
        daemon=True
    ).start()

    # Start GUI update loop
    update_gui()


def update_method_options():
    """
    Update the available sorting method options based on selected game.

    For MTG:
        - "name", "type", "color"

    For Pokémon:
        - "name", "type"
    """
    # Clear existing method radio buttons
    for widget in method_frame.winfo_children():
        widget.destroy()

    mode = game_var.get()

    if mode == "MTG":
        options = ["name", "type", "color"]
    else:
        options = ["name", "type"]

    # Default to first option
    method_var.set(options[0])

    # Label for method options
    tk.Label(
        method_frame,
        text=f"{mode} Options:",
        font=("arial", 12, "bold")
    ).pack(pady=5)

    # Create radio buttons for each method
    for opt in options:
        tk.Radiobutton(
            method_frame,
            text=opt,
            variable=method_var,
            value=opt,
            font=("arial", 11)
        ).pack(anchor="w", padx=40)


if __name__ == "__main__":
    # Initialize all servos to neutral position at startup
    for p in pins:
        Servo(p)

    # Create main Tkinter window
    root = tk.Tk()
    root.title("Card Sorter")
    root.geometry("800x480")

    # Game selection variable
    game_var = tk.StringVar(value="MTG")
    # Method selection variable
    method_var = tk.StringVar()

    # Game selection label
    tk.Label(
        root,
        text="Choose Game:",
        font=("arial", 12, "bold")
    ).pack(pady=10)

    # MTG radio button
    tk.Radiobutton(
        root,
        text="MTG",
        variable=game_var,
        value="MTG",
        font=("arial", 11),
        command=update_method_options
    ).pack(anchor="w", padx=50)

    # Pokémon radio button
    tk.Radiobutton(
        root,
        text="Pokémon",
        variable=game_var,
        value="Pokemon",
        font=("arial", 11),
        command=update_method_options
    ).pack(anchor="w", padx=50)

    # Frame for method options (name/type/color)
    method_frame = tk.Frame(root)
    method_frame.pack(pady=20)

    # Initialize method options based on default game
    update_method_options()

    # Confirm selection button
    tk.Button(
        root,
        text="Confirm Selection",
        command=handle_button_click,
        bg="white"
    ).pack(pady=20)

    # Start Tkinter main loop
    root.mainloop()

    # Cleanup GPIO on exit
    GPIO.cleanup()
