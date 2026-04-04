import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ── CONFIG ────────────────────────────────────────────────────────────────────
QR_SIZE_MM      = 50.8
CLASSIFIER_PATH = r"C:\Users\Navneet\Documents\ScrewSorter\screw_classifier.pt"
SAVE_PATH       = r"C:\Users\Navneet\Documents\ScrewSorter\result.png"
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD HEAD SHAPE CLASSIFIER ────────────────────────────────────────────────
print("Loading head shape classifier...")
ckpt         = torch.load(CLASSIFIER_PATH, map_location="cpu")
idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}

classifier = models.mobilenet_v2(weights=None)
classifier.classifier[1] = nn.Linear(classifier.last_channel, len(idx_to_class))
classifier.load_state_dict(ckpt["model_state_dict"])
classifier.eval()

infer_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
print(f"Classifier loaded. Classes: {list(idx_to_class.values())}\n")


# ── QR DETECTION ──────────────────────────────────────────────────────────────
def detect_qr(img):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if points is not None:
        return data, points
    scale = min(1.0, 1500 / max(img.shape[:2]))
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    data, points, _ = detector.detectAndDecode(small)
    if points is not None:
        return data, points / scale
    return None, None


# ── LENGTH MEASUREMENT ────────────────────────────────────────────────────────
def measure_length(img):
    data, points = detect_qr(img)
    if points is None:
        print("  QR not detected — make sure QR is fully visible.")
        return None, None, None, None

    pts        = points[0]
    sides_px   = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    qr_size_px = np.mean(sides_px)
    mm_per_px  = QR_SIZE_MM / qr_size_px
    qr_cx      = np.mean(pts[:, 0])
    qr_cy      = np.mean(pts[:, 1])

    lab             = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel       = lab[:, :, 0]
    white_level     = np.percentile(l_channel, 90)
    threshold_level = white_level * 0.80
    _, thresh = cv2.threshold(l_channel, threshold_level, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score   = -1
    best_contour = None
    best_rect    = None

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        if np.sqrt((cx - qr_cx)**2 + (cy - qr_cy)**2) < qr_size_px * 0.8:
            continue

        rect = cv2.minAreaRect(c)
        (rx, ry), (rw, rh), angle = rect
        length_px = max(rw, rh)
        width_px  = min(rw, rh)
        if width_px < 1:
            continue

        ratio     = length_px / width_px
        length_mm = length_px * mm_per_px

        if length_mm < 10.0 or length_mm > 200.0:
            continue
        if ratio < 2.5:
            continue

        score = ratio * length_mm
        if score > best_score:
            best_score   = score
            best_contour = c
            best_rect    = rect

    if best_contour is None:
        print("  Could not detect screw for length measurement.")
        return None, None, pts, None

    (rx, ry), (rw, rh), angle = best_rect
    shaft_length_mm = max(rw, rh) * mm_per_px
    shaft_width_mm  = min(rw, rh) * mm_per_px
    return shaft_length_mm, shaft_width_mm, pts, best_rect


# ── HEAD SHAPE CLASSIFICATION ─────────────────────────────────────────────────
def classify_head(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor  = infer_tfm(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx   = int(np.argmax(probs))
    head_type  = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
    return head_type, confidence, probs


# ── PROCESS IMAGE ─────────────────────────────────────────────────────────────
def process(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not open image: {image_path}")
        return

    print("\nRunning length measurement...")
    shaft_length_mm, shaft_width_mm, qr_pts, best_rect = measure_length(img)

    print("Running head shape classifier...")
    head_type, confidence, probs = classify_head(img)

    # ── Terminal output ───────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  SCREW SORTER RESULTS")
    print("=" * 45)

    if shaft_length_mm is not None:
        print(f"  Shaft length   : {shaft_length_mm:.1f} mm  ({shaft_length_mm/25.4:.2f} inches)")
        print(f"  Shaft diameter : {shaft_width_mm:.1f} mm")
    else:
        print("  Shaft length   : Could not measure")

    print(f"  Head type      : {head_type}")
    print(f"  Confidence     : {confidence*100:.1f}%")
    print("\n  All probabilities:")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 20)
        print(f"    {idx_to_class[i]:<20} {p*100:5.1f}%  {bar}")
    print("=" * 45)

    # ── Save annotated image ──────────────────────────────────────────────────
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if qr_pts is not None:
        cv2.polylines(result_img, [qr_pts.astype(int)], True, (80, 200, 120), 3)

    if best_rect is not None:
        (rx, ry), (rw, rh), angle = best_rect
        box_pts = cv2.boxPoints(best_rect).astype(int)
        cv2.drawContours(result_img, [box_pts], 0, (255, 120, 60), 3)
        if shaft_length_mm:
            cv2.putText(result_img, f"{shaft_length_mm:.1f}mm",
                        (int(rx) - 50, int(ry) - int(max(rw, rh) / 2) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 120, 60), 2)

    cv2.putText(result_img, f"{head_type} ({confidence*100:.0f}%)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 160, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(result_img)
    title = f"Head: {head_type} ({confidence*100:.0f}%)"
    if shaft_length_mm:
        title += f"   |   Length: {shaft_length_mm:.1f} mm"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nResult image saved → {SAVE_PATH}\n")


# ── MAIN: file picker dialog ──────────────────────────────────────────────────
print("=" * 45)
print("  SCREW SORTER")
print("=" * 45)

# Hide the tkinter root window
root = tk.Tk()
root.withdraw()

while True:
    print("\nOpening file picker — select a screw image...")
    image_path = filedialog.askopenfilename(
        title="Select screw image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )

    if not image_path:
        print("No file selected. Quitting.")
        break

    print(f"Selected: {image_path}")
    process(image_path)

    again = input("Analyse another image? (y/n): ").strip().lower()
    if again != 'y':
        print("Done.")
        break