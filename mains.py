import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os
from PIL import Image
from google import genai
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 1
QR_SIZE_MM     = 21.9  
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)
SAVE_PATH      = r"C:\Users\Navneet\Documents\ScrewSorter\result.png"
MODEL_ID = "gemini-2.5-flash"
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD GEMINI (New SDK) ─────────────────────────────────────────────────────
print("Connecting to Gemini...")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini ready.\n")
except Exception as e:
    print(f"Failed to connect: {e}")
    exit()

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

    H, W = img.shape[:2]

    # ── CROP around QR + screw area ───────────────────────────────────────────
    qr_xmin = int(np.min(pts[:, 0]))
    qr_xmax = int(np.max(pts[:, 0]))
    qr_ymin = int(np.min(pts[:, 1]))
    qr_ymax = int(np.max(pts[:, 1]))

    crop_x1 = max(0, int(qr_xmin - qr_size_px * 0.1))
    crop_x2 = min(W, int(qr_xmax + qr_size_px * 4.0))
    crop_y1 = max(0, int(qr_ymin - qr_size_px * 1.2))
    crop_y2 = min(H, int(qr_ymax + qr_size_px * 1.2))

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_h, crop_w = cropped.shape[:2]

    # QR centre in cropped coords (for exclusion zone)
    qr_cx_crop = qr_cx - crop_x1
    qr_cy_crop = qr_cy - crop_y1
    print(f"  Crop search area: {crop_w}x{crop_h} px")

    # ── Threshold on cropped image ────────────────────────────────────────────
    lab       = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    blur      = cv2.GaussianBlur(l_channel, (5, 5), 0)

    white_level     = np.percentile(blur, 90)
    threshold_level = max(120, white_level * 0.75)
    _, thresh1 = cv2.threshold(blur, threshold_level, 255, cv2.THRESH_BINARY_INV)
    thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)
    thresh = cv2.bitwise_or(thresh1, thresh2)

    qr_mask = np.zeros_like(thresh)
    qr_pts_crop = np.int32(pts - [crop_x1, crop_y1])
    cv2.fillPoly(qr_mask, [qr_pts_crop], 255)
    thresh[qr_mask > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Contours found: {len(contours)}")

    best_score   = -1
    best_contour = None
    best_rect    = None

    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Skip contours near the QR
        if np.sqrt((cx - qr_cx_crop)**2 + (cy - qr_cy_crop)**2) < qr_size_px * 0.8:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w > crop_w * 0.9 or h > crop_h * 0.9:
            continue
        if x <= 2 or y <= 2 or x + w >= crop_w - 2 or y + h >= crop_h - 2:
            continue

        rect = cv2.minAreaRect(c)
        (rx, ry), (rw, rh), angle = rect
        length_px = max(rw, rh)
        width_px  = min(rw, rh)
        if width_px < 1:
            continue

        ratio     = length_px / width_px
        length_mm = length_px * mm_per_px

        if length_mm < 5.0 or length_mm > 150.0:
            continue
        if ratio < 2.0:
            continue

        score = ratio * length_mm
        if score > best_score:
            best_score   = score
            best_contour = c
            best_rect    = rect

    if best_contour is None:
        print("  Could not detect screw in cropped region.")
        return None, None, pts, None

    (rx, ry), (rw, rh), angle = best_rect
    shaft_length_mm = max(rw, rh) * mm_per_px
    shaft_width_mm  = min(rw, rh) * mm_per_px

    # Convert best_rect back to full image coordinates for drawing
    best_rect_full = ((rx + crop_x1, ry + crop_y1), (rw, rh), angle)

    return shaft_length_mm, shaft_width_mm, pts, best_rect_full

# ── HEAD SHAPE CLASSIFICATION (With Rate Limit Handling) ─────────────────────
def classify_head(img, retries=3):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    prompt = """Identify the screw head type. Reply ONLY with one:
Flat_Head, Oval_Head, Round_Washer."""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt, pil_img]
            )
            return response.text.strip().replace(".", "").split()[0]
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"   Quota hit. Retrying in 5s... (Attempt {attempt+1}/{retries})")
                time.sleep(5)
            else:
                raise e
    return "Unknown"

# ── PROCESS FRAME ─────────────────────────────────────────────────────────────
def process(frame):
    print("\nRunning length measurement...")
    shaft_length_mm, shaft_width_mm, qr_pts, best_rect = measure_length(frame)

    print("Running head shape classification (Gemini)...")
    try:
        head_type = classify_head(frame)
    except Exception as e:
        print(f"   Gemini error: {e}")
        head_type = "Unknown"

    print("\n" + "=" * 45)
    print("   SCREW SORTER RESULTS")
    print("=" * 45)
    if shaft_length_mm is not None:
        print(f"   Shaft length   : {shaft_length_mm:.1f} mm  ({shaft_length_mm/25.4:.2f} inches)")
        print(f"   Shaft diameter : {shaft_width_mm:.1f} mm")
    else:
        print("   Shaft length   : Could not measure")
    print(f"   Head type      : {head_type}")
    print("=" * 45)

    result_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if qr_pts is not None:
        cv2.polylines(result_img, [qr_pts.astype(int)], True, (80, 200, 120), 3)

    if best_rect is not None:
        (rx, ry), (rw, rh), angle = best_rect
        box_pts = cv2.boxPoints(best_rect).astype(int)
        cv2.drawContours(result_img, [box_pts], 0, (255, 120, 60), 3)
        if shaft_length_mm:
            cv2.putText(result_img, f"{shaft_length_mm:.1f}mm",
                        (int(rx) - 50, int(ry) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 120, 60), 2)

    cv2.putText(result_img, head_type, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 160, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(result_img)
    plt.title(f"Head: {head_type} | Length: {shaft_length_mm:.1f} mm" if shaft_length_mm else f"Head: {head_type}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nResult saved → {SAVE_PATH}\nReady for next screw.\n")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("=" * 45)
print("   SCREW SORTER")
print("=" * 45)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Could not open camera {CAMERA_INDEX}.")
    raise SystemExit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera ready! Press SPACEBAR to capture, Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    preview = cv2.resize(frame, (800, 600))
    cv2.putText(preview, "SPACE = capture    Q = quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.imshow("Screw Sorter", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '):
        print("Capturing...")
        process(frame)

cap.release()
cv2.destroyAllWindows()