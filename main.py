import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_PATH   = r"C:\Users\Navneet\Downloads\IMG_0417.jpeg"
QR_SIZE_MM   = 50.8    # measure your printed QR with a ruler and set this
DARK_ON_LIGHT = True   # True = dark screw on white/light background (your setup)
                       # False = light screw on dark background
# ─────────────────────────────────────────────────────────────────────────────

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot open: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
print(f"Image loaded: {W}x{H}px")

# ── DETECT QR ─────────────────────────────────────────────────────────────────
detector = cv2.QRCodeDetector()
data, points, _ = detector.detectAndDecode(img)

if points is None:
    print("QR not detected — make sure QR is fully visible and in focus.")
    raise SystemExit()

pts        = points[0]
sides_px   = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
qr_size_px = np.mean(sides_px)
mm_per_px  = QR_SIZE_MM / qr_size_px
qr_cx      = np.mean(pts[:, 0])
qr_cy      = np.mean(pts[:, 1])

print(f"QR detected: '{data}'")
print(f"Scale: {mm_per_px:.4f} mm/px")

# ── FIND SCREW VIA THRESHOLD ──────────────────────────────────────────────────
# Simple threshold works well for dark screw on white paper background
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if DARK_ON_LIGHT:
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
else:
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_score   = -1
best_contour = None
best_rect    = None

MIN_LENGTH_MM = 10.0
MAX_LENGTH_MM = 200.0
MIN_RATIO     = 2.5

for c in contours:
    if cv2.contourArea(c) < 500:
        continue

    # Skip anything near the QR code
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

    if length_mm < MIN_LENGTH_MM or length_mm > MAX_LENGTH_MM:
        continue
    if ratio < MIN_RATIO:
        continue

    score = ratio * length_mm
    if score > best_score:
        best_score   = score
        best_contour = c
        best_rect    = rect

if best_contour is None:
    print("\nCould not detect screw. Tips:")
    print("  - Use a plain white or black background")
    print("  - Make sure screw contrasts clearly with background")
    print("  - Shoot from directly above")
    raise SystemExit()

# ── MEASURE ───────────────────────────────────────────────────────────────────
(rx, ry), (rw, rh), angle = best_rect
shaft_length_mm = max(rw, rh) * mm_per_px
shaft_width_mm  = min(rw, rh) * mm_per_px

# ── VISUALISE ─────────────────────────────────────────────────────────────────
result = img_rgb.copy()

cv2.polylines(result, [pts.astype(int)], True, (80, 200, 120), 3)
cv2.putText(result, f"QR={QR_SIZE_MM}mm",
            (int(pts[0][0]), int(pts[0][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 120), 2)

box_pts = cv2.boxPoints(best_rect).astype(int)
cv2.drawContours(result, [box_pts], 0, (255, 120, 60), 3)
cv2.putText(result, f"{shaft_length_mm:.1f} mm",
            (int(rx) - 60, int(ry) - int(max(rw, rh) / 2) - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 120, 60), 2)

plt.figure(figsize=(10, 8))
plt.imshow(result)
plt.title(f"Length: {shaft_length_mm:.1f} mm  |  Diameter: {shaft_width_mm:.1f} mm")
plt.axis("off")
plt.tight_layout()
plt.show()

print("\n" + "="*45)
print("  MEASUREMENT RESULTS")
print("="*45)
print(f"  Scale          : {mm_per_px:.4f} mm/px")
print(f"  Shaft length   : {shaft_length_mm:.1f} mm")
print(f"  Shaft diameter : {shaft_width_mm:.1f} mm")
print(f"  Screw angle    : {angle:.1f} degrees")
print("="*45)