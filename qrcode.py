import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = r"C:\Users\Navneet\Downloads\IMG_0433.jpeg"
QR_SIZE_MM  = 50.8    # measure your printed QR with a ruler
# ─────────────────────────────────────────────────────────────────────────────

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot open: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
print(f"Image loaded: {W}x{H}px")


# ── DETECT QR ─────────────────────────────────────────────────────────────────
# Downscale first — helps QR detector on high-res phone images
scale_factor = min(1.0, 1500 / max(img.shape[:2]))
small = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

detector = cv2.QRCodeDetector()
data, points, _ = detector.detectAndDecode(small)

if points is None:
    print("QR not detected — make sure QR is fully visible and in focus.")
    raise SystemExit()

# Scale points back up to original image coordinates
points = points / scale_factor

pts        = points[0]
sides_px   = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
qr_size_px = np.mean(sides_px)
mm_per_px  = QR_SIZE_MM / qr_size_px
qr_cx      = np.mean(pts[:, 0])
qr_cy      = np.mean(pts[:, 1])

print(f"QR detected: '{data}'")
print(f"Scale: {mm_per_px:.4f} mm/px")

# ── BACKGROUND REMOVAL (works for ANY screw color on white paper) ─────────────
# Step 1: Convert to LAB color space — better at separating light/dark
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel = lab[:, :, 0]  # L = lightness

# Step 2: Automatically find the white background level
# White paper will be the brightest region — use the 90th percentile as reference
white_level = np.percentile(l_channel, 90)

# Step 3: Anything significantly darker than the white background is an object
# This adapts automatically regardless of screw color
threshold_level = white_level * 0.80  # 80% of white = cutoff
_, thresh = cv2.threshold(l_channel, threshold_level, 255, cv2.THRESH_BINARY_INV)

# Step 4: Clean up noise with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ── FIND SCREW ────────────────────────────────────────────────────────────────
best_score   = -1
best_contour = None
best_rect    = None

MIN_LENGTH_MM = 10.0
MAX_LENGTH_MM = 200.0
MIN_RATIO     = 2.5

for c in contours:
    if cv2.contourArea(c) < 500:
        continue

    M = cv2.moments(c)
    if M["m00"] == 0:
        continue
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Skip contours near the QR code
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
    print("  - Use a plain WHITE background (paper works great)")
    print("  - Avoid shadows across the screw")
    print("  - Shoot from directly above")
    print("  - Make sure the whole screw is in frame")
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
plt.savefig(r"C:\Users\Navneet\Documents\ScrewSorter\result.png", dpi=150, bbox_inches='tight')
print("Result saved to result.png")

print("\n" + "="*45)
print("  MEASUREMENT RESULTS")
print("="*45)
print(f"  Scale          : {mm_per_px:.4f} mm/px")
print(f"  Shaft length   : {shaft_length_mm:.1f} mm")
print(f"  Shaft diameter : {shaft_width_mm:.1f} mm")
print(f"  Screw angle    : {angle:.1f} degrees")
print("="*45)