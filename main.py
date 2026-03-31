import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_PATH = r"C:\Users\Navneet\Documents\ScrewSorter\IMG_0405.jpeg"  # change this
QR_SIZE_MM  = 54.0    # printed QR size in mm (measure with ruler after printing)
MIN_AREA    = 300     # ignore contours smaller than this
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD IMAGE ────────────────────────────────────────────────────────────────
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not open image: {IMAGE_PATH}\nCheck the path is correct.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Image loaded: {img.shape[1]}x{img.shape[0]}px")

# ── DETECT QR CODE ────────────────────────────────────────────────────────────
detector = cv2.QRCodeDetector()
data, points, _ = detector.detectAndDecode(img)

if points is None or len(points) == 0:
    print("\nQR code NOT detected. Tips:")
    print("  - Make sure the QR is fully in frame and not blurry")
    print("  - Try better lighting, no glare")
    print("  - Shoot from directly above")
    raise SystemExit("Stopping — fix QR detection first.")

print(f"QR detected! Content: '{data}'")

pts = points[0]  # 4 corner points, shape (4, 2)

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

sides_px   = [dist(pts[i], pts[(i+1)%4]) for i in range(4)]
qr_size_px = np.mean(sides_px)
mm_per_px  = QR_SIZE_MM / qr_size_px

print(f"QR size in image : {qr_size_px:.1f} px")
print(f"Scale            : {mm_per_px:.4f} mm/px")

# ── FIND SCREW CONTOURS ───────────────────────────────────────────────────────
gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur    = cv2.GaussianBlur(gray, (5, 5), 0)
edges   = cv2.Canny(blur, 30, 100)
kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges_d = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

qr_cx = np.mean(pts[:, 0])
qr_cy = np.mean(pts[:, 1])

def near_qr(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return False
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return dist((cx, cy), (qr_cx, qr_cy)) < qr_size_px * 0.7

big_contours = [
    c for c in contours
    if cv2.contourArea(c) > MIN_AREA and not near_qr(c)
]

# ── SHOW CANDIDATES ───────────────────────────────────────────────────────────
vis = img_rgb.copy()

# Draw QR outline
cv2.polylines(vis, [pts.astype(int)], True, (80, 200, 120), 2)
cv2.putText(vis, f"QR ({QR_SIZE_MM}mm)", (int(pts[0][0]), int(pts[0][1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 120), 2)

# Draw candidate contours
for i, c in enumerate(big_contours):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis, (x, y), (x+w, y+h), (100, 160, 255), 2)
    cv2.putText(vis, str(i), (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 160, 255), 2)

plt.figure(figsize=(10, 8))
plt.imshow(vis)
plt.title("Green = QR (scale)     Blue = candidates\nClose this window then type the screw index in the terminal")
plt.tight_layout()
plt.show()  # window stays open until you close it

# ── USER PICKS SCREW ──────────────────────────────────────────────────────────
print("\nCandidate contours:")
for i, c in enumerate(big_contours):
    x, y, w, h = cv2.boundingRect(c)
    print(f"  [{i}]  area={int(cv2.contourArea(c)):>6}  "
          f"size={w}x{h}px  ~{max(w,h)*mm_per_px:.1f}mm long")

screw_idx     = int(input("\nEnter index of the SCREW contour: "))
screw_contour = big_contours[screw_idx]

# ── MEASURE ───────────────────────────────────────────────────────────────────
rect  = cv2.minAreaRect(screw_contour)
(rx, ry), (rw, rh), angle = rect

shaft_length_mm = max(rw, rh) * mm_per_px
shaft_width_mm  = min(rw, rh) * mm_per_px

# ── FINAL RESULT IMAGE ────────────────────────────────────────────────────────
result = img_rgb.copy()

cv2.polylines(result, [pts.astype(int)], True, (80, 200, 120), 2)
cv2.putText(result, f"QR = {QR_SIZE_MM}mm",
            (int(pts[0][0]), int(pts[0][1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 120), 2)

box_pts = cv2.boxPoints(rect).astype(int)
cv2.drawContours(result, [box_pts], 0, (255, 120, 60), 2)
cv2.putText(result, f"{shaft_length_mm:.1f} mm",
            (int(rx) - 40, int(ry) - int(max(rw, rh) / 2) - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 120, 60), 2)

plt.figure(figsize=(10, 8))
plt.imshow(result)
plt.title(f"Length: {shaft_length_mm:.1f} mm  |  "
          f"Diameter: {shaft_width_mm:.1f} mm  |  "
          f"Scale: {mm_per_px:.4f} mm/px")
plt.axis("off")
plt.tight_layout()
plt.show()

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*45)
print("  MEASUREMENT RESULTS")
print("="*45)
print(f"  Scale          : {mm_per_px:.4f} mm/px")
print(f"  Shaft length   : {shaft_length_mm:.1f} mm")
print(f"  Shaft diameter : {shaft_width_mm:.1f} mm")
print(f"  Screw angle    : {angle:.1f} degrees")
print("="*45)