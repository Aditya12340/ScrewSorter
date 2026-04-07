import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1        # 0 = laptop webcam, 1 = USB camera
QR_SIZE_MM   = 50.8     # measure your printed QR with a ruler and set this
SAVE_PATH    = r"C:\Users\Navneet\Documents\ScrewSorter\result.png"
# ─────────────────────────────────────────────────────────────────────────────

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


def measure_screw(img):
    data, points = detect_qr(img)
    if points is None:
        print("  QR not detected — make sure QR is fully visible.")
        return None

    pts        = points[0]
    sides_px   = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    qr_size_px = np.mean(sides_px)
    mm_per_px  = QR_SIZE_MM / qr_size_px
    qr_cx      = np.mean(pts[:, 0])
    qr_cy      = np.mean(pts[:, 1])
    print(f"  QR detected: '{data}' | Scale: {mm_per_px:.4f} mm/px")

    lab       = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

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
        print("  Could not detect screw.")
        print("  Tips: use plain WHITE background, avoid shadows, shoot from directly above.")
        return None

    (rx, ry), (rw, rh), angle = best_rect
    shaft_length_mm = max(rw, rh) * mm_per_px
    shaft_width_mm  = min(rw, rh) * mm_per_px

    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.polylines(result, [pts.astype(int)], True, (80, 200, 120), 3)
    cv2.putText(result, f"QR={QR_SIZE_MM}mm",
                (int(pts[0][0]), int(pts[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 120), 2)

    box_pts = cv2.boxPoints(best_rect).astype(int)
    cv2.drawContours(result, [box_pts], 0, (255, 120, 60), 3)
    cv2.putText(result, f"{shaft_length_mm:.1f}mm",
                (int(rx) - 60, int(ry) - int(max(rw, rh) / 2) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 120, 60), 3)

    plt.figure(figsize=(10, 8))
    plt.imshow(result)
    plt.title(f"Length: {shaft_length_mm:.1f} mm  |  Diameter: {shaft_width_mm:.1f} mm")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Result image saved to: {SAVE_PATH}")

    return {
        "length_mm" : shaft_length_mm,
        "width_mm"  : shaft_width_mm,
        "angle"     : angle,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
print("="*45)
print("  SCREW SORTER")
print("="*45)
print(f"Opening camera {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Could not open camera {CAMERA_INDEX}.")
    print("Try changing CAMERA_INDEX to 0 or 2 at the top of the script.")
    raise SystemExit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera open!")
print("Place screw on the QR sheet.")
print("Press SPACEBAR to capture and measure.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed — check USB connection.")
        break

    preview = cv2.resize(frame, (800, 600))
    cv2.putText(preview, "SPACE = capture    Q = quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.imshow("Screw Sorter - Live Preview", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quitting.")
        break

    elif key == ord(' '):
        print("\nCapturing...")
        result = measure_screw(frame)

        if result:
            print("\n" + "="*45)
            print("  MEASUREMENT RESULTS")
            print("="*45)
            print(f"  Shaft length   : {result['length_mm']:.1f} mm")
            print(f"  Shaft diameter : {result['width_mm']:.1f} mm")
            print(f"  Screw angle    : {result['angle']:.1f} degrees")
            print("="*45)
            print("\nReady for next screw — press SPACE again.\n")

cap.release()
cv2.destroyAllWindows()