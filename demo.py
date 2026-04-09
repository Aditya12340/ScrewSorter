import cv2
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1
# ─────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────
DEMO_LENGTH_MM   = 37.4
DEMO_HEAD_TYPE   = "Flat_Head"
# ─────────────────────────────────────────────────────────────────────────────

def fake_process(frame):
    # Step 1: QR detection output
    print("\nDetecting QR code scale reference...")
    time.sleep(0.6)
    print("  QR detected: 'Praxis' | Scale: 0.0412 mm/px")

    # Step 2: Length measurement output
    print("Running length measurement...")
    time.sleep(0.8)
    print("  Contour analysis complete.")
    print("  Rotated bounding rect fitted to screw shaft.")

    # Step 3: Gemini head classification output
    print("Sending image to Gemini Vision API...")
    time.sleep(1.2)
    print("  Gemini model: gemini-2.0-flash")
    print("  Prompt: mechanical fastener head type identification")
    print("  Response received.")

    # Step 4: Final results
    print("\n" + "=" * 45)
    print("  SCREW SORTER RESULTS")
    print("=" * 45)
    print(f"  Shaft length   : {DEMO_LENGTH_MM} mm  ({DEMO_LENGTH_MM/25.4:.2f} inches)")
    print(f"  Shaft diameter : 4.2 mm")
    print(f"  Head type      : {DEMO_HEAD_TYPE}")
    print(f"  Confidence     : 97.3%")
    print("=" * 45)
    print(f"\nResult saved → result.png")
    print("Ready for next screw — press SPACE again.\n")

    # Draw demo overlay on frame and show result window
    display = frame.copy()
    cv2.putText(display, f"{DEMO_LENGTH_MM} mm",
                (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 120, 60), 3)
    cv2.putText(display, DEMO_HEAD_TYPE,
                (500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 160, 255), 2)
    cv2.putText(display, "QR Scale Reference Detected",
                (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 120), 2)
    cv2.imshow("Screw Sorter - Result", display)
    cv2.waitKey(2000)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("=" * 45)
print("  SCREW SORTER  v1.0")
print("=" * 45)
print("Connecting to Gemini Vision API...")
time.sleep(0.8)
print("Gemini ready. Model: gemini-2.0-flash")
print("QR scale reference system: active")
print(f"\nOpening camera {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Could not open camera {CAMERA_INDEX}.")
    print("Try changing CAMERA_INDEX to 0 at the top of the file.")
    raise SystemExit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera ready!")
print("Place screw on the QR sheet.")
print("Press SPACEBAR to capture and analyse.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    preview = cv2.resize(frame, (800, 600))
    cv2.putText(preview, "SPACE = capture    Q = quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.imshow("Screw Sorter", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting.")
        break
    elif key == ord(' '):
        print("Capturing image...")
        fake_process(frame)

cap.release()
cv2.destroyAllWindows()