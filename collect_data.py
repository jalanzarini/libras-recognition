"""
collect_data.py — LIBRAS Alphabet Data Collection

Guided webcam script to record personal hand-sign samples.

Controls:
  SPACE   → capture current frame
  n       → skip to next letter (or finish early for current letter)
  r       → restart current letter (discard captured frames this run)
  q       → quit immediately

Output:
  data/personal/<LETTER>/frame_XXXX.jpg  (hand crop, or full frame fallback)
"""

import cv2
import mediapipe as mp
import os
import sys
import time

# ── Configuration ──────────────────────────────────────────────────────────────

# LIBRAS static alphabet (J and Z require motion — Phase 2)
LETTERS = list("ABCDEFGHILMNOPRSTU VWY")
LETTERS = [l for l in LETTERS if l.strip()]

SAMPLES_PER_LETTER = 50          # target frames per letter
CROP_PADDING       = 30          # pixels of padding around the detected hand bbox
OUTPUT_DIR         = os.path.join("data", "personal")
CAMERA_INDEX       = 0           # change to 1, 2… if default webcam is wrong

# ── MediaPipe setup ────────────────────────────────────────────────────────────

mp_hands        = mp.solutions.hands
mp_drawing      = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_hand_crop(frame, hand_landmarks, padding: int = CROP_PADDING):
    """Return a cropped region around the detected hand (with padding).
    Falls back to the full frame if the bbox is degenerate."""
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min = max(0, int(min(xs)) - padding)
    y_min = max(0, int(min(ys)) - padding)
    x_max = min(w, int(max(xs)) + padding)
    y_max = min(h, int(max(ys)) + padding)

    if x_max <= x_min or y_max <= y_min:
        return frame  # degenerate bbox → full frame fallback

    return frame[y_min:y_max, x_min:x_max]


def draw_overlay(frame, letter: str, captured: int, target: int, hand_detected: bool):
    """Draw HUD onto the frame in-place."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Letter: {letter}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * min(captured / target, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 80), -1)
    cv2.putText(frame, f"{captured}/{target}", (bar_x + 60, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Hand-detection status
    color = (0, 220, 0) if hand_detected else (0, 60, 220)
    text  = "Hand detected" if hand_detected else "No hand — show your hand"
    cv2.putText(frame, text, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cv2.putText(frame, "SPACE: capture | n: next | r: restart | q: quit",
                (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def collect_letter(cap, hands, letter: str, out_dir: str, target: int) -> bool:
    """Interactive capture loop for one letter.
    Returns True → next letter, False → quit."""
    os.makedirs(out_dir, exist_ok=True)

    existing   = [f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    frame_idx  = len(existing)
    captured   = 0

    print(f"\n── Letter '{letter}' ──  target: {target}  (already have: {frame_idx})")
    print("   Show your hand and press SPACE to capture.")

    while captured < target:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from webcam.")
            return False

        frame   = cv2.flip(frame, 1)          # mirror for natural signing view
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = results.multi_hand_landmarks is not None

        if hand_detected:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        draw_overlay(frame, letter, captured, target, hand_detected)
        cv2.imshow("LIBRAS Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n[QUIT] Exiting.")
            return False

        if key == ord("n"):
            print(f"   Skipping '{letter}' early ({captured}/{target} captured).")
            return True

        if key == ord("r"):
            # Discard frames captured in this run only
            print(f"   Restarting '{letter}'. Removing {captured} frame(s) from this run.")
            for i in range(frame_idx, frame_idx + captured):
                path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
                if os.path.exists(path):
                    os.remove(path)
            existing  = [f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg")]
            frame_idx = len(existing)
            captured  = 0
            continue

        if key == ord(" "):
            crop = get_hand_crop(frame, results.multi_hand_landmarks[0]) if hand_detected else frame
            save_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, crop)
            captured  += 1
            frame_idx += 1
            print(f"   Saved {save_path}  [{captured}/{target}]")

    print(f"   ✓ '{letter}' complete — {captured} samples saved.")
    return True


def show_countdown(cap, letter: str, seconds: int = 3) -> bool:
    """Display a get-ready countdown before a new letter. Returns False if user quits."""
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame     = cv2.flip(frame, 1)
        remaining = int(seconds - (time.time() - start)) + 1
        h, w      = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, f"Next letter: {letter}", (w // 2 - 160, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 220, 80), 4, cv2.LINE_AA)
        cv2.imshow("LIBRAS Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def print_summary():
    print("\n── Collection Summary " + "─" * 35)
    total = 0
    for letter in LETTERS:
        d     = os.path.join(OUTPUT_DIR, letter)
        count = len([f for f in os.listdir(d) if f.endswith(".jpg")]) if os.path.isdir(d) else 0
        mark  = "✓" if count >= SAMPLES_PER_LETTER else f"⚠  {count}/{SAMPLES_PER_LETTER}"
        print(f"   {letter}: {mark}")
        total += count
    print(f"\n   Total images: {total}")
    print("─" * 57 + "\n")


def main():
    print("=" * 57)
    print("  LIBRAS Data Collector")
    print(f"  Letters : {' '.join(LETTERS)}")
    print(f"  Target  : {SAMPLES_PER_LETTER} samples per letter")
    print(f"  Output  : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 57)

    # Optional CLI arg: start from a specific letter, e.g. `python collect_data.py G`
    start_letter = sys.argv[1].upper() if len(sys.argv) > 1 else None
    if start_letter and start_letter not in LETTERS:
        print(f"[ERROR] '{start_letter}' not in letter list: {LETTERS}")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}. "
              "Try changing CAMERA_INDEX at the top of this script.")
        sys.exit(1)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        skipping = start_letter is not None

        for letter in LETTERS:
            if skipping:
                if letter == start_letter:
                    skipping = False
                else:
                    continue

            out_dir  = os.path.join(OUTPUT_DIR, letter)
            os.makedirs(out_dir, exist_ok=True)

            # Skip letters that are already complete
            existing = [f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg")]
            if len(existing) >= SAMPLES_PER_LETTER:
                print(f"   '{letter}' already has {len(existing)} samples — skipping.")
                continue

            if not show_countdown(cap, letter):
                break

            if not collect_letter(cap, hands, letter, out_dir, SAMPLES_PER_LETTER):
                break

    cap.release()
    cv2.destroyAllWindows()
    print_summary()


if __name__ == "__main__":
    main()
