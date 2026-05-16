# src/collect_data.py
# ─────────────────────────────────────────────────────────────
# Interactive data collection with user-defined sample count
# and minimum hold-time enforcement per gesture.
#
# Flow:
#   1. Ask user how many samples per gesture (default 5)
#   2. For each gesture:
#      a. Show READY screen — user positions hand
#      b. SPACE to begin — 3-second stability countdown
#      c. Recording begins only after stable hold confirmed
#      d. Progress bar fills to target count
#      e. Auto-advance to next gesture
#
# Run: python src/collect_data.py
# ─────────────────────────────────────────────────────────────

import cv2
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from config import (
    GESTURES, FEATURE_DIM,
    DATA_CSV, DATA_DIR,
    CAM_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE
)
from hand_utils import (
    build_hand_detector, process_frame,
    get_landmark_list, extract_features
)

mp_draw        = mp.solutions.drawing_utils
mp_hands_mod   = mp.solutions.hands

# ── Timing constants ───────────────────────────────────────────
# Minimum seconds user must hold gesture BEFORE recording starts.
# This enforces the 2-3 second stable hold requirement.
PRE_RECORD_HOLD_SEC = 3.0

# Minimum seconds per gesture recording session regardless of count.
# Even if samples are collected fast, recording won't end before this.
MIN_RECORD_SEC      = 2.0

# Landmark drawing styles
STYLE_READY = mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3)
STYLE_REC   = mp_draw.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=3)
STYLE_CONN  = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)


# ── User input helpers ─────────────────────────────────────────

def ask_sample_count() -> int:
    """
    Ask user how many samples to collect per gesture.
    Validates input and shows a recommended range.
    Returns integer count.
    """
    print("\n" + "─" * 5)
    print("  How many samples per gesture?")
    print("  Recommended ranges:")
    print("    10   — quick test, lower accuracy")
    print("    20   — good balance")
    print("    50   — better robustness")
    print("    100   — high accuracy, takes longer")
    print("─" * 5)

    while True:
        raw = input("  Enter sample count [default: 5]: ").strip()
        if raw == "":
            return 5
        try:
            count = int(raw)
            if count < 5:
                print("  Minimum is 5. Try again.")
                continue
            if count > 100:
                print("  Maximum is 100. Try again.")
                continue
            return count
        except ValueError:
            print("  Enter a whole number.")


def ask_gesture_selection() -> list:
    """
    Ask user which gestures to collect.
    Allows collecting for specific gestures only
    (useful for topping up one weak class).
    """
    print("\n" + "─" * 5)
    print("  Which gestures to collect?")
    for i, g in enumerate(GESTURES):
        print(f"    {i+1}. {g}")
    print(f"    A. All gestures (default)")
    print("─" * 5)

    raw = input("  Enter numbers separated by comma, or A [default: A]: ").strip()
    if raw == "" or raw.upper() == "A":
        return GESTURES

    selected = []
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(GESTURES):
                selected.append(GESTURES[idx])
        except ValueError:
            pass

    if not selected:
        print("  Invalid selection — collecting all gestures.")
        return GESTURES

    print(f"  Selected: {', '.join(selected)}")
    return selected


# ── Drawing helpers ────────────────────────────────────────────

def draw_countdown_bar(frame, elapsed: float, total: float, label: str, color):
    """Draw a horizontal countdown/progress bar at top of frame."""
    h, w = frame.shape[:2]
    pct  = min(elapsed / total, 1.0)
    fill = int(pct * (w - 40))

    # Background
    cv2.rectangle(frame, (20, 8), (w - 20, 26), (40, 40, 40), -1)
    # Fill
    cv2.rectangle(frame, (20, 8), (20 + fill, 26), color, -1)
    # Label
    cv2.putText(frame, label,
                (24, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (240, 240, 240), 1)


def draw_sample_bar(frame, count: int, target: int):
    """Draw sample collection progress bar at bottom of frame."""
    h, w  = frame.shape[:2]
    pct   = count / max(target, 1)
    fill  = int(pct * (w - 40))

    cv2.rectangle(frame, (20, h - 28), (w - 20, h - 12), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, h - 28), (20 + fill, h - 12), (0, 200, 100), -1)
    cv2.putText(frame, f"Samples: {count} / {target}",
                (24, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1)


def draw_status(frame, line1: str, line2: str, color1, color2=(180,180,180)):
    """Draw two status lines in the middle of the frame."""
    h, w = frame.shape[:2]
    cv2.putText(frame, line1,
                (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, 2)
    if line2:
        cv2.putText(frame, line2,
                    (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color2, 1)


# ── Phase 1 — READY screen ─────────────────────────────────────

def phase_ready(gesture_name: str, cap, detector) -> bool:
    """
    Show READY screen. User positions their hand.
    Press SPACE to begin the stability countdown.
    Press Q to quit entirely.

    Returns True if user pressed SPACE, False if quit.
    """
    print(f"\n  Gesture: '{gesture_name}'")
    print("  Position your hand. Press SPACE when ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        # Draw skeleton if hand visible
        if result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS,
                STYLE_READY, STYLE_CONN
            )

        detected     = lms is not None
        detect_color = (0, 200, 100) if detected else (60, 60, 220)
        detect_text  = "Hand detected" if detected else "No hand — move into frame"

        draw_status(
            frame,
            f"GESTURE: {gesture_name.upper()}",
            detect_text,
            (255, 255, 255),
            detect_color
        )

        cv2.putText(frame,
                    "SPACE = start  |  Q = quit",
                    (20, frame.shape[0] - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (140, 140, 140), 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not detected:
                # Don't allow starting without a visible hand
                cv2.putText(frame, "Show your hand first!",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 80, 220), 2)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(800)
                continue
            return True

        if key == ord('q'):
            return False


# ── Phase 2 — STABILITY COUNTDOWN ─────────────────────────────

def phase_stability(gesture_name: str, cap, detector) -> bool:
    """
    Enforce PRE_RECORD_HOLD_SEC seconds of stable hand holding
    before recording begins.

    The timer resets if:
      - Hand disappears from frame
      - Hand is detected but MediaPipe loses tracking

    This enforces the 2-3 second stable hold requirement.
    Returns True when hold complete, False if user quit.
    """
    print(f"  Hold gesture steady for {PRE_RECORD_HOLD_SEC:.0f} seconds...")

    hold_start  = None
    hold_elapsed= 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        if lms is not None:
            # Hand visible — start or continue timer
            if hold_start is None:
                hold_start = time.time()
            hold_elapsed = time.time() - hold_start

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
                STYLE_CONN
            )
        else:
            # Hand lost — reset timer
            hold_start   = None
            hold_elapsed = 0.0

        # Countdown bar (fills left to right)
        remaining = PRE_RECORD_HOLD_SEC - hold_elapsed
        bar_color = (0, 200, 255)
        draw_countdown_bar(
            frame, hold_elapsed, PRE_RECORD_HOLD_SEC,
            f"Hold steady... {max(remaining, 0):.1f}s",
            bar_color
        )

        draw_status(
            frame,
            f"STABILISING: {gesture_name.upper()}",
            "Keep gesture perfectly still" if lms else "Hand lost — reposition!",
            (0, 200, 255),
            (180, 180, 180) if lms else (60, 60, 220)
        )

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return False

        if hold_elapsed >= PRE_RECORD_HOLD_SEC:
            print("  Stability confirmed. Recording...")
            return True


# ── Phase 3 — RECORDING ───────────────────────────────────────

def phase_record(
    gesture_name:  str,
    target_count:  int,
    cap,
    detector,
    csv_writer
) -> int:
    """
    Record target_count samples for one gesture.

    Rules:
      - Only records frames where MediaPipe detects a hand
      - Enforces MIN_RECORD_SEC minimum duration
        (won't finish even if count hit before time is up)
      - Shows live progress bar
      - Press Q to stop early

    Returns number of samples actually saved.
    """
    count      = 0
    rec_start  = time.time()

    while count < target_count:
        ret, frame = cap.read()
        if not ret:
            continue

        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        elapsed   = time.time() - rec_start
        time_done = elapsed >= MIN_RECORD_SEC
        count_done= count >= target_count

        if lms is not None:
            features = extract_features(lms)
            csv_writer.writerow([gesture_name] + features.tolist())
            count += 1

            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS,
                STYLE_REC, STYLE_CONN
            )

        # Recording bar (top) — time elapsed
        time_color = (0, 200, 100) if time_done else (0, 160, 255)
        draw_countdown_bar(
            frame, elapsed, max(MIN_RECORD_SEC, 1),
            f"Recording time: {elapsed:.1f}s  (min {MIN_RECORD_SEC:.0f}s)",
            time_color
        )

        # Sample bar (bottom)
        draw_sample_bar(frame, count, target_count)

        # Status
        status_color = (0, 200, 100) if lms else (60, 60, 220)
        status_text  = "RECORDING" if lms else "RECORDING — no hand!"
        draw_status(
            frame,
            f"{status_text}: {gesture_name.upper()}",
            f"Collected {count}/{target_count}  |  Time: {elapsed:.1f}s",
            (0, 160, 255),
            status_color
        )

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print(f"  Stopped early at {count} samples.")
            break

        # Only finish when BOTH conditions met:
        # count reached AND minimum time elapsed
        if count_done and time_done:
            break

    elapsed = time.time() - rec_start
    print(f"  Saved {count} samples in {elapsed:.1f}s")
    return count


# ── Summary ────────────────────────────────────────────────────

def show_summary(gesture_counts: dict):
    """Print collection summary and dataset state."""
    import pandas as pd

    print("\n" + "─" * 5)
    print("  Session summary")
    print("─" * 5)
    for g, c in gesture_counts.items():
        print(f"  {g:12s}: {c:4d} new samples collected")

    if os.path.exists(DATA_CSV):
        df    = pd.read_csv(DATA_CSV)
        print("\n  Dataset totals after this session:")
        counts = df['label'].value_counts()
        for label, count in counts.sort_index().items():
            bar = "█" * min(int(count / 100), 40)
            print(f"  {label:12s}: {count:6,}  {bar}")
        print(f"\n  Total rows: {len(df):,}")
        print(f"  Features  : {len(df.columns)-1}")

    print("─" * 5)


# ── CSV init ───────────────────────────────────────────────────

def init_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        with open(DATA_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'f{i}' for i in range(FEATURE_DIM)]
            writer.writerow(header)
        print(f"Created new dataset: {DATA_CSV}")
    else:
        print(f"Appending to existing dataset: {DATA_CSV}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("═" * 5)
    print("  Gesture Car — Data Collection")
    print("═" * 5)

    init_csv()

    # Ask user for configuration
    target_count      = ask_sample_count()
    selected_gestures = ask_gesture_selection()

    print(f"\n  Collecting {target_count} samples × {len(selected_gestures)} gestures")
    print(f"  Minimum hold time before recording : {PRE_RECORD_HOLD_SEC:.0f}s")
    print(f"  Minimum recording time per gesture : {MIN_RECORD_SEC:.0f}s")
    print(f"  Gestures: {', '.join(selected_gestures)}")
    input("\n  Press ENTER to open camera...")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAM_INDEX}.")
        print("Change CAM_INDEX in config.py and try again.")
        return

    detector      = build_hand_detector(
        MP_DETECTION_CONFIDENCE,
        MP_TRACKING_CONFIDENCE
    )
    gesture_counts= {}

    try:
        with open(DATA_CSV, 'a', newline='') as f:
            writer = csv.writer(f)

            for i, gesture in enumerate(selected_gestures):
                print(f"\n{'─'*5}")
                print(f"  Gesture {i+1}/{len(selected_gestures)}: {gesture.upper()}")
                print(f"{'─'*5}")

                # Phase 1: Ready
                ok = phase_ready(gesture, cap, detector)
                if not ok:
                    print("  Quit by user.")
                    break

                # Phase 2: Stability countdown
                ok = phase_stability(gesture, cap, detector)
                if not ok:
                    print("  Quit by user.")
                    break

                # Phase 3: Record
                saved = phase_record(gesture, target_count, cap, detector, writer)
                gesture_counts[gesture] = saved
                f.flush()

                # Brief pause between gestures
                if i < len(selected_gestures) - 1:
                    print(f"\n  Next gesture in 2 seconds...")
                    time.sleep(2)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    show_summary(gesture_counts)
    print("\n  Next step: python src/train_model.py")


if __name__ == "__main__":
    main()
