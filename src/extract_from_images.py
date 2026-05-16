# src/extract_from_images.py
# ─────────────────────────────────────────────────────────────
# Extract landmarks from pre-filtered images in external_images/
#
# Run AFTER filter_images.py — all images here are already
# quality-verified. No filtering logic needed here.
# This file has one job: MediaPipe → normalize → CSV.
#
# Run: python src/extract_from_images.py
# ─────────────────────────────────────────────────────────────

import cv2
import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GESTURES, DATA_CSV, DATA_DIR, FEATURE_DIM
from hand_utils import extract_features

import mediapipe as mp

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

IMAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external_images"
)


def init_csv():
    """Create CSV with header if it does not exist. Append if it does."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        with open(DATA_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label'] + [f'f{i}' for i in range(FEATURE_DIM)])
        print(f"Created new dataset: {DATA_CSV}")
    else:
        print(f"Appending to existing dataset: {DATA_CSV}")


def process_gesture(gesture: str, detector, csv_writer) -> dict:
    """
    Extract landmarks from all images in one gesture folder.

    No quality gates here — filter_images.py already guaranteed
    every image in this folder has a detectable, well-positioned hand.

    Returns stats dict for the final report.
    """
    folder = os.path.join(IMAGE_ROOT, gesture)

    if not os.path.exists(folder):
        print(f"\n  [SKIP] Folder not found: {folder}")
        return {"gesture": gesture, "saved": 0, "skipped": 0, "total": 0}

    all_files = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT
    ]

    total   = len(all_files)
    saved   = 0
    skipped = 0   # residual: images filter_images missed (corrupt file etc)
    t0      = time.time()

    if total == 0:
        print(f"\n  [SKIP] No images in: {folder}")
        return {"gesture": gesture, "saved": 0, "skipped": 0, "total": 0}

    print(f"\n  '{gesture}': {total:,} images")

    for i, fpath in enumerate(all_files):

        # Load
        img = cv2.imread(fpath)
        if img is None:
            skipped += 1
            continue

        # MediaPipe
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        # Skip if no hand — should be rare after filter_images.py
        if (result.multi_hand_landmarks is None or
                len(result.multi_hand_landmarks) == 0):
            skipped += 1
            continue

        # Extract + normalize
        lms      = result.multi_hand_landmarks[0].landmark
        features = extract_features(lms)

        # Write row
        csv_writer.writerow([gesture] + features.tolist())
        saved += 1

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 0.001)
            eta     = (total - i - 1) / max(rate, 0.001)
            print(f"    {i+1:5,}/{total:,}  "
                  f"saved={saved:,}  "
                  f"skipped={skipped}  "
                  f"rate={rate:.0f} img/s  "
                  f"ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"    Done — saved={saved:,}  skipped={skipped}  "
          f"time={elapsed:.0f}s")

    return {
        "gesture": gesture,
        "saved":   saved,
        "skipped": skipped,
        "total":   total,
        "time_s":  elapsed,
    }


def print_report(results: list[dict]):
    print(f"\n{'═'*55}")
    print(f"  EXTRACTION REPORT")
    print(f"{'═'*55}")

    total_saved   = sum(r["saved"]   for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_images  = sum(r["total"]   for r in results)
    total_time    = sum(r.get("time_s", 0) for r in results)

    for r in results:
        if r["total"] == 0:
            continue
        pct = r["saved"] / max(r["total"], 1) * 100
        print(f"  {r['gesture']:12s}: "
              f"saved={r['saved']:>5,}  "
              f"skipped={r['skipped']:>3}  "
              f"({pct:.1f}%)")

    print(f"{'─'*55}")
    print(f"  Total images     : {total_images:>8,}")
    print(f"  Total saved rows : {total_saved:>8,}")
    print(f"  Total skipped    : {total_skipped:>8,}")
    print(f"  Total time       : {total_time:.0f}s "
          f"({total_time/60:.1f} min)")

    # Class balance check
    counts = [r["saved"] for r in results if r["total"] > 0]
    if counts:
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 1.5:
            print(f"\n  WARNING: Class imbalance detected "
                  f"(ratio {ratio:.1f}x).")
            print(f"  Consider re-running filter_images.py with a "
                  f"lower KEEP_PER_GESTURE,")
            print(f"  or collect more webcam samples for smaller classes.")
        else:
            print(f"\n  Class balance: GOOD (ratio {ratio:.2f}x)")

    print(f"\n  CSV saved → {DATA_CSV}")
    print(f"  Next: python src/collect_data.py  (add webcam samples)")
    print(f"  Then: python src/train_model.py")
    print(f"{'═'*55}")


def main():
    print("═" * 55)
    print("  Gesture Car — Landmark Extractor")
    print("  (images pre-filtered by filter_images.py)")
    print("═" * 55)

    if not os.path.exists(IMAGE_ROOT):
        print(f"\nERROR: {IMAGE_ROOT} not found.")
        print("Run filter_images.py first.")
        return

    init_csv()

    # Static image mode — no temporal tracking between images
    detector = mp.solutions.hands.Hands(
        static_image_mode        = True,
        max_num_hands            = 1,
        min_detection_confidence = 0.5,
        model_complexity         = 1
    )

    results = []

    with open(DATA_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        for gesture in GESTURES:
            result = process_gesture(gesture, detector, writer)
            results.append(result)
            f.flush()   # flush after each gesture — safe against crashes

    detector.close()
    print_report(results)


if __name__ == "__main__":
    main()