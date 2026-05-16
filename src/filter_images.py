# src/filter_images.py
# ─────────────────────────────────────────────────────────────
# ONE-TIME IMAGE QUALITY FILTER.
#
# Scans all 135,000+ images in external_images/ exactly once.
# Scores every image on 4 quality metrics.
# Keeps the top KEEP_PER_GESTURE best images per gesture folder.
# PERMANENTLY DELETES the rest from disk.
#
# After this runs:
#   - external_images/ contains only clean, high-quality images
#   - extract_from_images.py needs zero filtering logic
#   - Training is faster, cleaner, more accurate
#
# Run ONCE before extract_from_images.py:
#   python src/filter_images.py
#
# WARNING: This permanently deletes files. It asks for
# confirmation before deleting anything.
# ─────────────────────────────────────────────────────────────

import cv2
import os
import sys
import time
import shutil
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GESTURES
import mediapipe as mp

# ── Configuration ──────────────────────────────────────────────
# How many images to KEEP per gesture after filtering.
# 1000–2000 per gesture is the sweet spot:
#   - Enough diversity for a robust Random Forest
#   - Fast enough extraction and training
#   - Total: 10,000 images across 5 gestures
KEEP_PER_GESTURE = 2000

# MediaPipe detection confidence threshold.
# Images below this are automatically scored zero — likely bad.
MIN_DETECT_CONF  = 0.5

# Image extensions to process
SUPPORTED_EXT    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Scoring weights — must sum to 1.0
W_AREA        = 0.35   # hand fills enough of the frame
W_CONFIDENCE  = 0.30   # MediaPipe's own detection confidence
W_SPREAD      = 0.20   # landmark vector has rich information
W_CENTRALITY  = 0.15   # wrist is not at the edge

IMAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external_images"
)

# ── Score dataclass ────────────────────────────────────────────

@dataclass
class ImageScore:
    path:       str
    score:      float    # 0.0 – 1.0, higher = better
    detected:   bool     # whether a hand was found at all

# ── Scoring function ───────────────────────────────────────────

def score_image(img_path: str, detector) -> ImageScore:
    """
    Score one image on 4 quality metrics.
    Returns ImageScore with composite score in [0, 1].

    Score = 0.0 if no hand detected (will be deleted).
    Score > 0.0 means a hand was found — higher is better quality.

    Scoring breakdown:

    1. Hand area score (weight 0.35):
       Fraction of image covered by hand bounding box,
       clipped to [0, 0.5] then normalized to [0, 1].
       A hand covering 25%+ of image = perfect score.
       Why: landmark precision improves with hand size.

    2. Detection confidence (weight 0.30):
       MediaPipe's internal confidence for the hand detection.
       Already in [0, 1]. Direct signal from the detector itself.
       Why: if MediaPipe is uncertain, landmarks are uncertain.

    3. Landmark spread (weight 0.20):
       Max absolute value of the raw (pre-normalization) landmark
       deviation from wrist. Higher = fingers more spread out =
       more discriminative vector.
       Clipped and normalized to [0, 1].
       Why: collapsed landmark vectors are useless for classification.

    4. Wrist centrality (weight 0.15):
       How far the wrist (landmark 0) is from the nearest frame edge.
       Wrist at center = 1.0. Wrist at edge = 0.0.
       Why: edge wrists produce corrupted normalization anchors.
    """
    img = cv2.imread(img_path)
    if img is None:
        return ImageScore(path=img_path, score=0.0, detected=False)

    img_h, img_w = img.shape[:2]
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    # No hand — score zero
    if (result.multi_hand_landmarks is None or
            len(result.multi_hand_landmarks) == 0):
        return ImageScore(path=img_path, score=0.0, detected=False)

    # Multiple hands — ambiguous, score zero
    if len(result.multi_hand_landmarks) > 1:
        return ImageScore(path=img_path, score=0.0, detected=False)

    lms = result.multi_hand_landmarks[0].landmark

    # ── Metric 1: Hand area ────────────────────────────────────
    xs    = [lm.x for lm in lms]
    ys    = [lm.y for lm in lms]
    box_w = (max(xs) - min(xs))        # fraction of image width
    box_h = (max(ys) - min(ys))        # fraction of image height
    area  = box_w * box_h              # fraction of image area
    # Normalize: 0.25 (25% image) = perfect score
    area_score = min(area / 0.25, 1.0)

    # ── Metric 2: Detection confidence ────────────────────────
    # MediaPipe stores confidence in multi_handedness
    confidence = 0.0
    if result.multi_handedness:
        confidence = result.multi_handedness[0].classification[0].score
    conf_score = confidence   # already [0, 1]

    # ── Metric 3: Landmark spread ──────────────────────────────
    wrist_x = lms[0].x
    wrist_y = lms[0].y
    wrist_z = lms[0].z
    deviations = []
    for lm in lms:
        dx = lm.x - wrist_x
        dy = lm.y - wrist_y
        dz = lm.z - wrist_z
        deviations.extend([dx, dy, dz])

    max_dev = max(abs(v) for v in deviations) if deviations else 0.0
    # Normalize: 0.3 deviation (30% of image) = perfect score
    spread_score = min(max_dev / 0.30, 1.0)

    # ── Metric 4: Wrist centrality ─────────────────────────────
    wrist_nx = lms[0].x    # already in [0, 1]
    wrist_ny = lms[0].y
    # Distance from nearest edge (0 = at edge, 0.5 = at center)
    dist_from_edge = min(
        wrist_nx,
        1.0 - wrist_nx,
        wrist_ny,
        1.0 - wrist_ny
    )
    # Normalize: 0.2 from edge = perfect score
    centrality_score = min(dist_from_edge / 0.20, 1.0)

    # ── Composite score ────────────────────────────────────────
    composite = (
        W_AREA       * area_score       +
        W_CONFIDENCE * conf_score       +
        W_SPREAD     * spread_score     +
        W_CENTRALITY * centrality_score
    )

    return ImageScore(
        path     = img_path,
        score    = round(composite, 4),
        detected = True
    )
# ── Per-gesture filtering ──────────────────────────────────────

def filter_gesture_folder(
    gesture:  str,
    detector,
    keep:     int,
    dry_run:  bool
) -> dict:
    """
    Score all images in one gesture folder.
    Sort by score descending.
    Keep top `keep` images.
    Delete the rest permanently.

    Returns summary dict for reporting.
    """
    folder = os.path.join(IMAGE_ROOT, gesture)

    if not os.path.exists(folder):
        print(f"\n  [SKIP] Folder not found: {folder}")
        return {}

    all_files = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT
    ]

    total = len(all_files)
    if total == 0:
        print(f"\n  [SKIP] No images in: {folder}")
        return {}

    print(f"\n  '{gesture}': scoring {total:,} images...")
    t0      = time.time()
    scores  = []
    no_hand = 0
    errors  = 0

    for i, fpath in enumerate(all_files):
        result = score_image(fpath, detector)
        scores.append(result)

        if not result.detected:
            no_hand += 1

        # Progress every 1000 images
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 0.001)
            eta     = (total - i - 1) / max(rate, 0.001)
            print(f"    {i+1:6,}/{total:,}  "
                  f"rate={rate:.0f} img/s  "
                  f"ETA={eta:.0f}s  "
                  f"no_hand={no_hand:,}")

    # Sort by score — highest first
    scores.sort(key=lambda x: x.score, reverse=True)

    # Partition: keep top N with hand detected, delete rest
    keep_list   = [s for s in scores if s.detected][:keep]
    keep_paths  = {s.path for s in keep_list}
    delete_list = [s for s in scores if s.path not in keep_paths]

    deleted   = 0
    kept      = len(keep_list)
    del_bytes = 0

    if not dry_run:
        for s in delete_list:
            try:
                size = os.path.getsize(s.path)
                os.remove(s.path)
                deleted   += 1
                del_bytes += size
            except OSError as e:
                errors += 1
    else:
        # Dry run: calculate size without deleting
        for s in delete_list:
            try:
                del_bytes += os.path.getsize(s.path)
                deleted   += 1
            except OSError:
                pass

    elapsed = time.time() - t0

    # Score distribution of kept images
    kept_scores = [s.score for s in keep_list]
    avg_score   = np.mean(kept_scores)  if kept_scores else 0
    min_score   = np.min(kept_scores)   if kept_scores else 0

    print(f"    Done in {elapsed:.0f}s │ "
          f"kept={kept:,} │ "
          f"deleted={deleted:,} │ "
          f"freed={del_bytes/1e6:.1f}MB │ "
          f"avg_score={avg_score:.3f}")

    return {
        "gesture":    gesture,
        "total":      total,
        "kept":       kept,
        "deleted":    deleted,
        "no_hand":    no_hand,
        "errors":     errors,
        "freed_mb":   del_bytes / 1e6,
        "avg_score":  avg_score,
        "min_score":  min_score,
        "time_s":     elapsed,
    }
# ── Final report ───────────────────────────────────────────────

def print_final_report(results: list[dict], dry_run: bool):
    tag = "DRY RUN — nothing deleted" if dry_run else "COMPLETED"
    print(f"\n{'═'*65}")
    print(f"  FILTER REPORT — {tag}")
    print(f"{'═'*65}")

    total_in    = sum(r.get("total",    0) for r in results)
    total_kept  = sum(r.get("kept",     0) for r in results)
    total_del   = sum(r.get("deleted",  0) for r in results)
    total_freed = sum(r.get("freed_mb", 0) for r in results)
    total_time  = sum(r.get("time_s",   0) for r in results)

    for r in results:
        if not r:
            continue
        pct = r["kept"] / max(r["total"], 1) * 100
        print(f"\n  {r['gesture'].upper()}")
        print(f"    Total scanned   : {r['total']:>8,}")
        print(f"    Kept (best)     : {r['kept']:>8,}  ({pct:.1f}%)")
        print(f"    Deleted         : {r['deleted']:>8,}")
        print(f"    No hand found   : {r['no_hand']:>8,}")
        print(f"    Avg kept score  : {r['avg_score']:.4f}")
        print(f"    Min kept score  : {r['min_score']:.4f}")
        print(f"    Disk freed      : {r['freed_mb']:.1f} MB")
        print(f"    Time            : {r['time_s']:.0f}s")

    print(f"\n{'─'*65}")
    print(f"  Total scanned    : {total_in:>10,}")
    print(f"  Total kept       : {total_kept:>10,}")
    print(f"  Total deleted    : {total_del:>10,}")
    print(f"  Disk freed       : {total_freed:>9.1f} MB  "
          f"({total_freed/1024:.2f} GB)")
    print(f"  Total time       : {total_time:.0f}s "
          f"({total_time/60:.1f} min)")

    if not dry_run:
        print(f"\n  external_images/ now contains {total_kept:,} "
              f"clean images only.")
        print(f"  Next step: python src/extract_from_images.py")
    else:
        print(f"\n  Dry run complete. Run again without --dry-run to delete.")

    print(f"{'═'*65}")

# ── Entry point ────────────────────────────────────────────────

def main():
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    print("═" * 65)
    print("  Gesture Car — Permanent Image Quality Filter")
    print(f"  Keep per gesture  : {KEEP_PER_GESTURE:,}")
    print(f"  Total target      : {KEEP_PER_GESTURE * len(GESTURES):,}")
    print(f"  Mode              : {'DRY RUN (preview only)' if dry_run else 'LIVE — will delete files'}")
    print("═" * 65)

    if not os.path.exists(IMAGE_ROOT):
        print(f"\nERROR: {IMAGE_ROOT} not found.")
        return

    # Count total images first
    total_count = 0
    for gesture in GESTURES:
        folder = os.path.join(IMAGE_ROOT, gesture)
        if os.path.exists(folder):
            count = sum(
                1 for fn in os.listdir(folder)
                if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT
            )
            total_count += count
            print(f"  {gesture:12s}: {count:>8,} images")

    print(f"  {'TOTAL':12s}: {total_count:>8,} images")
    print(f"\n  Will KEEP  : {KEEP_PER_GESTURE:,} per gesture "
          f"({KEEP_PER_GESTURE * len(GESTURES):,} total)")
    print(f"  Will DELETE: ~{max(total_count - KEEP_PER_GESTURE * len(GESTURES), 0):,} images")

    if not dry_run:
        print(f"\n  {'!'*55}")
        print(f"  ! PERMANENT DELETE — this cannot be undone.        !")
        print(f"  ! Run with --dry-run first to preview.             !")
        print(f"  {'!'*55}")
        confirm = input("\n  Type YES to proceed: ").strip()
        if confirm != "YES":
            print("  Cancelled.")
            return

    # Build MediaPipe detector
    detector = mp.solutions.hands.Hands(
        static_image_mode        = True,
        max_num_hands            = 2,    # detect 2 so we can reject multi-hand
        min_detection_confidence = MIN_DETECT_CONF,
        model_complexity         = 1
    )

    results = []
    for gesture in GESTURES:
        result = filter_gesture_folder(
            gesture  = gesture,
            detector = detector,
            keep     = KEEP_PER_GESTURE,
            dry_run  = dry_run
        )
        results.append(result)

    detector.close()
    print_final_report(results, dry_run)

if __name__ == "__main__":
    main()