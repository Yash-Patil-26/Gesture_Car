# src/hand_utils.py
# ─────────────────────────────────────────────────────────────
# Shared utilities: landmark extraction + asymmetric vote buffer.
# ─────────────────────────────────────────────────────────────

import numpy as np
import mediapipe as mp
from collections import deque


def build_hand_detector(detection_conf: float, tracking_conf: float):
    """
    Initialise MediaPipe Hands once at startup.
    Reuse across all frames — do not create per frame.
    """
    return mp.solutions.hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = detection_conf,
        min_tracking_confidence  = tracking_conf
    )


def extract_features(landmarks) -> np.ndarray:
    """
    21 MediaPipe landmarks → 63-dim normalized vector.

    Step 1 — Translation: subtract wrist (landmark 0).
    Step 2 — Scale: divide by max absolute value → values in [-1, +1].

    Both steps together make the vector invariant to hand
    position in frame AND distance from camera.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z

    coords = []
    for lm in landmarks:
        coords.append(lm.x - wrist_x)
        coords.append(lm.y - wrist_y)
        coords.append(lm.z - wrist_z)

    coords  = np.array(coords, dtype=np.float32)
    max_val = np.max(np.abs(coords))

    if max_val < 1e-6:
        # Degenerate detection — all landmarks at same point.
        # Returns zero vector — rejected by confidence gate.
        return np.zeros(63, dtype=np.float32)

    return coords / max_val


def process_frame(frame, detector):
    """
    Run MediaPipe on one BGR frame.
    Converts BGR → RGB (MediaPipe requirement).
    Uses writeable=False optimization to avoid memory copy.
    """
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = detector.process(rgb)
    rgb.flags.writeable = True
    return result, rgb


def get_landmark_list(result):
    """
    Extract first hand's landmarks from MediaPipe result.
    Returns landmark list or None if no hand detected.
    """
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0].landmark
    return None


class FastVoteBuffer:
    """
    Asymmetric vote buffer for real-time car control.

    ENGAGE (motion commands — FORWARD/REVERSE/LEFT/RIGHT):
      Requires MOTION_VOTES consecutive unanimous frames.
      Default 3 frames = 100ms at 30fps.
      Prevents false triggers from gesture transitions.

    STOP:
      Fires on STOP_VOTES consecutive frames OR immediately
      if confidence drops below threshold.
      Default 1 frame = 33ms — near-instant safety stop.

    NO HAND:
      Fires STOP immediately on first frame with no hand.
      Car must stop the moment user removes hand.

    Why asymmetric:
      Motion commands need confirmation to prevent jitter.
      STOP must be instant to prevent car damage.
      A car that is slow to stop is dangerous.
      A car that is slow to start is just slow.
    """

    MOTION_VOTES = 3    # frames needed to engage motion
    STOP_VOTES   = 1    # frames needed to engage STOP
    STOP_CMDS    = {"STOP"}
    MOTION_CMDS  = {"FORWARD", "REVERSE", "LEFT", "RIGHT"}

    def __init__(self, confidence_threshold: float):
        self.buf        = deque(maxlen=self.MOTION_VOTES)
        self.conf_thresh= confidence_threshold
        self.last_stable= "STOP"

    def update(self, label: str, confidence: float, hand_present: bool) -> str:
        """
        Update buffer with new frame result.

        Args:
            label:        classifier output class name
            confidence:   classifier probability [0,1]
            hand_present: whether MediaPipe found a hand

        Returns:
            Command string to send: FORWARD/REVERSE/LEFT/RIGHT/STOP
        """
        # Rule 1: No hand at all → instant STOP
        if not hand_present:
            self.buf.clear()
            self.last_stable = "STOP"
            return "STOP"

        # Rule 2: Low confidence → instant STOP
        # Model is uncertain — safest action is stop
        if confidence < self.conf_thresh:
            self.buf.clear()
            self.last_stable = "STOP"
            return "STOP"

        cmd = label.upper().replace(" (?)", "")

        # Rule 3: STOP gesture → fast path, needs only STOP_VOTES
        if cmd in self.STOP_CMDS:
            self.buf.clear()
            self.last_stable = "STOP"
            return "STOP"

        # Rule 4: Motion command → push to buffer, need MOTION_VOTES
        if cmd in self.MOTION_CMDS:
            self.buf.append(cmd)

            # Check if buffer is full and unanimous
            if (len(self.buf) >= self.MOTION_VOTES and
                    all(v == self.buf[0] for v in self.buf)):
                self.last_stable = self.buf[0]
                return self.last_stable

        # Buffer not yet full or not unanimous — hold last stable command
        # This prevents flicker during gesture transitions
        return self.last_stable

    def force_stop(self):
        """Call when camera closes or app exits."""
        self.buf.clear()
        self.last_stable = "STOP"