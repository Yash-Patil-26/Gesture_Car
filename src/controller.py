# src/controller.py
# ─────────────────────────────────────────────────────────────
# Real-time inference loop.
# Webcam → MediaPipe → features → RF → vote → UDP → ESP32
#
# Run AFTER training: python src/controller.py
# ─────────────────────────────────────────────────────────────

import cv2
import pickle
import socket
import sys
import os
import time
import numpy as np
from collections import deque

import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_FILE, ENCODER_FILE,
    CAM_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD, VOTE_WINDOW,
    ESP8266_IP, ESP8266_PORT
)
from hand_utils import (build_hand_detector, process_frame,
                        get_landmark_list, extract_features,
                        FastVoteBuffer)

mp_draw        = mp.solutions.drawing_utils
mp_hands_module= mp.solutions.hands


def load_model():
    """Load trained RF model and label encoder from disk."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}.\n"
            "Run train_model.py first."
        )
    with open(MODEL_FILE,   'rb') as f: model   = pickle.load(f)
    with open(ENCODER_FILE, 'rb') as f: encoder = pickle.load(f)
    print(f"Model loaded: {MODEL_FILE}")
    print(f"Classes: {list(encoder.classes_)}")
    return model, encoder


def create_udp_socket():
    """Create non-blocking UDP socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    return sock


class CommandSender:
    """
    Wraps UDP sending with two behaviours:
    1. Deduplication — only sends when command changes.
       Prevents flooding ESP32 with identical packets 30x/sec.
    2. Heartbeat — sends current command every 300ms even if
       unchanged, so ESP32 watchdog does not trigger on stable hold.
    """
    def __init__(self, sock, ip, port):
        self.sock         = sock
        self.addr         = (ip, port)
        self.last_command = None
        self.last_sent_t  = 0
        self.HEARTBEAT_S  = 0.3    # seconds between heartbeat sends

    def send(self, command: str):
        now      = time.time()
        changed  = (command != self.last_command)
        heartbeat= (now - self.last_sent_t) > self.HEARTBEAT_S

        if changed or heartbeat:
            try:
                self.sock.sendto(command.encode(), self.addr)
                self.last_command = command
                self.last_sent_t  = now
            except Exception as e:
                # Non-blocking socket raises BlockingIOError if
                # send buffer full — safe to ignore in this context
                pass


class VoteBuffer:
    """
    Sliding window majority vote over last VOTE_WINDOW predictions.
    Returns a command only when all votes in the window agree.
    Clears on low-confidence input.
    """
    def __init__(self, window: int):
        self.buffer = deque(maxlen=window)
        self.window = window

    def push(self, label: str):
        self.buffer.append(label)

    def clear(self):
        self.buffer.clear()

    def get_stable(self):
        """
        Returns label if all window entries agree, else None.
        Window must be full before returning anything.
        """
        if len(self.buffer) < self.window:
            return None
        first = self.buffer[0]
        if all(v == first for v in self.buffer):
            return first
        return None


def draw_hud(frame, label, confidence, stable_command, fps):
    """Render all status information onto the frame."""
    h, w = frame.shape[:2]

    # Gesture label — top left
    color = (0, 230, 120) if confidence >= CONFIDENCE_THRESHOLD else (80, 80, 220)
    cv2.putText(frame, f"Gesture : {label}",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Confidence bar
    bar_w = int(confidence * 220)
    cv2.rectangle(frame, (20, 60), (240, 76), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 60), (20 + bar_w, 76), color, -1)
    cv2.putText(frame, f"{confidence:.2f}",
                (248, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    # Active command
    if stable_command:
        cv2.putText(frame, f"CMD: {stable_command}",
                    (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 180), 2)

    # Threshold line on confidence bar
    thresh_x = 20 + int(CONFIDENCE_THRESHOLD * 220)
    cv2.line(frame, (thresh_x, 58), (thresh_x, 78), (255, 255, 100), 1)

    # FPS — bottom right
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 100, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (140, 140, 140), 1)

    # ESP8266 target
    cv2.putText(frame, f"ESP8266: {ESP8266_IP}:{ESP8266_PORT}",
                (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1)


def main():
    print("═" * 50)
    print("  Gesture Car — Controller")
    print("═" * 50)

    model, encoder = load_model()
    detector       = build_hand_detector(MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE)
    sock           = create_udp_socket()
    sender         = CommandSender(sock, ESP8266_IP, ESP8266_PORT)
    vote_buf = FastVoteBuffer(CONFIDENCE_THRESHOLD)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAM_INDEX}")
        return

    print(f"\nSending commands to ESP8266 at {ESP8266_IP}:{ESP8266_PORT}")
    print("Press Q to quit.\n")

    # FPS tracking
    fps      = 0.0
    t_prev   = time.time()
    label    = "—"
    confidence    = 0.0
    stable_cmd    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — retrying...")
            continue

        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        if lms is not None:
            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands_module.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,180,255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=1)
            )

            # Extract → classify
            features   = extract_features(lms).reshape(1, -1)
            proba      = model.predict_proba(features)[0]
            class_idx  = int(np.argmax(proba))
            confidence = float(proba[class_idx])
            label      = encoder.classes_[class_idx]

            command = vote_buf.update(label, confidence, lms is not None)
            sender.send(command)
            stable_cmd = command

        else:
            # No hand detected
            vote_buf.clear()
            label      = "No hand"
            confidence = 0.0
            stable_cmd = None
            sender.send("STOP")

        # FPS
        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        draw_hud(frame, label, confidence, stable_cmd, fps)
        cv2.imshow("Gesture Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    sender.send("STOP")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    sock.close()
    print("Controller stopped. STOP sent to ESP8266.")


if __name__ == "__main__":
    main()
    