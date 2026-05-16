# ─────────────────────────────────────────────────────────────
# LOCAL DEVELOPMENT DASHBOARD ONLY.
# Not used in production deployment (GitHub Pages).
#
# Runs Flask server with live camera feed and ML inference.
# Communicates with ESP8266 via WebSocket (AP mode, port 81).
#
# Run: python src/app.py
# Open: http://localhost:5000
# ─────────────────────────────────────────────────────────────

import os
import sys
import cv2
import pickle
import time
import threading
import numpy as np
from collections import deque

import mediapipe as mp
import websocket
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_FILE, ENCODER_FILE,
    CAM_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD, VOTE_WINDOW,
    ESP8266_IP, ESP8266_PORT,
    FLASK_HOST, FLASK_PORT,
    GESTURES
)
from hand_utils import (
    build_hand_detector, process_frame,
    get_landmark_list, extract_features,
    FastVoteBuffer    # ← updated import
)

# ── Flask + SocketIO ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config['SECRET_KEY'] = 'gesture_car_dev'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Shared state ───────────────────────────────────────────────
frame_lock = threading.Lock()
state = {
    "frame":         None,
    "gesture":       "—",
    "confidence":    0.0,
    "command":       "STOP",
    "car_connected": False,
    "cam_connected": False,
    "fps":           0.0,
    "session_stats": {g: 0 for g in GESTURES},
    "last_commands": [],
}


# ── Model loader ───────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run train_model.py first."
        )
    with open(MODEL_FILE,   'rb') as f: model   = pickle.load(f)
    with open(ENCODER_FILE, 'rb') as f: encoder = pickle.load(f)
    print(f"[ML] Loaded — classes: {list(encoder.classes_)}")
    return model, encoder


# ── WebSocket car sender ───────────────────────────────────────
class CarSender:
    """
    Sends commands to ESP8266 via WebSocket.
    ESP8266 is in AP mode — always at 192.168.4.1:81.
    Uses websocket-client library (not browser WebSocket).
    Runs in a background thread with auto-reconnect.
    """
    def __init__(self, ip, port):
        self.url       = f"ws://{ip}:{port}"
        self.ws        = None
        self.connected = False
        self.last_cmd  = None
        self.last_sent = 0
        self._connect()

    def _connect(self):
        def run():
            while True:
                try:
                    self.ws = websocket.create_connection(
                        self.url, timeout=3
                    )
                    self.connected = True
                    state["car_connected"] = True
                    print(f"[WS] Connected to car at {self.url}")
                    # Keep connection alive — read ACKs
                    while True:
                        self.ws.recv()
                except Exception as e:
                    self.connected = False
                    state["car_connected"] = False
                    print(f"[WS] Car disconnected — retry in 3s: {e}")
                    time.sleep(3)

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def send(self, command: str):
        now     = time.time()
        changed = command != self.last_cmd
        beat    = now - self.last_sent > 0.3

        if (changed or beat) and self.connected and self.ws:
            try:
                self.ws.send(command)
                self.last_cmd  = command
                self.last_sent = now
            except Exception:
                self.connected = False
                state["car_connected"] = False


# ── Frame annotation ───────────────────────────────────────────
mp_draw      = mp.solutions.drawing_utils
mp_hands_mod = mp.solutions.hands

CMD_COLORS = {
    "FORWARD": (0, 220, 100),
    "REVERSE": (0, 120, 255),
    "LEFT":    (200, 100, 255),
    "RIGHT":   (255, 180, 0),
    "STOP":    (60,  60,  220),
}

def annotate(frame, gesture, confidence, command):
    h, w  = frame.shape[:2]
    color = CMD_COLORS.get(command, (200, 200, 200))
    conf_color = (0, 220, 100) if confidence >= CONFIDENCE_THRESHOLD \
                 else (80, 80, 220)

    cv2.rectangle(frame, (0, 0), (w, 80), (18, 18, 18), -1)
    cv2.putText(frame, f"Gesture: {gesture}",
                (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

    bar = int(confidence * 200)
    cv2.rectangle(frame, (14, 50), (214, 64), (50,50,50), -1)
    cv2.rectangle(frame, (14, 50), (14+bar, 64), conf_color, -1)
    cv2.putText(frame, f"{confidence:.0%}",
                (220, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)
    cv2.putText(frame, command,
                (w-180, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return frame


# ── Inference thread ───────────────────────────────────────────
def inference_thread(model, encoder, sender):
    detector  = build_hand_detector(
        MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE)
    vote_buf  = FastVoteBuffer(CONFIDENCE_THRESHOLD)  # ← fixed

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)
    state["cam_connected"] = cap.isOpened()

    if not cap.isOpened():
        print(f"[CAM] Cannot open camera {CAM_INDEX}")
        return

    fps = 0.0
    t_prev = time.time()
    emit_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame     = cv2.flip(frame, 1)
        result, _ = process_frame(frame, detector)
        lms       = get_landmark_list(result)

        gesture    = "—"
        confidence = 0.0

        if lms is not None:
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(
                    color=(0,180,255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(
                    color=(255,255,255), thickness=1)
            )
            features   = extract_features(lms).reshape(1, -1)
            proba      = model.predict_proba(features)[0]
            idx        = int(np.argmax(proba))
            confidence = float(proba[idx])
            gesture    = encoder.classes_[idx]

        # FastVoteBuffer handles all cases
        command = vote_buf.update(gesture, confidence, lms is not None)
        sender.send(command)

        # Session stats
        if command != "STOP":
            state["session_stats"][gesture] = \
                state["session_stats"].get(gesture, 0) + 1

        log = state["last_commands"]
        log.append({
            "cmd":  command,
            "time": time.strftime("%H:%M:%S"),
            "conf": round(confidence, 3)
        })
        if len(log) > 20:
            log.pop(0)

        # FPS
        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        display = annotate(frame.copy(), gesture, confidence, command)

        with frame_lock:
            state["frame"]         = display
            state["gesture"]       = gesture
            state["confidence"]    = round(confidence, 3)
            state["command"]       = command
            state["fps"]           = round(fps, 1)

        emit_n += 1
        if emit_n % 3 == 0:
            socketio.emit('state_update', {
                "gesture":    gesture,
                "confidence": round(confidence, 3),
                "command":    command,
                "fps":        round(fps, 1),
                "car":        sender.connected,
                "stats":      state["session_stats"],
                "log":        state["last_commands"][-5:]
            })

    cap.release()
    detector.close()


# ── MJPEG stream ───────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = state["frame"]

        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Initialising camera…",
                        (160, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (80,80,80), 2)
            frame = blank

        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           esp_ip=ESP8266_IP,
                           esp_port=ESP8266_PORT)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "gesture":    state["gesture"],
        "confidence": state["confidence"],
        "command":    state["command"],
        "fps":        state["fps"],
        "car":        state["car_connected"],
        "cam":        state["cam_connected"],
        "stats":      state["session_stats"],
        "log":        state["last_commands"]
    })


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    print("═" * 50)
    print("  Gesture Car — Local Dev Dashboard")
    print("  NOTE: Production app is web/index.html")
    print("        deployed on GitHub Pages.")
    print("═" * 50)

    model, encoder = load_model()
    sender         = CarSender(ESP8266_IP, ESP8266_PORT)

    t = threading.Thread(
        target=inference_thread,
        args=(model, encoder, sender),
        daemon=True
    )
    t.start()

    print(f"[SERVER] http://localhost:{FLASK_PORT}")
    print(f"[CAR]    ws://{ESP8266_IP}:{ESP8266_PORT}")

    socketio.run(app,
                 host=FLASK_HOST,
                 port=FLASK_PORT,
                 debug=False,
                 use_reloader=False)