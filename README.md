# Gesture RC Car

Control a real RC car with hand gestures — using your phone's
front camera, on-device ML inference, and WiFi.

**Live demo →** `https://Yash-Patil-26.github.io/gesture-car`

No laptop needed during demo. No server. Works anywhere.

---

## How It Works
Phone camera → MediaPipe.js (landmarks) → ONNX model (gesture)
→ WebSocket → ESP8266 → L298N → 4 motors

The ML model runs entirely in the browser. The ESP8266 creates
its own WiFi hotspot. Phone connects to that hotspot, opens the
URL, and controls the car. Closing the tab stops the car.

---

## Demo — Step by Step

| Step | Action |
|---|---|
| 1 | Power on the car |
| 2 | Connect phone WiFi to **GestureCar** (open, no password) |
| 3 | Open `https://yourusername.github.io/gesture-car` |
| 4 | App auto-connects to car — no IP typing needed |
| 5 | Tap **Start** → show hand to front camera |

If another person is already controlling the car, you will see
**"Car Busy"** — ask them to close the app first.

---

## Gestures

| Gesture | Command |
|---|---|
| Open palm facing camera | FORWARD |
| Closed fist | STOP |
| Index finger pointing left | LEFT |
| Index finger pointing right | RIGHT |
| Thumbs down | REVERSE |

---

## Hardware

| Component | Detail |
|---|---|
| ESP8266 NodeMCU | WiFi AP + WebSocket server |
| L298N motor driver | Drives 4 TT motors |
| 2× 18650 battery (7.4V series) | Power supply |
| 4× TT motors + 65mm wheels | Movement |
| Acrylic chassis (200×160mm) | Structure |
| SPST switch | Master power |

### Pin Mapping

| NodeMCU | L298N | Function |
|---|---|---|
| D1 (GPIO5) | IN1 | Left forward |
| D2 (GPIO4) | IN2 | Left reverse |
| D5 (GPIO14) | IN3 | Right forward |
| D6 (GPIO12) | IN4 | Right reverse |
| D7 (GPIO13) | ENA | Left speed (PWM) |
| D8 (GPIO15) | ENB | Right speed (PWM) |
| VIN | 5V out | Logic power |
| GND | GND | Common ground |

---

## Software Pipeline

```bash
# 1. Setup
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Filter training images (one-time, deletes bad images permanently)
python src/filter_images.py --dry-run
python src/filter_images.py

# 3. Extract landmarks
python src/extract_from_images.py

# 4. Record your own gestures
python src/collect_data.py

# 5. Train
python src/train_model.py

# 6. Export to ONNX
python src/export_model.py

# 7. Push to GitHub → GitHub Pages auto-deploys
git add web/model.onnx web/labels.json
git commit -m "Update model"
git push
```

---

## ESP8266 Firmware

1. Install library: Arduino IDE → Library Manager → **WebSockets** by Markus Sattler
2. Board: `NodeMCU 1.0 (ESP-12E Module)`
3. Upload `esp8266/car_firmware.ino`
4. Open Serial Monitor at 115200 baud to confirm WiFi started

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | Random Forest (100 trees) |
| Input | 63 features (21 landmarks × x,y,z normalized) |
| Classes | 5 gestures |
| Training data | ~44,000 samples |
| CV accuracy | 97–99% |
| Format | ONNX opset 12 (runs in browser via onnxruntime-web) |

---

## Project Structure
gesture-car/
├── esp8266/car_firmware.ino     # ESP8266 firmware
├── src/
│   ├── config.py                # All constants
│   ├── hand_utils.py            # MediaPipe + vote buffer
│   ├── collect_data.py          # Webcam data collection
│   ├── filter_images.py         # Image quality filter
│   ├── extract_from_images.py   # Images → landmark CSV
│   ├── train_model.py           # Train Random Forest
│   ├── export_model.py          # Export to ONNX
│   └── app.py                   # Local dev dashboard
├── web/
│   ├── index.html               # Mobile web app
│   ├── model.onnx               # Trained model
│   └── labels.json              # Class labels
├── outputs/confusion_matrix.png
├── requirements.txt
└── .gitignore

---

## License

MIT