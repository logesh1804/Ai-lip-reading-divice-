from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import time
import requests
from keras.models import load_model
from gtts import gTTS
import dlib

# ---------------- CONFIG ---------------- #

ESP_IP = "192.168.1.100"   # Change to your ESP32 IP
SEQ_LEN = 22
IMG_H, IMG_W = 80, 112
CONFIDENCE_THRESHOLD = 0.6

LABELS = ['are', 'how']  # Keep simple for demo

# ---------------- INIT ---------------- #

app = Flask(__name__)
cap = cv2.VideoCapture(0)

model = load_model("model.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

output_frame = None
lock = threading.Lock()

sentence = []
last_word = ""

# ---------------- ESP SEND ---------------- #

def send_to_esp(word):
    try:
        requests.get(f"http://{ESP_IP}/display?word={word}", timeout=2)
        print("Sent to ESP:", word)
    except:
        print("ESP not connected")

# ---------------- CAMERA THREAD ---------------- #

def camera_stream():
    global output_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        with lock:
            output_frame = frame.copy()

# ---------------- VIDEO STREAM ---------------- #

def generate_frames():
    global output_frame

    while True:
        with lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            landmarks = predictor(gray, face)

            x_min = min([landmarks.part(i).x for i in range(48,68)])
            x_max = max([landmarks.part(i).x for i in range(48,68)])
            y_min = min([landmarks.part(i).y for i in range(48,68)])
            y_max = max([landmarks.part(i).y for i in range(48,68)])

            cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- PREPROCESS ---------------- #

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if np.std(gray) < 10:
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    equalized = clahe.apply(gray)

    resized = cv2.resize(equalized, (IMG_W, IMG_H))
    normalized = resized.astype(np.float32) / 255.0

    return normalized

# ---------------- EXTRACT MOUTH ---------------- #

def extract_mouth(frame, landmarks):
    x_min = min([landmarks.part(i).x for i in range(48,68)])
    x_max = max([landmarks.part(i).x for i in range(48,68)])
    y_min = min([landmarks.part(i).y for i in range(48,68)])
    y_max = max([landmarks.part(i).y for i in range(48,68)])

    mouth = frame[y_min:y_max, x_min:x_max]
    mouth = cv2.resize(mouth, (IMG_W, IMG_H))

    return mouth

# ---------------- PREDICT ---------------- #

@app.route('/predict', methods=['POST'])
def predict():
    global last_word

    frames = []

    while len(frames) < SEQ_LEN:
        with lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            mouth = extract_mouth(frame, landmarks)

            processed = preprocess_frame(mouth)

            if np.std(processed) > 0.02:
                frames.append(processed)

        time.sleep(0.02)

    clip = np.array(frames)
    clip = np.expand_dims(clip, axis=(0, -1))

    preds = model.predict(clip, verbose=0)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        word = LABELS[pred_idx]
        last_word = word
        send_to_esp(word)
        print(f"Prediction: {word} ({confidence:.2f})")

    return last_word

# ---------------- SENTENCE ---------------- #

@app.route('/add_word', methods=['POST'])
def add_word():
    global last_word

    if last_word:
        sentence.append(last_word)
        last_word = ""

    return jsonify({"sentence": " ".join(sentence)})

@app.route('/delete_word', methods=['POST'])
def delete_word():
    if sentence:
        sentence.pop()

    return jsonify({"sentence": " ".join(sentence)})

# ---------------- SPEAK ---------------- #

@app.route('/speak', methods=['POST'])
def speak():
    if sentence:
        text = " ".join(sentence)
        filename = f"static/output.mp3"

        tts = gTTS(text=text, lang='ta')
        tts.save(filename)

        return jsonify({"audio": filename})

    return jsonify({"audio": ""})

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    t = threading.Thread(target=camera_stream)
    t.daemon = True
    t.start()

    app.run(port=600, debug=False)
