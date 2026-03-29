from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time
import requests
from keras.models import load_model
from gtts import gTTS
import dlib

# ---------------- CONFIG ---------------- #
ESP_IP = "192.168.1.100"   # 🔥 change this
LABELS = ['are','how']     # keep simple for demo

SEQ_LEN = 22
IMG_H, IMG_W = 80,112
CONFIDENCE_THRESHOLD = 0.60

app = Flask(__name__)

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

output_frame = None
lock = threading.Lock()

# ---------------- MODEL ---------------- #
model = load_model("model.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sentence = []
last_predicted_word = ""

is_predicting = False
frame_counter = 0

# ---------------- ESP ---------------- #
def send_to_esp(word):
    try:
        url = f"http://{ESP_IP}/display?word={word}"
        requests.get(url, timeout=2)
        print("Sent to ESP:", word)
    except:
        print("ESP not connected")

# ---------------- CAMERA STREAM ---------------- #
def camera_stream():
    global output_frame

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame,1)   # mirror

        with lock:
            output_frame = frame.copy()

# ---------------- VIDEO ---------------- #
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
            x,y,w,h = face.left(),face.top(),face.width(),face.height()
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            landmarks = predictor(gray, face)

            x_min = min([landmarks.part(i).x for i in range(48,68)])
            x_max = max([landmarks.part(i).x for i in range(48,68)])
            y_min = min([landmarks.part(i).y for i in range(48,68)])
            y_max = max([landmarks.part(i).y for i in range(48,68)])

            cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),2)

        if is_predicting:
            text = f"CAPTURING {frame_counter}/{SEQ_LEN}"
            cv2.putText(frame,text,(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- PREPROCESS ---------------- #
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if np.std(gray) < 10:
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    equalized = clahe.apply(gray)

    resized = cv2.resize(equalized, (IMG_W, IMG_H))
    normalized = resized.astype(np.float32)/255.0

    return normalized

# ---------------- MOUTH ---------------- #
def extract_mouth(frame, landmarks):
    x_min = min([landmarks.part(i).x for i in range(48,68)])
    x_max = max([landmarks.part(i).x for i in range(48,68)])
    y_min = min([landmarks.part(i).y for i in range(48,68)])
    y_max = max([landmarks.part(i).y for i in range(48,68)])

    mouth = frame[y_min:y_max, x_min:x_max]
    return cv2.resize(mouth, (IMG_W, IMG_H))

# ---------------- PREDICT ---------------- #
@app.route('/predict', methods=['POST'])
def predict():
    global last_predicted_word, is_predicting, frame_counter

    frames = []
    is_predicting = True
    frame_counter = 0

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

            if np.std(processed) > 0.02:   # reject blank
                frames.append(processed)

            frame_counter = len(frames)

        time.sleep(0.02)

    is_predicting = False

    clip = np.array(frames, dtype=np.float32)
    clip = np.expand_dims(clip, axis=(0,-1))

    preds = model.predict(clip)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        word = LABELS[pred_idx]
        last_predicted_word = f"{word.upper()} ({confidence:.2f})"

        print(last_predicted_word)

        send_to_esp(word)   # 🔥 hardware call

    else:
        last_predicted_word = "Low Confidence"

    return last_predicted_word

# ---------------- SENTENCE ---------------- #
@app.route('/add_word', methods=['POST'])
def add_word():
    global last_predicted_word

    if last_predicted_word:
        sentence.append(last_predicted_word.split()[0])

    return jsonify({"sentence": " ".join(sentence)})

@app.route('/delete_word', methods=['POST'])
def delete_word():
    if sentence:
        sentence.pop()
    return jsonify({"sentence": " ".join(sentence)})

# ---------------- SPEECH ---------------- #
@app.route('/speak', methods=['POST'])
def speak():
    if sentence:
        text = " ".join(sentence)
        tts = gTTS(text=text, lang='en')
        tts.save("static/output.mp3")
        return jsonify({"audio": "/static/output.mp3"})

    return jsonify({"audio": ""})

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    t = threading.Thread(target=camera_stream)
    t.daemon = True
    t.start()

    app.run(port=600)
