from flask import Flask, render_template, Response, request, jsonify
import cv2
from mtcnn import MTCNN
from fer import FER
from pathlib import Path
import threading
import base64
import time

# Get the directory of the currently executing file (app.py)
base_dir = Path(__file__).resolve().parent

app = Flask(__name__)

mtcnn_detector = MTCNN()
fer_detector = FER()

camera = cv2.VideoCapture(0)

# Initialize _emotions with default values
_emotions = {
    "angry": 0.0,
    "disgust": 0.0,
    "fear": 0.0,
    "happy": 0.0,
    "neutral": 0.0,
    "sad": 0.0,
    "surprise": 0.0
}
_emotions_lock = threading.Lock()  # Lock for thread-safe access
frameHistory = []  # Store frames with emotions

def generate_frames():
    global _emotions
    last_saved_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        detected_faces = mtcnn_detector.detect_faces(frame)

        for face in detected_faces:
            x, y, width, height = face['box']
            face_roi = frame[y:y + height, x:x + width]

            # Detect emotions
            emots = fer_detector.detect_emotions(face_roi)
            if emots:
                with _emotions_lock:
                    _emotions = emots[0]["emotions"]  # Update with the latest emotion data

            emotion, confidence = fer_detector.top_emotion(face_roi)
            label = f"{emotion}: {confidence:.2f}" if emotion else "Unrecognized"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Save frame and label every second
        current_time = time.time()
        if current_time - last_saved_time >= 1:
            last_saved_time = current_time
            ret, buffer = cv2.imencode('.jpg', frame)
            frameHistory.append({"image": buffer.tobytes(), "emotion": emotion})  # Save frame with its emotion
            if len(frameHistory) > 11:
                frameHistory.pop(0)  # Keep only the latest 11 frames

        # Video feed generation
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotions_graph')
def emotions_graph():
    with _emotions_lock:
        return jsonify(_emotions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')
    
    if feedback:
        with open(Path(__file__).parent / 'feedback_log.txt', 'a') as f:
            f.write(f"Feedback: {feedback}\n")
        
        return "Your feedback has been submitted."
    else:
        return "Feedback cannot be empty."

@app.route('/report_misclassification', methods=['POST'])
def report_misclassification():
    data = request.get_json()
    frame_data = data.get('frame')
    emotion = data.get('emotion')

    # Decode base64 image and save to emotion-specific folder
    if frame_data and emotion:
        image_data = base64.b64decode(frame_data.split(',')[1])
        
        # Ensure the directory structure for saving images
        emotion_folder = base_dir / 'submittedEmotions' / emotion
        emotion_folder.mkdir(parents=True, exist_ok=True)
        
        # Save the image to the specified folder
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = emotion_folder / f"{emotion}_{timestamp}.jpg"
        with open(image_path, 'wb') as img_file:
            img_file.write(image_data)
        
        print(f"Saved misclassified image for '{emotion}' at {image_path}")

    return jsonify(message="Thank you for your submission!"), 200

@app.route('/last_frames', methods=['GET'])
def last_frames():
    # Send the last frames along with labels in JSON format
    return jsonify([{"frame": base64.b64encode(frame_data["image"]).decode('utf-8'), 
                     "label": frame_data["emotion"]} for frame_data in frameHistory[-10:]])

if __name__ == "__main__":
    #app.run(debug=True)
    app.run()
