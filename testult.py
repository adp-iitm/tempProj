import cv2
import time
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace
from gaze_tracking import GazeTracking
import mediapipe as mp
from datetime import datetime

# Initialize components
gaze = GazeTracking()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants
SKIP_FRAMES = 3
FRAME_WIDTH = 640
YAWN_THRESHOLD = 25
EYE_CLOSED_FRAMES = 15

# Counters
frame_count = 0
yawn_counter = 0
drowsy_counter = 0
start_time = time.time()

# Utils
def is_mouth_open(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    distance = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))
    return distance > YAWN_THRESHOLD

def draw_text(img, text, position, color=(0,255,0), scale=0.7, thickness=2):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# Video source
cap = cv2.VideoCapture("videoyt-lokshab.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- Face Detection ----
    faces = RetinaFace.detect_faces(frame)
    if isinstance(faces, dict):
        for key in faces:
            identity = faces[key]
            facial_area = identity["facial_area"]
            landmarks = identity["landmarks"]

            x1, y1, x2, y2 = facial_area
            face_img = frame[y1:y2, x1:x2]

            # ---- Emotion Recognition ----
            try:
                analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)[0]
                emotion = analysis['dominant_emotion']
                draw_text(frame, f'Emotion: {emotion}', (x1, y1 - 10), (255, 255, 0))
            except:
                draw_text(frame, 'Emotion: Error', (x1, y1 - 10), (0, 0, 255))

            # ---- Yawn Detection ----
            if is_mouth_open(list(landmarks.values())):
                yawn_counter += 1
                draw_text(frame, 'Yawning!', (x1, y2 + 30), (0, 0, 255))

            # ---- Drowsiness Detection ----
            eye_left = landmarks["left_eye"]
            eye_right = landmarks["right_eye"]
            eye_distance = np.linalg.norm(np.array(eye_left) - np.array(eye_right))
            if eye_distance < 10:
                drowsy_counter += 1
                if drowsy_counter > EYE_CLOSED_FRAMES:
                    draw_text(frame, 'Drowsy!', (x1, y2 + 50), (0, 0, 255))
            else:
                drowsy_counter = 0

            # ---- Gaze Tracking ----
            gaze.refresh(frame)
            if gaze.is_blinking():
                gaze_status = "Blinking"
            elif gaze.is_right():
                gaze_status = "Looking right"
            elif gaze.is_left():
                gaze_status = "Looking left"
            elif gaze.is_center():
                gaze_status = "Looking center"
            else:
                gaze_status = "Gaze undetected"
            draw_text(frame, f'Gaze: {gaze_status}', (x1, y2 + 70))

    # ---- Posture Detection ----
    result = pose.process(rgb)
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.1:
            draw_text(frame, 'Bad Posture!', (30, 50), (0, 0, 255))
        else:
            draw_text(frame, 'Good Posture', (30, 50), (0, 255, 0))

    # ---- Performance Metrics ----
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    draw_text(frame, f'FPS: {fps:.2f}', (30, 30))

    cv2.imshow("Behavior Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()