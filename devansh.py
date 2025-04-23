import cv2
import time
import numpy as np
from datetime import datetime
import os
import threading
import csv

# Import detection and tracking modules with error handling
# Face detection and recognition
try:
    import face_recognition
    print("✅ Face Recognition initialized")
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("❌ Face Recognition not available")
    FACE_RECOGNITION_AVAILABLE = False

# RetinaFace detection
try:
    from retinaface import RetinaFace
    print("✅ RetinaFace initialized")
    RETINAFACE_AVAILABLE = True
except ImportError:
    print("❌ RetinaFace not available")
    RETINAFACE_AVAILABLE = False

# Emotion recognition
try:
    from deepface import DeepFace
    print("✅ DeepFace initialized")
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("❌ DeepFace not available")
    DEEPFACE_AVAILABLE = False

# Gaze tracking
try:
    from gaze_tracking import GazeTracking
    gaze = GazeTracking()
    print("✅ GazeTracking initialized")
    GAZE_AVAILABLE = True
except ImportError:
    print("❌ GazeTracking not available")
    GAZE_AVAILABLE = False
    gaze = None

# MediaPipe
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh()
    print("✅ MediaPipe initialized")
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("❌ MediaPipe not available")
    MEDIAPIPE_AVAILABLE = False
    pose = None
    face_mesh = None
    mp_pose = None
    mp_drawing = None

# Object detection (for phone detection)
try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8n.pt")
    print("✅ YOLO initialized")
    YOLO_AVAILABLE = True
except ImportError:
    print("❌ YOLO not available, trying alternative")
    YOLO_AVAILABLE = False
    YOLO_MODEL = None
    
    # Try to use PyTorch YOLOv5 as fallback
    try:
        import torch
        TORCH_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        TORCH_MODEL.classes = [67]  # 67 is the class ID for cell phones in COCO
        print("✅ YOLOv5 (torch) initialized")
        TORCH_AVAILABLE = True
    except ImportError:
        print("❌ YOLOv5 (torch) not available")
        TORCH_AVAILABLE = False
        TORCH_MODEL = None

# Constants
SKIP_FRAMES = 4  # Process every 2nd frame for performance
FRAME_WIDTH = 720
YAWN_THRESHOLD = 25
EYE_CLOSED_THRESHOLD = 0.01
EYE_DISTANCE_THRESHOLD = 30
EYE_CLOSED_FRAMES = 10
DISTRACTION_THRESHOLD = 3  # Seconds
PHONE_CONFIDENCE_THRESHOLD = 0.5
BEHAVIOR_THRESHOLD = 20  # Frames to confirm behavior
MOUTH_OPEN_THRESHOLD = 0.03  # For talking detection

# Create directories for logs
LOG_DIR = "behavior_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"behavior_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Initialize log file
with open(LOG_FILE, 'w') as f:
    f.write("timestamp,name,attention_state,gaze_status,posture_status,emotion,behavior,phone_detected\n")

class State:
    """Class to maintain the current state of the monitoring system"""
    def __init__(self):
        self.frame_count = 0
        self.yawn_counter = 0
        self.drowsy_counter = 0
        self.start_time = time.time()
        self.last_face_time = time.time()
        self.last_focused_time = time.time()
        self.last_phone_detected = None
        self.attention_state = "Initializing..."
        self.attention_color = (255, 255, 255)
        self.emotion = "Unknown"
        self.gaze_status = "Unknown"
        self.posture_status = "Unknown"
        self.fps = 0
        self.phone_detected = False
        self.faces_data = {}
        self.current_face_key = None
        self.phone_box = None
        self.phone_confidence = 0
        self.behavior_counters = {}  # For tracking behavior per person
        self.identified_faces = {}   # Map face IDs to recognized names

# Initialize state and synchronization lock
state = State()
lock = threading.Lock()

# Load known face encodings if available
known_encodings = []
known_names = []

def load_known_faces(directory="img3"):
    """Load known face encodings from directory"""
    if not FACE_RECOGNITION_AVAILABLE:
        print("❌ Cannot load faces without face_recognition module")
        return
    
    if not os.path.exists(directory):
        print(f"❌ Faces directory '{directory}' not found")
        return
    
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            try:
                path = os.path.join(directory, filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    known_names.append(name)
                    print(f"✅ Loaded face: {name}")
            except Exception as e:
                print(f"❌ Error loading face {filename}: {e}")
    
    print(f"✅ Loaded {len(known_names)} faces")

# Helper functions
def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180 else angle

def is_mouth_open(landmarks):
    """Check if mouth is open based on face landmarks"""
    if isinstance(landmarks, dict):
        # RetinaFace format
        if len(landmarks) < 15:
            return False
        top_lip = landmarks.get("mouth_left", None)
        bottom_lip = landmarks.get("mouth_right", None)
        if top_lip and bottom_lip:
            distance = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))
            return distance > YAWN_THRESHOLD
    elif landmarks and hasattr(landmarks, "landmark"):
        # MediaPipe format
        top_lip = landmarks.landmark[13]
        bottom_lip = landmarks.landmark[14]
        return abs(top_lip.y - bottom_lip.y) > MOUTH_OPEN_THRESHOLD
    
    return False

def detect_sleeping(pose_landmarks, eyes_closed):
    """Detect if person appears to be sleeping"""
    if not pose_landmarks:
        return False
    
    left_shoulder = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    nose = [pose_landmarks[mp_pose.PoseLandmark.NOSE.value].x,
            pose_landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    
    head_angle = calculate_angle(left_shoulder, nose, right_shoulder)
    return eyes_closed and head_angle < 60

def detect_phone_use(pose_landmarks, phone_box):
    """Detect if person is using a phone"""
    if not pose_landmarks or phone_box is None:
        return False
    
    # Convert phone_box to relative coordinates if needed
    if isinstance(phone_box, (list, np.ndarray)) and len(phone_box) == 4:
        if phone_box[0] > 1:  # If absolute coordinates
            h, w = 1.0, 1.0  # Placeholder values
            phone_center = [(phone_box[0]+phone_box[2])/(2*w), (phone_box[1]+phone_box[3])/(2*h)]
        else:
            phone_center = [(phone_box[0]+phone_box[2])/2, (phone_box[1]+phone_box[3])/2]
    else:
        return False
        
    left_wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
    right_wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
    
    left_hand = [pose_landmarks[left_wrist_idx].x, pose_landmarks[left_wrist_idx].y]
    right_hand = [pose_landmarks[right_wrist_idx].x, pose_landmarks[right_wrist_idx].y]
    
    return (np.linalg.norm(np.array(left_hand) - np.array(phone_center)) < 0.1 or
            np.linalg.norm(np.array(right_hand) - np.array(phone_center)) < 0.1)

def detect_phone(frame):
    """Detect phones in the frame using available methods"""
    if YOLO_AVAILABLE and YOLO_MODEL is not None:
        # Use YOLO for phone detection (primary)
        results = YOLO_MODEL(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 67:  # 67 is COCO class for cell phone
                    box_coords = box.xyxy[0].cpu().numpy()
                    return True, [int(box_coords[0]), int(box_coords[1]), 
                                  int(box_coords[2]), int(box_coords[3])], box.conf.item()
    
    elif TORCH_AVAILABLE and TORCH_MODEL is not None:
        # Use YOLOv5 Torch for phone detection (fallback 1)
        results = TORCH_MODEL(frame)
        phones = results.pandas().xyxy[0]
        phones = phones[phones['confidence'] > PHONE_CONFIDENCE_THRESHOLD]
        phones = phones[phones['name'] == 'cell phone']
        
        if not phones.empty:
            # Get the box with highest confidence
            best_phone = phones.iloc[phones['confidence'].argmax()]
            box = [int(best_phone['xmin']), int(best_phone['ymin']), 
                   int(best_phone['xmax']), int(best_phone['ymax'])]
            return True, box, best_phone['confidence']
    else:
        # Simple edge-based detection as fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 2000:  # Filter small contours
                continue
                
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Check if shape resembles a phone (rectangular with specific aspect ratio)
            if len(approx) >= 4 and len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if 1.5 < aspect_ratio < 2.5 and w > 50:
                    return True, [x, y, x+w, y+h], 0.6
                    
    return False, None, 0

def analyze_attention_state(faces_data, gaze_status, posture_status, phone_detected):
    """Determine attention state based on multiple cues"""
    current_time = time.time()
    
    # If phone is detected, user is definitely distracted
    if phone_detected:
        return "DISTRACTED: Phone detected", (0, 0, 255)
    
    # If no face is detected for a while, user is away
    if not faces_data and (current_time - state.last_face_time > DISTRACTION_THRESHOLD):
        return "AWAY: No face detected", (0, 0, 255)
    
    # If user has a face detected, check gaze and posture
    if faces_data:
        if posture_status == "Bad Posture":
            return "DISTRACTED: Poor posture", (0, 165, 255)
        
        if gaze_status in ["Looking left", "Looking right", "Blinking", "Gaze undetected"]:
            # Short glances are ok, but sustained looking away is distraction
            if current_time - state.last_focused_time > DISTRACTION_THRESHOLD:
                return f"DISTRACTED: {gaze_status}", (0, 165, 255)
        
        # Check if user is yawning or drowsy
        for face_key in faces_data:
            face = faces_data[face_key]
            if face.get("yawning", False):
                return "TIRED: Yawning", (0, 165, 255)
            if face.get("drowsy", False):
                return "TIRED: Drowsy", (0, 0, 255)
            if face.get("sleeping", False):
                return "SLEEPING", (0, 0, 255)
            
            # Check behavior
            if face.get("behavior", "Attentive") == "Talking":
                return "TALKING", (0, 255, 255)
            
            # Check emotion for signs of disengagement
            emotion = face.get("emotion", "Unknown")
            if emotion in ["sad", "angry", "fear"]:
                return f"CONCERNED: {emotion}", (0, 165, 255)
    
        # If we've reached here, user is focused
        state.last_focused_time = current_time
        return "FOCUSED", (0, 255, 0)
    
    # Default case
    return "UNKNOWN", (128, 128, 128)

def draw_text(img, text, position, color=(0, 255, 0), scale=0.7, thickness=2):
    """Draw text with a subtle background for better visibility"""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_box(img, label, box, color=(0, 255, 0)):
    """Draw a labeled box on the image"""
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
    draw_text(img, label, (x1, y1 - 5), (255, 255, 255))

def find_working_webcam():
    # Enumerate available cameras and return the first working one
    for camera_info in enumerate_cameras():
        cap = cv2.VideoCapture(camera_info.index, camera_info.backend)
        if cap.isOpened():
            print(f"✅ Webcam accessed: {camera_info.name}")
            return cap
        cap.release()
    return None

def log_behavior_data(name="Unknown"):
    """Log behavior data to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion = "Unknown"
    behavior = "Unknown"
    
    for face_key in state.faces_data:
        face_data = state.faces_data[face_key]
        emotion = face_data.get("emotion", "Unknown")
        behavior = face_data.get("behavior", "Unknown")
        name = face_data.get("name", name)
        break
    
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp},{name},{state.attention_state},{state.gaze_status}," + 
                f"{state.posture_status},{emotion},{behavior},{state.phone_detected}\n")

def process_frame_thread(frame):
    """Process the frame for various detections in a separate thread"""
    try:
        with lock:
            state.frame_count += 1
            if state.frame_count % SKIP_FRAMES != 0:
                return
                
            # Make a copy to avoid conflicts
            frame_copy = frame.copy()
            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
            # Temporary data for this frame
            local_faces_data = {}
            
            # Phone Detection
            try:
                phone_detected, phone_box, confidence = detect_phone(frame_copy)
                if phone_detected:
                    if state.last_phone_detected is None:
                        state.last_phone_detected = time.time()
                    
                    # Only mark as phone detected if seen for more than a second (reduce false positives)
                    if time.time() - state.last_phone_detected > 1.0:
                        state.phone_detected = True
                        state.phone_box = phone_box
                        state.phone_confidence = confidence
                else:
                    state.last_phone_detected = None
                    state.phone_detected = False
            except Exception as e:
                print(f"Phone detection error: {e}")
            
            # Face Recognition (if available)
            face_locations = []
            face_names = []
            
            if FACE_RECOGNITION_AVAILABLE and known_encodings:
                try:
                    face_locations = face_recognition.face_locations(rgb)
                    face_encodings = face_recognition.face_encodings(rgb, face_locations)
                    
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        name = "Unknown"
                        
                        if True in matches:
                            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                        
                        face_names.append(name)
                        
                        # Initialize behavior counters for new faces
                        if name not in state.behavior_counters:
                            state.behavior_counters[name] = {
                                "sleep": 0, "talk": 0, "phone": 0, "drowsy": 0
                            }
                except Exception as e:
                    print(f"Face recognition error: {e}")
            
            # Face Detection with RetinaFace (if available)
            if RETINAFACE_AVAILABLE:
                try:
                    faces = RetinaFace.detect_faces(frame_copy)
                    if isinstance(faces, dict):
                        state.last_face_time = time.time()
                        
                        for idx, key in enumerate(faces):
                            identity = faces[key]
                            facial_area = identity["facial_area"]
                            landmarks = identity.get("landmarks", {})
                            
                            x1, y1, x2, y2 = facial_area
                            
                            # Match with recognized face if possible
                            name = "Unknown"
                            if idx < len(face_names):
                                name = face_names[idx]
                            
                            face_data = {
                                "box": (x1, y1, x2, y2),
                                "landmarks": landmarks,
                                "yawning": False,
                                "drowsy": False,
                                "sleeping": False,
                                "behavior": "Attentive",
                                "emotion": "Unknown",
                                "name": name
                            }
                            
                            # Ensure face region is valid
                            if y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0 and y2 < frame_copy.shape[0] and x2 < frame_copy.shape[1]:
                                face_img = frame_copy[y1:y2, x1:x2]
                                
                                # Emotion Detection with DeepFace (if available)
                                if DEEPFACE_AVAILABLE:
                                    try:
                                        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)[0]
                                        face_data["emotion"] = analysis['dominant_emotion']
                                    except Exception as e:
                                        face_data["emotion"] = "Unknown"
                                
                                # Yawn Detection
                                if isinstance(landmarks, dict) and is_mouth_open(landmarks):
                                    face_data["yawning"] = True
                                    state.yawn_counter += 1
                                
                                # Drowsiness Detection
                                if "left_eye" in landmarks and "right_eye" in landmarks:
                                    eye_left = landmarks["left_eye"]
                                    eye_right = landmarks["right_eye"]
                                    eye_distance = np.linalg.norm(np.array(eye_left) - np.array(eye_right))
                                    if eye_distance < EYE_DISTANCE_THRESHOLD:
                                        state.drowsy_counter += 1
                                        if state.drowsy_counter > EYE_CLOSED_FRAMES:
                                            face_data["drowsy"] = True
                                    else:
                                        state.drowsy_counter = 0
                            
                            local_faces_data[key] = face_data
                except Exception as e:
                    print(f"RetinaFace detection error: {e}")
            
            # MediaPipe processing (if available)
            eyes_closed = False
            if MEDIAPIPE_AVAILABLE:
                # Process pose
                if pose is not None:
                    try:
                        pose_result = pose.process(rgb)
                        if pose_result.pose_landmarks:
                            landmarks = pose_result.pose_landmarks.landmark
                            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                            
                            # Check if shoulders are level (good posture)
                            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                            state.posture_status = "Bad Posture" if shoulder_diff > 0.1 else "Good Posture"
                            
                            # Check for phone use
                            if state.phone_detected and state.phone_box:
                                phone_usage = detect_phone_use(landmarks, state.phone_box)
                                
                                # Update counters for each recognized face
                                for name in state.behavior_counters:
                                    if phone_usage:
                                        state.behavior_counters[name]["phone"] += 1
                                        if state.behavior_counters[name]["phone"] > BEHAVIOR_THRESHOLD:
                                            # Update behavior for all faces associated with this name
                                            for face_key in local_faces_data:
                                                if local_faces_data[face_key]["name"] == name:
                                                    local_faces_data[face_key]["behavior"] = "Using Phone"
                                    else:
                                        state.behavior_counters[name]["phone"] = max(0, state.behavior_counters[name]["phone"] - 1)
                        else:
                            state.posture_status = "Unknown"
                    except Exception as e:
                        state.posture_status = "Posture error"
                        print(f"Pose detection error: {e}")
                
                # Process face mesh
                if face_mesh is not None:
                    try:
                        face_result = face_mesh.process(rgb)
                        if face_result.multi_face_landmarks:
                            face_idx = 0
                            for face_landmarks in face_result.multi_face_landmarks:
                                # Check for eyes closed
                                left_eye_top = face_landmarks.landmark[159]
                                left_eye_bottom = face_landmarks.landmark[145]
                                eyes_closed = abs(left_eye_top.y - left_eye_bottom.y) < EYE_CLOSED_THRESHOLD
                                
                                # Check for talking
                                talking = is_mouth_open(face_landmarks)
                                
                                # Get matching face name
                                name = "Unknown"
                                if face_idx < len(face_names):
                                    name = face_names[face_idx]
                                
                                # Update behavior counters
                                if name in state.behavior_counters:
                                    if talking:
                                        state.behavior_counters[name]["talk"] += 1
                                        if state.behavior_counters[name]["talk"] > BEHAVIOR_THRESHOLD:
                                            # Update the corresponding face data
                                            for face_key in local_faces_data:
                                                if local_faces_data[face_key]["name"] == name:
                                                    local_faces_data[face_key]["behavior"] = "Talking"
                                    else:
                                        state.behavior_counters[name]["talk"] = max(0, state.behavior_counters[name]["talk"] - 1)
                                
                                face_idx += 1
                    except Exception as e:
                        print(f"Face mesh processing error: {e}")
            
            # Gaze Tracking (if available)
            if GAZE_AVAILABLE and gaze is not None:
                try:
                    gaze.refresh(frame_copy)
                    if gaze.is_blinking():
                        state.gaze_status = "Blinking"
                    elif gaze.is_right():
                        state.gaze_status = "Looking right"
                    elif gaze.is_left():
                        state.gaze_status = "Looking left"
                    elif gaze.is_center():
                        state.gaze_status = "Looking center"
                    else:
                        state.gaze_status = "Gaze undetected"
                except Exception as e:
                    state.gaze_status = "Gaze error"
                    print(f"Gaze tracking error: {e}")
            
            # Check for sleeping (combination of pose and eyes closed)
            if MEDIAPIPE_AVAILABLE and pose is not None and pose_result.pose_landmarks:
                for name in state.behavior_counters:
                    if eyes_closed and detect_sleeping(pose_result.pose_landmarks.landmark, eyes_closed):
                        state.behavior_counters[name]["sleep"] += 1
                        if state.behavior_counters[name]["sleep"] > BEHAVIOR_THRESHOLD:
                            # Update all faces with this name
                            for face_key in local_faces_data:
                                if local_faces_data[face_key]["name"] == name:
                                    local_faces_data[face_key]["sleeping"] = True
                                    local_faces_data[face_key]["behavior"] = "Sleeping"
                    else:
                        state.behavior_counters[name]["sleep"] = max(0, state.behavior_counters[name]["sleep"] - 1)
            
            # FPS calculation
            current_time = time.time()
            elapsed_time = current_time - state.start_time
            if elapsed_time > 1.0:  # Update FPS every second
                state.fps = state.frame_count / elapsed_time
                state.frame_count = 0
                state.start_time = current_time
            
            # Calculate attention state based on all factors
            state.attention_state, state.attention_color = analyze_attention_state(
                local_faces_data, state.gaze_status, state.posture_status, state.phone_detected
            )
            
            # Update global face data
            state.faces_data = local_faces_data
    except Exception as e:
        print(f"Frame processing error: {e}")




def main():
    # Try to load known faces
    load_known_faces()

    # Create window with controllable size
    cv2.namedWindow("Behavior Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Behavior Monitor", FRAME_WIDTH, int(FRAME_WIDTH * 9/16))

    # Try to find a working webcam
    print("🔍 Searching for available webcam...")
    cap = find_working_webcam()

    if cap is None:
        print("❌ No available webcam found. Exiting.")
        return

    # Create behavior log directory
    if not os.path.exists("behavior_logs"):
        os.makedirs("behavior_logs")

    process_thread = None
    last_log_time = time.time()
    log_interval = 5  # seconds

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("❌ Failed to read frame from webcam.")
                continue

            # DEBUG: Print frame shape
            print(f"✅ Frame read: shape={frame.shape}")

            # Start processing thread if not already running
            if process_thread is None or not process_thread.is_alive():
                process_thread = threading.Thread(target=process_frame_thread, args=(frame.copy(),))
                process_thread.daemon = True
                process_thread.start()

            with lock:
                faces_data = state['faces_data'].copy()
                attention_state = state['attention_state']
                attention_color = state['attention_color']
                gaze_status = state['gaze_status']
                posture_status = state['posture_status']
                fps = state['fps']
                phone_detected = state['phone_detected']

                phone_box = state.get('phone_box', None)
                phone_confidence = state.get('phone_confidence', 0.0)

            # Resize frame for display
            frame = cv2.resize(frame, (FRAME_WIDTH, int(FRAME_WIDTH * 9/16)))

            # Draw overlays
            for face_key in faces_data:
                face = faces_data[face_key]
                box = face.get("box")
                name = face.get("name", "Unknown")
                emotion = face.get("emotion", "Unknown")
                behavior = face.get("behavior", "Attentive")

                if box:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    color = (0, 255, 0) if behavior == "Attentive" else (0, 165, 255)
                    if face.get("sleeping", False):
                        color = (0, 0, 255)
                    draw_box(frame, f"{name} ({behavior})", (x1, y1, x2, y2), color)
                    draw_text(frame, f"Emotion: {emotion}", (x1, y2 + 20), color)

            if phone_detected and phone_box:
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in phone_box]
                    x1 = int(x1 * FRAME_WIDTH / frame.shape[1])
                    x2 = int(x2 * FRAME_WIDTH / frame.shape[1])
                    y1 = int(y1 * (FRAME_WIDTH * 9/16) / frame.shape[0])
                    y2 = int(y2 * (FRAME_WIDTH * 9/16) / frame.shape[0])
                    draw_box(frame, f"Phone ({phone_confidence:.2f})", (x1, y1, x2, y2), (0, 0, 255))
                except Exception as e:
                    print(f"⚠️ Error drawing phone box: {e}")

            # Draw status text
            overlay_y = 30
            draw_text(frame, f"Attention: {attention_state}", (10, overlay_y), attention_color)
            overlay_y += 30
            draw_text(frame, f"Gaze: {gaze_status}", (10, overlay_y), (255, 255, 255))
            overlay_y += 30
            draw_text(frame, f"Posture: {posture_status}", (10, overlay_y), (255, 255, 255))
            overlay_y += 30
            draw_text(frame, f"FPS: {fps:.2f}", (10, overlay_y), (255, 255, 255))

            # Show window
            cv2.imshow("Behavior Monitor", frame)

            # Periodic behavior log
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                log_behavior_data()
                last_log_time = current_time

            # Handle user input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("✅ Exiting per user request.")
                break

    except KeyboardInterrupt:
        print("✅ Program interrupted manually.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("🧹 Cleaning up...")
        if cap:
            cap.release()
        if process_thread and process_thread.is_alive():
            process_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        log_behavior_data()
        print(f"✅ Behavior log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()