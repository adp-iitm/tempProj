import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import csv
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Face recognition attendance system')
parser.add_argument('--frame-source', choices=['camera', 'file'], default='camera',
                    help='Source of frames: camera or shared file')
parser.add_argument('--frame-path', default='shared_frame.jpg',
                    help='Path to the shared frame file when using file as source')
args = parser.parse_args()

# Config
KNOWN_FACES_DIR = "img3"
VIDEO_PATH = 0  # Used only if frame-source is camera
ENCODINGS_FILE = "encodings4.pkl"
ATTENDANCE_FILE = "Attendance.csv"

known_face_encodings = []
known_face_names = []

# Load or encode known faces
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print("✅ Loaded known face encodings from cache")
else:
    print("🔄 Encoding known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✅ Encoded: {filename}")
            else:
                print(f"❌ No face found in {filename}, skipping.")
    
    # Save encodings to file
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"✅ Encoding completed and saved to '{ENCODINGS_FILE}'")

# Try to load DNN face detector models
print("Loading face detection models...")
use_dnn = False
try:
    # Check if files exist before loading
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    
    if os.path.exists(modelFile) and os.path.exists(configFile):
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        use_dnn = True
        print("Using DNN face detector")
    else:
        print("DNN model files not found. Download these files if you want to use DNN detection:")
        print("- deploy.prototxt")
        print("- res10_300x300_ssd_iter_140000.caffemodel")
except Exception as e:
    print(f"Warning: Could not load DNN model: {e}")
    print("Falling back to Haar Cascade detector")

# Load Haar Cascade detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Function to mark attendance
def mark_attendance(name):
    if not name or name == "Unknown":
        return
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Create the attendance file with headers if it doesn't exist
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time", "Date"])
    
    # Check if the person is already marked for today
    already_marked = False
    
    try:
        # First check if already marked without modifying the file
        with open(ATTENDANCE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) >= 3 and row[0] == name and row[2] == current_date:
                    already_marked = True
                    print(f"⚠️ {name} already marked present today!")
                    break
        
        # If not already marked, add the new entry
        if not already_marked:
            # Use append mode to avoid reading the whole file
            with open(ATTENDANCE_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, current_time, current_date])
            print(f"✅ Marked attendance for: {name}")
            
    except Exception as e:
        print(f"Error marking attendance: {e}")

def detect_faces(frame):
    """Detect faces in the frame using the selected method"""
    face_locations = []
    
    if use_dnn:
        # DNN detection (more accurate)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
                                     [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.6:  # Higher threshold for better accuracy
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Only add face if it has a meaningful size
                if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 400:
                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations.append((y1, x2, y2, x1))
    else:
        # Haar Cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in haar_faces:
            # Convert to face_recognition format (top, right, bottom, left)
            face_locations.append((y, x+w, y+h, x))
            
        # Try to detect profile faces if few faces were found
        if len(face_locations) < 2:
            # Try both original and flipped image to catch profiles facing both directions
            profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in profile_faces:
                face_locations.append((y, x+w, y+h, x))
            
            # Try flipped image for profiles facing the other way
            flipped = cv2.flip(gray, 1)
            flipped_profile_faces = profile_cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Convert flipped coordinates back to original image
            frame_width = gray.shape[1]
            for (x, y, w, h) in flipped_profile_faces:
                # Flip x-coordinate: new_x = width - (x + w)
                new_x = frame_width - (x + w)
                face_locations.append((y, new_x+w, y+h, new_x))
    
    return face_locations

def get_frame_from_file(file_path):
    """Read frame from file if it exists and is not being written to"""
    if not os.path.exists(file_path):
        return None
    try:
        # Read the frame
        frame = cv2.imread(file_path)
        return frame
    except Exception as e:
        print(f"Error reading frame from file: {e}")
        return None

def process_video_from_file(file_path):
    """Process video by reading frames from a file"""
    
    # Initialize stats
    start_time = time.time()
    frames_processed = 0
    faces_detected = 0
    last_update_time = time.time()
    frame_interval = 0.1  # Check for frame update every 100ms
    skip_counter = 0
    frame_skip = 3  # Process every 3rd frame
    
    # Initialize window
    cv2.namedWindow('Face Recognition with Attendance', cv2.WINDOW_NORMAL)
    
    # Initialize recognition counter
    recognition_interval = 5  # Process face recognition every 5 frames
    recognition_counter = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Read the latest frame at regular intervals
            if current_time - last_update_time >= frame_interval:
                frame = get_frame_from_file(file_path)
                last_update_time = current_time
                
                if frame is None:
                    print("Waiting for frames...")
                    time.sleep(0.5)
                    continue
                
                # Initialize display frame and skip counter
                display_frame = frame.copy()
                skip_counter += 1
                
                # Only process every nth frame
                if skip_counter % frame_skip != 0:
                    # Just show frame without processing
                    cv2.imshow('Face Recognition with Attendance', display_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                    continue
                
                # Face recognition happens at a different interval
                do_recognition = (recognition_counter % recognition_interval == 0)
                
                # Detect faces
                face_locations = detect_faces(frame)
                faces_detected += len(face_locations)
                
                # Process faces for recognition
                if face_locations and do_recognition and known_face_encodings:
                    # Get face encodings for detected faces
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    
                    # Match each face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Try to match against known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        
                        # If match found, use the name of the first match
                        if True in matches:
                            match_index = matches.index(True)
                            name = known_face_names[match_index]
                            mark_attendance(name)
                        
                        # Draw on display frame
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(display_frame, name, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif face_locations:
                    # Just draw boxes without recognition
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
                # Display performance metrics
                elapsed_time = current_time - start_time
                fps_text = f"FPS: {frames_processed / max(1, elapsed_time):.1f}"
                cv2.putText(display_frame, fps_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Face Recognition with Attendance', display_frame)
                frames_processed += 1
                recognition_counter += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        
        print("\n--- Performance Summary ---")
        print(f"Total frames processed: {frames_processed}")
        print(f"Total faces detected: {faces_detected}")
        print(f"Average FPS: {frames_processed / max(1, time.time() - start_time):.2f}")

def process_video_from_camera():
    """Process video directly from camera (original method)"""
    # Try multiple camera indices if the default doesn't work
    video_capture = None
    for camera_index in range(3):  # Try indices 0, 1, 2
        video_capture = cv2.VideoCapture(camera_index)
        if video_capture.isOpened():
            print(f"Successfully opened camera at index {camera_index}")
            break
        video_capture.release()
        
    if not video_capture or not video_capture.isOpened():
        print(f"❌ Error: Could not open any camera. Please check connections and permissions.")
        return

    # Get camera properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_skip = 5  # Process every 5th frame for better performance
    start_time = time.time()
    frames_processed = 0
    faces_detected = 0
    
    # Recognition settings
    recognition_interval = 10  # Process face recognition every 10 frames
    recognition_counter = 0
    
    print(f"Starting video processing: {width}x{height} at {fps:.2f} FPS")

    try:
        frame_counter = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Maximum number of consecutive errors before giving up
        
        while True:
            # Try to read a frame
            ret, frame = video_capture.read()
            
            # If frame reading failed
            if not ret:
                consecutive_errors += 1
                print(f"Failed to read frame - attempt {consecutive_errors}/{max_consecutive_errors}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive frame reading errors. Exiting.")
                    break
                    
                # Try to reconnect to the camera
                time.sleep(1)
                video_capture.release()
                video_capture = cv2.VideoCapture(camera_index)
                continue
            
            # Reset error counter on successful frame read
            consecutive_errors = 0
                
            # Process every nth frame for detection
            if frame_counter % frame_skip == 0:
                # Create window if it doesn't exist
                cv2.namedWindow('Face Recognition with Attendance', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Face Recognition with Attendance', width, height)
                
                # Always show current frame
                display_frame = frame.copy()
                
                # Face recognition happens at a different interval
                do_recognition = (recognition_counter % recognition_interval == 0)
                
                # Detect faces
                face_locations = detect_faces(frame)
                faces_detected += len(face_locations)
                
                # Process faces for recognition
                if face_locations and do_recognition and known_face_encodings:
                    # Get face encodings for detected faces
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    
                    # Match each face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Try to match against known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        
                        # If match found, use the name of the first match
                        if True in matches:
                            match_index = matches.index(True)
                            name = known_face_names[match_index]
                            mark_attendance(name)
                        
                        # Draw on display frame
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(display_frame, name, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif face_locations:
                    # Just draw boxes without recognition
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
                # Display performance metrics
                elapsed_time = time.time() - start_time
                fps_text = f"FPS: {frames_processed / max(1, elapsed_time):.1f}"
                cv2.putText(display_frame, fps_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Face Recognition with Attendance', display_frame)
                frames_processed += 1
                recognition_counter += 1
            
            frame_counter += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        
        print("\n--- Performance Summary ---")
        print(f"Total frames processed: {frames_processed}")
        print(f"Total faces detected: {faces_detected}")
        print(f"Average FPS: {frames_processed / max(1, time.time() - start_time):.2f}")

if __name__ == "__main__":
    if args.frame_source == 'file':
        print(f"Reading frames from file: {args.frame_path}")
        process_video_from_file(args.frame_path)
    else:
        print("Reading frames directly from camera")
        process_video_from_camera()