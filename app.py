import cv2
import threading
import subprocess
import os
import signal
import time
import numpy as np
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Global variables
analyzing = False
attendance_process = None
capture_active = False
capture_error = None
latest_frame_path = "shared_frame.jpg"
frame_ready = False

def video_capture():
    """Continuously capture and store video."""
    global capture_active, capture_error, frame_ready
    
    try:
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            capture_error = "Could not open camera. Please check connections and permissions."
            print(f"Error: {capture_error}")
            return

        # Get frame size from webcam
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        # VideoWriter for continuous storage (optional)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/recorded_output.avi', fourcc, 20.0, (width, height))
        
        capture_active = True
        consecutive_errors = 0
        
        while capture_active:
            ret, frame = cap.read()
            
            if not ret:
                consecutive_errors += 1
                print(f"Frame capture failed ({consecutive_errors}). Attempting to reconnect...")
                
                if consecutive_errors > 5:
                    capture_error = "Lost connection to camera and couldn't reconnect."
                    break
                    
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                continue
            
            consecutive_errors = 0
            
            # Store the video (optional)
            out.write(frame)
            
            # Save the current frame for the attendance process to use
            cv2.imwrite(latest_frame_path, frame)
            frame_ready = True
            
            # Optional: Display the frame in a window (can be disabled in production)
            # cv2.imshow("Live Feed", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # Brief sleep to reduce CPU usage
            time.sleep(0.01)
    
    except Exception as e:
        capture_error = f"Error in video capture: {str(e)}"
        print(capture_error)
    
    finally:
        capture_active = False
        frame_ready = False
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None: 
            out.release()
        cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera-status')
def camera_status():
    global capture_active, capture_error, frame_ready
    return jsonify({
        'active': capture_active,
        'error': capture_error,
        'frame_ready': frame_ready
    })

@app.route('/start-analyze')
def start_analyze():
    global analyzing, attendance_process, capture_active, capture_error
    
    if analyzing:
        return "Analysis already running."
    
    try:
        # Start the attendance.py script with a parameter to read frames from file
        attendance_process = subprocess.Popen(['python', 'analyzer/attendance.py', '--frame-source', 'file', 
                                              '--frame-path', latest_frame_path])
        analyzing = True
        print("Started analyzing. Process ID:", attendance_process.pid)
        return "Analysis started."
    except Exception as e:
        return f"Error starting analysis: {str(e)}"

@app.route('/stop-analyze')
def stop_analyze():
    global analyzing, attendance_process
    
    if not analyzing:
        return "Analysis not running."
    
    try:
        # Terminate the attendance process
        if attendance_process:
            attendance_process.terminate()
            # On Windows you might need: os.kill(attendance_process.pid, signal.SIGTERM)
            attendance_process = None
        
        analyzing = False
        print("Stopped analyzing.")
        return "Analysis stopped."
    except Exception as e:
        return f"Error stopping analysis: {str(e)}"

if __name__ == '__main__':
    # Start the video capture thread before the Flask app
    # global capture_error
    capture_error = None
    video_thread = threading.Thread(target=video_capture, daemon=True)
    video_thread.start()
    
    # Give the camera a moment to initialize
    time.sleep(2)
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0')  # Set debug=False to avoid thread issues