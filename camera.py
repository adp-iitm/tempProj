import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ Webcam opened successfully")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Webcam Test", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("❌ Webcam failed to open")

cap.release()
cv2.destroyAllWindows()
