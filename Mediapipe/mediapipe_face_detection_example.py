import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        if results.detections:
            for detection in results.detections:
                # Draw detection with MediaPipe utility (handles multiple faces)
                mp_drawing.draw_detection(frame, detection)
                # Also overlay confidence score for clarity
                score = detection.score[0] if detection.score else 0.0
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
                cv2.putText(frame, f"{score:.2f}", (max(0, x), max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
