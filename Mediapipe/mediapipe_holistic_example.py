import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        # Draw pose landmarks
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        # Draw hand landmarks
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # Draw face landmarks
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 0, 255), cv2.FILLED)
        cv2.imshow("MediaPipe Holistic", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
