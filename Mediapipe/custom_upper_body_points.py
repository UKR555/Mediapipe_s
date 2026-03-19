import cv2
import mediapipe as mp

# List of upper body landmark indices you want to use (add or remove as needed)
UPPER_BODY_LANDMARKS = [
    # mp.solutions.pose.PoseLandmark.NOSE,
    mp.solutions.pose.PoseLandmark.LEFT_EYE,
    mp.solutions.pose.PoseLandmark.RIGHT_EYE,
    mp.solutions.pose.PoseLandmark.MOUTH_LEFT,
    mp.solutions.pose.PoseLandmark.MOUTH_RIGHT,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_EAR,
    mp.solutions.pose.PoseLandmark.LEFT_THUMB,
    mp.solutions.pose.PoseLandmark.LEFT_INDEX,
    mp.solutions.pose.PoseLandmark.LEFT_PINKY,
    # Add or remove landmarks here
]


# Define connections between upper body landmarks (pairs of indices)
UPPER_BODY_CONNECTIONS = [
    # (start, end)
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    # Add more connections if needed
]

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            h, w, _ = frame.shape
            # Draw circles at each selected landmark
            for idx in UPPER_BODY_LANDMARKS:
                lm = results.pose_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            # Draw lines between connected landmarks
            for start, end in UPPER_BODY_CONNECTIONS:
                lm_start = results.pose_landmarks.landmark[start]
                lm_end = results.pose_landmarks.landmark[end]
                x1, y1 = int(lm_start.x * w), int(lm_start.y * h)
                x2, y2 = int(lm_end.x * w), int(lm_end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.imshow("Custom Upper Body Landmarks", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
