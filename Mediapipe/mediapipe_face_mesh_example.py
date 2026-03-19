import numpy as np
import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# Indices for lips and eyes (468 landmark model)
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

mouth_open = False
mouth_open_count = 0
eye_closed = False
#number of faces detected
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        h, w, _ = frame.shape
        talking = False
        sleeping = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks
                num_landmarks = len(face_landmarks.landmark)
                for idx, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Rainbow color: map idx to HSV, then convert to BGR
                    hue = int(179 * idx / num_landmarks)  # OpenCV hue range: 0-179
                    color_hsv = np.uint8([[[hue, 255, 255]]])
                    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    cv2.circle(frame, (cx, cy), 2, color_bgr, cv2.FILLED)
                # Get mouth and eye points
                upper_lip = face_landmarks.landmark[UPPER_LIP]
                lower_lip = face_landmarks.landmark[LOWER_LIP]
                left_eye_upper = face_landmarks.landmark[LEFT_EYE_UPPER]
                left_eye_lower = face_landmarks.landmark[LEFT_EYE_LOWER]
                right_eye_upper = face_landmarks.landmark[RIGHT_EYE_UPPER]
                right_eye_lower = face_landmarks.landmark[RIGHT_EYE_LOWER]
                # Calculate distances
                mouth_dist = euclidean_distance((upper_lip.x * w, upper_lip.y * h), (lower_lip.x * w, lower_lip.y * h))
                left_eye_dist = euclidean_distance((left_eye_upper.x * w, left_eye_upper.y * h), (left_eye_lower.x * w, left_eye_lower.y * h))
                right_eye_dist = euclidean_distance((right_eye_upper.x * w, right_eye_upper.y * h), (right_eye_lower.x * w, right_eye_lower.y * h))
                # Thresholds (may need tuning)
                if mouth_dist > 15:
                    talking = True
                    if not mouth_open:
                        mouth_open_count += 1
                        mouth_open = True
                else:
                    mouth_open = False
                if left_eye_dist < 5 and right_eye_dist < 5:
                    sleeping = True
                # Draw status
                cv2.putText(frame, f"Talking: {'Yes' if talking else 'No'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, f"Mouth Opens: {mouth_open_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(frame, f"State: {'Sleeping' if sleeping else 'Awake'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("MediaPipe Face Mesh", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
