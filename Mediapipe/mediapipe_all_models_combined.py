
import cv2
import mediapipe as mp
import numpy as np
import math
import time
output_last_time = time.time()
output_interval = 120  # seconds (2 minutes)
latest_values = {}
def euclidean_distance_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle_3d(a, b, c):
    ab = [a.x - b.x, a.y - b.y, a.z - b.z]
    cb = [c.x - b.x, c.y - b.y, c.z - b.z]
    dot_product = sum([ab[i]*cb[i] for i in range(3)])
    ab_mag = math.sqrt(sum([ab[i]**2 for i in range(3)]))
    cb_mag = math.sqrt(sum([cb[i]**2 for i in range(3)]))
    if ab_mag * cb_mag == 0:
        return 0.0
    angle = math.acos(dot_product / (ab_mag * cb_mag))
    return math.degrees(angle)

mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Holistic: pose, hands, face mesh
        results_holistic = holistic.process(image_rgb)
        # Face Detection
        results_face = face_detection.process(image_rgb)
        # Selfie Segmentation
        results_seg = selfie_segmentation.process(image_rgb)

        # Optionally, you can use the segmentation mask for other effects, but no background change is applied now
        # mask = results_seg.segmentation_mask if results_seg.segmentation_mask is not None else None
        # if mask is not None:
        #     # Example: overlay mask as alpha, or ignore for no background effect
        #     pass


        # Face detection box removed as requested


        # Draw pose, hand, and face mesh landmarks with connections
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        if results_holistic.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Extract keypoints for calculations
            landmarks = results_holistic.pose_landmarks.landmark
            try:
                nose = landmarks[mp_holistic.PoseLandmark.NOSE]
                left_eye = landmarks[mp_holistic.PoseLandmark.LEFT_EYE]
                right_eye = landmarks[mp_holistic.PoseLandmark.RIGHT_EYE]
                left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW]
                right_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]

                # Head orientation: use vector from nose to midpoint between eyes for yaw (left/right turn)
                mid_eye_x = (left_eye.x + right_eye.x) / 2
                mid_eye_y = (left_eye.y + right_eye.y) / 2
                mid_eye_z = (left_eye.z + right_eye.z) / 2
                nose_to_eye = [mid_eye_x - nose.x, mid_eye_y - nose.y, mid_eye_z - nose.z]
                # Shoulder vector (left to right)
                shoulder_vec = [right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y, right_shoulder.z - left_shoulder.z]
                # Head yaw: project nose_to_eye onto shoulder_vec (left/right)
                yaw = (nose_to_eye[0]*shoulder_vec[0] + nose_to_eye[1]*shoulder_vec[1] + nose_to_eye[2]*shoulder_vec[2])
                if yaw > 0.01:
                    head_direction = "Right"
                elif yaw < -0.01:
                    head_direction = "Left"
                else:
                    head_direction = "Center"

                # Use shoulder width for normalization
                shoulder_width = euclidean_distance_3d(left_shoulder, right_shoulder)
                # Arm extension distances (normalized)
                left_arm_length = (euclidean_distance_3d(left_shoulder, left_elbow) + euclidean_distance_3d(left_elbow, left_wrist)) / shoulder_width
                right_arm_length = (euclidean_distance_3d(right_shoulder, right_elbow) + euclidean_distance_3d(right_elbow, right_wrist)) / shoulder_width

                # Elbow angles (3D)
                left_elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)

                # Display results on frame
                cv2.putText(frame, f"Head: {head_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(frame, f"L Arm: {left_arm_length:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                cv2.putText(frame, f"R Arm: {right_arm_length:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"L Elbow: {left_elbow_angle:.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
                cv2.putText(frame, f"R Elbow: {right_elbow_angle:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                # Store latest values for periodic output
                latest_values = {
                    "Head": head_direction,
                    "L_Arm": left_arm_length,
                    "R_Arm": right_arm_length,
                    "L_Elbow": left_elbow_angle,
                    "R_Elbow": right_elbow_angle
                }

                # Every 2 minutes, write the latest values to a file
                if latest_values and (time.time() - output_last_time) > output_interval:
                    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    for k, v in latest_values.items():
                        print(f"{k}: {v}")
                    print("---")
                    with open("movement_output.txt", "a") as f:
                        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        for k, v in latest_values.items():
                            f.write(f"{k}: {v}\n")
                        f.write("---\n")
                    output_last_time = time.time()
            except Exception as e:
                pass

        if results_holistic.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)

        if results_holistic.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)

        if results_holistic.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        cv2.imshow("MediaPipe All Models Combined", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
