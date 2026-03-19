import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ab = [a.x - b.x, a.y - b.y]
    cb = [c.x - b.x, c.y - b.y]
    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    ab_mag = math.sqrt(ab[0]**2 + ab[1]**2)
    cb_mag = math.sqrt(cb[0]**2 + cb[1]**2)
    angle = math.acos(dot_product / (ab_mag * cb_mag))
    return math.degrees(angle)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check if your webcam is connected and not used by another application.")
    input("Press Enter to exit...")
    exit()

# Initialize Pose model
with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam. Try restarting your computer or checking webcam drivers.")
            input("Press Enter to exit...")
            break

        # Convert frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks and extract key points
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # Extract keypoints
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Head orientation: eye vector
            head_vector_x = left_eye.x - right_eye.x
            head_direction = "Left" if head_vector_x > 0.02 else "Right" if head_vector_x < -0.02 else "Center"

            # Arm extension distances
            left_arm_length = euclidean_distance(left_shoulder, left_elbow) + euclidean_distance(left_elbow, left_wrist)
            right_arm_length = euclidean_distance(right_shoulder, right_elbow) + euclidean_distance(right_elbow, right_wrist)

            # Elbow angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Print results
            print(f"Head Direction: {head_direction}")
            print(f"Left Arm Length: {left_arm_length:.2f}")
            print(f"Right Arm Length: {right_arm_length:.2f}")
            print(f"Left Elbow Angle: {left_elbow_angle:.2f}°")
            print(f"Right Elbow Angle: {right_elbow_angle:.2f}°")
            print("---")

        # Display the frame
        try:
            cv2.imshow("Pose Detection", frame)
        except Exception as e:
            print(f"Error displaying window: {e}\nAre you running in a remote or headless environment? GUI windows may not be supported.")
            input("Press Enter to exit...")
            break

        # Exit loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
