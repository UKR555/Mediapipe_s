import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation
cap = cv2.VideoCapture(0)

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image_rgb)
        mask = results.segmentation_mask
        if mask is not None:
            condition = mask > 0.5
            bg = np.zeros(frame.shape, dtype=np.uint8)
            bg[:] = (0, 0, 255)  # Red background
            output = np.where(condition[..., None], frame, bg)
            cv2.imshow("MediaPipe Selfie Segmentation", output)
        else:
            cv2.imshow("MediaPipe Selfie Segmentation", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
