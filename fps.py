import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time
import os

# === CONFIGURATION ===
VIDEO_PATH = r"C:\Users\João Matias\Desktop\MVI_39051.avi"
SCORE_THRESHOLD = 0.4  # example threshold, adjust as needed
MODEL_PATH = r"C:\Users\João Matias\Desktop\Nova pasta\models\YOLOv12n.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize tracker
tracker = Tracker()

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    raise RuntimeError(f"Could not read video: {VIDEO_PATH}")

# Get total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded: {VIDEO_PATH}, Total frames: {total_frames}")

# Measure FPS
frame_idx = 0
start_time = time.time()

while ret:
    frame_idx += 1

    # YOLO detection
    results = model(frame, verbose=False)
    detections_for_tracker = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score >= SCORE_THRESHOLD:
                detections_for_tracker.append([x1, y1, x2, y2, score])

    detections_np = np.array(detections_for_tracker) if detections_for_tracker else np.empty((0, 5))

    # Update tracker
    tracker.update(frame, detections_np)

    # Read next frame
    ret, frame = cap.read()

end_time = time.time()
cap.release()
cv2.destroyAllWindows()

# Compute FPS
elapsed_time = end_time - start_time
fps = frame_idx / elapsed_time
print(f"\nProcessed {frame_idx} frames in {elapsed_time:.2f} seconds")
print(f"Approximate FPS: {fps:.2f}")
