import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker  # your DeepSORT tracker - MUST BE PRESENT

# === CONFIGURATION AND PATHS ===
output_videos_dir = r"C:\Users\João Matias\Desktop\outputs"
results_dir = r"C:\Users\João Matias\Desktop\UA_DETRAC_results\YOLOv10n"
model_path = r"C:\Users\João Matias\Desktop\Nova pasta\models\YOLOv10n.pt"

# === TARGET VIDEO PATH ===
SINGLE_VIDEO_PATH = r"C:\Users\João Matias\Desktop\540p-Test\MVI_39031.avi"

# NOTE: detection_threshold is still defined but is NOT used to filter detections.
# This ensures ALL of YOLO's output is sent to the tracker.
detection_threshold = 0.5 

# Ensure output directories exist
os.makedirs(output_videos_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# === LOAD YOLO MODEL ===
model = YOLO(model_path)

# === COLORS FOR VISUALIZATION ===
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(200)]


def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n▶ Processing: {video_name}")

    video_out_path = os.path.join(output_videos_dir, f"{video_name}_out.mp4")
    result_file = os.path.join(results_dir, f"{video_name}.txt")

    # === Video setup ===
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"⚠ Could not read video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer setup
    cap_out = cv2.VideoWriter(
        video_out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # === Tracker Initialization ===
    # If possible, you should try setting custom DeepSORT parameters here,
    # for example: tracker = Tracker(max_age=20) 
    tracker = Tracker()
    frame_idx = 0

    # Clear old results file
    if os.path.exists(result_file):
        os.remove(result_file)

    # === Main Processing Loop ===
    while ret:
        frame_idx += 1
        
        # 1. RUN DETECTION
        results = model(frame, verbose=False)
        detections = []

        # 2. COLLECT ALL DETECTIONS (NO CONFIDENCE FILTER)
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                
                # --- CORRECTION: NO FILTER APPLIED ---
                # All detections, regardless of score, are passed to the tracker.
                
                # Format: [x1, y1, x2, y2, score]
                detections.append([x1, y1, x2, y2, score])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # 3. UPDATE TRACKER
        tracker.update(frame, detections)

        # 4. PROCESS TRACKING RESULTS AND OUTPUT
        frame_results = []
        for track in tracker.tracks:
            # Get bbox in (x_tl, y_tl, w, h) format
            x, y, w, h = track.bbox
            if w <= 0 or h <= 0:
                continue

            # Convert to (x1, y1, x2, y2) for drawing and output
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            track_id = track.track_id
            color = colors[track_id % len(colors)]

            # Draw bbox on frame for video output
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # UA-DETRAC line: frame,id,x,y,w,h,1,-1,-1,-1
            # Note: w = x2 - x1; h = y2 - y1
            frame_results.append(
                [frame_idx, track_id, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1]
            )

        # Append to file only if there are active tracks
        if frame_results:
            frame_results = np.array(frame_results)
            with open(result_file, "a") as f:
                np.savetxt(f, frame_results, fmt="%d,%d,%d,%d,%d,%d,%d,%d,%d,%d")

        cap_out.write(frame)
        print(f"Processing frame {frame_idx}/{total_frames}", end="\r")

        ret, frame = cap.read()

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Done: {video_name}")
    print(f"   Output video: {video_out_path}")
    print(f"   Results file: {result_file}")


# ==========================================
# === MAIN EXECUTION: Process single video ===
# ==========================================
if os.path.exists(SINGLE_VIDEO_PATH):
    print(f"Starting tracking for video: {SINGLE_VIDEO_PATH}")
    process_video(SINGLE_VIDEO_PATH)
    print("\n🎯 Single video processed successfully!")
else:
    print(f"\n❌ Error: Video file not found at {SINGLE_VIDEO_PATH}")