import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta
from ultralytics import YOLO
from tracker import Tracker 

# =========================
# CONFIGURATION
# =========================
VIDEO_PATH = r"C:\Users\João Matias\Desktop\UA-DETRAC\UA_DETRAC_2016\540p-Test\MVI_39361.avi"
OUTPUT_VIDEO_PATH = r"C:\Users\João Matias\Desktop\output_tracked_classes.mp4"
MODEL_PATH = r"C:\Users\João Matias\Desktop\Nova pasta\models\YOLOv8n.pt"

FRAME_LOG_PATH = r"C:\Users\João Matias\Desktop\tracking_log.csv"
SUMMARY_LOG_PATH = r"C:\Users\João Matias\Desktop\tracking_summary.csv"
SUMMARY_TXT_PATH = r"C:\Users\João Matias\Desktop\tracking_summary.txt"

SCORE_THRESHOLD = 0.4
CLASS_DECISION_FRAMES = 20

# =========================
# LOAD MODEL + TRACKER
# =========================
model = YOLO(MODEL_PATH)
tracker = Tracker()

# =========================
# CLASS MEMORY STORAGE
# =========================
track_class_memory = {}

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not read video: {VIDEO_PATH}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# =========================
# VIDEO OUTPUT
# =========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(f"Processing: {os.path.basename(VIDEO_PATH)}")
print(f"Total frames: {total_frames}")

# =========================
# COLORS
# =========================
CLASS_COLORS = {
    'car': (0, 255, 255),
    'bus': (255, 0, 255),
    'van': (0, 0, 255),
    'other': (255, 255, 0)
}

# =========================
# CLASS MAP
# =========================
CLASS_ID_TO_NAME = {
    0: 'car',
    1: 'bus',
    2: 'van'
}

# =========================
# DATA STORAGE
# =========================
records = []

track_lifetimes = defaultdict(lambda: {
    "first": None,
    "last": None,
    "class": None
})

frame_idx = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1

    # ---------------------
    # YOLO DETECTION
    # ---------------------
    results = model(frame, verbose=False)

    detections_for_tracker = []
    class_ids_for_detections = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score >= SCORE_THRESHOLD:
                detections_for_tracker.append([x1, y1, x2, y2, score])
                class_ids_for_detections.append(int(class_id))

    detections_np = (
        np.array(detections_for_tracker)
        if len(detections_for_tracker) > 0
        else np.empty((0, 5))
    )

    # ---------------------
    # TRACKER UPDATE
    # ---------------------
    tracker.update(frame, detections_np)

    # Remove dead tracks
    active_ids = {t.track_id for t in tracker.tracks}
    track_class_memory = {k: v for k, v in track_class_memory.items() if k in active_ids}

    # ---------------------
    # PROCESS TRACKS
    # ---------------------
    for idx, track in enumerate(tracker.tracks):

        if hasattr(track, 'time_since_update') and track.time_since_update > 1:
            continue

        x, y, w, h = track.bbox
        track_id = track.track_id

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # CURRENT CLASS
        if idx < len(class_ids_for_detections):
            detected_class_id = class_ids_for_detections[idx]
            detected_class_name = CLASS_ID_TO_NAME.get(detected_class_id, 'other')
        else:
            detected_class_name = 'other'

        # INIT MEMORY
        if track_id not in track_class_memory:
            track_class_memory[track_id] = {
                "votes": [],
                "final_class": None,
                "locked": False
            }

        memory = track_class_memory[track_id]

        # VOTING
        if not memory["locked"]:
            memory["votes"].append(detected_class_name)

            if len(memory["votes"]) >= CLASS_DECISION_FRAMES:
                memory["final_class"] = max(
                    set(memory["votes"]),
                    key=memory["votes"].count
                )
                memory["locked"] = True

        # FINAL CLASS
        if memory["locked"]:
            class_name = memory["final_class"]
        else:
            class_name = detected_class_name

        # ---------------------
        # STORE DATA
        # ---------------------
        time_seconds = frame_idx / fps

        records.append({
            "frame": frame_idx,
            "time_sec": time_seconds,
            "object_id": track_id,
            "class": class_name
        })

        # lifetime tracking
        if track_lifetimes[track_id]["first"] is None:
            track_lifetimes[track_id]["first"] = frame_idx

        track_lifetimes[track_id]["last"] = frame_idx
        track_lifetimes[track_id]["class"] = class_name

        # ---------------------
        # DRAW
        # ---------------------
        color = CLASS_COLORS[class_name]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} ID:{track_id}",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

    if frame_idx % 50 == 0:
        print(f"Progress: {frame_idx}/{total_frames}", end="\r")

# =========================
# CREATE DATAFRAME
# =========================
df = pd.DataFrame(records)

# =========================
# SUMMARY TABLE
# =========================
summary_rows = []

for track_id, info in track_lifetimes.items():

    duration_frames = info["last"] - info["first"]
    duration_seconds = duration_frames / fps

    summary_rows.append({
        "object_id": track_id,
        "class": info["class"],
        "first_frame": info["first"],
        "last_frame": info["last"],
        "duration_frames": duration_frames,
        "duration_seconds": duration_seconds,
        "duration_hms": str(timedelta(seconds=duration_seconds))
    })

summary_df = pd.DataFrame(summary_rows)

# =========================
# SAVE CSV FILES
# =========================
df.to_csv(FRAME_LOG_PATH, index=False)
summary_df.to_csv(SUMMARY_LOG_PATH, index=False)

# =========================
# WRITE SUMMARY TXT
# =========================
with open(SUMMARY_TXT_PATH, "w") as f:
    f.write("Tracked objects summary:\n\n")
    for idx, row in summary_df.iterrows():
        f.write(f"ID: {row['object_id']}, Class: {row['class']}, Duration: {row['duration_seconds']:.2f} sec\n")

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nDone!")
print(f"Video → {OUTPUT_VIDEO_PATH}")
print(f"Frame log → {FRAME_LOG_PATH}")
print(f"Summary CSV → {SUMMARY_LOG_PATH}")
print(f"Summary TXT → {SUMMARY_TXT_PATH}")
