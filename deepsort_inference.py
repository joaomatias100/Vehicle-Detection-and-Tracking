import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO
from tracker import Tracker 
from collections import defaultdict

# === CONFIGURATION AND PATHS (USER-DEFINED) ===
# Base directory for all videos
VIDEO_BASE_DIR = r"C:\Users\João Matias\Desktop\540p-Test"
# Base directory for all results (Threshold subfolders will be created here)
BASE_RESULTS_DIR = r"C:\DETRAC-Test-Tra\DETRAC-Test-Tra\results\DeepSORT\YOLOv12n"
# Define the threshold range: 0.0, 0.1, 0.2, ..., 1.0
# The round(1) ensures clean float representation for folder names
THRESHOLDS = np.arange(0.0, 1.01, 0.1).round(1) 
# Path to your YOLO model
model_path = r"C:\Users\João Matias\Desktop\Nova pasta\models\YOLOv12n.pt" 

# === LOAD YOLO MODEL ===
model = YOLO(model_path)

# --- Detection and Tracking Function ---
def process_video(video_path, score_threshold, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n▶ Starting tracking for: {video_name} @ Threshold {score_threshold:.1f} (to: {os.path.basename(output_dir)})")

    # This is the temporary file needed for the pivoting step
    temp_result_file = os.path.join(output_dir, f"{video_name}_uadetrac.txt")

    # Clean up any old temporary file
    if os.path.exists(temp_result_file):
        os.remove(temp_result_file) 

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"⚠ Could not read video: {video_path}")
        return

    # Get video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Tracker()
    frame_idx = 0

    while ret:
        frame_idx += 1
        results = model(frame, verbose=False)
        
        frame_results_uadetrac = [] 
        detections_for_tracker = []
        
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                
                # Filter detections using the current threshold
                if score >= score_threshold: 
                    detections_for_tracker.append([x1, y1, x2, y2, score])
        
        detections_np = np.array(detections_for_tracker) if len(detections_for_tracker) > 0 else np.empty((0, 5))
        tracker.update(frame, detections_np)

        # Collect results for writing
        for track in tracker.tracks:
            try:
                if track.time_since_update > 1: 
                    continue
            except AttributeError:
                pass

            x_start, y_start, w, h = track.bbox
            
            if w <= 0 or h <= 0:
                continue

            track_id = track.track_id
            
            # UADETRAC/MOT Format: 
            # [Frame ID, Track ID, x_start, y_start, width, height, Conf_Score(1), -1, -1, -1]
            frame_results_uadetrac.append(
                [frame_idx, track_id, x_start, y_start, w, h, 1, -1, -1, -1]
            )

        print(f"Processing frame {frame_idx}/{total_frames}", end="\r")

        # Write the current frame's results to the temporary text file
        if frame_results_uadetrac:
            frame_results_uadetrac_np = np.array(frame_results_uadetrac)
            with open(temp_result_file, "a") as f:
                np.savetxt(f, frame_results_uadetrac_np, 
                            fmt="%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d",
                            delimiter=',')
        
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Tracking Done: {video_name}")


# --- DETRAC FORMAT CONVERSION FUNCTION (Handles cleanup of temp file) ---
def convert_to_detrac_format(video_name, results_dir):
    mot_file = os.path.join(results_dir, f"{video_name}_uadetrac.txt")
    
    if not os.path.exists(mot_file):
        print(f"❌ Error: Temporary MOT results file not found at {mot_file}. Cannot convert.")
        return

    # Load and pivot data
    data = np.loadtxt(mot_file, delimiter=',')
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    max_frame = int(data[:, 0].max())
    max_id = int(data[:, 1].max())
    
    LX = np.zeros((max_frame, max_id))
    LY = np.zeros((max_frame, max_id))
    W = np.zeros((max_frame, max_id))
    H = np.zeros((max_frame, max_id))

    for row in data:
        frame_idx = int(row[0]) - 1
        track_idx = int(row[1]) - 1
        LX[frame_idx, track_idx] = row[2] 
        LY[frame_idx, track_idx] = row[3] 
        W[frame_idx, track_idx] = row[4]  
        H[frame_idx, track_idx] = row[5]  
    
    # Save the four final matrices
    np.savetxt(os.path.join(results_dir, f"{video_name}_LX.txt"), LX, fmt='%.4f', delimiter=' ')
    np.savetxt(os.path.join(results_dir, f"{video_name}_LY.txt"), LY, fmt='%.4f', delimiter=' ')
    np.savetxt(os.path.join(results_dir, f"{video_name}_W.txt"), W, fmt='%.4f', delimiter=' ')
    np.savetxt(os.path.join(results_dir, f"{video_name}_H.txt"), H, fmt='%.4f', delimiter=' ')
    
    # === CRITICAL STEP: DELETE THE TEMPORARY FILE ===
    os.remove(mot_file) 
    print(f"✅ DETRAC pivot conversion complete. Temporary {video_name}_uadetrac.txt removed.")


# =========================================================================
# === MAIN EXECUTION: NESTED LOOPS for VIDEOS and THRESHOLDS ================
# =========================================================================

video_paths = glob.glob(os.path.join(VIDEO_BASE_DIR, "*.avi"))
total_runs = len(video_paths) * len(THRESHOLDS)
print(f"Starting FINAL BATCH processing. Total runs: {total_runs}.")


for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for threshold in THRESHOLDS:
        
        # 1. Create the threshold folder name (e.g., 0.3 -> '0.3')
        threshold_folder_name = str(threshold).replace('.', '.')
        current_output_dir = os.path.join(BASE_RESULTS_DIR, threshold_folder_name)
        os.makedirs(current_output_dir, exist_ok=True)
        
        # 2. Process the video with the current threshold, creating the temporary .txt
        process_video(video_path, threshold, current_output_dir)
        
        # 3. Convert the temporary .txt to the four final files and delete the temporary file
        convert_to_detrac_format(video_name, current_output_dir) 

print("\n\n✨ FINAL BATCH PROCESSING COMPLETE! ✨")
print("All four DETRAC files have been generated for all videos and all thresholds (0.0 to 1.0).")
print("The intermediate '_uadetrac.txt' files were successfully deleted.")