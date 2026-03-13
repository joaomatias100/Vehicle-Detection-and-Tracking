import os
import glob
import numpy as np

# === CONFIGURATION ===
# Source directory where your videos are located (to get MVI names)
VIDEO_BASE_DIR = r"C:\Users\João Matias\Desktop\540p-Test"

# Target directory for the 1.0 threshold results
# NOTE: The MATLAB script often expects '1_0' instead of '1.0', so we use '1_0' for robustness.
TARGET_RESULTS_DIR = r"C:\DETRAC-Test-Tra\DETRAC-Test-Tra\results\DeepSORT\YOLOv12n\1.0"

# List of the four required file suffixes
FILE_SUFFIXES = ['_LX.txt', '_LY.txt', '_W.txt', '_H.txt']

# The minimal zero matrix for the placeholder file
# np.zeros((1, 1)) creates a 1x1 matrix containing 0.0
ZERO_MATRIX = np.zeros((1, 1))
# =====================

print(f"Targeting directory: {TARGET_RESULTS_DIR}")

# 1. Ensure the target directory exists
os.makedirs(TARGET_RESULTS_DIR, exist_ok=True)

# 2. Find all video files to get their base names (e.g., MVI_39031)
video_paths = glob.glob(os.path.join(VIDEO_BASE_DIR, "*.avi"))

if not video_paths:
    print(f"❌ Error: No .avi files found in {VIDEO_BASE_DIR}")
else:
    print(f"Found {len(video_paths)} videos. Generating placeholder files...")
    
    # 3. Loop through each video and create the four files
    for video_path in video_paths:
        # Extract the base name (e.g., 'MVI_39031')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"   Generating placeholders for: {video_name}...")
        
        # 4. Create the four placeholder files
        for suffix in FILE_SUFFIXES:
            file_name = f"{video_name}{suffix}"
            full_path = os.path.join(TARGET_RESULTS_DIR, file_name)
            
            # Use np.savetxt to save the 1x1 zero matrix with the required format
            np.savetxt(full_path, ZERO_MATRIX, fmt='%.4f', delimiter=' ')

    print("\n✨ ALL 1.0 PLACEHOLDER FILES GENERATED SUCCESSFULLY. ✨")
    print(f"You can now safely restart your MATLAB evaluation script.")