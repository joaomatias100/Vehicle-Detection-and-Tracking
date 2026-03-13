import cv2
import os

video_path = r"C:\output_tracked_classes.mp4"
output_path = r"C:\frame_1min.png"

print("Opening video...")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print("FPS:", fps)
print("Total frames:", total_frames)

# Target time: 1 minute 0 seconds → 60 seconds
target_time_sec = 60.2
target_frame = int(fps * target_time_sec)
print("Target frame:", target_frame)

# Set video position to target frame
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()

print("Frame read success:", ret)

if ret:
    success = cv2.imwrite(output_path, frame)
    print("Write success:", success)
    print("Saved to:", output_path)
else:
    print("ERROR: Could not read frame")

cap.release()