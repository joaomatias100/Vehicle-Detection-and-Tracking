import os
import glob
import cv2
import xml.etree.ElementTree as ET

# ---------- CONFIG ----------
yolo_txt_dir = r"C:\Users\João Matias\Desktop\Nova pasta\runs\detect\train - YOLOv10n new\weights\runs\detect\predict\labels"  # folder containing all YOLO .txt files
video_dir = r"C:\Users\João Matias\Desktop\MVI_39031.avi"
output_xml_dir = r"C:\Users\João Matias\Desktop\UA-DETRAC-XML"  # where XMLs will be saved
os.makedirs(output_xml_dir, exist_ok=True)

# Mapping class indices to vehicle types (adjust if you have more classes)
class_map = {0: "car", 1: "bus", 2: "van", 3: "truck"}  

# ---------- PROCESS ALL VIDEOS ----------
videos = [video_dir]

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {video_name}...")

    # Get frame size
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create XML root
    root = ET.Element("sequence", {"name": video_name})

    # Find all YOLO txts for this video
    pattern = os.path.join(yolo_txt_dir, f"{video_name}_*.txt")
    txt_files = sorted(glob.glob(pattern), key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))

    for txt_file in txt_files:
        # Frame number from filename
        frame_num = int(os.path.splitext(os.path.basename(txt_file))[0].split("_")[-1])
        frame_elem = ET.SubElement(root, "frame", {"num": str(frame_num)})
        target_list = ET.SubElement(frame_elem, "target_list")

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for obj_id, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_idx, x_c, y_c, w, h, conf = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

            # Convert normalized coords to pixel coordinates
            left = (x_c - w/2) * frame_width
            top  = (y_c - h/2) * frame_height
            width = w * frame_width
            height = h * frame_height

            target = ET.SubElement(target_list, "target", {"id": str(obj_id)})
            ET.SubElement(target, "box", {
                "left": f"{left:.2f}",
                "top": f"{top:.2f}",
                "width": f"{width:.2f}",
                "height": f"{height:.2f}"
            })
            ET.SubElement(target, "attribute", {"vehicle_type": class_map.get(cls_idx, "unknown")})

    # Write XML to file
    tree = ET.ElementTree(root)
    xml_out_path = os.path.join(output_xml_dir, f"{video_name}.xml")
    tree.write(xml_out_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved {xml_out_path}")
