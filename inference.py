import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    model_yolov8n = YOLO(r"C:\Users\João Matias\Desktop\Nova pasta\runs\detect\train - YOLOv12n new\weights\best.pt")

    results = model_yolov8n.val(
        data = r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml",  # image, video, or folder
        split = 'test',
        batch = 16,
        imgsz = 640
    )

    print(results.results_dict)
    print("mAP50:", results.results_dict["metrics/mAP50(B)"])
    print("mAP50-95:", results.results_dict["metrics/mAP50-95(B)"])