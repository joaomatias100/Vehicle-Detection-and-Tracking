

if __name__ == "__main__":
    from ultralytics import YOLO
    from roboflow import Roboflow
    import torch  
    
    #model_8n = YOLO('yolov8n.yaml').load("yolov8n.pt")
    #model_8s = YOLO('yolov8s.yaml').load("yolov8s.pt") 28.7
    #model_8m = YOLO('yolov8m.yaml').load("yolov8m.pt")
    #model_9t = YOLO('yolov9t.yaml').load("yolov9t.pt")
    #model_9s = YOLO('yolov9s.yaml').load("yolov9s.pt") 27.4
    #model_10n = YOLO('yolov10n.yaml').load("yolov10n.pt")
    #model_10s = YOLO('yolov10s.yaml').load("yolov10s.pt")
    #model_10m = YOLO('yolov10m.yaml').load("yolov10m.pt")
    #model_11n = YOLO(r"C:\Users\João Matias\Desktop\Nova pasta\yolo11n.pt")
    #model_11s = YOLO(r"C:\Users\João Matias\Desktop\Nova pasta\yolo11s.pt")
    #model_12n = YOLO(r"C:\Users\João Matias\Desktop\Nova pasta\yolo12n.pt")
    #model_12s = YOLO(r"C:\Users\João Matias\Desktop\Nova pasta\yolo12s.pt")
    #model_6n = YOLO('yolov6n.yaml').load('yolov6n.pt')
    #model_6s = YOLO('yolov7s.yaml').load('yolov7s.pt')
    model_5n = YOLO('yolov5n.yaml').load('yolov5n.pt')
    #model_5s = YOLO('yolov5s.yaml').load('yolov5s.pt')
   # model_6n = YOLO("yolov6n.yaml")

    torch.cuda.empty_cache()

    #results8n = model_8n.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 2, cache = True)
    #results8s = model_8s.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results8m = model_8m.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results9t = model_9t.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 4, cache = False)
    #results9s = model_9s.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results10n = model_10n.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 4, cache = False)
    #results10s = model_10s.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results10m = model_10m.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results11n = model_11n.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results12n = model_12n.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results12s = model_12s.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results6n = model_6n.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results7s = model_7s.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    results5n = model_5n.train(data=r"C:\Users\João Matias\Desktop\Vehicle detection and tracking.v1i.yolov8\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = False)
    #results5s = model_5s.train(data=r"C:\Users\João Matias\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
    #results6n = model_6n.train(data=r"C:\Vehicle-detection-and-tracking-1\data.yaml", epochs=70, device='cuda', batch=6, amp=True, workers = 1, cache = True)
