from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        # Absolute path to the Re-ID model
        encoder_model_filename = r"C:\Users\João Matias\Desktop\Nova pasta\model_data\veri_vehicle_model.pb"

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # Initializes the core DeepSORT tracker with default max_age/n_init parameters
        self.tracker = DeepSortTracker(metric)
        
        # Loads the Re-ID model
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        # 1. Prepare Detections for DeepSORT
        bboxes = np.asarray([d[:-1] for d in detections])
        # Converts [x1, y1, x2, y2] to [x, y, w, h] for Kalman Filter
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        # 2. Extract Re-ID Features
        features = self.encoder(frame, bboxes)

        # 3. Create DeepSORT Detection Objects
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        # 4. Run Tracking Core
        self.tracker.predict()
        self.tracker.update(dets)
        
        # 5. Filter and Output Results
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # Gets the bounding box in TLBR [x1, y1, x2, y2] format
            bbox_tlbr = track.to_tlbr()
            
            # === CRITICAL CHANGE: Convert TLBR (x1, y1, x2, y2) to TLWH (x, y, w, h) ===
            # The deepsort_inference.py file expects [x, y, w, h] format when unpacking
            # x, y, w, h = track.bbox (in the inference file)
            x1, y1, x2, y2 = bbox_tlbr
            w = x2 - x1
            h = y2 - y1
            
            # The final box must be [x1, y1, w, h] to match the unpacking in the inference script
            bbox_tlwh = [x1, y1, w, h]
            # =========================================================================

            id = track.track_id

            # Packages the ID and the new TLWH box into your custom Track object
            tracks.append(Track(id, bbox_tlwh))

        self.tracks = tracks


class Track:
    # Class attributes (optional, but harmless)
    track_id = None
    bbox = None 

    def __init__(self, id, bbox):
        # Ensure the attribute is correctly named 'self.bbox'
        self.track_id = id
        self.bbox = bbox