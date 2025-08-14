# import os
# import random

# import cv2
# from ultralytics import YOLO

# from tracker import Tracker


# video_path = os.path.join('.', 'data', 'video.mp4')
# video_out_path = os.path.join('.', 'output_video.mp4')

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()

# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
#                           (frame.shape[1], frame.shape[0]))

# model = YOLO("model_colab.pt")

# tracker = Tracker()

# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# detection_threshold = 0.5
# while ret:

#     results = model(frame)

#     for result in results:
#         detections = []
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             if score > detection_threshold:
#                 detections.append([x1, y1, x2, y2, score])

#         tracker.update(frame, detections)

#         for track in tracker.tracks:
#             bbox = track.bbox
#             x1, y1, x2, y2 = bbox
#             track_id = track.track_id

#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

#     cap_out.write(frame)
#     ret, frame = cap.read()

# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()


import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

# Video yolunu ve çıkış yolunu tanımla
video_path = os.path.join('video.mp4')
video_out_path = os.path.join('.', 'output_video.mp4')

# Video kaynağını aç
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video could not be opened.")
    exit()

# İlk kareyi oku
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: The first frame could not be read.")
    cap.release()
    exit()

# Video çıkış dosyasını ayarla
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# YOLO modelini yükle
model = YOLO("model_colab.pt")

# Tracker'ı başlat
tracker = Tracker()

# Rastgele renkler oluştur
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

while ret:

    # Nesne tespiti yap
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        # Tracker'ı güncelle
        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            # Takip edilen nesneyi çerçevele
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)

    # Çerçeveyi çıkış videosuna yaz
    cap_out.write(frame)

    # Bir sonraki kareyi oku
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Frame could not be read.")
        break

# Kaynakları serbest bırak
cap.release()
cap_out.release()
cv2.destroyAllWindows()


from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from filterpy.kalman import KalmanFilter

import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'model_data/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox