import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

def has_crossed_line(mid_x, mid_y, start_point, end_point):
    if end_point[0] == start_point[0]:
        return mid_x < start_point[0]  # or mid_x > start_point[0], depending on the desired direction
    else:
        slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        line_y_at_mid_x = slope * (mid_x - start_point[0]) + start_point[1]
        return mid_y > line_y_at_mid_x

# Path to input and output videos
video_path = 'video.mp4'
video_out_path = 'output_video.mp4'

# Capture input video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

input_fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer for output video
cap_out = cv2.VideoWriter(
    video_out_path, 
    cv2.VideoWriter_fourcc(*'MP4V'), 
    2,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

model = YOLO("model_colab.pt")
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

detection_threshold = 0.5
frame_skip = 20

frame_count = 0
total_count = 0
crossed_ids = set()
start_point = (200, 890)
end_point = (200, 145)

while ret:
    if frame_count % frame_skip == 0:
        results = model(frame)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if score > detection_threshold:
                    detections.append([int(x1), int(y1), int(x2), int(y2), score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                track_id = track.track_id
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2

                if has_crossed_line(mid_x, mid_y, start_point, end_point):
                    if track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        total_count += 1
                        tracker.delete_track(track_id)
                else:
                    if track_id not in crossed_ids:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,192,203) , 2)
                        label = f'ID: {track_id}'
                        cv2.circle(frame, (mid_x, mid_y), 5, (224,255,255), -1)

        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(frame, f'Total Count: {total_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,254,156), 2, cv2.LINE_AA)

        cap_out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ret, frame = cap.read()
    frame_count += 1

# Release resources properly
cap.release()
cap_out.release()
cv2.destroyAllWindows()