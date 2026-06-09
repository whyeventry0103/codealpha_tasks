"""
Real-time Object Detection and Tracking
- Uses YOLOv8 (ultralytics) for detection
- Uses SORT tracker for multi-object tracking with unique IDs
- Works with webcam or a video file
- Run: python object_detection.py              (webcam)
       python object_detection.py video.mp4   (video file)
"""

import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Minimal SORT tracker (no external sort package needed)
# Based on: https://github.com/abewley/sort (MIT License)
# ---------------------------------------------------------------------------

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2] - bb_gt[0])   * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection + 1e-6
    return intersection / union

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)
        self.kf.R[2:,2:] *= 10.0
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self._to_z(bbox)
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hit_streak = 0

    def _to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _to_bbox(self, x):
        x0,x1,x2,x3 = x[0].item(),x[1].item(),x[2].item(),x[3].item()
        w = float(np.sqrt(abs(x2 * x3)))
        h = x2 / (w + 1e-6)
        return [x0-w/2, x1-h/2, x0+w/2, x1+h/2]

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self._to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        return self._to_bbox(self.kf.x)

    def get_state(self):
        return self._to_bbox(self.kf.x)

class SORTTracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: np.array of shape (N, 5) -> [x1, y1, x2, y2, score]
        returns: np.array of shape (M, 5) -> [x1, y1, x2, y2, id]
        """
        self.frame_count += 1
        predictions = np.zeros((len(self.trackers), 4))
        to_del = []
        for i, t in enumerate(self.trackers):
            p = t.predict()
            predictions[i] = p
            if np.any(np.isnan(p)):
                to_del.append(i)
        for i in reversed(to_del):
            self.trackers.pop(i)
            predictions = np.delete(predictions, i, axis=0)

        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predictions, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i, :4]))

        results = []
        for t in reversed(self.trackers):
            if t.time_since_update <= self.max_age:
                if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = t.get_state()
                    results.append([*bbox, t.id])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(results) if results else np.empty((0, 5))

    def _associate(self, dets, trks, iou_thresh):
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det[:4], trk)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched, unmatched_dets, unmatched_trks = [], [], []

        for d in range(len(dets)):
            if d not in row_ind:
                unmatched_dets.append(d)
        for t in range(len(trks)):
            if t not in col_ind:
                unmatched_trks.append(t)
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < iou_thresh:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append([r, c])

        return matched, unmatched_dets, unmatched_trks

# ---------------------------------------------------------------------------
# Color palette for tracking IDs
# ---------------------------------------------------------------------------

def get_color(track_id):
    palette = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 165, 0),   (0, 165, 255),   (200, 50, 200),
    ]
    return palette[track_id % len(palette)]

# ---------------------------------------------------------------------------
# Main detection + tracking loop
# ---------------------------------------------------------------------------

def run(source=0, conf_threshold=0.4, model_name="yolov8n.pt"):
    from ultralytics import YOLO

    print(f"Loading YOLOv8 model: {model_name} ...")
    model = YOLO(model_name)
    tracker = SORTTracker(max_age=8, min_hits=2, iou_threshold=0.3)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source '{source}'")
        sys.exit(1)

    print("Running detection. Press Q to quit.")
    fps_display = 0
    import time
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)[0]
        boxes = results.boxes
        class_names = model.names

        dets = []
        det_labels = {}
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                dets.append([x1, y1, x2, y2, conf])
                # We'll match label by closest box later; store by rough key
                det_labels[(round(x1), round(y1))] = (class_names[cls], conf)

        dets_np = np.array(dets) if dets else np.empty((0, 5))
        tracks = tracker.update(dets_np)

        # Draw tracked objects
        for track in tracks:
            x1, y1, x2, y2, tid = [int(v) for v in track]
            tid = int(track[4])
            color = get_color(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Find closest detection label to this track box
            best_label = "object"
            best_conf  = 0.0
            min_dist   = float("inf")
            for (dx, dy), (lbl, cf) in det_labels.items():
                dist = abs(dx - x1) + abs(dy - y1)
                if dist < min_dist:
                    min_dist = dist
                    best_label = lbl
                    best_conf  = cf

            label_text = f"ID:{tid} {best_label} {best_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS counter
        curr_time = time.time()
        fps_display = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Object Detection & Tracking (YOLOv8 + SORT) — Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        source = int(source)
    except ValueError:
        pass
    run(source=source)
