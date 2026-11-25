
"""Member 1 tracking module: YOLO detection + ByteTrack association."""

import cv2
from ultralytics import YOLO

from src.lifecycle import TrackLifecycleManager


class TrackingEngine:
    """YOLO detection, ByteTrack association and lifecycle monitoring."""

    def __init__(self, model_path="yolov8n.pt", confidence=0.35, device="cpu"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

        # Monitor the lifecycle of temporary ByteTrack IDs.
        self.lifecycle = TrackLifecycleManager(max_lost_frames=30)
        self.frame_number = 0

    def process_frame(self, frame):
        """Process one frame and return (annotated_frame, tracks)."""

        self.frame_number += 1

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        annotated = frame.copy()
        tracks = []

        if result.boxes is not None:
            boxes = result.boxes
            ids = boxes.id

            id_values = (
                ids.int().cpu().tolist()
                if ids is not None
                else [None] * len(boxes)
            )

            xyxy_values = boxes.xyxy.cpu().tolist()
            conf_values = boxes.conf.cpu().tolist()
            cls_values = boxes.cls.int().cpu().tolist()

            for bbox, track_id, confidence, class_id in zip(
                xyxy_values, id_values, conf_values, cls_values
            ):
                # ARGUS-X focuses on human tracking.
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = [int(value) for value in bbox]
                class_name = self.model.names.get(class_id, str(class_id))

                tracks.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": class_name,
                })

        # Update lifecycle even when no objects are detected.
        tracks = self.lifecycle.update(tracks, self.frame_number)

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["track_id"]
            status = track["lifecycle_status"]

            label = (
                f"{track['class_name']} | ID {track_id} | {status}"
            )

            color = (40, 220, 100)
            if status == "recovered":
                color = (0, 200, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated, tracks
