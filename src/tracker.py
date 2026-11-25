"""Member 1 tracking module: YOLO detection + ByteTrack association."""

from ultralytics import YOLO


class TrackingEngine:
    """Runs YOLO detection and ByteTrack on successive frames.

    Track IDs from this module are temporary/local tracker IDs. Member 2's
    identity module should maintain separate persistent global identities.
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.35, device="cpu"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def process_frame(self, frame):
        """Process one BGR OpenCV frame and return (annotated_frame, tracks).

        Each track is a dictionary:
        {
            "track_id": int | None,
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int,
            "class_name": str
        }
        """
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

        if result.boxes is None:
            return annotated, tracks

        boxes = result.boxes
        ids = boxes.id
        id_values = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)

        xyxy_values = boxes.xyxy.cpu().tolist()
        conf_values = boxes.conf.cpu().tolist()
        cls_values = boxes.cls.int().cpu().tolist()

        for bbox, track_id, confidence, class_id in zip(
            xyxy_values, id_values, conf_values, cls_values
        ):
            x1, y1, x2, y2 = [int(value) for value in bbox]
            class_name = self.model.names.get(class_id, str(class_id))

            track = {
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": class_name,
            }
            tracks.append(track)

            label = f"{class_name} | ID {track_id if track_id is not None else '?'}"
            cv2_color = (40, 220, 100)
            import cv2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), cv2_color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cv2_color,
                2,
                cv2.LINE_AA,
            )

        return annotated, tracks
