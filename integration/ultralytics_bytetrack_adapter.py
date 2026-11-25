class UltralyticsByteTrackAdapter:
    # Reuse one adapter instance throughout one stream (persist=True).
    def __init__(self, model_path="yolo11n.pt", device="cpu",
                 classes=None, confidence=0.25, tracker_config="bytetrack.yaml"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: python -m pip install -r requirements.txt") from exc
        self.model = YOLO(model_path)
        self.device = device
        self.classes = classes
        self.confidence = confidence
        self.tracker_config = tracker_config

    def process(self, frame):
        results = self.model.track(source=frame, persist=True,
                                   tracker=self.tracker_config, device=self.device,
                                   classes=self.classes, conf=self.confidence,
                                   verbose=False)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        return [{"track_id": int(tid),
                 "bbox": [float(v) for v in bbox],
                 "confidence": float(conf),
                 "class_id": int(cls)}
                for bbox, tid, conf, cls in zip(xyxy, ids, confs, classes)]
