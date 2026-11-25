
import cv2
import numpy as np

from src.reid import AppearanceExtractor
from src.tracker import TrackingEngine


# Read one frame from our demo video.
cap = cv2.VideoCapture("data/solutions_ci_demo.mp4")
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read demo video.")

# Detect objects using our existing YOLO + ByteTrack engine.
tracker = TrackingEngine()
_, tracks = tracker.process_frame(frame)

if not tracks:
    raise RuntimeError("No objects detected in the first frame.")

# Load DINOv2 appearance extractor.
extractor = AppearanceExtractor(device="cpu")

# Extract features for the first detected object.
feature = extractor.extract(frame, tracks[0]["bbox"])

if feature is None:
    raise RuntimeError("Appearance feature extraction failed.")

print("Object:", tracks[0]["class_name"])
print("Local Track ID:", tracks[0]["track_id"])
print("Feature shape:", feature.shape)
print("Feature norm:", np.linalg.norm(feature))

# Compare the feature with itself.
score = extractor.similarity(feature, feature)
print("Self similarity:", score)

assert feature.shape == (384,)
assert np.isclose(score, 1.0, atol=1e-5)

print("DINOv2 appearance extraction test PASSED!")
