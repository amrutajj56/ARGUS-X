
import cv2

from src.recognition import ARGUSXPipeline


VIDEO_PATH = "data/solutions_ci_demo.mp4"

pipeline = ARGUSXPipeline(
    camera_id="camera_1",
    device="cpu",
    feature_refresh_interval=15,
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_count = 0
unique_global_ids = set()
unique_local_ids = set()

try:
    while frame_count < 30:
        success, frame = cap.read()

        if not success:
            break

        annotated, tracks = pipeline.process_frame(frame)
        frame_count += 1

        for track in tracks:
            if track["track_id"] is not None:
                unique_local_ids.add(
                    (track["camera_id"], track["track_id"])
                )

            if track["global_id"] is not None:
                unique_global_ids.add(track["global_id"])

        if frame_count % 10 == 0:
            print(
                f"Frame: {frame_count} | "
                f"Visible objects: {len(tracks)} | "
                f"Global IDs observed: {len(unique_global_ids)}"
            )

finally:
    cap.release()

print("\nFrames processed:", frame_count)
print("Unique local tracks:", len(unique_local_ids))
print("Unique global identities:", len(unique_global_ids))

assert frame_count > 0, "No video frames were processed."

print("\nARGUS-X integrated pipeline test PASSED!")
