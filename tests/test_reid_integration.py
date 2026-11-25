
"""ARGUS-X real appearance-based identity recovery integration test."""

import cv2

from src.recognition import ARGUSXPipeline
from src.identity import GlobalIdentityManager


def test_real_appearance_recovery():
    video_path = "data/solutions_ci_demo.mp4"

    # Step 1: Read a real video frame.
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), f"Cannot open video: {video_path}"

    success, frame = cap.read()
    cap.release()

    assert success, "Could not read first video frame"

    # Step 2: Create a shared global identity manager.
    manager = GlobalIdentityManager(
        similarity_threshold=0.75
    )

    # Step 3: Process the frame using YOLO + DINOv2.
    pipeline = ARGUSXPipeline(
        camera_id="camera_1",
        identity_manager=manager,
        device="cpu",
    )

    _, tracks = pipeline.process_frame(frame)

    assert len(tracks) > 0, "YOLO did not detect any objects"

    # Step 4: Select a detected object.
    track = tracks[0]
    original_global_id = track["global_id"]
    local_track_id = track["track_id"]

    assert original_global_id is not None, (
        "No global identity assigned"
    )

    # Step 5: Retrieve its actual DINOv2 feature.
    cache_key = ("camera_1", local_track_id)

    assert cache_key in pipeline.feature_cache, (
        f"No cached appearance feature for track {local_track_id}"
    )

    feature = pipeline.feature_cache[cache_key]["feature"]

    assert feature is not None, "DINOv2 feature extraction failed"

    # Step 6: Simulate the same appearance arriving from camera 2
    # with a different local tracking ID.
    simulated_track = dict(track)
    simulated_track["track_id"] = 9999

    recovered_ids = manager.update(
        tracks=[simulated_track],
        features=[feature],
        camera_id="camera_2",
        frame_number=1,
    )

    recovered_global_id = recovered_ids[0]["global_id"]

    # Step 7: Display results.
    print("\n========== APPEARANCE RECOVERY TEST ==========")
    print("Object:", track["class_name"])
    print("Original camera: camera_1")
    print("New camera: camera_2")
    print("Original local ID:", local_track_id)
    print("New local ID: 9999")
    print("Original Global ID:", original_global_id)
    print("Recovered Global ID:", recovered_global_id)

    # Step 8: Verify identity association.
    assert recovered_global_id == original_global_id, (
        "Appearance-based identity recovery failed"
    )

    print("\nDINOv2 APPEARANCE RECOVERY TEST PASSED!")


if __name__ == "__main__":
    test_real_appearance_recovery()
