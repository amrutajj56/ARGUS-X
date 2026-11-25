
import numpy as np

from src.identity import GlobalIdentityManager


def make_track(track_id):
    return {
        "track_id": track_id,
        "class_id": 2,
        "class_name": "car",
    }


manager = GlobalIdentityManager(
    similarity_threshold=0.75,
    max_local_mapping_age=60,
)

feature_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
feature_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# Create identities in two different cameras.
camera_1_result = manager.update(
    [make_track(1)], [feature_a], "camera_1", 1
)

camera_2_result = manager.update(
    [make_track(2)], [feature_b], "camera_2", 1
)

camera_1_id = camera_1_result[0]["global_id"]
camera_2_id = camera_2_result[0]["global_id"]

# Advance only Camera 1 beyond its expiry limit.
manager.update([], [], "camera_1", 62)

# Camera 1's old mapping should expire.
assert ("camera_1", 1) not in manager.local_to_global

# Camera 2's mapping should remain untouched.
assert manager.local_to_global[("camera_2", 2)] == camera_2_id

print("Camera 1 mapping expiry: PASSED")
print("Camera 2 mapping isolation: PASSED")

# Camera 2 should still reuse its original global identity.
result = manager.update(
    [make_track(2)], [feature_b], "camera_2", 2
)

assert result[0]["global_id"] == camera_2_id
assert result[0]["identity_match_status"] == "known_local_track"

print("Camera 2 identity stability: PASSED")

print("\nALL MULTI-CAMERA ISOLATION TESTS PASSED!")
