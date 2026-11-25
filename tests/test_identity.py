
import numpy as np

from src.identity import GlobalIdentityManager


manager = GlobalIdentityManager(similarity_threshold=0.75)

# Simulate two different objects.
track_1 = {
    "track_id": 1,
    "class_id": 2,
    "class_name": "car",
}

track_2 = {
    "track_id": 2,
    "class_id": 2,
    "class_name": "car",
}

# Create two normalized appearance features.
feature_1 = np.zeros(384, dtype=np.float32)
feature_1[0] = 1.0

feature_2 = np.zeros(384, dtype=np.float32)
feature_2[1] = 1.0

# Frame 1: register both objects.
result_1 = manager.update(
    [track_1, track_2],
    [feature_1, feature_2],
    camera_id="camera_1",
    frame_number=1,
)

print("Frame 1:", [item["global_id"] for item in result_1])

# Frame 2: a new local track has an appearance similar to object 1.
new_track = {
    "track_id": 10,
    "class_id": 2,
    "class_name": "car",
}

result_2 = manager.update(
    [new_track],
    [feature_1.copy()],
    camera_id="camera_2",
    frame_number=2,
)

print("Frame 2:", [item["global_id"] for item in result_2])

# It should recover the first object's global identity.
assert result_2[0]["global_id"] == result_1[0]["global_id"]

print("Global identity matching test PASSED!")

# Test 2: A visually different object must receive a new global ID.

different_track = {
    "track_id": 20,
    "class_id": 2,
    "class_name": "car",
}

different_feature = np.zeros(384, dtype=np.float32)
different_feature[100] = 1.0

result_3 = manager.update(
    [different_track],
    [different_feature],
    camera_id="camera_2",
    frame_number=3,
)

print("Frame 3:", [item["global_id"] for item in result_3])

# It must NOT match global ID 1 or 2.
assert result_3[0]["global_id"] not in (1, 2)

print("Different-object rejection test PASSED!")

# Test 3: Existing local track keeps its global ID
# even when its appearance feature changes.

changed_feature = feature_1.copy()
changed_feature[50] = 0.1

result_4 = manager.update(
    [track_1],
    [changed_feature],
    camera_id="camera_1",
    frame_number=4,
)

print("Frame 4:", [item["global_id"] for item in result_4])

assert result_4[0]["global_id"] == result_1[0]["global_id"]

print("Local identity stability test PASSED!")

