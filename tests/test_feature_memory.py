
import numpy as np

from src.identity import GlobalIdentityManager


manager = GlobalIdentityManager(max_features=10)

track = {
    "track_id": 1,
    "class_id": 2,
    "class_name": "car",
}

for frame in range(1, 16):
    feature = np.array(
        [1.0, frame * 0.001, 0.0],
        dtype=np.float32,
    )

    manager.update(
        [track],
        [feature],
        "camera_1",
        frame,
    )

identity = manager.identities[1]

print("Total observations:", identity["observations"])
print("Stored features:", len(identity["features"]))

assert identity["observations"] == 15
assert len(identity["features"]) == 10

print("\nFEATURE MEMORY LIMIT TEST PASSED!")
