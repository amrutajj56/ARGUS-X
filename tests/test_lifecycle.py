
from src.lifecycle import TrackLifecycleManager

manager = TrackLifecycleManager(max_lost_frames=3)

def make_track(track_id):
    return {
        "track_id": track_id,
        "bbox": [100, 100, 200, 200],
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "person",
    }

# Frame 1: Object appears
print("Frame 1:", manager.update([make_track(1)], 1)[0]["lifecycle_status"])

# Frames 2 and 3: Object disappears
manager.update([], 2)
manager.update([], 3)
print("Frame 3: Lost =", manager.records[1]["status"])

# Frame 4: Same ID returns
print("Frame 4:", manager.update([make_track(1)], 4)[0]["lifecycle_status"])

# Frames 5 to 9: Object disappears for too long
for frame in range(5, 9):
    manager.update([], frame)

print("Frame 8: Removed =", manager.records[1]["status"])

print("Statistics:", manager.get_statistics())
