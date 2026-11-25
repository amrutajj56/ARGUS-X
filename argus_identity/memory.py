from collections import deque
from dataclasses import dataclass, field

@dataclass
class IdentityRecord:
    global_id: int
    class_id: int | None
    created_frame: int
    last_seen_frame: int
    gallery_size: int
    gallery: deque = field(default_factory=deque)
    last_bbox: tuple | None = None
    observations: int = 0
    active_track_id: str | None = None

    def __post_init__(self):
        self.gallery = deque(self.gallery, maxlen=self.gallery_size)

    def add(self, feature, frame_index, bbox, track_id):
        if feature is not None:
            self.gallery.append(feature.copy())
        self.last_seen_frame = frame_index
        self.last_bbox = tuple(float(v) for v in bbox)
        self.observations += 1
        self.active_track_id = str(track_id)

    def mark_lost(self):
        self.active_track_id = None

    def best_similarity(self, feature, similarity_fn):
        if feature is None or not self.gallery:
            return -1.0
        return max(similarity_fn(feature, item) for item in self.gallery)

class IdentityMemory:
    def __init__(self, gallery_size=12):
        self.gallery_size = int(gallery_size)
        self.records = {}
        self.next_global_id = 1

    def create(self, class_id, frame_index, feature, bbox, track_id):
        gid = self.next_global_id
        self.next_global_id += 1
        record = IdentityRecord(gid, class_id, frame_index, frame_index, self.gallery_size)
        record.add(feature, frame_index, bbox, track_id)
        self.records[gid] = record
        return record

    def prune(self, frame_index, max_lost_frames):
        expired = [gid for gid, rec in self.records.items()
                   if frame_index - rec.last_seen_frame > max_lost_frames]
        for gid in expired:
            del self.records[gid]
        return expired
