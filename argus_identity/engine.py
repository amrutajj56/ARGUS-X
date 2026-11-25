import math
import numpy as np
from .config import IdentityConfig
from .features import AppearanceExtractor, cosine_similarity
from .memory import IdentityMemory

class IdentityEngine:
    # Call update once for every processed frame, including frames with no tracks.
    def __init__(self, config=None, extractor=None):
        self.config = config or IdentityConfig()
        self.extractor = extractor or AppearanceExtractor(self.config.descriptor_bins)
        self.memory = IdentityMemory(self.config.gallery_size)
        self.frame_index = -1
        self.track_to_global = {}

    @staticmethod
    def _center(bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def _eligible(self, rec, class_id, bbox, frame_shape):
        if self.frame_index - rec.last_seen_frame > self.config.max_lost_frames:
            return False
        if (self.config.require_class_match and rec.class_id is not None
                and class_id is not None and rec.class_id != class_id):
            return False
        if self.config.use_spatial_gate and rec.last_bbox is not None:
            h,w = frame_shape[:2]
            diagonal = max(math.hypot(w,h), 1.0)
            if math.dist(self._center(rec.last_bbox), self._center(bbox))/diagonal > self.config.max_center_distance:
                return False
        return True

    def _match(self, feature, class_id, bbox, frame_shape, claimed):
        if feature is None:
            return None, -1.0, False
        candidates = []
        for gid, rec in self.memory.records.items():
            if gid in claimed or rec.active_track_id is not None:
                continue
            if self._eligible(rec, class_id, bbox, frame_shape):
                candidates.append((rec.best_similarity(feature, cosine_similarity), gid))
        candidates.sort(reverse=True)
        if not candidates:
            return None, -1.0, False
        best, gid = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else -1.0
        ambiguous = second >= self.config.match_threshold and best-second < self.config.ambiguity_margin
        if best < self.config.match_threshold or ambiguous:
            return None, best, ambiguous
        return gid, best, False

    def update(self, frame, tracks):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV BGR image")
        if not isinstance(tracks, (list, tuple)):
            raise TypeError("tracks must be a list of dictionaries")
        self.frame_index += 1
        self.memory.prune(self.frame_index, self.config.max_lost_frames)
        visible = {str(t["track_id"]) for t in tracks if "track_id" in t}
        for rec in self.memory.records.values():
            if rec.active_track_id not in visible:
                rec.mark_lost()

        output, claimed, seen = [], set(), set()
        for track in tracks:
            if "track_id" not in track or "bbox" not in track:
                raise ValueError("each track requires track_id and bbox")
            tid = str(track["track_id"])
            if tid in seen:
                raise ValueError("duplicate track_id in one frame: " + tid)
            seen.add(tid)
            bbox = [float(v) for v in track["bbox"]]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("bbox must be valid xyxy coordinates")
            class_id = track.get("class_id")
            det_conf = float(track.get("confidence", 1.0))
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            crop_ok = (area >= self.config.min_box_area
                       and bbox[2]-bbox[0] >= self.config.min_crop_width
                       and bbox[3]-bbox[1] >= self.config.min_crop_height
                       and det_conf >= self.config.min_detection_confidence)
            feature = self.extractor.extract(frame, bbox) if crop_ok else None

            # Trust the tracker ID only if its identity stayed continuously active.
            mapped_gid = self.track_to_global.get(tid)
            mapped_rec = self.memory.records.get(mapped_gid)
            if mapped_rec is not None and mapped_rec.active_track_id == tid:
                gid, status, score = mapped_gid, "confirmed", 1.0
            else:
                gid, score, ambiguous = self._match(feature, class_id, bbox, frame.shape, claimed)
                if gid is not None:
                    status = "recovered"
                else:
                    rec = self.memory.create(class_id, self.frame_index, feature, bbox, tid)
                    gid = rec.global_id
                    status = "uncertain" if ambiguous else "new"

            rec = self.memory.records[gid]
            if status in ("confirmed", "recovered") and feature is not None:
                rec.add(feature, self.frame_index, bbox, tid)
            else:
                rec.last_seen_frame = self.frame_index
                rec.last_bbox = tuple(bbox)
                rec.active_track_id = tid
            self.track_to_global[tid] = gid
            rec.active_track_id = tid
            rec.last_seen_frame = self.frame_index
            rec.last_bbox = tuple(bbox)
            claimed.add(gid)

            if status == "confirmed":
                identity_conf = 0.90 if feature is not None else 0.65
            elif status == "recovered":
                span = max(1e-6, 1.0-self.config.match_threshold)
                identity_conf = float(np.clip(0.60 + 0.35*(score-self.config.match_threshold)/span, 0.60, 0.95))
            elif status == "uncertain":
                identity_conf = 0.35
            else:
                identity_conf = 0.55 if feature is not None else 0.25

            item = dict(track)
            item.update({"global_id": int(gid),
                         "identity_confidence": round(float(identity_conf), 3),
                         "identity_status": status,
                         "identity_match_score": None if score < 0 else round(float(score), 4)})
            output.append(item)
        return output

    def reset(self):
        self.memory = IdentityMemory(self.config.gallery_size)
        self.frame_index = -1
        self.track_to_global.clear()

    def diagnostics(self):
        return {"frame_index": self.frame_index,
                "known_identities": len(self.memory.records),
                "active_identities": sum(r.active_track_id is not None for r in self.memory.records.values())}
