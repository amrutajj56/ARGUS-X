
"""ARGUS-X track lifecycle management.

Monitors active, lost, recovered and removed local track IDs.
This module does not perform appearance-based re-identification.
"""


class TrackLifecycleManager:
    def __init__(self, max_lost_frames=30):
        self.max_lost_frames = max_lost_frames
        self.records = {}

    def update(self, tracks, frame_number):
        """Update lifecycle states and enrich current track dictionaries."""

        current_ids = {
            track["track_id"]
            for track in tracks
            if track.get("track_id") is not None
        }

        # Mark tracks missing from the current frame.
        for track_id, record in self.records.items():
            if track_id in current_ids:
                continue

            if record["status"] == "removed":
                continue

            missed = frame_number - record["last_seen"]

            if missed > self.max_lost_frames:
                record["status"] = "removed"
            else:
                record["status"] = "lost"

            record["missed_frames"] = missed

        enriched_tracks = []

        for track in tracks:
            track = track.copy()
            track_id = track.get("track_id")

            if track_id is None:
                track["lifecycle_status"] = "untracked"
                track["track_age"] = 0
                track["hits"] = 0
                track["missed_frames"] = 0
                enriched_tracks.append(track)
                continue

            record = self.records.get(track_id)

            if record is None or record["status"] == "removed":
                record = {
                    "first_seen": frame_number,
                    "last_seen": frame_number,
                    "hits": 0,
                    "status": "active",
                }
                self.records[track_id] = record
                status = "active"

            elif record["status"] == "lost":
                status = "recovered"

            else:
                status = "active"

            record["status"] = status
            record["last_seen"] = frame_number
            record["hits"] += 1
            record["missed_frames"] = 0

            track["lifecycle_status"] = status
            track["track_age"] = frame_number - record["first_seen"] + 1
            track["hits"] = record["hits"]
            track["missed_frames"] = 0

            enriched_tracks.append(track)

        return enriched_tracks

    def get_statistics(self):
        """Return the current lifecycle statistics."""

        return {
            "active": sum(
                r["status"] in ("active", "recovered")
                for r in self.records.values()
            ),
            "lost": sum(
                r["status"] == "lost"
                for r in self.records.values()
            ),
            "removed": sum(
                r["status"] == "removed"
                for r in self.records.values()
            ),
            "total_tracks": len(self.records),
        }
