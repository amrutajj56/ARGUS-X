
"""ARGUS-X integrated detection, appearance and identity pipeline."""

import cv2

from src.tracker import TrackingEngine
from src.reid import AppearanceExtractor
from src.identity import GlobalIdentityManager


class ARGUSXPipeline:
    """Combine local tracking with global appearance-based identity."""

    def __init__(
        self,
        camera_id="camera_1",
        identity_manager=None,
        device="cpu",
        feature_refresh_interval=15,
        feature_cache_max_age=60,
        model_path="yolov8n.pt",
        confidence=0.35,
    ):
        self.camera_id = camera_id

        self.tracker = TrackingEngine(
            model_path=model_path,
            confidence=confidence,
            device=device,
        )

        self.extractor = AppearanceExtractor(device=device)

        self.identity_manager = (
            identity_manager
            if identity_manager is not None
            else GlobalIdentityManager()
        )

        self.feature_refresh_interval = feature_refresh_interval
        self.feature_cache_max_age = feature_cache_max_age

        self.frame_number = 0
        self.feature_cache = {}

    def _get_track_generation(self, track):
        """
        Identify a local track lifecycle generation.

        If ByteTrack reuses a local ID after its previous track
        was removed, its estimated first-seen frame will change.
        """

        track_age = track.get("track_age")

        if track_age is None:
            return None

        return self.frame_number - track_age + 1

    def _cleanup_feature_cache(self):
        """Remove stale or removed local-track appearance features."""

        expired_keys = []

        lifecycle_records = self.tracker.lifecycle.records

        for cache_key, cached in self.feature_cache.items():

            if cache_key[0] != self.camera_id:
                continue

            track_id = cache_key[1]

            age = self.frame_number - cached["last_seen"]

            lifecycle_record = lifecycle_records.get(track_id)

            # A removed local track must not retain its old feature.
            if (
                lifecycle_record is not None
                and lifecycle_record["status"] == "removed"
            ):
                expired_keys.append(cache_key)
                continue

            # Also enforce the independent cache-age limit.
            if age > self.feature_cache_max_age:
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            self.feature_cache.pop(cache_key, None)

    def process_frame(self, frame):
        """Process a frame and return annotated output and identified tracks."""

        self.frame_number += 1

        annotated, tracks = self.tracker.process_frame(frame)

        # Clean stale appearance history for this camera.
        self._cleanup_feature_cache()

        features = []

        for track in tracks:
            track_id = track["track_id"]

            if track_id is None:
                feature = self.extractor.extract(
                    frame,
                    track["bbox"],
                )

                features.append(feature)
                continue

            cache_key = (self.camera_id, track_id)
            cached = self.feature_cache.get(cache_key)

            current_generation = self._get_track_generation(track)

            # Never reuse features from an older lifecycle that
            # happened to receive the same ByteTrack local ID.
            if (
                cached is not None
                and cached.get("generation") is not None
                and current_generation is not None
                and cached["generation"] != current_generation
            ):
                self.feature_cache.pop(cache_key, None)
                cached = None

            should_refresh = (
                cached is None
                or self.frame_number - cached["frame"]
                >= self.feature_refresh_interval
            )

            if should_refresh:
                feature = self.extractor.extract(
                    frame,
                    track["bbox"],
                )

                if feature is not None:
                    self.feature_cache[cache_key] = {
                        "feature": feature,
                        "frame": self.frame_number,
                        "last_seen": self.frame_number,
                        "generation": current_generation,
                    }

                elif cached is not None:
                    # Keep the last valid feature for this same
                    # local-track generation if extraction fails.
                    feature = cached["feature"]

            else:
                feature = cached["feature"]

            if cache_key in self.feature_cache:
                self.feature_cache[cache_key]["last_seen"] = (
                    self.frame_number
                )

            features.append(feature)

        # Associate local tracks with persistent global identities.
        identified_tracks = self.identity_manager.update(
            tracks=tracks,
            features=features,
            camera_id=self.camera_id,
            frame_number=self.frame_number,
        )

        # Draw global identity information.
        for track in identified_tracks:
            x1, y1, x2, y2 = track["bbox"]

            global_id = track["global_id"]
            status = track["identity_match_status"]

            if global_id is None:
                label = f"{track['class_name']} | Unassigned"
            else:
                label = (
                    f"{track['class_name']} | "
                    f"Global ID: {global_id} | {status}"
                )

            color = (255, 200, 0)

            if status == "new_identity":
                color = (0, 255, 0)
            elif status == "appearance_matched":
                color = (0, 200, 255)

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated, identified_tracks
