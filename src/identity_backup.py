
"""ARGUS-X global identity memory and confidence-aware association."""

import numpy as np
from scipy.optimize import linear_sum_assignment


class GlobalIdentityManager:
    """Assign global IDs using appearance and confidence-aware matching."""

    def __init__(
        self,
        similarity_threshold=0.75,
        max_features=10,
        ambiguity_margin=0.05,
        max_local_mapping_age=60,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.ambiguity_margin = ambiguity_margin
        self.max_local_mapping_age = max_local_mapping_age

        self.identities = {}
        self.next_global_id = 1

        # Maps (camera_id, local_track_id) to global_id.
        self.local_to_global = {}

        # Last observed frame for each local mapping.
        self.local_last_seen = {}

    def _expire_local_mappings(self, camera_id, frame_number):
        """Expire stale mappings for one camera only."""

        expired_keys = []

        for local_key, last_seen in self.local_last_seen.items():
            if local_key[0] != camera_id:
                continue

            age = frame_number - last_seen

            if age > self.max_local_mapping_age:
                expired_keys.append(local_key)

        for local_key in expired_keys:
            self.local_to_global.pop(local_key, None)
            self.local_last_seen.pop(local_key, None)

    def _create_identity(self, track, feature, camera_id, frame_number):
        """Create a new global identity."""

        global_id = self.next_global_id
        self.next_global_id += 1

        local_key = (camera_id, track["track_id"])

        self.identities[global_id] = {
            "class_id": track["class_id"],
            "class_name": track["class_name"],
            "features": [feature.copy()],
            "camera_ids": {camera_id},
            "first_seen": frame_number,
            "last_seen": frame_number,
            "observations": 1,
            "local_track_ids": {local_key},
        }

        self.local_to_global[local_key] = global_id
        self.local_last_seen[local_key] = frame_number

        return global_id

    @staticmethod
    def _cosine_similarity(feature_a, feature_b):
        """Calculate cosine similarity between two feature vectors."""

        if feature_a is None or feature_b is None:
            return 0.0

        feature_a = np.asarray(feature_a, dtype=np.float32)
        feature_b = np.asarray(feature_b, dtype=np.float32)

        if feature_a.shape != feature_b.shape:
            return 0.0

        norm_a = np.linalg.norm(feature_a)
        norm_b = np.linalg.norm(feature_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(feature_a, feature_b) / (
            norm_a * norm_b
        )

        return float(np.clip(similarity, -1.0, 1.0))

    def _identity_similarity(self, feature, identity):
        """Compare a feature with stored appearance history."""

        similarities = [
            self._cosine_similarity(feature, old_feature)
            for old_feature in identity["features"]
        ]

        if not similarities:
            return 0.0

        similarities.sort(reverse=True)

        return float(np.mean(similarities[:2]))

    def update(self, tracks, features, camera_id, frame_number):
        """Assign global IDs using appearance and confidence-aware matching."""

        if len(tracks) != len(features):
            raise ValueError(
                "Each track must have one feature entry."
            )

        self._expire_local_mappings(camera_id, frame_number)

        results = [None] * len(tracks)
        assigned_global_ids = set()
        pending_indices = []

        # STEP 1: Reuse valid local-to-global mappings.
        for index, (track, feature) in enumerate(
            zip(tracks, features)
        ):
            track_id = track.get("track_id")
            local_key = (camera_id, track_id)

            global_id = self.local_to_global.get(local_key)

            if global_id is not None:
                identity = self.identities.get(global_id)

                if (
                    identity is not None
                    and global_id not in assigned_global_ids
                ):
                    identity["last_seen"] = frame_number
                    identity["camera_ids"].add(camera_id)
                    identity["observations"] += 1

                    if feature is not None:
                        identity["features"].append(feature.copy())
                        identity["features"] = identity["features"][
                            -self.max_features:
                        ]

                    self.local_last_seen[local_key] = frame_number
                    assigned_global_ids.add(global_id)

                    results[index] = {
                        **track,
                        "global_id": global_id,
                        "camera_id": camera_id,
                        "identity_match_status": "known_local_track",
                        "match_similarity": None,
                    }
                    continue

            if feature is not None:
                pending_indices.append(index)
            else:
                results[index] = {
                    **track,
                    "global_id": None,
                    "camera_id": camera_id,
                    "identity_match_status": "no_appearance_feature",
                    "match_similarity": None,
                }

        # STEP 2: Find candidate global identities.
        candidate_ids = [
            global_id
            for global_id in self.identities
            if global_id not in assigned_global_ids
        ]

        matched_indices = set()

        if pending_indices and candidate_ids:

            cost_matrix = np.full(
                (len(pending_indices), len(candidate_ids)),
                1e6,
                dtype=np.float32,
            )

            similarity_matrix = np.zeros(
                (len(pending_indices), len(candidate_ids)),
                dtype=np.float32,
            )

            for row, track_index in enumerate(pending_indices):
                track = tracks[track_index]
                feature = features[track_index]

                for col, global_id in enumerate(candidate_ids):
                    identity = self.identities[global_id]

                    # Never match different object classes.
                    if track["class_id"] != identity["class_id"]:
                        continue

                    similarity = self._identity_similarity(
                        feature,
                        identity,
                    )

                    similarity_matrix[row, col] = similarity
                    if similarity > 0.40:
                        print(
                            f"[IDENTITY CHECK] "
                            f"Frame={frame_number} | "
                            f"Local ID={tracks[pending_indices[row]]['track_id']} | "
                            f"Candidate Global ID={global_id} | "
                            f"Similarity={similarity:.3f}"
                        )
                    if similarity > 0.40:
                        print(
                            f"[IDENTITY CHECK] "
                            f"Frame={frame_number} | "
                            f"Local ID={tracks[pending_indices[row]]['track_id']} | "
                            f"Candidate Global ID={global_id} | "
                            f"Similarity={similarity:.3f}"
                        )
                    cost_matrix[row, col] = 1.0 - similarity

            # STEP 3: Hungarian one-to-one assignment.
            rows, cols = linear_sum_assignment(cost_matrix)

            for row, col in zip(rows, cols):
                similarity = float(similarity_matrix[row, col])

                if cost_matrix[row, col] >= 1e5:
                    continue

                if similarity < self.similarity_threshold:
                    continue

                # STEP 4: Confidence and ambiguity rejection.
                valid_scores = [
                    float(similarity_matrix[row, candidate_col])
                    for candidate_col in range(len(candidate_ids))
                    if cost_matrix[row, candidate_col] < 1e5
                ]

                valid_scores.sort(reverse=True)

                # Reject when the two best candidates are too similar.
                if (
                    len(valid_scores) > 1
                    and valid_scores[0] - valid_scores[1]
                    < self.ambiguity_margin
                ):
                    continue

                # Reject an assignment that is not the best candidate.
                if similarity < valid_scores[0] - 1e-6:
                    continue

                track_index = pending_indices[row]
                global_id = candidate_ids[col]
                track = tracks[track_index]
                feature = features[track_index]

                identity = self.identities[global_id]
                local_key = (camera_id, track["track_id"])

                identity["features"].append(feature.copy())
                identity["features"] = identity["features"][
                    -self.max_features:
                ]

                identity["camera_ids"].add(camera_id)
                identity["last_seen"] = frame_number
                identity["observations"] += 1
                identity["local_track_ids"].add(local_key)

                self.local_to_global[local_key] = global_id
                self.local_last_seen[local_key] = frame_number

                assigned_global_ids.add(global_id)
                matched_indices.add(track_index)

                results[track_index] = {
                    **track,
                    "global_id": global_id,
                    "camera_id": camera_id,
                    "identity_match_status": "appearance_matched",
                    "match_similarity": similarity,
                }

        # STEP 5: Create new identities for unmatched tracks.
        for index in pending_indices:
            if index in matched_indices:
                continue

            track = tracks[index]
            feature = features[index]

            global_id = self._create_identity(
                track,
                feature,
                camera_id,
                frame_number,
            )

            results[index] = {
                **track,
                "global_id": global_id,
                "camera_id": camera_id,
                "identity_match_status": "new_identity",
                "match_similarity": None,
            }

        return results
