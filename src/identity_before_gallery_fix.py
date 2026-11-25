
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
        self.local_to_global = {}
        self.local_last_seen = {}

    def _expire_local_mappings(self, camera_id, frame_number):
        """Expire stale mappings for one camera only."""

        expired_keys = []

        for local_key, last_seen in self.local_last_seen.items():
            if local_key[0] != camera_id:
                continue

            if frame_number - last_seen > self.max_local_mapping_age:
                expired_keys.append(local_key)

        for local_key in expired_keys:
            self.local_to_global.pop(local_key, None)
            self.local_last_seen.pop(local_key, None)

    def _store_feature(self, identity, feature):
        """Store non-duplicate appearance features."""

        if feature is None:
            return False

        feature = np.asarray(feature, dtype=np.float32)

        for old_feature in identity["features"]:
            if np.array_equal(feature, old_feature):
                return False

        identity["features"].append(feature.copy())
        identity["features"] = identity["features"][-self.max_features:]
        return True

    def _create_identity(self, track, feature, camera_id, frame_number):
        """Create a new global identity."""

        global_id = self.next_global_id
        self.next_global_id += 1

        local_key = (camera_id, track["track_id"])

        self.identities[global_id] = {
            "class_id": track["class_id"],
            "class_name": track["class_name"],
            "features": [],
            "camera_ids": {camera_id},
            "first_seen": frame_number,
            "last_seen": frame_number,
            "observations": 1,
            "local_track_ids": {local_key},
        }

        self._store_feature(self.identities[global_id], feature)

        self.local_to_global[local_key] = global_id
        self.local_last_seen[local_key] = frame_number

        return global_id

    @staticmethod
    def _cosine_similarity(feature_a, feature_b):
        """Calculate cosine similarity."""

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

        similarity = np.dot(feature_a, feature_b) / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))

    def _identity_similarity(self, feature, identity):
        """Compare against stored appearance history."""

        similarities = [
            self._cosine_similarity(feature, old_feature)
            for old_feature in identity["features"]
        ]

        if not similarities:
            return 0.0

        similarities.sort(reverse=True)
        return float(np.mean(similarities[:2]))

    def _build_similarity_matrix(
        self,
        tracks,
        features,
        pending_indices,
        candidate_ids,
        frame_number,
    ):
        """Calculate valid appearance similarities."""

        matrix = np.full(
            (len(pending_indices), len(candidate_ids)),
            -1.0,
            dtype=np.float32,
        )

        for row, track_index in enumerate(pending_indices):
            track = tracks[track_index]
            feature = features[track_index]

            for col, global_id in enumerate(candidate_ids):
                identity = self.identities[global_id]

                if track["class_id"] != identity["class_id"]:
                    continue

                similarity = self._identity_similarity(
                    feature,
                    identity,
                )

                matrix[row, col] = similarity

                if True:
                    print(
                        f"[IDENTITY CHECK] "
                        f"Frame={frame_number} | "
                        f"Local ID={track['track_id']} | "
                        f"Candidate Global ID={global_id} | "
                        f"Similarity={similarity:.3f}"
                    )

        return matrix

    def _match_pending_tracks(
        self,
        tracks,
        features,
        pending_indices,
        candidate_ids,
        similarity_matrix,
    ):
        """Iteratively assign strongest unambiguous matches."""

        remaining_rows = set(range(len(pending_indices)))
        remaining_cols = set(range(len(candidate_ids)))

        matches = {}

        while remaining_rows and remaining_cols:
            rows = sorted(remaining_rows)
            cols = sorted(remaining_cols)

            sub_matrix = similarity_matrix[np.ix_(rows, cols)]

            # Invalid class combinations must never be assigned.
            cost_matrix = np.where(
                sub_matrix >= 0,
                1.0 - sub_matrix,
                1e6,
            )

            assigned_rows, assigned_cols = linear_sum_assignment(
                cost_matrix
            )

            accepted = []

            for local_row, local_col in zip(
                assigned_rows,
                assigned_cols,
            ):
                row = rows[local_row]
                col = cols[local_col]

                similarity = float(similarity_matrix[row, col])

                if similarity < self.similarity_threshold:
                    continue

                valid_scores = [
                    float(similarity_matrix[row, other_col])
                    for other_col in remaining_cols
                    if similarity_matrix[row, other_col] >= 0
                ]

                valid_scores.sort(reverse=True)

                if (
                    len(valid_scores) > 1
                    and valid_scores[0] - valid_scores[1]
                    < self.ambiguity_margin
                ):
                    continue

                if similarity < valid_scores[0] - 1e-6:
                    continue

                accepted.append((row, col, similarity))

            if not accepted:
                break

            for row, col, similarity in accepted:
                track_index = pending_indices[row]
                global_id = candidate_ids[col]

                matches[track_index] = (global_id, similarity)

                remaining_rows.discard(row)
                remaining_cols.discard(col)

        return matches

    def update(self, tracks, features, camera_id, frame_number):
        """Assign global identities to current tracks."""

        if len(tracks) != len(features):
            raise ValueError(
                "Each track must have one feature entry."
            )

        self._expire_local_mappings(camera_id, frame_number)

        results = [None] * len(tracks)
        assigned_global_ids = set()
        pending_indices = []

        # Step 1: Reuse known local mappings.
        for index, (track, feature) in enumerate(zip(tracks, features)):
            local_key = (camera_id, track.get("track_id"))
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

                    self._store_feature(identity, feature)

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

        # Step 2: Build candidates not already assigned this frame.
        candidate_ids = [
            global_id
            for global_id in self.identities
            if global_id not in assigned_global_ids
        ]

        matches = {}

        if pending_indices and candidate_ids:
            similarity_matrix = self._build_similarity_matrix(
                tracks,
                features,
                pending_indices,
                candidate_ids,
                frame_number,
            )

            matches = self._match_pending_tracks(
                tracks,
                features,
                pending_indices,
                candidate_ids,
                similarity_matrix,
            )

        # Step 3: Apply accepted matches.
        matched_indices = set()

        for track_index, (global_id, similarity) in matches.items():
            track = tracks[track_index]
            feature = features[track_index]
            identity = self.identities[global_id]

            local_key = (camera_id, track["track_id"])

            self._store_feature(identity, feature)

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

        # Step 4: Create identities for unmatched tracks.
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
