
"""ARGUS-X confidence-aware identity matching tests."""

import numpy as np

from src.identity import GlobalIdentityManager


def make_track(track_id, class_id=2, class_name="car"):
    return {
        "track_id": track_id,
        "bbox": [10, 20, 100, 120],
        "confidence": 0.95,
        "class_id": class_id,
        "class_name": class_name,
    }


def test_strong_match():
    manager = GlobalIdentityManager(similarity_threshold=0.75)

    feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    first = manager.update(
        [make_track(1)], [feature], "camera_1", 1
    )[0]

    second = manager.update(
        [make_track(2)], [feature.copy()], "camera_2", 2
    )[0]

    assert first["global_id"] == second["global_id"]
    assert second["identity_match_status"] == "appearance_matched"

    print("Strong appearance match: PASSED")


def test_weak_match():
    manager = GlobalIdentityManager(similarity_threshold=0.75)

    feature_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    feature_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    first = manager.update(
        [make_track(1)], [feature_a], "camera_1", 1
    )[0]

    second = manager.update(
        [make_track(2)], [feature_b], "camera_2", 2
    )[0]

    assert first["global_id"] != second["global_id"]
    assert second["identity_match_status"] == "new_identity"

    print("Weak appearance rejection: PASSED")


def test_different_class_rejection():
    manager = GlobalIdentityManager(similarity_threshold=0.75)

    feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    first = manager.update(
        [make_track(1, 2, "car")],
        [feature],
        "camera_1",
        1,
    )[0]

    second = manager.update(
        [make_track(2, 5, "bus")],
        [feature.copy()],
        "camera_2",
        2,
    )[0]

    assert first["global_id"] != second["global_id"]

    print("Different-class rejection: PASSED")


def test_ambiguous_match():
    manager = GlobalIdentityManager(
        similarity_threshold=0.75,
        ambiguity_margin=0.05,
    )

    # Two different identities with identical appearance features.
    feature_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    feature_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    manager.update(
        [make_track(1), make_track(2)],
        [feature_a, feature_b],
        "camera_1",
        1,
    )

    # This feature matches both identities equally.
    ambiguous_feature = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    result = manager.update(
        [make_track(3)],
        [ambiguous_feature],
        "camera_2",
        2,
    )[0]

    assert result["identity_match_status"] == "new_identity"
    assert result["global_id"] == 3

    print("Ambiguous-match rejection: PASSED")


if __name__ == "__main__":
    print("\n========== CONFIDENCE TESTS ==========\n")

    test_strong_match()
    test_weak_match()
    test_different_class_rejection()
    test_ambiguous_match()

    print("\nALL CONFIDENCE TESTS PASSED!")
