
"""ARGUS-X: Video and webcam demonstration runner."""

import argparse
import csv
from pathlib import Path

import cv2

from src.recognition import ARGUSXPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARGUS-X multi-object identity tracking"
    )

    parser.add_argument(
        "--source",
        default="data/solutions_ci_demo.mp4",
        help="Video file path or 0 for webcam",
    )
    parser.add_argument(
        "--output",
        default="outputs/argusx_output.mp4",
    )
    parser.add_argument(
        "--csv",
        default="outputs/argusx_tracks.csv",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--camera-id",
        default="camera_1",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the tracking window",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source = 0 if args.source == "0" else args.source

    if isinstance(source, str) and not Path(source).is_file():
        raise FileNotFoundError(
            f"Input video not found: {source}"
        )

    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )
    Path(args.csv).parent.mkdir(
        parents=True, exist_ok=True
    )

    print("\n========== ARGUS-X ==========")
    print(f"Input source: {args.source}")
    print(f"Camera ID: {args.camera_id}")
    print("Loading detection and identity models...")

    pipeline = ARGUSXPipeline(
        camera_id=args.camera_id,
        device="cpu",
        model_path=args.model,
        confidence=args.confidence,
    )

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input source: {args.source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video dimensions.")

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {args.output}"
        )

    csv_fields = [
        "frame",
        "camera_id",
        "local_track_id",
        "global_id",
        "class_name",
        "confidence",
        "identity_match_status",
        "match_similarity",
        "lifecycle_status",
        "track_age",
    ]

    frame_number = 0
    total_detections = 0
    observed_global_ids = set()

    print("\nProcessing started. Press Q to stop the display.\n")

    try:
        with open(
            args.csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=csv_fields,
            )
            csv_writer.writeheader()

            while True:
                success, frame = capture.read()

                if not success:
                    break

                frame_number += 1

                annotated, tracks = pipeline.process_frame(frame)

                total_detections += len(tracks)

                for track in tracks:
                    global_id = track.get("global_id")

                    if global_id is not None:
                        observed_global_ids.add(global_id)

                    csv_writer.writerow({
                        "frame": frame_number,
                        "camera_id": args.camera_id,
                        "local_track_id": track.get("track_id"),
                        "global_id": global_id,
                        "class_name": track.get("class_name"),
                        "confidence": track.get("confidence"),
                        "identity_match_status": track.get(
                            "identity_match_status"
                        ),
                        "match_similarity": track.get(
                            "match_similarity"
                        ),
                        "lifecycle_status": track.get(
                            "lifecycle_status"
                        ),
                        "track_age": track.get("track_age"),
                    })

                writer.write(annotated)

                if args.show:
                    cv2.imshow("ARGUS-X Identity Tracking", annotated)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Display stopped by user.")
                        break

                if frame_number % 30 == 0:
                    print(
                        f"Frames: {frame_number} | "
                        f"Visible tracks: {len(tracks)} | "
                        f"Global IDs observed: {len(observed_global_ids)}"
                    )

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\n========== RESULTS ==========")
    print(f"Frames processed: {frame_number}")
    print(f"Total track observations: {total_detections}")
    print(f"Unique global IDs observed: {len(observed_global_ids)}")
    print(f"Annotated video: {args.output}")
    print(f"Tracking CSV: {args.csv}")
    print("ARGUS-X processing completed.")


if __name__ == "__main__":
    main()
