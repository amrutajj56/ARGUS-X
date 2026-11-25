"""Run ARGUS-X Member 1 tracking on a video file or webcam."""

import argparse
import csv
from pathlib import Path

import cv2

from src.tracker import TrackingEngine


def parse_args():
    parser = argparse.ArgumentParser(description="ARGUS-X object tracking demo")
    parser.add_argument(
        "--source",
        default="data/test_video.mp4",
        help="Video path, or 0 for the default webcam",
    )
    parser.add_argument("--output", default="outputs/tracked_output.mp4")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--show", action="store_true", help="Display the video window")
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open source: {args.source}\n"
            "Check the video path, or try --source 0 for a webcam."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output_path}")

    csv_path = output_path.with_suffix(".csv")
    engine = TrackingEngine(
        model_path=args.model,
        confidence=args.confidence,
        device="cpu",
    )

    frame_number = 0
    print("ARGUS-X tracking started. First run may download the YOLO nano model.")
    print("Press Q in the video window to stop when --show is enabled.")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "lifecycle_status",
                "track_age",
                "hits",
                "missed_frames",
            ],
        )
        csv_writer.writeheader()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                annotated, tracks = engine.process_frame(frame)
                writer.write(annotated)
                frame_number += 1

                for track in tracks:
                    x1, y1, x2, y2 = track["bbox"]
                    csv_writer.writerow(
                        {
                            "frame": frame_number,
                            "track_id": track["track_id"],
                            "class_id": track["class_id"],
                            "class_name": track["class_name"],
                            "lifecycle_status": track["lifecycle_status"],
                            "track_age": track["track_age"],
                            "hits": track["hits"],
                            "missed_frames": track["missed_frames"],
                            "confidence": round(track["confidence"], 4),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

                if frame_number % 30 == 0:
                    print(
                        f"Processed {frame_number} frames | "
                        f"visible tracks: {len(tracks)}"
                    )

                if args.show:
                    cv2.imshow("ARGUS-X | Member 1 Tracking Engine", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

    print(f"Finished. Annotated video: {output_path}")
    print(f"Frame-by-frame tracking data: {csv_path}")


if __name__ == "__main__":
    main()
