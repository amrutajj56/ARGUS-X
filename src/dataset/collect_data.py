
import cv2
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "custom_reid"
MANIFEST = DATASET_DIR / "manifest.csv"

CAMERA_INDEX = 0


def collect_data():
    print("\n======================================")
    print("       ARGUS-X DATA COLLECTOR")
    print("======================================")

    person_id = input("Enter participant ID: ").strip()
    session_id = input("Enter session ID: ").strip()

    if not person_id or not session_id:
        print("Participant ID and session ID are required.")
        return

    if any(c in person_id + session_id for c in r'\/:*?"<>|'):
        print("Invalid ID. Use letters, numbers, _ or -.")
        return

    save_dir = DATASET_DIR / person_id / session_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Windows webcam backend
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    # Warm up the camera
    for _ in range(10):
        cap.read()

    print("\nWebcam started successfully!")
    print("Capture only consenting participants.")
    print("\nS = Save image")
    print("Q = Finish session")

    image_count = 0
    manifest_exists = MANIFEST.exists()

    try:
        with open(MANIFEST, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not manifest_exists:
                writer.writerow([
                    "person_id",
                    "session_id",
                    "image_path",
                    "timestamp"
                ])

            while True:
                success, frame = cap.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                display = frame.copy()

                cv2.putText(
                    display,
                    f"Participant: {person_id}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"Session: {session_id} | Images: {image_count}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    "S = Save | Q = Quit",
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.imshow("ARGUS-X Dataset Collector", display)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("s"), ord("S")):
                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )

                    filename = f"image_{timestamp}.jpg"
                    image_path = save_dir / filename

                    if cv2.imwrite(str(image_path), frame):
                        relative_path = image_path.relative_to(
                            PROJECT_ROOT
                        ).as_posix()

                        writer.writerow([
                            person_id,
                            session_id,
                            relative_path,
                            datetime.now().isoformat()
                        ])

                        file.flush()
                        image_count += 1

                        print(f"Saved image {image_count}")

                elif key in (ord("q"), ord("Q"), 27):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n======================================")
    print("SESSION COMPLETED")
    print("Participant:", person_id)
    print("Session:", session_id)
    print("Images saved:", image_count)
    print("Folder:", save_dir)
    print("======================================")


if __name__ == "__main__":
    collect_data()
