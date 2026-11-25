import argparse
import cv2
from argus_identity import IdentityEngine
from integration.ultralytics_bytetrack_adapter import UltralyticsByteTrackAdapter

def parse_source(value):
    return int(value) if value.isdigit() else value

def run(source, model_path, people_only=False, show=True, output_path=None):
    tracker = UltralyticsByteTrackAdapter(model_path=model_path, device="cpu",
                                          classes=[0] if people_only else None)
    identity = IdentityEngine()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {source}")
    writer = None
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tracks = tracker.process(frame)
            results = identity.update(frame, tracks)  # also called with [] on empty frames
            for item in results:
                x1,y1,x2,y2 = map(int, item["bbox"])
                status = item["identity_status"]
                color = (0,255,0) if status in ("confirmed","recovered") else (0,165,255)
                label = f"GID:{item['global_id']} TID:{item['track_id']} {status} {item['identity_confidence']:.2f}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1,max(20,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if status == "recovered":
                    print(f"[RECOVERED] frame={frame_no} GID={item['global_id']} TID={item['track_id']} score={item['identity_match_score']}")
            cv2.putText(frame, f"ARGUS-X | Frame: {frame_no}", (15,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if output_path and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output: {output_path}")
            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("ARGUS-X Identity Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_no += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    print("Session complete:", identity.diagnostics())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGUS-X YOLO + ByteTrack + Identity Intelligence")
    parser.add_argument("--source", required=True, help="Video file path or webcam index (0)")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights path")
    parser.add_argument("--people-only", action="store_true")
    parser.add_argument("--output", default=None, help="Optional annotated MP4 output path")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    run(parse_source(args.source), args.model, args.people_only,
        not args.no_display, args.output)
