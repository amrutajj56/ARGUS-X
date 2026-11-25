# ARGUS-X — Complete Identity Integration Starter

Includes Member 2's CPU appearance-based identity engine and an adapter/runner for Ultralytics YOLO + ByteTrack.

## Important scope
The appearance descriptor is a handcrafted HSV-histogram + grayscale-texture baseline, not a trained deep Re-ID network. The synthetic demo validates the software flow, not real-world Re-ID accuracy. Real video testing and metric-based evaluation are still required.

## Recommended setup (Windows)
Use stable 64-bit Python 3.11–3.13. Python 3.15 release candidates may not have compatible PyTorch wheels.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tests and synthetic demo
```powershell
python -m unittest discover -s tests -v
python demo_identity.py
```

## Video input
```powershell
python run_argus.py --source test_video.mp4 --people-only
```

## Webcam input
```powershell
python run_argus.py --source 0 --people-only
```

## Use Member 1's trained model and save annotated output
```powershell
python run_argus.py --source test_video.mp4 --model best.pt --people-only --output argus_output.mp4
```

The default model is `yolo11n.pt`; Ultralytics may download it on first use. If Member 1 has a trained model, use its actual `.pt` path.

## Adapter contract
Each track passed to `IdentityEngine.update(frame, tracks)` must contain:
- `track_id`: ByteTrack temporary ID
- `bbox`: `[x1, y1, x2, y2]` pixel coordinates
- `confidence`: detection confidence
- `class_id`: detector class index

Keep one adapter instance alive throughout a stream because Ultralytics `persist=True` retains tracker state. Start a new adapter and IdentityEngine for each independent stream. Call `identity.update(frame, [])` on frames with no detections.

## Team integration
Member 1 should provide the exact model weights, class mapping, tracker YAML (if custom), and a short sample video. This starter uses Ultralytics `model.track(..., persist=True, tracker="bytetrack.yaml")`.

## CPU notes
The runner sets `device="cpu"` and defaults to the small `yolo11n.pt` model. Speed depends on CPU, frame resolution and model. Reduce video resolution if needed.

## Limitations / next research tasks
- Similar clothing, illumination and viewpoint changes can confuse HSV/texture descriptors.
- No explicit camera-motion compensation or deep Re-ID embedding is included.
- Confidence is a heuristic score, not a calibrated probability.
- Tune thresholds on validation footage.
- Evaluate using ground-truth video and appropriate MOT/Re-ID metrics such as IDF1, HOTA, ID switches, recovery precision and recall.
- Identity memory is in RAM and resets per run; database persistence is not included.
