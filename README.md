# ARGUS-X — Member 1 Tracking Engine

Starter project for the ARGUS-X hackathon. This module provides CPU-based YOLO object detection and ByteTrack multi-object tracking.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for the first package/model download

## Setup in VS Code PowerShell

Open this folder in VS Code, then run:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Add a test video

Copy a video into `data` and name it `test_video.mp4`.

Then run:

```powershell
python -m src.main --source data/test_video.mp4 --show
```

Or specify another video:

```powershell
python -m src.main --source "data/my_video.mp4" --show
```

For a webcam:

```powershell
python -m src.main --source 0 --show
```

The first run downloads the small `yolov8n.pt` model automatically. Keep an internet connection until it finishes.

## Outputs

- `outputs/tracked_output.mp4` — annotated video with bounding boxes and local track IDs
- `outputs/tracked_output.csv` — per-frame boxes, class labels, confidence and local track IDs

## Member 1 / Member 2 interface

`TrackingEngine.process_frame(frame)` returns:

```python
annotated_frame, tracks = engine.process_frame(frame)
```

Each track has this structure:

```python
{
    "track_id": 1,                 # temporary/local ByteTrack ID
    "bbox": [x1, y1, x2, y2],     # pixel coordinates
    "confidence": 0.94,
    "class_id": 0,
    "class_name": "person"
}
```

Important: `track_id` is a local tracker ID, not a guaranteed long-term identity. Member 2's identity module should produce a separate persistent `global_id` and should abstain when the match confidence is low.

## Git collaboration

This starter is intended for the `feature/tracking-engine` branch. Do not commit `.venv`, test videos, model weights or generated outputs.
