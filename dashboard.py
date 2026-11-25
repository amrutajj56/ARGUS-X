
import streamlit as st
import cv2
import tempfile
import csv
import os
import time
from pathlib import Path
from src.recognition import ARGUSXPipeline

st.set_page_config(
    page_title="ARGUS-X Command Center",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background: #0b1120; color: white;}
[data-testid="stSidebar"] {background: #111c30;}
.metric-card {
    background: #17243a; padding: 18px;
    border: 1px solid #263b59; border-radius: 12px;
}
h1, h2, h3 {color: #e5efff;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ ARGUS-X")
st.caption("AI-POWERED PERSON TRACKING & GLOBAL IDENTITY INTELLIGENCE")

st.sidebar.title("COMMAND CENTER")
source_type = st.sidebar.radio(
    "Select video source",
    ["Built-in Demo Video", "Upload Video", "Live Camera"]
)

camera_id = st.sidebar.text_input("Camera ID", "camera_1")
confidence = st.sidebar.slider("Detection confidence", 0.10, 0.90, 0.35)

uploaded_file = None
demo_path = "data/solutions_ci_demo.mp4"

if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Choose video", type=["mp4", "avi", "mov", "mkv"]
    )

if source_type == "Built-in Demo Video":
    st.sidebar.success("Built-in demo selected")

start = st.sidebar.button("▶ START ANALYSIS", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("YOLOv8 • ByteTrack • DINOv2 • Global Identity Manager")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True

if st.session_state.running:
    input_source = None
    temp_path = None

    if source_type == "Built-in Demo Video":
        if not Path(demo_path).exists():
            st.error(f"Demo video not found: {demo_path}")
            st.session_state.running = False
            st.stop()
        input_source = demo_path

    elif source_type == "Upload Video":
        if uploaded_file is None:
            st.warning("Please upload a video first.")
            st.session_state.running = False
            st.stop()

        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.getbuffer())
            temp_path = f.name
        input_source = temp_path

    else:
        input_source = 0

    st.info("Loading ARGUS-X detection and identity models...")
    pipeline = ARGUSXPipeline(
        camera_id=camera_id,
        device="cpu",
        model_path="yolov8n.pt",
        confidence=confidence
    )

    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        st.error("Could not open selected video source.")
        st.session_state.running = False
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        st.stop()

    video_area = st.empty()
    status_area = st.empty()
    progress = st.progress(0)
    c1, c2, c3, c4 = st.columns(4)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    observations = 0
    global_ids = set()
    appearance_matches = 0
    rows = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            annotated, tracks = pipeline.process_frame(frame)
            observations += len(tracks)

            for track in tracks:
                gid = track.get("global_id")
                if gid is not None:
                    global_ids.add(gid)

                match_status = track.get("identity_match_status")
                if match_status == "appearance_matched":
                    appearance_matches += 1

                rows.append({
                    "frame": frame_number,
                    "camera_id": camera_id,
                    "local_track_id": track.get("track_id"),
                    "global_id": gid,
                    "class_name": track.get("class_name"),
                    "confidence": track.get("confidence"),
                    "identity_match_status": match_status,
                    "match_similarity": track.get("match_similarity"),
                    "lifecycle_status": track.get("lifecycle_status"),
                    "track_age": track.get("track_age")
                })

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_area.image(rgb, channels="RGB", use_container_width=True)

            c1.metric("Frames processed", frame_number)
            c2.metric("Person observations", observations)
            c3.metric("Global identities", len(global_ids))
            c4.metric("Appearance matches", appearance_matches)

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))

            status_area.success(
                f"Monitoring {camera_id} | Frame {frame_number} | "
                f"Visible people: {len(tracks)}"
            )

        cap.release()

        os.makedirs("outputs", exist_ok=True)
        csv_path = "outputs/dashboard_events.csv"

        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        st.success("Analysis completed successfully!")

        if rows:
            st.subheader("Identity Event Log")
            st.dataframe(rows, use_container_width=True)

            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇ Download tracking CSV",
                    f,
                    file_name="argus_events.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Analysis error: {e}")
    finally:
        cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        st.session_state.running = False
