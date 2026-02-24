"""
open_cv_detection.py
--------------------
Streamlit UI for the OpenCV Detection page.

All pure CV logic lives in open_cv_core.py.
This file handles:
  - Environment detection (local vs. Streamlit Cloud)
  - Cascade loading (with Streamlit warnings)
  - Webcam mode   (_run_local_webcam / WebRTC)
  - Video upload  (_run_video_upload)
  - Image mode    (upload / sample)
  - Main page     (opencv_detection_page)
"""

import os
import platform
import tempfile
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.open_cv.open_cv_core import (
    CASCADE_PATHS,
    run_eye_smile_detection,
    run_face_count,
    run_face_detection,
    run_stop_sign_detection,
)


# ---------------------------------------------------------------------------
# Environment detection — local vs. Streamlit Cloud
# ---------------------------------------------------------------------------
def _is_local() -> bool:
    """
    Returns True when running on the user's local machine.

    Detection strategy (any match → cloud):
      1. IS_STREAMLIT_CLOUD env var is set  (Streamlit Community Cloud)
      2. STREAMLIT_SHARING_MODE env var is set
      3. HOSTNAME contains 'streamlit'  (Linux cloud runners)
      4. No DISPLAY and not Windows     (headless Linux server)
    """
    if os.environ.get("IS_STREAMLIT_CLOUD"):
        return False
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        return False
    hostname = os.environ.get("HOSTNAME", "").lower()
    if "streamlit" in hostname:
        return False
    if platform.system() != "Windows" and not os.environ.get("DISPLAY"):
        return False
    return True


IS_LOCAL = _is_local()


# ---------------------------------------------------------------------------
# Cascade loader (Streamlit warnings live here, not in core)
# ---------------------------------------------------------------------------
def _load_cascade(key: str) -> cv2.CascadeClassifier | None:
    """Load a Haar cascade by key; return None and warn if missing/invalid."""
    path = CASCADE_PATHS.get(key, "")
    if not os.path.exists(path):
        st.warning(f"⚠️ Cascade not found: {path}")
        return None
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        st.warning(f"⚠️ Failed to load cascade: {path}")
        return None
    return clf


# ---------------------------------------------------------------------------
# Helper: dispatch detection to the correct function
# ---------------------------------------------------------------------------
def _apply_detection(img, detection_type, face_cascades, eye_cascade, smile_cascade):
    if detection_type == "Real Time Face Count":
        img, _ = run_face_count(img, face_cascades)
    elif detection_type == "Stop Sign Detection":
        img = run_stop_sign_detection(img)
    elif detection_type == "Face Detection":
        img = run_face_detection(img, face_cascades)
    else:  # Eye + Smile Detection
        img = run_eye_smile_detection(img, face_cascades, eye_cascade, smile_cascade)
    return img


# ---------------------------------------------------------------------------
# Local webcam mode — cv2.VideoCapture (no WebRTC, no lag)
# ---------------------------------------------------------------------------
def _run_local_webcam(detection_type, face_cascades, eye_cascade, smile_cascade):
    st.info(
        "🖥️ **Local mode detected** — using `cv2.VideoCapture` for high-quality, "
        "low-latency webcam access."
    )

    FRAME_WINDOW = st.empty()
    status_text  = st.empty()

    col1, col2 = st.columns(2)
    run  = col1.button("▶ Start Webcam", key="local_start")
    stop = col2.button("⏹ Stop",         key="local_stop")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "webcam_det_type" not in st.session_state:
        st.session_state.webcam_det_type = None

    # If the detection type changed while the webcam was running, stop cleanly
    if (
        st.session_state.webcam_running
        and st.session_state.webcam_det_type != detection_type
    ):
        st.session_state.webcam_running  = False
        st.session_state.webcam_det_type = None
        time.sleep(0.4)  # give Windows time to release the device

    if run:
        st.session_state.webcam_running  = True
        st.session_state.webcam_det_type = detection_type
    if stop:
        st.session_state.webcam_running  = False
        st.session_state.webcam_det_type = None

    if st.session_state.webcam_running:
        # Retry opening the camera a few times (handles brief release delay on Windows)
        cap = None
        for _ in range(4):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                break
            cap.release()
            time.sleep(0.3)

        if cap is None or not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure it is connected and not in use.")
            st.session_state.webcam_running  = False
            st.session_state.webcam_det_type = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    status_text.warning("⚠️ Frame capture failed — retrying…")
                    continue

                frame = _apply_detection(
                    frame, detection_type, face_cascades, eye_cascade, smile_cascade
                )
                FRAME_WINDOW.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width='stretch',
                )
        finally:
            cap.release()
            status_text.info("📷 Webcam stopped.")


# ---------------------------------------------------------------------------
# Video upload mode — process uploaded video frame-by-frame
# ---------------------------------------------------------------------------
def _run_video_upload(detection_type, face_cascades, eye_cascade, smile_cascade):
    st.info("📹 Upload a video file. Detection will be applied to every frame.")

    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("❌ Could not open the uploaded video.")
        os.unlink(tmp_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    col1, col2 = st.columns([3, 1])
    frame_idx   = col1.slider(
        "Preview frame", min_value=0, max_value=max(total_frames - 1, 0), value=0
    )
    process_all = col2.checkbox("Process full video", value=False)

    # --- Single frame preview ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        preview = _apply_detection(
            frame.copy(), detection_type, face_cascades, eye_cascade, smile_cascade
        )
        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"Frame {frame_idx} / {total_frames - 1}",
            width='stretch',
        )

    # --- Full video processing ---
    if process_all:
        st.warning(
            "⚠️ Processing the full video may take a while depending on its length. "
            "The processed video will be available for download when done."
        )
        if st.button("🚀 Process & Download Video"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = tmp_path.replace(".mp4", "_out.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress = st.progress(0)
            frame_no = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = _apply_detection(
                    frame, detection_type, face_cascades, eye_cascade, smile_cascade
                )
                writer.write(frame)
                frame_no += 1
                if total_frames > 0:
                    progress.progress(min(frame_no / total_frames, 1.0))

            writer.release()
            progress.progress(1.0)
            st.success("✅ Video processed!")

            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=f,
                    file_name="detected_output.mp4",
                    mime="video/mp4",
                )
            os.unlink(out_path)

    cap.release()
    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main Streamlit page
# ---------------------------------------------------------------------------
def opencv_detection_page():
    """OpenCV Detection Page — Face, Eye, Smile, Stop Sign, and Face Count Detection."""

    st.title("Detection using OpenCV")
    st.info(
        "ℹ️ Haar Cascade classifiers are used for detection. "
        "Result maybe not 100% accurate."
    )

    # ------------------------------------------------------------------
    # Detection type selector
    # ------------------------------------------------------------------
    detection_type = st.radio(
        "Select Detection Type",
        ("Face Detection", "Eye + Smile Detection", "Stop Sign Detection", "Real Time Face Count"),
        horizontal=True,
    )

    # ------------------------------------------------------------------
    # Load cascades
    # ------------------------------------------------------------------
    face_cascades: list = []
    eye_cascade         = None
    smile_cascade       = None

    if detection_type != "Stop Sign Detection":
        face_cascades = [
            _load_cascade("default"),
            _load_cascade("alt"),
            _load_cascade("alt_tree"),
        ]
        face_cascades = [c for c in face_cascades if c is not None]

        if not face_cascades:
            st.error("❌ No face cascade could be loaded. Please check the cascade files.")
            return

        if detection_type == "Eye + Smile Detection":
            eye_cascade   = _load_cascade("eye")
            smile_cascade = _load_cascade("smile")

    # ------------------------------------------------------------------
    # Input mode selector
    # ------------------------------------------------------------------
    mode = st.radio(
        "Select Input Method",
        ("Webcam", "Upload Video", "Image"),
        horizontal=True,
    )

    # ==================================================================
    # WEBCAM MODE
    # ==================================================================
    if mode == "Webcam":

        with st.expander("🔍 Environment debug info", expanded=False):
            st.write({
                "IS_LOCAL":               IS_LOCAL,
                "platform":               platform.system(),
                "HOSTNAME":               os.environ.get("HOSTNAME", "(not set)"),
                "IS_STREAMLIT_CLOUD":     os.environ.get("IS_STREAMLIT_CLOUD", "(not set)"),
                "STREAMLIT_SHARING_MODE": os.environ.get("STREAMLIT_SHARING_MODE", "(not set)"),
                "DISPLAY":                os.environ.get("DISPLAY", "(not set)"),
            })
            force_local = st.toggle(
                "🖥️ Force local cv2 mode (use this if you're on your own machine but "
                "auto-detection says cloud)",
                value=IS_LOCAL,
                key="force_local_webcam",
            )
            st.caption(
                "When ON: uses `cv2.VideoCapture` directly — zero lag, no WebRTC.  "
                "When OFF: uses WebRTC (needed for Streamlit Cloud)."
            )

        use_local = st.session_state.get("force_local_webcam", IS_LOCAL)

        if use_local:
            _run_local_webcam(detection_type, face_cascades, eye_cascade, smile_cascade)

        else:
            st.warning(
                "⚠️ **You appear to be running on Streamlit Cloud.**\n\n"
                "Streamlit Cloud has limited CPU resources, so the webcam stream "
                "may lag or stutter. For the best experience, we strongly recommend "
                "**running this app on your local machine** where `cv2.VideoCapture` "
                "is used directly for high-quality, lag-free video.\n\n"
                "The WebRTC stream below will still work, but performance may vary."
            )

            class VideoProcessor(VideoProcessorBase):
                """WebRTC video processor — runs ensemble face detection per frame."""

                def __init__(self):
                    self._det_type      = detection_type
                    self._face_cascades = face_cascades
                    self._eye_cascade   = eye_cascade
                    self._smile_cascade = smile_cascade

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    img = _apply_detection(
                        img,
                        self._det_type,
                        self._face_cascades,
                        self._eye_cascade,
                        self._smile_cascade,
                    )
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(
                key="opencv-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width":     {"ideal": 1280, "min": 640},
                        "height":    {"ideal": 720,  "min": 480},
                        "frameRate": {"ideal": 30,   "min": 15},
                    },
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [
                        {"urls": "stun:stun.l.google.com:19302"},
                        {"urls": "stun:stun1.l.google.com:19302"},
                        {"urls": "stun:stun2.l.google.com:19302"},
                        {"urls": "stun:stun.stunprotocol.org:3478"},
                    ],
                    "iceCandidatePoolSize": 10,
                },
                async_processing=True,
            )

    # ==================================================================
    # VIDEO UPLOAD MODE
    # ==================================================================
    elif mode == "Upload Video":
        _run_video_upload(detection_type, face_cascades, eye_cascade, smile_cascade)

    # ==================================================================
    # IMAGE MODE
    # ==================================================================
    else:
        image_source = st.radio(
            "Select Image Source", ("Upload Image", "Sample Image"), horizontal=True
        )

        image: np.ndarray | None = None

        if image_source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    st.error("❌ Could not decode the uploaded image.")

        else:  # Sample Image
            sample_map = {
                "Face Detection":        "src/open_cv/sample/group_pic.jpg",
                "Eye + Smile Detection": "src/open_cv/sample/henry.jpg",
                "Real Time Face Count":  "src/open_cv/sample/group_pic.jpg",
                "Stop Sign Detection":   "src/open_cv/sample/stop_sign.png",
            }
            sample_path = sample_map.get(detection_type, "")

            if st.button("Load Sample Image"):
                image = cv2.imread(sample_path)
                if image is None:
                    st.error(f"❌ Could not load sample image from: {sample_path}")

        # ---- Run detection on the loaded image ----
        if image is not None:
            if detection_type == "Real Time Face Count":
                result, count = run_face_count(image, face_cascades)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")
                st.success(f"👥 Faces Detected: **{count}**")

            elif detection_type == "Stop Sign Detection":
                result = run_stop_sign_detection(image)
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Stop Sign Detection", channels="RGB",
                )

            elif detection_type == "Face Detection":
                result = run_face_detection(image, face_cascades)
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Face Detection", channels="RGB",
                )

            else:  # Eye + Smile Detection
                result = run_eye_smile_detection(
                    image, face_cascades, eye_cascade, smile_cascade
                )
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Eye + Smile Detection", channels="RGB",
                )
