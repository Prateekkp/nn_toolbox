import os
import platform
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.application_pages.open_cv.open_cv_shared import apply_detection


def is_local_environment() -> bool:
    """Detect whether app is running locally instead of Streamlit Cloud."""
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


def run_local_webcam(detection_type, face_cascades, eye_cascade, smile_cascade):
    st.info(
        "Local mode detected. Using cv2.VideoCapture for low-latency webcam access."
    )

    frame_window = st.empty()
    status_text = st.empty()

    col1, col2 = st.columns(2)
    run = col1.button("Start Webcam", key="opencv_local_start")
    stop = col2.button("Stop", key="opencv_local_stop")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "webcam_det_type" not in st.session_state:
        st.session_state.webcam_det_type = None

    if (
        st.session_state.webcam_running
        and st.session_state.webcam_det_type != detection_type
    ):
        st.session_state.webcam_running = False
        st.session_state.webcam_det_type = None
        time.sleep(0.4)

    if run:
        st.session_state.webcam_running = True
        st.session_state.webcam_det_type = detection_type
    if stop:
        st.session_state.webcam_running = False
        st.session_state.webcam_det_type = None

    if st.session_state.webcam_running:
        cap = None
        for _ in range(4):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                break
            cap.release()
            time.sleep(0.3)

        if cap is None or not cap.isOpened():
            st.error("Could not open webcam. Make sure it is connected and not in use.")
            st.session_state.webcam_running = False
            st.session_state.webcam_det_type = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    status_text.warning("Frame capture failed. Retrying...")
                    continue

                frame = apply_detection(
                    frame, detection_type, face_cascades, eye_cascade, smile_cascade
                )
                frame_window.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch",
                )
        finally:
            cap.release()
            status_text.info("Webcam stopped.")


def run_webcam_use_case(detection_type, face_cascades, eye_cascade, smile_cascade):
    is_local = is_local_environment()

    with st.expander("Environment debug info", expanded=False):
        st.write(
            {
                "IS_LOCAL": is_local,
                "platform": platform.system(),
                "HOSTNAME": os.environ.get("HOSTNAME", "(not set)"),
                "IS_STREAMLIT_CLOUD": os.environ.get("IS_STREAMLIT_CLOUD", "(not set)"),
                "STREAMLIT_SHARING_MODE": os.environ.get("STREAMLIT_SHARING_MODE", "(not set)"),
                "DISPLAY": os.environ.get("DISPLAY", "(not set)"),
            }
        )
        st.toggle(
            "Force local cv2 mode (turn on if you are local but auto-detection says cloud)",
            value=is_local,
            key="force_local_webcam",
        )
        st.caption(
            "When ON: uses cv2.VideoCapture directly. When OFF: uses WebRTC."
        )

    use_local = st.session_state.get("force_local_webcam", is_local)

    if use_local:
        run_local_webcam(detection_type, face_cascades, eye_cascade, smile_cascade)
        return

    st.warning(
        "You appear to be on Streamlit Cloud. Webcam streaming may lag due to "
        "limited CPU resources."
    )

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self._det_type = detection_type
            self._face_cascades = face_cascades
            self._eye_cascade = eye_cascade
            self._smile_cascade = smile_cascade

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = apply_detection(
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
                "width": {"ideal": 1280, "min": 640},
                "height": {"ideal": 720, "min": 480},
                "frameRate": {"ideal": 30, "min": 15},
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
