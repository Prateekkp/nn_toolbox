import os
import tempfile

import cv2
import streamlit as st

from src.application_pages.open_cv.open_cv_shared import apply_detection


def run_video_upload_use_case(detection_type, face_cascades, eye_cascade, smile_cascade):
    st.info("Upload a video file. Detection will be applied to every frame.")

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
        st.error("Could not open the uploaded video.")
        os.unlink(tmp_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    col1, col2 = st.columns([3, 1])
    frame_idx = col1.slider(
        "Preview frame", min_value=0, max_value=max(total_frames - 1, 0), value=0
    )
    process_all = col2.checkbox("Process full video", value=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        preview = apply_detection(
            frame.copy(), detection_type, face_cascades, eye_cascade, smile_cascade
        )
        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"Frame {frame_idx} / {total_frames - 1}",
            width="stretch",
        )

    if process_all:
        st.warning(
            "Processing the full video may take a while depending on its length."
        )
        if st.button("Process and Download Video", key="opencv_process_video"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = tmp_path.replace(".mp4", "_out.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress = st.progress(0)
            frame_no = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = apply_detection(
                    frame, detection_type, face_cascades, eye_cascade, smile_cascade
                )
                writer.write(frame)
                frame_no += 1

                if total_frames > 0:
                    progress.progress(min(frame_no / total_frames, 1.0))

            writer.release()
            progress.progress(1.0)
            st.success("Video processed successfully.")

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    data=f,
                    file_name="detected_output.mp4",
                    mime="video/mp4",
                )
            os.unlink(out_path)

    cap.release()
    os.unlink(tmp_path)
