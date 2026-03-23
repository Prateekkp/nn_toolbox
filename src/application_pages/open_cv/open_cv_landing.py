import streamlit as st

from src.application_pages.open_cv.open_cv_image import run_image_use_case
from src.application_pages.open_cv.open_cv_shared import DETECTION_OPTIONS, prepare_detectors
from src.application_pages.open_cv.open_cv_video import run_video_upload_use_case
from src.application_pages.open_cv.open_cv_webcam import run_webcam_use_case


INPUT_OPTIONS = ("Webcam", "Upload Video", "Image")


def open_cv_landing_page():
    st.title("Detection using OpenCV")
    st.info(
        "Haar Cascade classifiers are used for detection. "
        "Results may not be 100 percent accurate."
    )

    detection_type = st.radio(
        "Select Detection Type",
        DETECTION_OPTIONS,
        horizontal=True,
    )

    detectors = prepare_detectors(detection_type)
    if detectors is None:
        return
    face_cascades, eye_cascade, smile_cascade = detectors

    mode = st.radio("Select Input Method", INPUT_OPTIONS, horizontal=True)

    if mode == "Webcam":
        run_webcam_use_case(detection_type, face_cascades, eye_cascade, smile_cascade)
    elif mode == "Upload Video":
        run_video_upload_use_case(
            detection_type,
            face_cascades,
            eye_cascade,
            smile_cascade,
        )
    else:
        run_image_use_case(detection_type, face_cascades, eye_cascade, smile_cascade)
