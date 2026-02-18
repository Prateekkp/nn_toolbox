import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import os

# ---------------------------------------------------------------------------
# Cascade paths
# ---------------------------------------------------------------------------
_CASCADE_DIR = "src/open_cv/cascades"

_CASCADE_PATHS = {
    "default":   os.path.join(_CASCADE_DIR, "haarcascade_frontalface_default.xml"),
    "alt":       os.path.join(_CASCADE_DIR, "haarcascade_frontalface_alt.xml"),
    "alt_tree":  os.path.join(_CASCADE_DIR, "haarcascade_frontalface_alt_tree.xml"),
    "eye":       os.path.join(_CASCADE_DIR, "haarcascade_eye.xml"),
    "smile":     os.path.join(_CASCADE_DIR, "haarcascade_smile.xml"),
}


def _load_cascade(key: str) -> cv2.CascadeClassifier | None:
    """Load a Haar cascade by key; return None and warn if missing/invalid."""
    path = _CASCADE_PATHS.get(key, "")
    if not os.path.exists(path):
        st.warning(f"⚠️ Cascade not found: {path}")
        return None
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        st.warning(f"⚠️ Failed to load cascade: {path}")
        return None
    return clf


# ---------------------------------------------------------------------------
# NMS helper
# ---------------------------------------------------------------------------
def _nms_boxes(boxes, overlap_thresh: float = 0.3):
    """
    Non-Maximum Suppression over a list of (x, y, w, h) boxes.
    Returns filtered list of (x, y, w, h).
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= overlap_thresh)[0] + 1]

    return [tuple(map(int, boxes[k])) for k in keep]


# ---------------------------------------------------------------------------
# Multi-cascade face detection (ensemble of 3 cascades + NMS)
# ---------------------------------------------------------------------------
def detect_faces_ensemble(
    gray: np.ndarray,
    cascades: list,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    min_size: tuple = (30, 30),
    nms_thresh: float = 0.3,
) -> list:
    """
    Run all provided cascades on *gray* and merge results with NMS.

    Parameters
    ----------
    gray          : Grayscale, equalised image.
    cascades      : List of loaded CascadeClassifier objects (Nones are skipped).
    scale_factor  : Detection scale factor (lower = more detections, slower).
    min_neighbors : Minimum neighbours required (higher = fewer false positives).
    min_size      : Minimum face size in pixels.
    nms_thresh    : IoU threshold for NMS deduplication.

    Returns
    -------
    List of (x, y, w, h) tuples after NMS.
    """
    all_boxes = []
    for clf in cascades:
        if clf is None:
            continue
        detections = clf.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(detections) > 0:
            all_boxes.extend(detections.tolist())

    return _nms_boxes(all_boxes, overlap_thresh=nms_thresh)


# ---------------------------------------------------------------------------
# Pre-process frame for faster, more robust detection
# ---------------------------------------------------------------------------
def _preprocess(frame: np.ndarray, max_width: int = 640) -> tuple[np.ndarray, float]:
    """
    Resize frame so its width ≤ max_width (keeps aspect ratio),
    convert to grayscale, and equalise histogram.

    Returns (gray_eq, scale) where scale = original_width / resized_width.
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = w / max_width
        new_w, new_h = max_width, int(h / scale)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    return gray_eq, scale


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_face_box(frame, x, y, w, h, label="Face", color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Background pill for text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_count_overlay(frame, count: int):
    label = f"Faces Detected: {count}"
    cv2.rectangle(frame, (10, 8), (10 + len(label) * 13, 42), (0, 0, 0), -1)
    cv2.putText(frame, label, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# High-level detection functions (used by both Image and Webcam modes)
# ---------------------------------------------------------------------------
def run_face_detection(frame: np.ndarray, face_cascades: list) -> np.ndarray:
    """Detect faces and draw labelled bounding boxes."""
    gray_eq, scale = _preprocess(frame)
    faces = detect_faces_ensemble(gray_eq, face_cascades)

    for (x, y, w, h) in faces:
        sx, sy, sw, sh = (int(v * scale) for v in (x, y, w, h))
        _draw_face_box(frame, sx, sy, sw, sh, label="Face")

    return frame


def run_face_count(frame: np.ndarray, face_cascades: list) -> tuple[np.ndarray, int]:
    """Detect faces, draw boxes, and overlay the count."""
    gray_eq, scale = _preprocess(frame)
    faces = detect_faces_ensemble(gray_eq, face_cascades)

    for (x, y, w, h) in faces:
        sx, sy, sw, sh = (int(v * scale) for v in (x, y, w, h))
        _draw_face_box(frame, sx, sy, sw, sh, label=f"#{faces.index((x,y,w,h))+1}",
                       color=(0, 200, 255))

    _draw_count_overlay(frame, len(faces))
    return frame, len(faces)


def run_eye_smile_detection(
    frame: np.ndarray,
    face_cascades: list,
    eye_cascade,
    smile_cascade,
) -> np.ndarray:
    """Detect faces, then eyes and smiles within each face ROI."""
    gray_eq, scale = _preprocess(frame)
    faces = detect_faces_ensemble(gray_eq, face_cascades)

    for (x, y, w, h) in faces:
        sx, sy, sw, sh = (int(v * scale) for v in (x, y, w, h))
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 100, 0), 2)

        # Work on the original-resolution ROI for sub-detection accuracy
        roi_gray = cv2.cvtColor(frame[sy:sy + sh, sx:sx + sw], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.equalizeHist(roi_gray)
        roi_color = frame[sy:sy + sh, sx:sx + sw]

        if eye_cascade is not None:
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        if smile_cascade is not None:
            smiles = smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.6, minNeighbors=18, minSize=(25, 25)
            )
            for (smx, smy, smw, smh) in smiles:
                cv2.rectangle(roi_color, (smx, smy), (smx + smw, smy + smh), (0, 0, 255), 2)
                cv2.putText(roi_color, "Smile", (smx, smy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    return frame


def run_stop_sign_detection(image: np.ndarray) -> np.ndarray:
    """Detect stop signs via HSV colour + contour shape analysis."""
    image = cv2.resize(image, (640, 480))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 80, 50]),   np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 80, 50]), np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.75 < (w / float(h)) < 1.35:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(image, "STOP SIGN", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return image


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
    # Detection type
    # ------------------------------------------------------------------
    detection_type = st.radio(
        "Select Detection Type",
        ("Face Detection", "Eye + Smile Detection", "Stop Sign Detection", "Real Time Face Count"),
        horizontal=True,
    )

    # ------------------------------------------------------------------
    # Load cascades once (cached via st.cache_resource would be ideal,
    # but cascade objects aren't pickle-able; load per session instead)
    # ------------------------------------------------------------------
    face_cascades = []   # ensemble list
    eye_cascade   = None
    smile_cascade = None

    if detection_type != "Stop Sign Detection":
        face_cascades = [
            _load_cascade("default"),
            _load_cascade("alt"),
            _load_cascade("alt_tree"),
        ]
        # Filter out any that failed to load
        face_cascades = [c for c in face_cascades if c is not None]

        if not face_cascades:
            st.error("❌ No face cascade could be loaded. Please check the cascade files.")
            return

        if detection_type == "Eye + Smile Detection":
            eye_cascade   = _load_cascade("eye")
            smile_cascade = _load_cascade("smile")

    # ------------------------------------------------------------------
    # Input mode
    # ------------------------------------------------------------------
    mode = st.radio("Select Input Method", ("Webcam", "Image"), horizontal=True)

    # ==================================================================
    # WEBCAM MODE
    # ==================================================================
    if mode == "Webcam":
        st.info("🎥 Real-time webcam detection. Click **START** to begin streaming.")

        class VideoProcessor(VideoProcessorBase):
            """WebRTC video processor — runs ensemble face detection per frame."""

            def __init__(self):
                self._det_type      = detection_type
                self._face_cascades = face_cascades
                self._eye_cascade   = eye_cascade
                self._smile_cascade = smile_cascade

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")

                if self._det_type == "Real Time Face Count":
                    img, _ = run_face_count(img, self._face_cascades)
                elif self._det_type == "Stop Sign Detection":
                    img = run_stop_sign_detection(img)
                elif self._det_type == "Face Detection":
                    img = run_face_detection(img, self._face_cascades)
                else:  # Eye + Smile Detection
                    img = run_eye_smile_detection(
                        img, self._face_cascades, self._eye_cascade, self._smile_cascade
                    )

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="opencv-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
        )

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
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                         caption="Stop Sign Detection", channels="RGB")

            elif detection_type == "Face Detection":
                result = run_face_detection(image, face_cascades)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                         caption="Face Detection", channels="RGB")

            else:  # Eye + Smile Detection
                result = run_eye_smile_detection(
                    image, face_cascades, eye_cascade, smile_cascade
                )
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                         caption="Eye + Smile Detection", channels="RGB")
