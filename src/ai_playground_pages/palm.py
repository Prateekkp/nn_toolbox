import streamlit as st
import base64
try:
    import cv2
    import numpy as np
    CV_IMPORT_ERROR = None
except Exception as import_err:
    cv2 = None
    np = None
    CV_IMPORT_ERROR = import_err
import os
import re
import requests
import shutil
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_URL = os.getenv("NVIDIA_API_URL") or "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL") or "meta/llama-4-maverick-17b-128e-instruct"
ASSETS_PALM_DIR = Path(__file__).resolve().parents[1] / "assets" / "palm"
MODEL_PATH = ASSETS_PALM_DIR / "hand_landmarker.task"
LEGACY_MODEL_PATH = Path("hand_landmarker.task")
PALM_SECTIONS = ["Life line", "Heart line", "Head line", "Fate line", "Overall vibe"]


def is_streamlit_cloud_env():
    cloud_markers = [
        os.getenv("STREAMLIT_SHARING_MODE"),
        os.getenv("STREAMLIT_CLOUD"),
        os.getenv("STREAMLIT_RUNTIME"),
    ]
    if any(cloud_markers):
        return True

    cwd = str(Path.cwd()).replace("\\", "/")
    return cwd.startswith("/mount/src")


def call_nvidia_vision_llm(prompt, image_bytes):
    if not NVIDIA_API_KEY:
        raise ValueError("Missing NVIDIA_API_KEY in .env file.")

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.8,
        "stream": False,
    }

    response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=25)
    response.raise_for_status()
    response_json = response.json()
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("NVIDIA response did not include any choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")

    # NVIDIA can return content as a plain string or as a list of typed chunks.
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if text_part:
                    chunks.append(str(text_part))
            elif item:
                chunks.append(str(item))
        content = " ".join(chunks)

    content = str(content or "").strip()
    if not content:
        raise ValueError("NVIDIA response did not include readable content.")
    return content


def normalize_palm_reading(raw_text):
    def clean_text(value):
        return str(value or "").strip(" *-:\t\n\r")

    text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
    extracted = {section: "" for section in PALM_SECTIONS}

    label_patterns = {
        "Life line": r"life line",
        "Heart line": r"heart line",
        "Head line": r"head line",
        "Fate line": r"fate line",
        "Overall vibe": r"overall vibe|vibe",
    }

    for section, label_pattern in label_patterns.items():
        pattern = re.compile(
            rf"(?:\*\*)?\b{label_pattern}\b(?:\*\*)?\s*[:\-]\s*(.*?)(?=\s*(?:\b(?:life line|heart line|head line|fate line|overall vibe|vibe)\b(?:\*\*)?\s*[:\-])|$)",
            re.I,
        )
        match = pattern.search(text)
        if match:
            extracted[section] = clean_text(match.group(1))

    if not any(extracted.values()):
        chunks = re.split(r"\s*(?=\b(?:life line|heart line|head line|fate line|overall vibe|vibe)\b)\s*", text, flags=re.I)
        for chunk in chunks:
            lower_chunk = chunk.lower()
            if lower_chunk.startswith("life line"):
                extracted["Life line"] = clean_text(chunk.split(":", 1)[-1])
            elif lower_chunk.startswith("heart line"):
                extracted["Heart line"] = clean_text(chunk.split(":", 1)[-1])
            elif lower_chunk.startswith("head line"):
                extracted["Head line"] = clean_text(chunk.split(":", 1)[-1])
            elif lower_chunk.startswith("fate line"):
                extracted["Fate line"] = clean_text(chunk.split(":", 1)[-1])
            elif lower_chunk.startswith("overall vibe") or lower_chunk.startswith("vibe"):
                extracted["Overall vibe"] = clean_text(chunk.split(":", 1)[-1])

    if not extracted["Life line"]:
        extracted["Life line"] = "Reads as steady and grounded, with low-drama energy."
    if not extracted["Heart line"]:
        extracted["Heart line"] = "Emotionally private, but you do care more than you show."
    if not extracted["Head line"]:
        extracted["Head line"] = "Mental energy looks practical, sharp, and a little overactive."
    if not extracted["Fate line"]:
        extracted["Fate line"] = "Career path looks self-made, with moves that depend on your own choices."
    if not extracted["Overall vibe"]:
        extracted["Overall vibe"] = "Big independent energy with a no-nonsense mindset."

    return "\n".join([f"- **{section}:** {extracted[section]}" for section in PALM_SECTIONS])


# -------------------------------
# DOWNLOAD MEDIAPIPE MODEL
# -------------------------------
@st.cache_resource
def load_hand_landmarker():
    ASSETS_PALM_DIR.mkdir(parents=True, exist_ok=True)

    # Migrate existing model from old root path to assets/palm.
    if not MODEL_PATH.exists() and LEGACY_MODEL_PATH.exists():
        try:
            shutil.move(str(LEGACY_MODEL_PATH), str(MODEL_PATH))
        except Exception:
            pass

    if not MODEL_PATH.exists():
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        with st.spinner("Downloading hand detection model (one time only)..."):
            urllib.request.urlretrieve(url, str(MODEL_PATH))

    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return HandLandmarker.create_from_options(options)


# -------------------------------
# STEP 1 — DETECT HAND + CROP PALM
# -------------------------------
def detect_and_crop_palm(img_bgr):
    import mediapipe as mp

    landmarker = load_hand_landmarker()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None, "No hand detected. Show your full palm clearly in good light."

    landmarks = result.hand_landmarks[0]
    h, w = img_bgr.shape[:2]

    all_pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_min = max(0, min(xs) - 40)
    x_max = min(w, max(xs) + 40)
    y_min = max(0, min(ys) - 40)
    y_max = min(h, max(ys) + 40)

    palm_crop = img_bgr[y_min:y_max, x_min:x_max]

    crop_landmarks = [
        (int(lm.x * w) - x_min, int(lm.y * h) - y_min)
        for lm in landmarks
    ]

    palm_boundary_indices = [0, 1, 2, 5, 9, 13, 17, 18]
    palm_poly = np.array(
        [crop_landmarks[i] for i in palm_boundary_indices],
        dtype=np.int32
    )

    landmark_names = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    landmark_dict = {
        name: (lm.x, lm.y)
        for name, lm in zip(landmark_names, landmarks)
    }

    return palm_crop, palm_poly, landmark_dict, None


# -------------------------------
# STEP 2 — DETECT PALM LINES (masked)
# -------------------------------
def detect_palm_lines(palm_crop, palm_poly):
    h, w = palm_crop.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [palm_poly], 255)

    erode_kernel = np.ones((12, 12), np.uint8)
    mask_eroded = cv2.erode(mask, erode_kernel, iterations=1)

    masked_bgr = cv2.bitwise_and(palm_crop, palm_crop, mask=mask_eroded)

    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask_eroded)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 130)
    edges = cv2.bitwise_and(edges, edges, mask=mask_eroded)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_length = min(h, w) * 0.10
    significant = [c for c in contours if cv2.arcLength(c, False) > min_length]
    significant = sorted(significant, key=lambda c: cv2.arcLength(c, False), reverse=True)

    annotated = palm_crop.copy()
    cv2.polylines(annotated, [palm_poly], isClosed=True, color=(200, 200, 200), thickness=1)
    cv2.drawContours(annotated, significant, -1, (0, 220, 255), 2)

    line_summary = summarize_lines(significant, palm_crop.shape)

    return annotated, line_summary, len(significant)


# -------------------------------
# STEP 3 — SUMMARIZE LINE DATA
# -------------------------------
def summarize_lines(contours, shape):
    h, w = shape[:2]
    summary = []

    for i, c in enumerate(contours[:12]):
        length = cv2.arcLength(c, False)
        x, y, cw, ch = cv2.boundingRect(c)

        center_x = (x + cw / 2) / w
        center_y = (y + ch / 2) / h

        if center_y < 0.35:
            zone = "upper palm (heart line region)"
        elif center_y < 0.55:
            zone = "mid palm (head line region)"
        elif center_y < 0.75:
            zone = "lower mid palm (life line region)"
        else:
            zone = "lower palm (fate line / wrist region)"

        orientation = "horizontal" if cw > ch else "vertical"
        relative_length = "long" if length > min(h, w) * 0.3 else "medium" if length > min(h, w) * 0.15 else "short"

        summary.append(
            f"Line {i+1}: {relative_length}, {orientation}, in {zone}, "
            f"normalized length {length/min(h,w):.2f}"
        )

    return "\n".join(summary)


# -------------------------------
# LLM CALL WITH CV DATA
# -------------------------------
def analyze_with_llm(image_bgr, line_summary, line_count, landmark_dict):
    max_dim = 1024
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(image_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image_bgr

    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 72])
    image_bytes = buffer.tobytes()

    wrist = landmark_dict.get("WRIST", (0, 0))
    middle_mcp = landmark_dict.get("MIDDLE_MCP", (0, 0))
    palm_height = abs(middle_mcp[1] - wrist[1])
    hand_context = f"Palm height ratio: {palm_height:.2f} (0=flat, 1=tall palm)"

    prompt = f"""You are a Gen Z palm reader with real palmistry knowledge.

A computer vision system has already analyzed this palm and detected the following lines:

DETECTED LINES ({line_count} significant lines found):
{line_summary}

HAND STRUCTURE:
{hand_context}

The image you're seeing has the detected lines highlighted in cyan/yellow overlay.

Based on BOTH the CV data above AND what you can see in the image, give a palm reading covering:

1. Life Line — describe its length and curve based on what was detected in the lower-mid palm region
2. Heart Line — describe the upper palm lines and what they say about love/emotions
3. Head Line — describe the mid-palm lines and what they say about thinking style
4. Fate Line — any vertical lines detected? Career and purpose reading.
5. Overall Vibe — final verdict, 1-2 lines

Rules:
- Ground your reading in the actual CV data. Mention specifics like "your life line is long" only if the data supports it.
- If lines in a region are short or absent, say so honestly.
- Fun and Gen Z tone, no emojis, no fortune-teller drama.
- Output exactly 5 lines and nothing else.
- Use this exact format on every line:
    **Life line:** ...
    **Heart line:** ...
    **Head line:** ...
    **Fate line:** ...
    **Overall vibe:** ...
- Do not add numbering, bullets, titles, or extra commentary.
- Keep each line short.
"""
    fallback_text = call_nvidia_vision_llm(prompt, image_bytes)
    return normalize_palm_reading(fallback_text)


# -------------------------------
# PAGE
# -------------------------------
def palm_reader_page():
    for key, default in {
        "phase": "capture",
        "result": "",
        "image_bytes": None,
        "annotated_img": None,
        "line_count": 0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    def reset():
        st.session_state.phase = "capture"
        st.session_state.result = ""
        st.session_state.image_bytes = None
        st.session_state.annotated_img = None
        st.session_state.line_count = 0

    st.title("Palm Reader")

    if is_streamlit_cloud_env():
        st.warning("Palm Reader is currently supported on local machine only.")
        st.info("Please run this app locally to use OpenCV and MediaPipe features reliably.")
        return

    if CV_IMPORT_ERROR is not None:
        st.error(
            "OpenCV/NumPy failed to load in this environment. "
            "Please redeploy after updating requirements and packages. "
            f"Original import error: {CV_IMPORT_ERROR}"
        )
        return

    if st.session_state.phase == "capture":
        st.write("Hold your dominant hand flat, face the camera, fingers slightly apart. Take a clear photo.")
        st.write("---")
        st.info("Good lighting matters. Natural light or bright indoor light works best.")

        img_file = st.camera_input("Take a photo of your palm")

        if img_file is not None:
            st.session_state.image_bytes = img_file.getvalue()
            st.session_state.phase = "confirm"
            st.rerun()

    elif st.session_state.phase == "confirm":
        st.write("Good shot? Make sure lines are visible.")
        st.write("---")

        st.image(st.session_state.image_bytes, caption="Your palm", use_container_width=True)

        col1, col2 = st.columns([7, 1])
        with col1:
            if st.button("Read my palm"):
                st.session_state.phase = "processing"
                st.rerun()
        with col2:
            if st.button("Retake"):
                reset()
                st.rerun()

    elif st.session_state.phase == "processing":
        st.write("---")

        with st.spinner("Detecting hand landmarks..."):
            img_array = np.frombuffer(st.session_state.image_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            palm_crop, palm_poly, landmark_dict, error = detect_and_crop_palm(img_bgr)

        if error:
            st.error(error)
            if st.button("Try again"):
                reset()
                st.rerun()
            st.stop()

        with st.spinner("Detecting palm lines..."):
            annotated, line_summary, line_count = detect_palm_lines(palm_crop, palm_poly)
            st.session_state.line_count = line_count

            _, buf = cv2.imencode(".jpg", annotated)
            st.session_state.annotated_img = buf.tobytes()

        with st.spinner("Analyzing your palm..."):
            try:
                result = analyze_with_llm(annotated, line_summary, line_count, landmark_dict)
                st.session_state.result = result
                st.session_state.phase = "result"
                st.rerun()
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                if st.button("Try again"):
                    reset()
                    st.rerun()

    elif st.session_state.phase == "result":
        st.write("---")

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Your palm (original)")
            st.image(st.session_state.image_bytes, use_container_width=True)
        with col2:
            st.caption(f"Detected lines ({st.session_state.line_count} found)")
            st.image(st.session_state.annotated_img, use_container_width=True)

        st.write("---")
        st.markdown(st.session_state.result)
        st.write("---")

        if st.button("Read again"):
            reset()
            st.rerun()
