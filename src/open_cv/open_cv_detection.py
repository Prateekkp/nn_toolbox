import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av


def opencv_detection_page():
    """OpenCV Detection Page - Face, Eye, Smile, Stop Sign, and Face Count Detection"""
    
    st.title("Detection using OpenCV")
    
    st.info("ℹ️ We are using Haar Cascade classifiers for detection. Results may not be 100% accurate as this is a classical computer vision approach.")

    # Detection type selection
    detection_type = st.radio(
        "Select Detection Type",
        ("Face Detection", "Eye + Smile Detection", "Stop Sign Detection", "Real Time Face Count"),
        horizontal=True
    )

    # Only show further options after detection type is selected
    if detection_type:
        # Load appropriate cascades based on detection type
        if detection_type != "Stop Sign Detection":
            face_cascade = cv2.CascadeClassifier(
                "src/open_cv/cascades/haarcascade_frontalface_default.xml"
            )

            if detection_type == "Eye + Smile Detection":
                eye_cascade = cv2.CascadeClassifier(
                    "src/open_cv/cascades/haarcascade_eye.xml"
                )
                smile_cascade = cv2.CascadeClassifier(
                    "src/open_cv/cascades/haarcascade_smile.xml"
                )
        
        # Face counting function for Real Time Face Count mode
        def detect_and_count_faces(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
            
            face_count = len(faces)
            
            cv2.putText(
                frame,
                f"People Count: {face_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )
            
            return frame, face_count

        # Detection function for STOP sign
        def detect_stop_sign(image):
            image = cv2.resize(image, (600, 400))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Red color ranges
            lower_red1 = np.array([0, 80, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 80, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 | mask2

            # Morphology
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > 1500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h)

                    if 0.8 < aspect_ratio < 1.2:
                        cv2.rectangle(
                            image, (x, y), (x + w, y + h),
                            (0, 255, 0), 3
                        )
                        cv2.putText(
                            image, "STOP SIGN",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0), 2
                        )

            return image

        # Mode selection
        mode = st.radio(
            "Select Input Method",
            ("Webcam", "Image"),
            horizontal=True
        )

        # -----------------------------
        # WEBCAM MODE
        # -----------------------------
        if mode == "Webcam":
            st.info("🎥 Real-time webcam detection. Click 'START' to begin streaming.")
            
            # Create a video processor class for real-time detection
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.detection_type = detection_type
                    self.face_cascade = face_cascade if detection_type != "Stop Sign Detection" else None
                    self.eye_cascade = eye_cascade if detection_type == "Eye + Smile Detection" else None
                    self.smile_cascade = smile_cascade if detection_type == "Eye + Smile Detection" else None
                
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    
                    if self.detection_type == "Real Time Face Count":
                        img, count = self.detect_and_count_faces(img)
                    elif self.detection_type == "Stop Sign Detection":
                        img = self.detect_stop_sign(img)
                    elif self.detection_type == "Face Detection":
                        img = self.detect_faces(img)
                    else:  # Eye + Smile Detection
                        img = self.detect_eyes_smile(img)
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                def detect_and_count_faces(self, frame):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    face_count = len(faces)
                    cv2.putText(frame, f"People Count: {face_count}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    return frame, face_count
                
                def detect_stop_sign(self, image):
                    image = cv2.resize(image, (600, 400))
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    
                    lower_red1 = np.array([0, 80, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 80, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    mask = mask1 | mask2
                    
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 1500:
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect_ratio = w / float(h)
                            if 0.8 < aspect_ratio < 1.2:
                                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(image, "STOP SIGN", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    return image
                
                def detect_faces(self, frame):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, "Face", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    return frame
                
                def detect_eyes_smile(self, frame):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_color = frame[y:y + h, x:x + w]
                        
                        # Eyes
                        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        
                        # Smile
                        smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
                        for (sx, sy, sw, sh) in smiles:
                            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                            cv2.putText(roi_color, "Smile", (sx, sy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    return frame
            
            # Start the webrtc streamer with size constraints
            webrtc_streamer(
                key="opencv-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480}
                    },
                    "audio": False
                },
                async_processing=True,
            )

        # -----------------------------
        # IMAGE MODE
        # -----------------------------
        else:
            image_source = st.radio(
                "Select Image Source",
                ("Upload Image", "Sample Image"),
                horizontal=True
            )
            
            if image_source == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload an image",
                    type=["jpg", "jpeg", "png"]
                )

                if uploaded_file is not None:
                    file_bytes = np.asarray(
                        bytearray(uploaded_file.read()),
                        dtype=np.uint8
                    )

                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if detection_type == "Real Time Face Count":
                        result, count = detect_and_count_faces(image)
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, channels="RGB")
                        st.success(f"People Count: {count}")
                    elif detection_type == "Stop Sign Detection":
                        result = detect_stop_sign(image)
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, caption="Stop Sign Detection", channels="RGB")
                    else:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.3,
                            minNeighbors=5
                        )

                        for (x, y, w, h) in faces:
                            if detection_type == "Face Detection":
                                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(
                                    image,
                                    "Face",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )
                            else:  # Eye + Smile Detection
                                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                                roi_gray = gray[y:y + h, x:x + w]
                                roi_color = image[y:y + h, x:x + w]

                                # Eyes
                                eyes = eye_cascade.detectMultiScale(
                                    roi_gray,
                                    scaleFactor=1.1,
                                    minNeighbors=10
                                )

                                for (ex, ey, ew, eh) in eyes:
                                    cv2.rectangle(roi_color,
                                                  (ex, ey),
                                                  (ex + ew, ey + eh),
                                                  (0, 255, 0), 2)

                                # Smile
                                smiles = smile_cascade.detectMultiScale(
                                    roi_gray,
                                    scaleFactor=1.7,
                                    minNeighbors=20
                                )

                                for (sx, sy, sw, sh) in smiles:
                                    cv2.rectangle(roi_color,
                                                  (sx, sy),
                                                  (sx + sw, sy + sh),
                                                  (0, 0, 255), 2)
                                    cv2.putText(roi_color, "Smile",
                                                (sx, sy - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)

                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption="Detected Faces", channels="RGB")

            
            else:  # Sample Image
                # Select sample image based on detection type
                if detection_type == "Face Detection":
                    sample_path = "src/open_cv/sample/group_pic.jpg"
                elif detection_type == "Eye + Smile Detection":
                    sample_path = "src/open_cv/sample/henry.jpg"
                elif detection_type == "Real Time Face Count":
                    sample_path = "src/open_cv/sample/group_pic.jpg"
                else:  # Stop Sign Detection
                    sample_path = "src/open_cv/sample/stop_sign.png"
                
                if st.button("Load Sample Image"):
                    image = cv2.imread(sample_path)
                    
                    if image is not None:
                        if detection_type == "Real Time Face Count":
                            result, count = detect_and_count_faces(image)
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            st.image(result_rgb, channels="RGB")
                            st.success(f"People Count: {count}")
                        elif detection_type == "Stop Sign Detection":
                            result = detect_stop_sign(image)
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            st.image(result_rgb, caption="Stop Sign Detection - Sample", channels="RGB")
                        else:
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.3,
                                minNeighbors=5
                            )

                            for (x, y, w, h) in faces:
                                if detection_type == "Face Detection":
                                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(
                                        image,
                                        "Face",
                                        (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (0, 255, 0),
                                        2
                                    )
                                else:  # Eye + Smile Detection
                                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                                    roi_gray = gray[y:y + h, x:x + w]
                                    roi_color = image[y:y + h, x:x + w]

                                    # Eyes
                                    eyes = eye_cascade.detectMultiScale(
                                        roi_gray,
                                        scaleFactor=1.1,
                                        minNeighbors=10
                                    )

                                    for (ex, ey, ew, eh) in eyes:
                                        cv2.rectangle(roi_color,
                                                      (ex, ey),
                                                      (ex + ew, ey + eh),
                                                      (0, 255, 0), 2)

                                    # Smile
                                    smiles = smile_cascade.detectMultiScale(
                                        roi_gray,
                                        scaleFactor=1.7,
                                        minNeighbors=20
                                    )

                                    for (sx, sy, sw, sh) in smiles:
                                        cv2.rectangle(roi_color,
                                                      (sx, sy),
                                                      (sx + sw, sy + sh),
                                                      (0, 0, 255), 2)
                                        cv2.putText(roi_color, "Smile",
                                                    (sx, sy - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6,
                                                    (0, 0, 255), 2)

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption="Detected Faces - Sample Image", channels="RGB")
                    else:
                        st.error(f"Could not load sample image from: {sample_path}")
