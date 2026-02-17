import cv2
import streamlit as st
import numpy as np


def opencv_detection_page():
    """OpenCV Detection Page - Face, Eye, Smile, Stop Sign, and Face Count Detection"""
    
    st.title("Detection using OpenCV")
    
    st.info("ℹ️ We are using Haar Cascade classifiers for detection. Results may not be 100% accurate as this is a classical computer vision approach.")

    # Session state init
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

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

            start = st.button("Start Camera")

            if start:
                st.session_state.camera_on = True

            if st.session_state.camera_on:
                cap = cv2.VideoCapture(0)
                frame_placeholder = st.empty()

                stop = st.button("Stop Camera")

                while st.session_state.camera_on:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera not accessible")
                        break

                    if detection_type == "Real Time Face Count":
                        output, count = detect_and_count_faces(frame)
                        frame_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    elif detection_type == "Stop Sign Detection":
                        output = detect_stop_sign(frame)
                        frame_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.3,
                            minNeighbors=5
                        )

                        for (x, y, w, h) in faces:
                            if detection_type == "Face Detection":
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    "Face",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )
                            else:  # Eye + Smile Detection
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                                roi_gray = gray[y:y + h, x:x + w]
                                roi_color = frame[y:y + h, x:x + w]

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

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(frame_rgb, channels="RGB")

                    if stop:
                        st.session_state.camera_on = False
                        cap.release()
                        frame_placeholder.empty()
                        break

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
                    sample_path = r"C:\PK\Data Analysis\Projects\nn_toolbox\src\open_cv\sample\group_pic.jpg"
                elif detection_type == "Eye + Smile Detection":
                    sample_path = r"C:\PK\Data Analysis\Projects\nn_toolbox\src\open_cv\sample\henry.jpg"
                elif detection_type == "Real Time Face Count":
                    sample_path = r"C:\PK\Data Analysis\Projects\nn_toolbox\src\open_cv\sample\group_pic.jpg"
                else:  # Stop Sign Detection
                    sample_path = r"C:\PK\Data Analysis\Projects\nn_toolbox\src\open_cv\sample\stop_sign.png"
                
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
