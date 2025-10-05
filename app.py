import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(page_title="Real-Time Face Mask Detection", page_icon="😷", layout="wide")

# --------------------------
# Load Models
# --------------------------
@st.cache_resource
def load_mask_model():
    return load_model("final_mobilenetv2_facemask.h5")

@st.cache_resource
def load_face_detector():
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    return cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_mask_model()
faceNet = load_face_detector()

# --------------------------
# Detection Function
# --------------------------
def detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        preds = maskNet.predict(np.array(faces, dtype="float32"), batch_size=32, verbose=0)

    return locs, preds

# --------------------------
# Video Transformer
# --------------------------
class MaskDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.confidence_threshold = 0.5

    def update_settings(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        locs, preds = detect_and_predict_mask(img, faceNet, maskNet, self.confidence_threshold)

        labels = ["Without Mask", "With Mask", "Incorrect Mask"]
        colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0)]

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            class_index = np.argmax(pred)
            confidence = pred[class_index] * 100
            label = f"{labels[class_index]}: {confidence:.1f}%"
            color = colors[class_index]

            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --------------------------
# UI
# --------------------------
st.title("😷 Real-Time Face Mask Detection")
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, 0.1)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_transformer:
    ctx.video_transformer.update_settings(confidence_threshold)
