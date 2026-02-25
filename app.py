from PIL import Image
import streamlit as st
import os
from collections import Counter

from face_module.predict_face import predict_face
from speech_module.predict_speech import predict_speech
from text_module.predict_text import predict_text


# -----------------------------
# Fusion Function (Inside app.py)
# -----------------------------
def majority_fusion(results):
    count = Counter(results)
    return count.most_common(1)[0][0]


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Multimodal Emotion Detection", layout="centered")

st.title("🎭 Multimodal Emotion Detection System")

# -----------------------------
# Emojis
# -----------------------------
emotion_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
    "neutral": "😐"
}

# -----------------------------
# Inputs
# -----------------------------
st.subheader("📸 Upload Face Image")
image_file = st.file_uploader("Upload JPG/PNG Image", type=["jpg", "png", "jpeg"])

st.subheader("📷 OR Capture From Webcam")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image)
    image_path = "temp_camera.jpg"
    image.save(image_path)
    image_file = image_path

st.subheader("🎤 Upload Speech Audio")
audio_file = st.file_uploader("Upload WAV File", type=["wav"])

st.subheader("💬 Enter Text")
text_input = st.text_input("Type a sentence")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Emotion"):

    results = []

    # -------- FACE --------

    if image_file is not None:

     # If uploaded image
     if isinstance(image_file, str):
        image_path = image_file
     else:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.read())

     face_emotion, face_conf = predict_face(image_path)

    st.write(f"### 📸 Face Emotion: {emotion_emojis.get(face_emotion, '')} {face_emotion}")
    st.progress(int(face_conf))
    st.write(f"Confidence: {face_conf}%")

    results.append(face_emotion)

    # -------- SPEECH --------
    if audio_file is not None:
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        speech_emotion, speech_conf = predict_speech(audio_path)

        st.write(f"### 🎤 Speech Emotion: {emotion_emojis.get(speech_emotion, '')} {speech_emotion}")
        st.progress(int(speech_conf))
        st.write(f"Confidence: {speech_conf}%")

        results.append(speech_emotion)
        if isinstance(image_file, str):
                image_path = image_file
        else:
                image_path = "temp_image.jpg"
                with open(image_path, "wb") as f:
                    f.write(image_file.read())

        face_emotion, face_conf = predict_face(image_path)
    # -------- TEXT --------
    if text_input.strip() != "":
        text_emotion, text_conf = predict_text(text_input)

        st.write(f"### 💬 Text Emotion: {emotion_emojis.get(text_emotion, '')} {text_emotion}")
        st.progress(int(text_conf))
        st.write(f"Confidence: {text_conf}%")

        results.append(text_emotion)

    # -------- FUSION --------
    if len(results) > 0:
        final_emotion = majority_fusion(results)

        st.markdown("---")
        st.success(
            f"{emotion_emojis.get(final_emotion, '')} Final Emotion (Fusion): *{final_emotion.upper()}*"
        )