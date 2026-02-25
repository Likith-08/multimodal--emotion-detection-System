import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from face_module.predict_face import predict_face
from speech_module.predict_speech import predict_speech
from text_module.predict_text import predict_text


def majority_vote(predictions):
    count = {}
    for emotion in predictions:
        if emotion in count:
            count[emotion] += 1
        else:
            count[emotion] = 1

    return max(count, key=count.get)


if __name__ == "__main__":

    image_path = input("Enter image path: ")
    audio_path = input("Enter audio path: ")
    text_input = input("Enter text: ")

    face_emotion = predict_face(image_path)
    speech_emotion = predict_speech(audio_path)
    text_emotion = predict_text(text_input)

    print("\nFace Emotion:", face_emotion)
    print("Speech Emotion:", speech_emotion)
    print("Text Emotion:", text_emotion)

    final_emotion = majority_vote([face_emotion, speech_emotion, text_emotion])

    print("\n🔥 Final Emotion (Fusion):", final_emotion)