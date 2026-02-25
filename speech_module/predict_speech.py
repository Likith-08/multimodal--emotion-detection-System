import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import pickle

# Device
device = torch.device("cpu")

# Emotion classes (MUST match training order)
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Model
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpeechEmotionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


# Feature extraction
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc


def predict_speech(file_path):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../models/speech_emotion_model.pth")
    encoder_path = os.path.join(current_dir, "../models/speech_label_encoder.pkl")

    # Load label encoder
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    input_size = 13
    num_classes = len(label_encoder.classes_)

    model = SpeechEmotionModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Extract features
    features = extract_features(file_path)

    # Convert to tensor
    audio_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    emotion = classes[predicted.item()]
    confidence_score = float(confidence.item()) * 100

    return emotion, round(confidence_score, 2)