import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_folder = os.path.join(current_dir, "audio")

features = []
labels = []

for file in os.listdir(audio_folder):
    if file.endswith(".wav"):
        label = file.split("_")[0]
        file_path = os.path.join(audio_folder, file)

        signal, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)

        features.append(mfcc)
        labels.append(label)

features = np.array(features)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(encoded_labels, dtype=torch.long)

class SpeechEmotionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpeechEmotionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

model = SpeechEmotionModel(X.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Speech model trained!")

torch.save(model.state_dict(), "models/speech_emotion_model.pth")

with open("models/speech_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Speech model saved!")