import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Sample dataset
texts = [
    "I am very happy",
    "This is amazing",
    "I feel great",
    "I am sad",
    "This is terrible",
    "I feel depressed",
    "I am angry",
    "This makes me furious",
    "I am scared",
    "I feel nervous"
]

labels = [
    "happy",
    "happy",
    "happy",
    "sad",
    "sad",
    "sad",
    "angry",
    "angry",
    "fear",
    "fear"
]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(encoded_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Model
class TextEmotionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextEmotionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

model = TextEmotionModel(X_train.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
epochs = 50
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Text model trained!")

# Save model
torch.save(model.state_dict(), "models/text_emotion_model.pth")

with open("models/text_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/text_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Text model saved!")