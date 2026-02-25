import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Device
device = torch.device("cpu")

# Emotion classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Same CNN model (must match training architecture)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Load model
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("models/face_emotion_model.pth", map_location=device))
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test image path
image_path = "test_image.jpg"  # change this

image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Predicted Emotion:", classes[predicted.item()])
from PIL import Image
def predict_face(image_path):

    # Load image properly
    image = Image.open(image_path)

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 48x48 (FER standard size)
    image = image.resize((48, 48))

    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    face_tensor = transform(image)

    # Add batch dimension
    face_tensor = face_tensor.unsqueeze(0)

    # Move to device
    face_tensor = face_tensor.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    emotion = classes[predicted.item()]
    confidence_score = float(confidence.item()) * 100

    return emotion, round(confidence_score, 2)