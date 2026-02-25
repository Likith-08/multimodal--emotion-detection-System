import os
import torch
import torch.nn as nn
import pickle

class TextEmotionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextEmotionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


def predict_text(sentence):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models")

    model_path = os.path.join(model_dir, "text_emotion_model.pth")
    vectorizer_path = os.path.join(model_dir, "text_vectorizer.pkl")
    encoder_path = os.path.join(model_dir, "text_label_encoder.pkl")

    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Load label encoder
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load model
    input_size = len(vectorizer.get_feature_names_out())
    num_classes = len(label_encoder.classes_)

    model = TextEmotionModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Transform text
    X = vectorizer.transform([sentence]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    emotion = label_encoder.inverse_transform(predicted.numpy())[0]
    confidence_score = float(confidence.numpy()[0]) * 100

    return emotion, round(confidence_score, 2)