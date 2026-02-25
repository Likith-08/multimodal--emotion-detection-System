# 🎭 Multimodal Emotion Detection System

A complete AI-based emotion recognition system that detects human emotions using:

- 🧠 Text (NLP)
- 🎤 Speech (Audio Processing)
- 📷 Face (Computer Vision)
- 🔥 Fusion Model (Majority Voting)

This project combines multiple deep learning models to produce a final emotion prediction using multimodal fusion.

---

## 🚀 Project Overview

This system predicts emotions from three different input modalities:

1. Facial Expression (Image / Webcam)
2. Speech Emotion (Audio File)
3. Text Emotion (Sentence Input)

Each model predicts independently, and a Majority Fusion model determines the final emotion.

---

## 🧠 Architecture

```
          Image  ──► CNN Model ──► Face Emotion
          Audio  ──► Speech Model ──► Speech Emotion
          Text   ──► NLP Model ──► Text Emotion
                                │
                                ▼
                    Majority Voting Fusion
                                │
                                ▼
                      🎯 Final Emotion
```

---

## 🧩 Modules Used

### 📷 Face Emotion Module
- CNN-based deep learning model
- Trained on FER2013 dataset
- Converts image to grayscale
- Outputs emotion + confidence score

### 🎤 Speech Emotion Module
- Extracts audio features (MFCC)
- Deep learning classifier
- Supports .wav files
- Outputs emotion + confidence score

### 🧠 Text Emotion Module
- TF-IDF Vectorizer
- Machine Learning classifier
- Preprocessed text input
- Outputs emotion + confidence score

### 🔥 Fusion Module
- Majority voting strategy
- Combines predictions from all three models
- Outputs final emotion

---

## 🛠 Tech Stack

- Python
- PyTorch
- Scikit-learn
- OpenCV
- Librosa
- NumPy
- Streamlit (Frontend Interface)

---

## 📁 Project Structure

```
multimodal_emotion_system/
│
├── app.py
├── requirements.txt
│
├── face_module/
│   ├── train_cnn.py
│   └── predict_face.py
│
├── speech_module/
│   ├── train_speech.py
│   └── predict_speech.py
│
├── text_module/
│   ├── train_text.py
│   └── predict_text.py
│
├── fusion_module/
│   └── fusion_predict.py
│
└── models/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/multimodal-emotion-detection-system.git
cd multimodal-emotion-detection-system
```

### 2️⃣ Create Environment

```
conda create -n emotion_env python=3.9
conda activate emotion_env
```

### 3️⃣ Install Requirements

```
pip install -r requirements.txt
```

---

## ▶️ Run Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🎯 Features

- Real-time Webcam Emotion Detection
- Audio File Emotion Detection
- Text-based Emotion Detection
- Confidence Score Display
- Multimodal Fusion Prediction
- Clean Interactive UI

---

## 📊 Example Output

Face Emotion: Sad (22.91%)  
Speech Emotion: Happy (99.13%)  
Text Emotion: Sad (50.3%)  

🔥 Final Emotion (Fusion): Sad  

---

## 📈 Future Improvements

- Real-time microphone recording
- Live emotion tracking dashboard
- Model accuracy improvements
- Deployment on cloud (AWS / GCP)

---

## 👨‍💻 Author

Likith Reddy  
B.Tech - Computer Science  
Aspiring AI & Python Developer  

---

## ⭐ If You Like This Project

Give it a star ⭐ on GitHub!
