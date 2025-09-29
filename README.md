# 🎙️ Voice Emotion Classifier

This project is a **Streamlit-based web app** that classifies emotions from voice recordings using a Convolutional Neural Network (CNN) trained on Mel-spectrogram features.

---

## 🚀 Features
- Upload a `.wav` audio file and get the predicted **emotion**.
- Preprocessing includes:
  - Resampling to 22,050 Hz
  - Converting stereo to mono
  - Extracting **Mel-spectrograms**
  - Normalizing and resizing features
- CNN model (`cnn_mel_model.h5`) used for classification.
- Supported emotion classes:  
  **Neutral, Happy, Sad, Angry**

---

## 📂 DataSet 
  - RAVDESS Emotional Speech Dataset 
