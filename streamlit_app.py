
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
SR = 22050 
N_MFCC = 13
N_MELS = 128
HOP_LENGTH = 512
st.title('Voice Emotion Classifier - Demo')

model = None
@st.cache_resource(show_spinner=False)
def load_my_model(path='cnn_mel_model.h5'):
    return load_model(path)

if st.button('Load Model'):
    model = load_model('cnn_mel_model.h5')
    st.success('Model loaded')



uploaded_file = st.file_uploader('Upload a WAV file', type=['wav'])

if uploaded_file is not None:
    data, sr = sf.read(uploaded_file)
    y = data.astype('float32')
    st.audio(uploaded_file)
    
    y_mono = librosa.to_mono(y.T) if y.ndim > 1 else y
    y_res = librosa.resample(y_mono, orig_sr=sr, target_sr=SR)

  
    y_res = np.asarray(y_res, dtype=np.float32).flatten()

    S = librosa.feature.melspectrogram(y=y_res, sr=SR, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-6)
    
    import cv2
    img = cv2.resize(S_norm, (128, 128)).astype('float32')
    x = img[np.newaxis, ..., np.newaxis]
    
    if model is None:
        model = load_my_model()
    pred = model.predict(x)
    idx = np.argmax(pred, axis=1)[0]
    
    # label mapping - match these to training labels
    labels = ['neutral', 'happy', 'sad', 'angry']
    st.write('Predicted emotion:', labels[idx])


