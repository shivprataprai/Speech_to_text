import time, os
import logging
import streamlit as st
import torch
import librosa
import soundfile as sf
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from settings import  DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import wave
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('app')

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")    
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 




def main():
    title = "Speech to Text"
    st.title(title)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Convert'):
        with st.spinner("Converting Speech into Text"):
            data = wavfile.read(WAVE_OUTPUT_FILE) 
            framerate = data[0] 
            sounddata = data[1]
            time = np.arange(0,len(sounddata))/framerate 
            input_audio, _ = librosa.load(WAVE_OUTPUT_FILE, sr=16000) 
            input_values = tokenizer(input_audio, return_tensors="pt").input_values   
            logits = model(input_values).logits  
            predicted_ids = torch.argmax(logits, dim=-1) 
            transcription = tokenizer.batch_decode(predicted_ids)[0] 
        #with st.spinner("Converting Speech into Text"):
            #chord = cnn.predict(WAVE_OUTPUT_FILE, False)
        st.success("Classification completed")
        st.write("### The recorded speech is **", transcription + "**")
        if transcription == 'N/A':
            st.write("Please record sound first")
        st.write("\n")

if __name__ == '__main__':
    main()