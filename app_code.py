import streamlit as st
from algorithms import *
import librosa
st.set_page_config(layout="wide")


st.title('Audio analysis app')


with st.container(border=True, height=200):
    uploaded_file = st.file_uploader("Choose audio file to upload", type=["wav"])
with st.container():
    st.write('Uploaded file:')
    if uploaded_file is not None:
        st.audio(uploaded_file)


st.header('Audio timeline')  
with st.container(border=True):
    if uploaded_file is not None:
        y,sr = librosa.load(uploaded_file, sr=None)
        plot_audio(y,sr)

st.header('Audio analysis')
with st.container(border=True, height=100):
    frame_check = st.checkbox('Show frame-level analysis')
    clip_check = st.checkbox('Show clip-level analysis')

if frame_check:
        
    st.header('Audio frame characteristics')
    with st.container(border=True):
        if uploaded_file is not None:
            with st.container(border=True):
                st.subheader('Volume analysis')
                loudness(y,sr, 1024)
            with st.container(border=True):
                st.subheader('Short Time Energy analysis')
                short_time_energy(y,sr,1024)
            with st.container(border=True):
                st.subheader('Zero Crossing Rate analysis')
                zero_crossing_rate(y,sr,1024)
            with st.container():
                st.subheader('Silence detection')
                volume = loudness(y,sr,1024, plot=False)
                min_volume = np.min(volume)
                max_volume = np.max(volume)
                mid = (max_volume - min_volume)/2
                zcr = zero_crossing_rate(y,sr,frame_size=1024, plot=False)
                min_zcr = np.min(zcr)
                max_zcr = np.max(zcr)
                mid_zcr = (max_zcr-min_zcr)/2

                
                vol_slider = st.slider('Choose volume threshold', min_value=float(min_volume),max_value = float(max_volume), value=float(mid))
                #zcr_slider = st.slider('Choose ZCR threshold', min_value=float(min_zcr), max_value=float(max_zcr), value = float(mid_zcr))
                #ste_slider = st.slider('Choose STE threshold', min_value=float(min_ste), max_value=float(max_ste), value=float(mid_ste), step=1e-6)
                classify_silence(y,sr,frame_size=1024, vol_t=vol_slider)
            with st.container():
                st.subheader('Fundamental Frequency')
                #f0_autocorrelation(y,sr,1024)
       