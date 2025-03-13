import streamlit as st
from algorithms import *
st.set_page_config(layout="wide")


st.title('Audio analysis app')


with st.container(border=True, height=200):
    uploaded_file = st.file_uploader("Choose audio file to upload", type=["wav"])

st.header('Basic audio plots')  
with st.container(border=True):
    if uploaded_file is not None:
        y,sr = librosa.load(uploaded_file, sr=None)
        plot_audio(y,sr)
        
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
            st.table(zero_crossing_rate(y,sr,1024)[0])
       