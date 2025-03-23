import streamlit as st
from algorithms import *
from clip_algorithms import *
import librosa
from fpdf import FPDF
import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go
import io
st.set_page_config(layout="wide")


st.title('Audio analysis app')

with st.sidebar:
        st.subheader('File upload')
        uploaded_file = st.file_uploader("Choose audio file to upload", type=["wav"])
      
        st.subheader("Analysis Options")
        frame_check = st.checkbox('Show frame-level analysis')
        clip_check = st.checkbox('Show clip-level analysis')
        
with st.container():
    if uploaded_file is None:
        st.markdown(
    '<span style="color: #b11f47; font-weight: bold;font-size: 20px;">Please upload a .wav file to analyse.</span>', 
    unsafe_allow_html=True
)
    if uploaded_file is not None:
        st.write('Uploaded file:')
        st.audio(uploaded_file)


 
with st.container(border=True):
    if uploaded_file is not None:
        st.header('Audio timeline') 
        y,sr = librosa.load(uploaded_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        frame_length =0.05*duration
        plot_audio(y,sr)



if frame_check:
    st.header('Audio frame characteristics')
    with st.container(border=True):
        if uploaded_file is not None:
            with st.container(border=True):
                st.subheader('Volume analysis')
                loudness(y,sr, frame_length)
            with st.container(border=True):
                st.subheader('Short Time Energy analysis')
                short_time_energy(y,sr, frame_length)
            with st.container(border=True):
                st.subheader('Zero Crossing Rate analysis')
                zero_crossing_rate(y,sr, frame_length)
            with st.container():
                st.subheader('Silence detection')
                volume = loudness(y,sr,frame_length, plot=False)
                min_volume = np.min(volume)
                max_volume = np.max(volume)
                mid = (max_volume - min_volume)/2

                zcr = zero_crossing_rate(y,sr, frame_length, plot=False)
                min_zcr = np.min(zcr)
                max_zcr = np.max(zcr)
                mid_zcr = (max_zcr-min_zcr)/2

                
                vol_slider = st.slider('Choose volume threshold', min_value=float(min_volume),max_value = float(max_volume), value=float(mid))
                zcr_slider = st.slider('Choose ZCR threshold', min_value=float(min_zcr), max_value=float(max_zcr), value = float(mid_zcr))
                #ste_slider = st.slider('Choose STE threshold', min_value=float(min_ste), max_value=float(max_ste), value=float(mid_ste), step=1e-6)
                classify_silence(y,sr,frame_length=frame_length, vol_t=vol_slider, zcr_t=zcr_slider)
            with st.container():
                st.subheader('Fundamental Frequency')
                f0(y,sr)

# clip based analysis
if clip_check:
    st.header('Clip based analysis')
    st.write('Volume related metrics')
    with st.container():
        v1 = vstd(y,sr)
        vdr = volume_dynamic_range(y,sr)
        vu = volume_undulation(y,sr)
        df =pd.DataFrame({
            "Metric" : ['VSTD', 'VDR', 'VU'],
            "Value":[v1,vdr, vu]
        })
        df = df.reset_index(drop=True)
        st.table(df)
        if st.button('Save Volume Metrics', key='save_volume_metrics'):
            file_path = os.path.join('report', 'volume_metrics.csv')
            df.to_csv(file_path, index=False)
            st.success(f"Table saved as {file_path}")

    st.write('Energy related metrics')
    with st.container():
        l = lster(y,sr)
        entropy = energy_entropy(y,sr)
        df =pd.DataFrame({
            "Metric" : ['LSTER', 'ENTROPY'],
            "Value":[l,entropy]
        })
        df = df.reset_index(drop=True)
        st.table(df)
        if st.button('Save Energy Metrics', key='save_energy_metrics'):
            file_path = os.path.join('report', 'energy_metrics.csv')
            df.to_csv(file_path, index=False)
            st.success(f"Table saved as {file_path}")

    st.write('Zero Crossing Rate related metrics')
    with st.container():
        hzcr = hzcrr(y,sr)
        df =pd.DataFrame({
            "Metric" : ['HZCRR'],
            "Value":[hzcr]
        })
        df = df.reset_index(drop=True)
        st.table(df)
        if st.button('Save ZCR Metrics', key='save_zcr_metrics'):
            file_path = os.path.join('report', 'zcr_metrics.csv')
            df.to_csv(file_path, index=False)
            st.success(f"Table saved as {file_path}")


  