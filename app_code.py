import streamlit as st
from algorithms import *
from clip_algorithms import *
import librosa
import pandas as pd
import plotly.graph_objs as go
from algorithms2 import *
st.set_page_config(layout="wide")


# Here is code for the app interface using algorithms for both project 1 and 2


st.title('Audio analysis app')

with st.sidebar:
        st.subheader('File upload')
        uploaded_file = st.file_uploader("Choose audio file to upload", type=["wav"])
        
        if uploaded_file is not None:
            y,sr = librosa.load(uploaded_file, sr=None)
            normalize = st.checkbox('Normalize signal')
            if normalize:
                y = y/np.max(np.abs(y))
            with st.container(border=True):
                st.markdown('Uploaded file classified as:')
            
                classif,l = speech_music(y,sr)
                st.markdown(f'<span style="color: #b11f47; font-weight: bold;font-size: 15px;">{classif}</span>', unsafe_allow_html=True)
                st.write(l)
            with st.container(border=True):
                if classif=='Music':
                    cl,bp = beats(y,sr)
                    st.markdown(f'<span style="color: #b11f47; font-weight: bold;font-size: 15px;">{cl}</span>', unsafe_allow_html=True
                                )
                    st.write(bp)

            st.subheader("Analysis Options")
            frame_check = st.checkbox('Show frame-level analysis')
            clip_check = st.checkbox('Show clip-level analysis')
            fourier_check = st.checkbox('Show frequency analysis')
        
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
       
        duration = librosa.get_duration(y=y, sr=sr)
        frame_length =0.1
        plot_audio(y,sr)


if uploaded_file is not None:
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
                    st.subheader('Fundamental Frequency - Autocorrelation')
                    estimate_f0_auto(y,sr)
                    st.subheader('Fundamental Frequency - AMDF')
                    estimate_f0_amdf(y,sr)

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
            zs = zstd(y,sr)
            df =pd.DataFrame({
                "Metric" : ['ZSTD','HZCRR'],
                "Value":[zs, hzcr]
            })
            df = df.reset_index(drop=True)
            st.table(df)
            if st.button('Save ZCR Metrics', key='save_zcr_metrics'):
                file_path = os.path.join('report', 'zcr_metrics.csv')
                df.to_csv(file_path, index=False)
                st.success(f"Table saved as {file_path}")
        st.subheader('Extra audio features')
        spectral_centroid(y,sr,frame_length)
# -------------------------------------------------------Projekt 2 --------------------------------------------------
if uploaded_file is not None:
    y = y/np.max(np.abs(y))
    if fourier_check:
        st.header('Frequency analysis')
       
        granularity = st.radio(label='Chose analysis level', options=['Whole clip', 'Single Frame'])
        if granularity == 'Single Frame':
            
            with st.container(border=True):
                frame_lengths = [0.05,0.1,0.2,0.3,0.4]
                frame_len = st.select_slider('Choose frame length:',options=frame_lengths,value=0.1)
                frames = split_into_frames(y,sr,frame_len)
                frame_number = st.number_input(label='Choose frame to analyse', min_value=0, max_value=(len(frames)-1), step=1) 
                frame = frames[frame_number]
                frame_size = int(frame_len*sr)
                window_functions =['Rectangle','Triangle','Hann', 'Hamming', 'Blackman']
                chosen_window = st.radio(label='Choose window function', options=window_functions, index=0)
                window_func = get_window(chosen_window, frame_size)
                windowed_frame = frame* window_func
                times = np.arange(0,len(frame))/sr
            st.subheader(f'Window functions comparison for frame {frame_number}')
            plot_after_window(frame, times, windowed_frame, chosen_window)
            st.subheader(f'Frequency magnitude spectrum for frame {frame_number}')
            plot_fourier(windowed_frame, sr, f'Fourier for frame with {chosen_window} window',freq_ratio=0.5)
        else:
            with st.container(border=True):
                window_functions =['Rectangle','Triangle','Hann', 'Hamming', 'Blackman']
                chosen_window = st.radio(label='Choose window function', options=window_functions, index=0)
                window_func = get_window(chosen_window, len(y))
                windowed_signal = y* window_func
                times = np.arange(0,len(y))/sr
            st.subheader('Window functions comparison for the whole clip')
            plot_after_window(y, times, windowed_signal, chosen_window)
            st.subheader('Frequency magnitude spectrum for the whole clip')
            
            plot_fourier(windowed_signal, sr, f'Fourier for the whole signal with {chosen_window} window',freq_ratio=0.5)
        st.header('Frequency frame based metrics')
        with st.container(border=True):
            frame_sizes = [128, 256, 512, 1024, 2048]
            frame_size1 = st.select_slider('Choose frame size for clip analysis:',options=frame_sizes,value=1024)
            fc_values, eb_values = clip_functions(y,sr, chosen_window, frame_size1)

            st.subheader('Frequency Centroids and Bandwidth')
            plot_centroids(fc_values, eb_values, frame_size1, sr)

            st.subheader('Band Energy')
            band_en = band_energies(y, sr, chosen_window, frame_size1)
            plot_band_energies(band_en, frame_size1, sr)

            st.subheader('Band Energy Ratio')
            band_enery_ratios(band_en, frame_size1, sr)

            st.subheader('Spectral Flatness Measure')
            spectral_flatness(y,sr,chosen_window, frame_size1)

            st.subheader('Spectral Crest Factor')
            spectral_crest(y,sr,chosen_window, frame_size1)

        st.header('Spectrogram')
        with st.container(border=True):
            overlap = st.number_input(label='Choose overlap ratio', min_value=float(0.1), max_value=float(0.90), step=float(0.1), value=0.5)
            frame_size3 = st.select_slider(label='Chose frame size', options= frame_sizes, value=1024 )
            plot_spectrogram(y,frame_size3,sr,chosen_window,overlap)



           

            



            





  