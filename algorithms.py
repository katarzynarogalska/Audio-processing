import librosa
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks
import os


def interactive_plot(x, y, title, x_axis, y_axis, text, key, width=1):
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x, y=y, mode ='lines', text= text,
    #                          hoverinfo='text', line=dict(color='#0d0469', width=width)))
    # fig.update_layout(
    #     title=title,
    #     xaxis_title=x_axis,
    #     yaxis_title= y_axis,
    #     showlegend=False
    # )
    # st.plotly_chart(fig, key=key)
    
    # # save button for the plot
    # if st.button(f'Save {title}', key=f'save_{key}'):
    #     file_path = os.path.join('report', f"{title.replace(' ', '_')}.html")
    #     fig.write_html(file_path)
    #     st.success(f"Plot saved as {title.replace(' ', '_')}.html")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color='#0d0469', linewidth=width)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.grid(True)
    st.pyplot(fig)

    # przycisk zapisywania
    if st.button(f'Save {title}', key=f'save_{key}'):
        if not os.path.exists('report'):
            os.makedirs('report')
        file_path = os.path.join('report', f"{title.replace(' ', '_')}.png")
        fig.savefig(file_path)
        st.success(f"Plot saved as {title.replace(' ', '_')}.png")
    
def plot_audio(y,sr):
    times = np.arange(0,len(y))/sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=y, mode ='markers', text= [f'Time: {t:.2f}s, Amplitude: {a:.2f}' for t, a in zip(times, y)],
                             hoverinfo='text', marker=dict(size=1.2, color='blue')))
    fig.update_layout(
        title='Audio timecourse',
        xaxis_title='Time [s]',
        yaxis_title= 'Amplitude',
        showlegend=False
    )
    st.plotly_chart(fig,key=None)
    if st.button(f'Save Audio Timeline', key=f'save_timeline'):
        file_path = os.path.join('report', f"Audio_timeline.html")
        fig.write_html(file_path)
        st.success(f"Plot saved as Audio_timeline.html")
    
def split_into_frames(y, sr, frame_length_sec=0.1):
    frame_length = int(frame_length_sec * sr)  
    num_frames = len(y) // frame_length  # liczba pełnych ramek
    frames = [y[i * frame_length: (i + 1) * frame_length] for i in range(num_frames)]
    
    remainder = len(y) % frame_length
    if remainder:
        frames.append(y[-remainder:])
    return frames


def loudness(y,sr,frame_length=0.1, plot=True):
    frames = split_into_frames(y,sr, frame_length)
    frame_size = int(frame_length * sr)

    frame_loudness =[]
    for f in frames:
        energy = np.sum(f**2)
        mean_energy = energy/frame_size
        loudness = np.sqrt(mean_energy)
        frame_loudness.append(loudness)

    frame_numbers = np.arange(len(frame_loudness))
    if plot:
     interactive_plot(x=frame_numbers, y= frame_loudness, title='Volume per frame', x_axis='Frame number', y_axis='Volume (RMS)', text = [f'Frame: {f}, Volume: {v:.2f}' for f,v in zip(frame_numbers, frame_loudness)], key='volume_button')
    return frame_loudness

def short_time_energy(y,sr,frame_length=0.1,plot=True):
    frames = split_into_frames(y,sr, frame_length)
    frame_size = int(frame_length * sr)
    frame_energy =[]
    for f in frames:
        energy = np.sum(f**2)
        mean_energy = energy/frame_size
        frame_energy.append(mean_energy)
    frame_numbers = np.arange(len(frame_energy))
    if plot:
        interactive_plot(x=frame_numbers, y=frame_energy, title='Short time energy', x_axis='Frame number', y_axis='Short time energy (STE)', text=[f'Frame: {f}, STE: {v:.2f}' for f,v in zip(frame_numbers, frame_energy)],key='ste_button')
    return frame_energy


def zero_crossing_rate(y, sr, frame_length=0.1, plot=True):
    frames = split_into_frames(y,sr,frame_length)
    frame_size= int(frame_length * sr)

    def signum(x):
        if x==0:
            return 0
        elif x <0:
            return -1
        else:
            return 1
    zcr_values =[]
    for f in frames:
        zcr =0
        for i in range(1, len(f)):
            zcr += abs(signum(f[i]) - signum(f[i-1]))
        zcr_values.append(zcr/(2*frame_size))
    frame_numbers = np.arange(len(zcr_values))
    if plot:
        interactive_plot(x = frame_numbers, y=zcr_values, title='Zero crossing rate per frame', x_axis='Frame number', y_axis='ZCR', text=[f'Frame: {f}, ZCR: {v:.2f}' for f,v in zip(frame_numbers, zcr_values)] , key='zcr_button')
    return  zcr_values


def classify_silence(y,sr,vol_t, zcr_t, frame_length=0.1):
   
    volume = loudness(y,sr,frame_length, plot=False)
    zcr = zero_crossing_rate(y,sr,frame_length=frame_length, plot=False)
    frame_size = int(frame_length*sr)

    #klasyfikacja ramki jako cisza lub nie
    silence_frames = []
    for i, (v,z) in enumerate(zip(volume, zcr)):
        if  v<vol_t and z>zcr_t:
            silence_frames.append(i)

    silence_mask = np.zeros(len(y), dtype=bool)
    for frame_idx in silence_frames:
        start = frame_idx * frame_size
        end = start + frame_size
        silence_mask[start:end] = True

    times = np.arange(0, len(y)) / sr
    loud_times = times[~silence_mask]
    loud_amplitudes = y[~silence_mask]
    silent_times = times[silence_mask]
    silent_amplitudes = y[silence_mask]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=loud_times, y=loud_amplitudes, mode='markers',
                             marker = dict(size=1.2, color='blue'),
                             name='Volume'))
    
    fig.add_trace(go.Scatter(x=silent_times, y=silent_amplitudes, mode='markers',
                            marker = dict(size=1, color='gray'),
                             name='Silence'))
    
    fig.update_layout(
        title='Audio time course with detected silence',
        xaxis_title='Time [s]',
        yaxis_title='Amplitude',
        showlegend=True
    )
    
    st.plotly_chart(fig, key='1234')
    if st.button(f'Save Silence Detection', key=f'save_silence'):
        file_path = os.path.join('report', f"silence.html")
        fig.write_html(file_path)
        st.success(f"Plot saved as silence.html")


def estimate_f0_auto(y, sr,frame_length=0.1, min_frequency=50, max_frequency=200):
    frames = split_into_frames(y,sr,frame_length)
    f0_values =[]
    min_period = int(sr / max_frequency)
    max_period = int(sr / min_frequency)
    for frame in frames:
        N=len(frame)
        autocorr=np.zeros(max_period)
        for lag in range(min_period, max_period):
            if lag >= N:  # czy lag jest większe niż rozmiar ramki
                continue
            autocorr[lag] = np.sum(frame[:N-lag] * frame[lag:N])
        autocorr = autocorr[min_period:max_period] #dodatnie opoznienia
        peak_indices, _ = find_peaks(autocorr)
        if len(peak_indices) == 0:
            return 0  # Jeśli nie znaleziono maksimum, zwracamy 0 Hz
        best_lag = peak_indices[0] + min_period
        f0 = sr / best_lag
        f0_values.append(f0)
    frame_numbers=np.arange(len(frames))
    interactive_plot(x = frame_numbers, y=f0_values, title='Fundamental Frequency Autocorrelation', x_axis='Frame number', y_axis='F0', text=[f'Frame: {f}, F0: {v:.2f}' for f,v in zip(frame_numbers, f0_values)] , key='f0_corr_button')
    return f0_values


def estimate_f0_amdf(y,sr,frame_length=0.1,min_frequency=50, max_frequency=200):
    frames = split_into_frames(y,sr,frame_length)
    f0_vals =[]
    min_lag = int(sr/max_frequency)
    max_lag = int(sr/min_frequency)
    for frame in frames:
        N=len(frame)
        amdf = np.zeros(max_lag)
        for l in range(min_lag, max_lag):
            if l >= N:  # Sprawdź, czy lag jest większe niż rozmiar ramki
                continue
            amdf[l] = np.sum(np.abs(frame[:N-l] - frame[l:N]))
        #find min after adding real lag
        best_lag = np.argmin(amdf[min_lag:max_lag]) + min_lag
        f0 = sr / best_lag if best_lag > 0 else 0  # Convert lag to frequency
        f0_vals.append(f0)
    frame_numbers = np.arange(len(frames))
    interactive_plot(x = frame_numbers, y=f0_vals, title='Fundamental Frequency AMDF', x_axis='Frame number', y_axis='F0', text=[f'Frame: {f}, F0: {v:.2f}' for f,v in zip(frame_numbers, f0_vals)] , key='f0_amdf_button')
    return f0_vals




  






if __name__ == '__main__':
    audio = 'sample_data/.wav'
    y,sr = librosa.load(audio, sr=None)

    f0(y,sr)
    
    
    




    
    