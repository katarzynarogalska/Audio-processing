import librosa
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def interactive_plot(x, y, title, x_axis, y_axis, text,  width=1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode ='lines', text= text,
                             hoverinfo='text', line=dict(color='#0d0469', width=width)))
    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title= y_axis,
        showlegend=False
    )
    st.plotly_chart(fig,key=None)

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
    

def split_into_frames(y,sr, frame_size = 1024):
    step=int(frame_size)
    frames = [y[i:i+frame_size] for i in range(0, len(y)-frame_size, step)]
    return np.array(frames)

def loudness(y,sr, frame_size, plot=True):
    frames = split_into_frames(y,sr, frame_size)
    frame_loudness =[]
    for f in frames:
        energy = np.sum(f**2)
        mean_energy = energy/frame_size
        loudness = np.sqrt(mean_energy)
        frame_loudness.append(loudness)

    frame_numbers = np.arange(len(frame_loudness))
    if plot:
     interactive_plot(x=frame_numbers, y= frame_loudness, title='Volume per frame', x_axis='Frame number', y_axis='Volume (RMS)', text = [f'Frame: {f}, Volume: {v:.2f}' for f,v in zip(frame_numbers, frame_loudness)])
    return frame_loudness

def short_time_energy(y,sr,frame_size,plot=True):
    frames = split_into_frames(y,sr, frame_size)
    frame_energy =[]
    for f in frames:
        energy = np.sum(f**2)
        mean_energy = energy/frame_size
        frame_energy.append(mean_energy)
    frame_numbers = np.arange(len(frame_energy))
    if plot:
        interactive_plot(x=frame_numbers, y=frame_energy, title='Short time energy', x_axis='Frame number', y_axis='Short time energy (STE)', text=[f'Frame: {f}, STE: {v:.2f}' for f,v in zip(frame_numbers, frame_energy)])
    return frame_energy
def zero_crossing_rate(y, sr, frame_size, plot=True):
    frames = split_into_frames(y,sr,frame_size)
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
        interactive_plot(x = frame_numbers, y=zcr_values, title='Zero crossing rate per frame', x_axis='Frame number', y_axis='ZCR', text=[f'Frame: {f}, ZCR: {v:.2f}' for f,v in zip(frame_numbers, zcr_values)] )
    return  zcr_values

def classify_silence(y,sr,frame_size,vol_t):
   
    volume = loudness(y,sr,frame_size, plot=False)
    zcr= zero_crossing_rate(y,sr,frame_size, plot=False)


    #klasyfikacja ramki jako cisza lub nie
    silence_frames = []
    for i, (v) in enumerate(volume):
        if  v<vol_t:
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


if __name__ == '__main__':
    audio = 'sample_data/zdanie_1.wav'
    y,sr = librosa.load(audio, sr=None)

    classify_silence(y,sr,1024, 0.05,0.05)
    
    
    




    
    