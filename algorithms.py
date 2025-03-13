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
    interactive_plot(times, y, title='Audio time course', x_axis='Time [s]', y_axis='Amplitude', text=[f'Time: {t:.2f}s, Amplitude: {a:.2f}' for t, a in zip(times, y)])
    

def split_into_frames(y,sr, frame_size = 1024):
    step=int(frame_size)
    frames = [y[i:i+frame_size] for i in range(0, len(y)-frame_size, step)]
    return np.array(frames)

def loudness(y,sr, frame_size):
    frames = split_into_frames(y,sr, frame_size)
    frame_loudness =[]
    for f in frames:
        energy = np.sum(f**2)
        mean_energy = energy/frame_size
        loudness = np.sqrt(mean_energy)
        frame_loudness.append(loudness)

    frame_numbers = np.arange(len(frame_loudness))
    interactive_plot(x=frame_numbers, y= frame_loudness, title='Volume per frame', x_axis='Frame number', y_axis='Volume (RMS)', text = [f'Frame: {f}, Volume: {v:.2f}' for f,v in zip(frame_numbers, frame_loudness)])
    return frame_loudness

def short_time_energy(y,sr,frame_size):
    frames = split_into_frames(y,sr, frame_size)
    frame_energy =[]
    for f in frames:
        energy = np.sum(f**2)
        mean_energy = energy/frame_size
        frame_energy.append(mean_energy)
    frame_numbers = np.arange(len(frame_energy))
    interactive_plot(x=frame_numbers, y=frame_energy, title='Short time energy', x_axis='Frame number', y_axis='Short time energy (STE)', text=[f'Frame: {f}, STE: {v:.2f}' for f,v in zip(frame_numbers, frame_energy)])
    
def zero_crossing_rate(y, sr, frame_size):
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
    zcr_table = pd.DataFrame({
        'Numer ramki': frame_numbers,
        'Zero Crossing Rate': zcr_values
    })
    return zcr_table, zcr_values

def classify_silence(y, sr, frame_size, volume_threshold, zcr_threshold):
    volumes = loudness(y,sr,frame_size)
    zcr = zero_crossing_rate(y,sr,frame_size)

    classified_frames =[]
    for volume,zcr in zip(volumes, zcr):
        if volume < volume_threshold and zcr > zcr_threshold:
            classified_frames.append('silence')  
        else:
            classified_frames.append('sound') 
            
    return classified_frames


if __name__ == '__main__':
    audio = 'sample_data/zdanie_1.wav'
    print(loudness(audio, 1024))
    
    
    




    
    