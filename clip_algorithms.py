from algorithms import *
import librosa
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Volume based ----------------------------------------------------------------------------------------
def vstd(y,sr,frame_length=1):
    volumes = loudness(y,sr,frame_length,plot=False)
    std = np.std(volumes)
    max_vol = np.max(volumes)
    return std/max_vol

def volume_dynamic_range(y,sr,frame_length=1):
    volume = loudness(y,sr,frame_length, plot=False)
    min_vol = np.min(volume)
    max_vol = np.max(volume)
    vdr = (max_vol-min_vol)/max_vol
    return vdr

def volume_undulation(y,sr,frame_length=1):
    volumes = np.array(loudness(y,sr,frame_length, plot=False))
    peaks, _ = find_peaks(volumes)
    valleys, _ = find_peaks(-volumes)
    extrema = np.sort(np.concatenate([peaks, valleys])) #sortowanie po indeksach
    diffs = np.diff(volumes[extrema]) #różnice między sasiednimi ekstremami
    vu = np.sum(np.abs(diffs))  # Suma różnic
    return vu


# Energy based --------------------------------------------------------------------------------------------------------

def lster(y,sr,frame_length=1):
    energies = short_time_energy(y,sr,frame_length, plot=False)
    avste = np.average(energies)
    sum =0
    for en in energies:
        sum+= np.sign(0.5*avste - en)+ 1
    return sum/(2*len(energies))


def energy_entropy(y,sr,frame_length=1):
    ste = short_time_energy(y,sr,frame_length, plot=False)
    frame_size = int(frame_length * sr) #ilość próbek w ramce
    K = 441
    normalized_ste =[] 
    for i in range(len(ste)):
        frame_start = i * frame_size
        frame_end = (i + 1) * frame_size
        frame = y[frame_start:frame_end]
        total_energy = ste[i]
        for start in range(0, len(frame), K):
            segment_end = min(start+K, len(ste))
            segment = frame[start:segment_end]
            segment_energy = np.sum(segment**2)

            if total_energy > 0:  #dzielenie przez 0 
                normalized_energy = segment_energy / total_energy
            else:
                normalized_energy = 0
            normalized_ste.append(normalized_energy)
    #clipping to avoid log(0)
    normalized_ste = np.clip(normalized_ste, 1e-10, 1)
    entropy = -np.sum(normalized_ste * np.log(normalized_ste))
    return entropy

# ZCR Based ---------------------------------------------------------------------------------------------------------
def zstd(y,sr,frame_length=1):
    zcrs = zero_crossing_rate(y,sr,frame_length, plot=False)
    return np.std(zcrs)

def hzcrr(y,sr,frame_length=1):
    zcrs = zero_crossing_rate(y,sr,frame_length, plot=False)
    avgzcr = np.average(zcrs)
    sum =0
    for zcr in zcrs:
        sum+= np.sign(zcr  -1.5*avgzcr)+ 1
    return sum/(2*len(zcrs))


# music speech detection ------------------------------------------------------------------------------------------------
def speech_music(y,sr):
    l = lster(y,sr)
   
    if l>0.2:
        return (f'Speech', f'LSER = {l:.3f}')
    elif l<0.14:
        return (f'Music', f'LSER = {l:.3f}')
    else:
        return (f'Uncetrain', f'LSER = {l:.3f}')
    

# Extras --------------------------------------------------------------------------------------------------------------

def spectral_centroid(y,sr,frame_length):
    frame_size = int(frame_length*sr)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size)[0]
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_size)[0]
    times = librosa.frames_to_time(range(len(sc)), sr=sr)
    # interactive_plot(x=times, y=sc,title='Spectral Centroid plot', x_axis='Time [s]', y_axis='Spectral Centroid', text='', key='spectral_centroid')
    # return sc
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, sc, label="Spectral Centroid", color='#0d0469', linewidth=2)
    ax.fill_between(times, sc - band, sc + band, 
                    color='#a0c1db', alpha=0.3, label="Spectral Bandwidth")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectral Centroid & Bandwidth Over Time")
    ax.legend()
    st.pyplot(fig)


#only for clips classified as music
def beats(y,sr):
    tempo= librosa.feature.tempo(y=y, sr=sr)[0]
    tempo = np.round(tempo,2)
    if tempo<86:
        return ('Slow music', f'{tempo}BPM')
    elif tempo<100:
        return ('Medium pace music', f'{tempo}BPM')
    elif tempo<140:
        return ('Fast music', f'{tempo}BPM')
    else:
        return ('Very fast music', f'{tempo}BPM')


if __name__ == '__main__':
    audio = 'sample_data/m2.wav'
    y,sr = librosa.load(audio, sr=None)

    print(beats(y,sr))