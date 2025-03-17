from algorithms import *
import librosa
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks

def volume_dynamic_range(y,sr,frame_size):
    volume = loudness(y,sr,frame_size, plot=False)
    min_vol = np.min(volume)
    max_vol = np.max(volume)
    vdr = (max_vol-min_vol)/max_vol
    return vdr

def volume_undulation(y,sr,frame_size):
    volumes = np.array(loudness(y,sr,frame_size, plot=False))
    peaks, _ = find_peaks(volumes)
    valleys, _ = find_peaks(-volumes)
    extrema = np.sort(np.concatenate([peaks, valleys]))
    diffs = np.diff(volumes[extrema]) #różnice między sasiednimi ekstremami
    vu = np.sum(np.abs(diffs))  # Suma różnic bez uwzględniania znaku (rozpiętość amplitud)
    return vu

def energy_entropy(y,K):
    pass

def low_short_time_energy_ratio(y,sr,frame_size):
    pass

if __name__ == '__main__':
    audio = 'sample_data/zdanie_1.wav'
    y,sr = librosa.load(audio, sr=None)

    print(volume_undulation(y,sr,1024))