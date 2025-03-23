from algorithms import *
import librosa
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
from scipy.signal import find_peaks

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
    extrema = np.sort(np.concatenate([peaks, valleys]))
    diffs = np.diff(volumes[extrema]) #różnice między sasiednimi ekstremami
    vu = np.sum(np.abs(diffs))  # Suma różnic bez uwzględniania znaku (rozpiętość amplitud)
    return vu


# Energy based

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
    #dla każdej ramki

    for i in range(len(ste)):
        frame_start = i * frame_size
        frame_end = (i + 1) * frame_size
        frame = y[frame_start:frame_end]
        total_energy = ste[i]

        #przejscie po probkach w danej ramce i podzial na segmenty
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

# ZCR Based
def hzcrr(y,sr,frame_length=1):
    zcrs = zero_crossing_rate(y,sr,frame_length, plot=False)
    avgzcr = np.average(zcrs)
    sum =0
    for zcr in zcrs:
        sum+= np.sign(zcr  -1.5*avgzcr)+ 1
    return sum/(2*len(zcrs))


# music speech detection
def speech_music(y,sr):
    l = lster(y,sr)
    print(l)
    if l>0.17 and l<0.5:
        return 'Speech'
    elif l<0.17:
        return 'Music'





if __name__ == '__main__':
    audio = 'sample_data/zdanie_1.wav'
    y,sr = librosa.load(audio, sr=None)

    print(speech_music(y,sr))