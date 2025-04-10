import math
import matplotlib.pyplot as plt
import streamlit as st
from scipy.fft import fft, fftfreq
import numpy as np

def split_to_size_frames(audio, frame_size):
    num_frames = len(audio)//frame_size
    frames=[]
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio[start:end]
        frames.append(frame)
    return frames


def rectengular_window(frame_lenght):
    return [1 for i in range(frame_lenght)]

def traingle_window(frame_length):
    return [(1 - 2*(abs(n-(frame_length-1)/2)) /(frame_length-1)) for n in range(frame_length)]

def hann_window(frame_length):
    return [(0.5*(1-math.cos(2*math.pi*n/(frame_length-1)))) for n in range(frame_length)]

def hamming_window(frame_length):
    return [(0.54-0.46*math.cos(2*math.pi*n/(frame_length-1))) for n in range(frame_length)]

def blackman_window(frame_length):
    return [(0.42 - 0.5*math.cos(2*math.pi*n/(frame_length-1)) + 0.08*math.cos(4*math.pi*n/(frame_length-1))) for n in range(frame_length)]

def get_window(chosen_window, frame_len):
    if chosen_window =='Rectangle':
        return rectengular_window(frame_len)
    elif chosen_window=='Triangle':
        return traingle_window(frame_len)
    elif chosen_window=='Hann':
        return hann_window(frame_len)
    elif chosen_window=='Hamming':
        return hamming_window(frame_len)
    elif chosen_window=='Blackman':
        return blackman_window(frame_len)
    
def plot_after_window(frame, times, windowed_frame, chosen_window):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4)
    axs[0].plot(times, frame, color='green')
    axs[0].set_title('Frame timecourse before windowing')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    # Po zastosowaniu okna
    axs[1].plot(times, windowed_frame)
    axs[1].set_title(f'Frame timecourse after {chosen_window} window')
    axs[1].set_xlabel('Time[s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()

    st.pyplot(fig)

def plot_fourier(frame, sr):
    n = len(frame)
    y_fft = fft(frame)
    x_fft = fftfreq(n, 1 / sr)[:n//2]

    amplitude = np.abs(y_fft[0:n//2]) / n
    amplitude_db = 20 * np.log10(amplitude + 1e-12) 
    plt.figure(figsize=(10, 5))
    plt.plot(x_fft, amplitude_db)
    plt.title('Fourier plot')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda [dB]')
    plt.grid(True)
    st.pyplot(plt)
