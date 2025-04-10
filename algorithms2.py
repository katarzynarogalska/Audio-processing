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

def plot_fourier(frame, sr, title, freq_ratio=1):
    ft = np.fft.fft(frame)
    magnitude_spectrum = np.abs(ft)
    #frequency = np.linspace(0, sr, len(magnitude_spectrum)) #frequency between 0 and sample rate + create bins 
    frequency = np.fft.fftfreq(len(frame), d=1/sr)
    number_of_bins = int(len(frequency) * freq_ratio)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequency[:number_of_bins], magnitude_spectrum[:number_of_bins], color='#0d0469')
    ax.set_title(title)
    ax.set_xlabel('Frequency [Hz]')
    st.pyplot(fig)

def clip_functions(y,sr, window, frame_size = 1024):
    frames = split_to_size_frames(y, frame_size)
    fc_values =[]
    eb_values =[]
    for frame in frames:
        frame = frame * get_window(window, frame_size)
        ft = np.fft.fft(frame)
        magnitude_spectrum = np.abs(ft)[:len(ft)//2] #only half because its symmetrical
        freqs = np.fft.fftfreq(len(frame), 1/sr)[:len(ft)//2]

        fc = frame_frequency_centroid(magnitude_spectrum, freqs)
        fc_values.append(fc)
        eb_values.append(frame_effective_bandwidth(magnitude_spectrum, freqs, fc))
    return fc_values, eb_values



def frame_frequency_centroid(magnitude_spectrum, frequencies):
    if np.sum(magnitude_spectrum) == 0:
            fc = 0 
    fc = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
    return fc 
def frame_effective_bandwidth(magnitude_spectrum, frequencies, frequency_centroid):
    eb = np.sqrt(np.sum((frequencies - frequency_centroid)**2* magnitude_spectrum**2)/np.sum(magnitude_spectrum**2))
    return eb

def plot_parameters(values, title, x, y):
    frame_numbers = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame_numbers, values, color='#0d0469')
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    st.pyplot(fig)

def plot_centroids(fc_values, eb_values):
    frame_numbers = np.arange(len(fc_values))
    print(fc_values[:10])
    print(eb_values[:10])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame_numbers, np.array(fc_values), label="Frequency Centroid", color='#0d0469', linewidth=2)
    ax.fill_between(frame_numbers, np.array(fc_values)-np.array(eb_values), np.array(fc_values)+np.array(eb_values), 
                    color='#a0c1db', alpha=0.3, label="Effective Bandwidth")

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Frequency Centroid & Bandwidth for each frame")
    ax.legend()
    st.pyplot(fig)



