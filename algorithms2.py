import math
import matplotlib.pyplot as plt
import streamlit as st
from scipy.fft import fft, fftfreq
import numpy as np
import librosa
import plotly.graph_objs as go
import io
import wave

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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frequency[:number_of_bins],
        y=magnitude_spectrum[:number_of_bins],
        mode='lines',
        line=dict(color='#0d0469'),
        hovertemplate='Częstotliwość: %{x:.1f} Hz<br>Amplituda: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Frequency [Hz]',
        yaxis_title='Spectrum magnitude',
        hovermode='x unified',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

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

def plot_centroids(fc_values, eb_values, frame_size, sr):
    frame_numbers = np.arange(len(fc_values))
    times = frame_numbers * frame_size / sr
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, np.array(fc_values), label="Frequency Centroid", color='#0d0469', linewidth=2)
    ax.fill_between(times, np.array(fc_values)-np.array(eb_values), np.array(fc_values)+np.array(eb_values), 
                    color='#a0c1db', alpha=0.3, label="Effective Bandwidth")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Frequency Centroid & Bandwidth for each frame")
    ax.legend()
    st.pyplot(fig)

def band_energies(y,sr,window, frame_size):
    frames = split_to_size_frames(y, frame_size)
    window = get_window(window, frame_size)
    window_sum = np.sum(window)
    bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
    band_energies = {i: [] for i in range(len(bands))}
    for frame in frames:
        frame = window*frame
        ft = np.fft.fft(frame)
        magnitude_spectrum = np.abs(ft)[:len(ft)//2] #only half because its symmetrical
        freqs = np.fft.fftfreq(len(frame), 1/sr)[:len(ft)//2]
        power = magnitude_spectrum**2

        for idx, (f0, f1) in enumerate(bands): #for each band
            band_mask = (freqs >= f0) & (freqs < f1)
            band_energy = np.sum(power[band_mask]) / window_sum
            band_energies[idx].append(band_energy)
    return band_energies

def plot_band_energies(band_energies, frame_size, sr):
    frame_numbers = np.arange(len(next(iter(band_energies.values()))))
    times = frame_numbers * frame_size/sr
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

    for idx in band_energies:
        ax.plot(times, band_energies[idx], label=labels[idx], color=colors[idx])

    ax.set_xlabel('Time')
    ax.set_ylabel('Band Energy')
    ax.set_title('Band Energy per frame')
    ax.legend()
    st.pyplot(fig)
        
def band_enery_ratios(band_energies, frame_size, sr):
    num_frames = len(next(iter(band_energies.values())))  # ile jest ramek
    num_bands = len(band_energies)
    volumes = []
    for i in range(num_frames):
        total_energy = sum(band_energies[idx][i] for idx in range(num_bands))
        volumes.append(total_energy)

    band_energy_ratios = {i: [] for i in range(num_bands)}
    for i in range(num_frames):
        vol = volumes[i]
        for idx in range(num_bands):
            ratio = band_energies[idx][i] / vol if vol != 0 else 0
            band_energy_ratios[idx].append(ratio)
    
    frame_numbers = np.arange(len(next(iter(band_energies.values()))))  # liczba ramek
    times = frame_numbers * frame_size / sr  # czas w sekundach dla każdej ramki

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

    for idx in band_energy_ratios:
        ax.plot(times, band_energy_ratios[idx], label=labels[idx], color=colors[idx])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Band Energy Ratio')
    ax.set_title('Band Energy Ratio per frame')
    ax.legend()
    st.pyplot(fig)

def spectral_flatness(y,sr,window, frame_size):
    frames = split_to_size_frames(y, frame_size)
    window = get_window(window, frame_size)
    bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
    band_flatness = {i: [] for i in range(len(bands))}
    for frame in frames:
        frame = window*frame
        ft = np.fft.fft(frame)
        magnitude_spectrum = np.abs(ft)[:len(ft)//2] #only half because its symmetrical
        freqs = np.fft.fftfreq(len(frame), 1/sr)[:len(ft)//2]
        power = magnitude_spectrum**2

        for i, (f0,f1) in enumerate(bands):
            idx = np.where((freqs>=f0) & (freqs<=f1))[0]
            band_power = power[idx]
            multiplied = np.prod(band_power)
            root = multiplied**(1/(f1-f0+1)) if multiplied >0 else 0 
            denominator = np.mean(band_power)

            band_flatness[i].append(root/denominator)

    frame_numbers = np.arange(len(next(iter(band_flatness.values()))))  # liczba ramek
    times = frame_numbers * frame_size / sr  # czas w sekundach dla każdej ramki
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

    for idx in band_flatness:
        ax.plot(times, band_flatness[idx], label=labels[idx], color=colors[idx])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spectral Flatness')
    ax.set_title('Spectral Flatness per frame')
    ax.legend()
    st.pyplot(fig)

def spectral_crest(y,sr,window, frame_size):
    frames = split_to_size_frames(y, frame_size)
    window = get_window(window, frame_size)
    bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
    band_crest = {i: [] for i in range(len(bands))}

    for frame in frames:
        frame = window*frame
        ft = np.fft.fft(frame)
        magnitude_spectrum = np.abs(ft)[:len(ft)//2] #only half because its symmetrical
        freqs = np.fft.fftfreq(len(frame), 1/sr)[:len(ft)//2]
        power = magnitude_spectrum**2

        for i, (f0,f1) in enumerate(bands):
            idx = np.where((freqs>=f0) & (freqs<=f1))[0]
            band_power = power[idx]
            nominator = np.max(band_power)
            denominator = np.mean(band_power)
            band_crest[i].append(nominator/denominator)

    frame_numbers = np.arange(len(next(iter(band_crest.values()))))  # liczba ramek
    times = frame_numbers * frame_size / sr  # czas w sekundach dla każdej ramki
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['0-630 Hz', '630-1720 Hz', '1720-4400 Hz', '4400-11025 Hz']

    for idx in band_crest:
        ax.plot(times, band_crest[idx], label=labels[idx], color=colors[idx])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spectral Crest Factor')
    ax.set_title('Spectral Crest Factor per frame')
    ax.legend()
    st.pyplot(fig)


# ------------------------------------------------ additional -------------------------------------------
def split_to_size_frames_with_overlap(audio, frame_size, overlap):
        step = int(frame_size * (1 - overlap)) 
        frames = []
        for start in range(0, len(audio) - frame_size +1, step):
            end = start + frame_size
            frame = audio[start:end]
            frames.append(frame)
        print('Frames number: ', len(frames))
        return frames


def compute_spectrogram(y, sr, frame_size, window, overlap):
    
    frames = split_to_size_frames_with_overlap(y, frame_size, overlap)
    window = get_window(window, frame_size)
    spectrogram=[]

    for i,frame in enumerate(frames):
        frame = window*frame
        ft = np.fft.fft(frame)
        magnitude_spectrum = np.abs(ft)[:len(ft)//2] #only half because its symmetrical
        spectrogram.append(magnitude_spectrum)
        
    spectrogram = np.array(spectrogram).T
    freqs = np.fft.fftfreq(len(frames[0]), 1/sr)[:len(ft)//2]
    return spectrogram, freqs

def plot_spectrogram(y, frame_size, sr,window, overlap):
    step = int(frame_size * (1 - overlap))  # Krok zależny od overlap
    num_frames = (len(y) - frame_size) // step + 1
    times = np.arange(num_frames) * step/sr

    spectrogram, freqs = compute_spectrogram(y,sr,frame_size,window, overlap)
    ref_value = np.max(spectrogram)
    spectrogram_db = 10 * np.log10(spectrogram / ref_value)

    fig, ax = plt.subplots(figsize=(12, 6))
    c = ax.pcolormesh(times, freqs,spectrogram_db, cmap='viridis')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    ax.set_title('Manual Spectrogram')
    fig.colorbar(c, ax=ax, label='Intensity [dB]')
    st.pyplot(fig)



if __name__ == "__main__":
    y,sr = librosa.load('sample_data/female.wav', sr=None)
    y = y/np.max(np.abs(y))
    frames = split_to_size_frames(y, 1024)
    frame = frames[0]
    rect = frame * rectengular_window(1024)
    tr = frame * traingle_window(1024)
    hann = frame* hann_window(1024)
    hamming = frame*hamming_window(1024)
    blackman = frame*blackman_window(1024)
    plt.plot(rect, label='Rectangle', color="#3698e5")
    plt.plot(tr, label='Triangle', color="#364be5")
    plt.plot(hann, label='Hann', color='#b836e5')
    plt.plot(hamming, label='Hamming', color='#e56e36')
    plt.plot(blackman, label='Blackman', color='#d54031')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    

        
        






