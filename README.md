# 🎧 Audio Analysis Streamlit App

This repository contains a two-part audio analysis project implemented in Python using Streamlit. It provides tools for both **time-domain** and **frequency-domain** signal processing, allowing for interactive exploration of audio signals.

---

## 📁 Project Structure

- **`algorithms.py`** – Contains functions for **time-based** audio analysis.
- **`algorithms2.py`** – Focuses on **frequency-based** analysis using Fourier Transform.
- **`app_code.py`** – The main Streamlit web app that integrates both modules into an interactive interface.

---
## 🔍 Features

###  Time-Based Analysis

- Audio Timecourse Visualization
- Frame-based parameters, such as:
  - Volume
  - Short-Time Energy (STE)
  - Zero Crossing Rate (ZCR)
  - Silence Ratio
  - Volume Dynamic Range
  - Volume Undulation
  - LSTER (Low Short-Time Energy Ratio)
  - Energy Entropy
- Silence Detection
- Speech vs. Music Classification based on LSTER
- Fundamental Frequency Estimation using:
  - Autocorrelation
  - AMDF (Average Magnitude Difference Function)
- Spectral Centroid and Bandwidth

### Frequency-Based Analysis

- Support for multiple window functions:
  - Rectangle
  - Triangle
  - Hann
  - Blackman
  - Hamming
- Fourier Transform (FFT)
- Spectrogram Visualization
- Frequency-domain metrics, such as:
  - Frequency Centroid
  - Bandwidth
  - Band Energy Ratio (BER)
  - Spectral Flatness
  - Spectral Crest

## How to Run

To launch the Streamlit web application, run the following command in your terminal:

```bash
streamlit run app_code.py
