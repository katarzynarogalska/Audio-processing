from algorithms import *
import librosa
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def volume_dynamic_range(y,sr,frame_size):
    volume = loudness(y,sr,frame_size, plot=False)
    min_vol = np.min(volume)
    max_vol = np.max(volume)
    vdr = (max_vol-min_vol)/max_vol
    return vdr

def low_short_time_energy_ratio(y,sr,frame_size):
    pass