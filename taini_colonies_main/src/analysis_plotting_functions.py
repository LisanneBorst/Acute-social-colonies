from pynwb import NWBFile
from datetime import datetime
from dateutil import tz
from pynwb.file import Subject
import numpy as np
from pynwb.behavior import SpatialSeries, Position, IntervalSeries, BehavioralEpochs
import pandas as pd
import mne
from pynwb.ecephys import ElectricalSeries, LFP
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
# from nwbwidgets import nwb2widget
import seaborn as sns
from ndx_events import LabeledEvents, AnnotatedEventsTable, TTLs
from .nwb_data_retrieval_functions import *
from scipy import signal
import re
import os


def eeg_plotter(eeg_dict, r):
    fig, ax = plt.subplots(nrows=1, figsize=(10,2))
    for location, signal in eeg_dict.items():
        ax.plot(signal[r[0]:r[1]])
        ax.set_ylabel(location)
        ax.spines[['right', 'top', 'bottom']].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_channel_psd(epochs, channel, fmin = 0, fmax = 100, method = 'multitaper', save_title=False, **kwargs):
    if method == 'multitaper':
        psds, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(picks=channel), fmin=fmin, fmax=fmax, sfreq=epochs.info['sfreq'], **kwargs)
    elif method == 'welch':
        psds, freqs = epochs.compute_psd(method='welch', picks=channel, fmin=fmin, fmax=fmax, **kwargs).get_data(picks=channel)
    else:
        raise NotImplementedError('Please chose either "welch", or "multitaper" for method')
    
    mean_psd = np.mean(psds[:, 0, :], axis=0)
    conf_int = 1.96 * np.std(psds[:, 0, :], axis=0) / np.sqrt(psds.shape[0])  # 95% confidence interval
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(freqs, 10 * np.log10(mean_psd), label='Average PSD', color='b')
    ax.fill_between(freqs, 10 * np.log10(mean_psd - conf_int), 10 * np.log10(mean_psd + conf_int), alpha=0.2, color='b', label='95% CI')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power/Frequency (dB/Hz)')
    ax.set_title(f'Average (PSD) - {channel}')
    if save_title:
        plt.savefig(f'averagePSD_{channel}_{save_title}.pdf')


def plot_two_channel_psd(epochs1, epochs2, label1, label2, channel, fmin = 0, fmax = 100, method = 'multitaper', save_title=False, pname='tab10', **kwargs):

    ncols = 2
    palette = list(reversed(sns.color_palette(pname, ncols).as_hex()))

    if isinstance(channel, list):
        fig, ax = plt.subplots(figsize=(8, 6))
    elif channel == 'all':
        pass
    else:
        if method == 'multitaper':
            psds1, freqs1 = mne.time_frequency.psd_array_multitaper(epochs1.get_data(picks=channel), fmin=fmin, fmax=fmax, sfreq=epochs1.info['sfreq'], **kwargs)
            psds2, freqs2 = mne.time_frequency.psd_array_multitaper(epochs2.get_data(picks=channel), fmin=fmin, fmax=fmax, sfreq=epochs2.info['sfreq'], **kwargs)
        elif method == 'welch':
            psds1, freqs1 = epochs1.compute_psd(method='welch', picks=channel, fmin=fmin, fmax=fmax, **kwargs).get_data(picks=channel)
            psds2, freqs2 = epochs2.compute_psd(method='welch', picks=channel, fmin=fmin, fmax=fmax, **kwargs).get_data(picks=channel)
        else:
            raise NotImplementedError('Please chose either "welch", or "multitaper" for method')
        
        mean_psd1 = np.mean(psds1[:, 0, :], axis=0)
        conf_int1 = 1.96 * np.std(psds1[:, 0, :], axis=0) / np.sqrt(psds1.shape[0])  # 95% confidence interval
        mean_psd2 = np.mean(psds2[:, 0, :], axis=0)
        conf_int2 = 1.96 * np.std(psds2[:, 0, :], axis=0) / np.sqrt(psds2.shape[0])  # 95% confidence interval 

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(freqs1, 10 * np.log10(mean_psd1), label=label1, color=palette[0])
        ax.plot(freqs2, 10 * np.log10(mean_psd2), label=label2, color=palette[1])
        ax.fill_between(freqs1, 10 * np.log10(mean_psd1 - conf_int1), 10 * np.log10(mean_psd1 + conf_int1), alpha=0.2, color=palette[0])
        ax.fill_between(freqs2, 10 * np.log10(mean_psd2 - conf_int2), 10 * np.log10(mean_psd2 + conf_int2), alpha=0.2, color=palette[1])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.set_title(f'Average (PSD) - {channel}')
        ax.legend()
        if save_title:
            plt.savefig(f'averagePSD_{channel}_{save_title}.pdf')

