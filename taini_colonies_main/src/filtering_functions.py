'''
Filter EEG data using multithreading
'''
import numpy as np
import mne
import os
import re
from scipy import signal
import pickle
import threading


def interpolate_nan(padata, pkind='linear'):
    """
    Interpolates data to fill nan values

    Parameters:
        padata : nd array 
            source data with np.NaN values
        
    Returns:
        nd array 
            resulting data with interpolated values instead of nans
    """
    from scipy.interpolate import interp1d
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes
            , padata[agood_indexes]
            , bounds_error=False
            , copy=False
            , fill_value="extrapolate"
            , kind=pkind)
    return f(aindexes)

# Define general functions
def filtering(x, sfreq, lp=0.5, hp=200, lower_val = 0.006, higher_val=0.013, art=3):
    '''
        Returns filtered_eeg_array
    '''
    # artifact rejection
    rej = np.where(x > lower_val, x , np.nan)
    rej = interpolate_nan(rej, pkind='linear')
    rej = np.where(rej < higher_val, rej , np.nan)
    rej = interpolate_nan(rej, pkind='linear')
    
    # filter
    b, a = signal.butter(N=5, Wn=[lp/(sfreq/2), hp/(sfreq/2)], btype='bandpass')
    rej = signal.filtfilt(b, a, rej)

    if art:
        # artifact rejection
        rej = np.where((rej > np.mean(rej) + art*np.std(rej)) | (rej < np.mean(rej) - art*np.std(rej)), np.nan, rej)
        return interpolate_nan(rej, pkind='linear')
    return rej

def time_to_samples(time_str, sfreq):
    # split the time string into its components
    hour, minute, second = time_str.split('-')

    # convert the components to integers
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # calculate the total number of seconds
    total_seconds = hour * 3600 + minute * 60 + second

    return total_seconds*int(sfreq)

def main(edf):
    global electrode_info, export_path

    # Find export name
    info = re.split('_', edf)
    export_name = f'filtered_{info[2]}_{info[3]}_{info[6]}.pickle'

    data = mne.io.read_raw_edf(edf, preload=False)
    sfreq = data.info['sfreq']

    # Filter channels of interest
    filt = {}
    for channel, name in electrode_info.items():

        print(f'\tFiltering {channel}')

        raw = data[channel][0][0]
        filt[name] = filtering(raw, sfreq, art=5)
        del raw
    
    # Export
    with open(f'{export_path}/{export_name}', "wb") as f:
        pickle.dump(filt, f, pickle.HIGHEST_PROTOCOL)
    
    

# Main
if __name__ == "__main__":
    export_path = 'filtered'
    electrode_info = {
        'EEG 2' : 'OFC_R',
        'EEG 3' : 'OFC_L',
        'EEG 4' : 'CG',
        'EEG 13': 'STR_R',
        'EEG 6' : 'S1_L',
        'EEG 11': 'S1_R',
        'EEG 12': 'V1_R',
        'EEG 7' : 'EMG_L',
        'EEG 10': 'EMG_R',
    } # Rememebr that it's zero indexed

    for file in os.listdir('edfs'):
        if not os.path.exists(f'psds/{file}'):
            thread = threading.Thread(target=main, args=(f'edfs/{file}',))
            thread.start()
        else:
            print(f'Path psds/{file} exists')
    print('Done filtering')
