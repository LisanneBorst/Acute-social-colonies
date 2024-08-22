'''
Collection of various data retrieval functions to get data from NWB files

'''

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

def get_raw_eeg(nwb_file, segment, channel_names=True):
    '''
        Retrieves that raw EEG data from an nwb_file
        Returns dict:
            keys: electorde brain locations
            values: EEG array
    '''
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        raw_eeg = nwb.acquisition['raw_EEG'].data[segment[0]: segment[1]].T
        if channel_names==False:
            return raw_eeg
        channel_data = {}
        locations = nwb.electrodes.location.data[:]

        for signal, location in zip(raw_eeg, locations):
            channel_data[location] = signal
        return channel_data

def get_filtered_eeg(nwb_file, segment, channel_names=True):
    '''
        Retrieves that filtered EEG data from an nwb_file
        Returns dict:
            keys: electorde brain locations
            values: 1D-array with filtered EEG samples, 
    '''
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        filtered_eeg = nwb.acquisition['filtered_EEG'].data[segment[0]: segment[1]].T
        if channel_names==False:
            return filtered_eeg
        channel_data = {}
        locations = nwb.electrodes.location.data[:]

        for signal, location in zip(filtered_eeg, locations):
            channel_data[location] = signal
        return channel_data

def get_ttl(nwb_file, arena_num, as_samples=True):
    '''
        Retrieves that TTL pulse data from an nwb_file for a given arena number (from 1 to 4)
        if as_samples == True: Then the onsets are multipled with the sampling frequency
        else the raw timestamps are returned (in seconds)
        Returns:
            - array of TTL onsets (in seconds or in sample numbers)
    '''
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        onsets = nwb.acquisition[f'TTL_{arena_num}'].timestamps[:]
        if as_samples:
            return (onsets*nwb.acquisition['raw_EEG'].rate).astype(int)
        return onsets

def get_event_trace(nwb_file, version='last'):
    '''
        Retrieves the behavioral event trace data from an nwb files
        Returns:
            - pd.DataFrame of event trace
    '''

    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        # Check version arg
        if version != 'last' and version not in nwb.processing.keys():
            print("version no last")
            raise IndexError(f'Version {version} invalid. Pick between {nwb.processing.keys()} for this nwb file')
        if version == 'last':
            try:
                version = [i for i in nwb.processing.keys() if i != 'coordinate_data'][-1]
                print("version is "+ version)
            except IndexError:
                print(f"No event trace found in {nwb_file}")
                return pd.DataFrame({
                    'start_frame' : None,
                    'end_frame' : None,
                    'event' : None
                }, index = [0])

        df = pd.DataFrame()
        for behavior in nwb.processing[version]['all_colony_behaviors'].interval_series.keys():
            print(f'Behavior type: {behavior}')
            start_timestamps = nwb.processing[version]['all_colony_behaviors'].interval_series[behavior].timestamps[::2]
            end_timestamps = nwb.processing[version]['all_colony_behaviors'].interval_series[behavior].timestamps[1::2]
            print(f"start_timestamps: {len(start_timestamps)}")
            print(f"end_timestamps: {len(end_timestamps)}")
            tmp = pd.DataFrame({
                'start_frame' : np.array(start_timestamps).astype(int),
                'end_frame' : np.array(end_timestamps).astype(int),
                'event' : behavior
            })
            print(tmp)
            df = pd.concat([df, tmp])
        return df
    
def get_package_loss(nwb_file, segment):
    '''
        Retrieves the raw EEG from the NWB file, searches and returns package loss sample numbers
        To save time
        Returns (ploss_signal, ploss_samples)
            - ploss_signal : raw signal containing np.nan values (without interpolating)
            - ploss_samples : sample indexes where there is package loss (more usefull)
    '''
    import re

    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()

        # Parse filtering info
        finfo = re.search('low_val:(.+),.+high_val:(.+),.+art:(.+)', nwb.acquisition['filtered_EEG'].filtering)
        low_val, high_val, art = float(finfo[1]), float(finfo[2]), finfo[3]

        # Find package loss in raw_eeg
        raw_eeg = nwb.acquisition['raw_EEG'].data[segment[0]: segment[1]].T
        locations = nwb.electrodes.location.data[:]

        ploss_signal = {}
        ploss_samples = {}

        for signal, location in zip(raw_eeg, locations):
            rej = np.where(signal > low_val, signal , np.nan)
            rej = np.where(signal < high_val, rej , np.nan)
            if art != 'None':
                art = float(finfo[3])
                rej = np.where((rej > np.mean(rej) + art*np.std(rej)) | (rej < np.mean(rej) - art*np.std(rej)), np.nan, rej)
            ploss_signal[location] = rej
            ploss_samples[location] = np.where(np.isnan(rej))[0]
        return ploss_signal, ploss_samples

def get_sfreq(nwb_file, filtered=True):
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        if filtered:
            return nwb.acquisition['filtered_EEG'].rate
        return nwb.acquisition['raw_EEG'].rate

def get_metadata(nwb_file, picks='all'):
    with NWBHDF5IO(nwb_file, "r") as io:
        if picks == 'all':
            return io.read().fields
        else:
            return io.read().fields[picks]
        
def get_xy_coordinates(nwb_file, animal, body_point = 'center'):

    if body_point not in ['center', 'nose']:
        raise ValueError('body_point must be either center or nose')
    
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        data = nwb.processing['coordinate_data'][f'xy_{body_point}_{animal}'].data[:]
        timestamps = nwb.processing['coordinate_data'][f'xy_{body_point}_{animal}'].timestamps[:].astype(int)
        return timestamps, data
    
def get_motion_data(nwb_file, animal):
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        data = nwb.processing['coordinate_data'][f'motion_{animal}'].data[:]
        timestamps = nwb.processing['coordinate_data'][f'motion_{animal}'].timestamps[:].astype(int)
        return timestamps, data

def get_orientation_data(nwb_file, animal):
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        data = nwb.processing['coordinate_data'][f'orientation_{animal}'].data[:]
        timestamps = nwb.processing['coordinate_data'][f'orientation_{animal}'].timestamps[:].astype(int)
        return timestamps, data

def get_animal_id(nwb_file):
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        return nwb.subject.subject_id
        
        
# def get_genotype(nwb_file):
#     with NWBHDF5IO(nwb_file, "r") as io:
#         nwb = io.read()
#         return nwb.subject.genotype

def get_arena_id(nwb_file):
    from re import search
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        ses = nwb.experiment_description
        return search('Colony\/Arena_(\d+)', ses)[1]
    
def get_arena_position(nwb_file):
    from re import search
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        ses = nwb.experiment_description
        return search('Position_(\d+)', ses)[1]

def get_day(nwb_file):
    from re import search
    with NWBHDF5IO(nwb_file, "r") as io:
        nwb = io.read()
        return search("Day(\d+)", nwb.identifier)[1]
