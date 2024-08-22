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
import seaborn as sns
from ndx_events import LabeledEvents, AnnotatedEventsTable, TTLs
from scipy import signal
import re
import os
import sys

# taini_colonies stuff
from nwb_data_retrieval_functions import *


def get_behavior_eeg_onsets(nwb_file, behavior):
    '''
        Gets the EEG sample onsets of a specific social behavior
        which is going to be used later to epoch the EEG

        Args:
            - nwb_file: path of nwb file
            - behavior: str of the behavior name to analyze
        Returns:
            - 1D np.array of EEG sample onsets for a specific behavior in the input nwb file
    '''

    # Get data
    arena_num = get_arena_id(nwb_file)
    print(f'Arena number: {arena_num}')
    # arena_position = get_arena_position(nwb_file)
    ttl_onsets = get_ttl(nwb_file, arena_num)
    events = get_event_trace(nwb_file)
    sfreq = get_sfreq(nwb_file)

    def find_ttl_pulse(frame_number):
        # Check if the input frame number falls exactly on a TTL pulse
        if (frame_number % 30) == 0:
            pulse_number = frame_number // 30
            return pulse_number, frame_number
        else:
            # Calculate the last TTL pulse number
            last_pulse = (frame_number - 1) // 30 + 1
            last_frame = (last_pulse - 1) * 30 
            
            # Calculate the next TTL pulse number
            next_pulse = (frame_number + 29) // 30 + 1
            next_frame = (next_pulse - 1) * 30
            return last_pulse, last_frame, next_pulse, next_frame
    
    behavior_onsets =  events[events['event']==behavior]['start_frame'].to_numpy()
    behavior_ends =  events[events['event']==behavior]['end_frame'].to_numpy()
    b_on_corr = []
    b_end_corr = []
    sample_onsets = []
    sample_ends = []
    for behavior_onset_f, behavior_end_f in zip(behavior_onsets, behavior_ends):
        behavior_duration = behavior_end_f - behavior_onset_f
        
        # First check if behavior length is ok
        try: 
            if len(find_ttl_pulse(behavior_onset_f)) == 2:
                pulse_number, frame_number = find_ttl_pulse(behavior_onset_f)
                computed_onset = ttl_onsets[pulse_number - 1]
                sample_onsets.append(computed_onset)
                sample_ends.append(int(computed_onset + sfreq*behavior_duration/30))
                b_on_corr.append(behavior_onset_f)
                b_end_corr.append(behavior_end_f)

            else:
                last_pulse, last_frame, next_pulse, next_frame = find_ttl_pulse(behavior_onset_f)

                # Find these in EEG recording
                last_eeg_pulse = ttl_onsets[last_pulse - 1]
                next_eeg_pulse = ttl_onsets[next_pulse - 1]
                
                # Distance in seconds in recording
                delta_F1 = (behavior_onset_f - last_frame) / 30
                delta_F2 = (next_frame - behavior_onset_f) / 30
                
                # Solve for EEG sample onsets
                x1 = int(delta_F1 * sfreq + last_eeg_pulse)
                x2 = int(next_eeg_pulse - delta_F2 * sfreq)
                
                computed_onset = np.mean([x1, x2], dtype=int)
                sample_onsets.append(computed_onset)
                sample_ends.append(int(computed_onset + sfreq * behavior_duration / 30))
                b_on_corr.append(behavior_onset_f)
                b_end_corr.append(behavior_end_f)

        except IndexError as e:
            print(f"IndexError: {e}. Check if 'ttl_onsets' has sufficient elements.")
            continue

    return np.array(sample_onsets), np.array(sample_ends), np.array(b_on_corr), np.array(b_end_corr)



def epoch_eeg(nwb_file, behavior, epoch_length=1.0, relative_start = 0, ploss_threshold = 10):
    '''
        Args:
            - nwb_file: path, of the nwb_file
            - behavior: str, of behavior label (e.g. social_sniff)
            - epoch_len: float, length of the epoch in seconds
            - relative_start: seconds relative to the behavior onset which we use to get the eeg sample
                for example if relative start=-1 we get the eeg sample 1 second before the onset of the behavior
            - ploss_threshold: int or float, milliseconds of packageloss above which an epoch is excluded
        Returns:
            - mne.EpochsArray of behavioral EEG epochs (bad epochs are removed)

    '''
    print(f"Gonna epoch now for {nwb_file}")

    behavior_onsets, behavior_ends, frame_onsets, frame_ends = get_behavior_eeg_onsets(nwb_file, behavior)

    if behavior_onsets.size == 0:
        print(f'No Behaviors were scored for {nwb_file}')
        return None
    
    sfreq = get_sfreq(nwb_file, filtered=False)
    relative_start = int(relative_start*sfreq)
    samples_per_epoch = int(epoch_length * sfreq)

    # Initialize empty dict to store epochs
    data = {}
    bad_epochs = []

    # Extract epochs from behavior onsets
    for i, start_sample in enumerate(behavior_onsets):
        epoch_start = start_sample + relative_start
        epoch_end = epoch_start + samples_per_epoch

        # Load EEG data for the current epoch
        filt = get_filtered_eeg(nwb_file, segment=(epoch_start, epoch_end))
        ploss, _ = get_package_loss(nwb_file, segment=(epoch_start, epoch_end))

        for location, eeg in filt.items():
            if location not in data:
                data[location] = np.zeros((behavior_onsets.size, samples_per_epoch))
            data[location][i] = eeg

            # Check package loss threshold
            if np.sum(np.isnan(ploss[location])) > int(sfreq * ploss_threshold / 1000):
                bad_epochs.append(i)
    
    # Create channel info for MNE
    ch_names = list(data.keys())
    ch_types = []
    for chan in ch_names:
        if 'EMG' in chan:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # Function to find circardian phase from frame number
    def find_circ_phase(behavior_ends):
        '''
            Find whether a behavior was done on the light or dark phase.
            Assumes that recordings always start on the start of the dark phase.
            If a behavior lasts from one phase to the other, we score it as it happened during the second phase

            Args:
                behavior_ends: array, of all end frames of the scored behaviors
            Returns:
                array of str, of corresponding phases (light or dark)
        '''
        fps = 30
        seconds_in_hour = 3600
        hours_per_phase = 12

        # Convert frame numbers to hours
        behavior_end_hours = np.array(behavior_ends) / (fps * seconds_in_hour)

        # Determine the phase for each behavior based on the end hour
        end_phase = (behavior_end_hours // hours_per_phase) % 2  # 0 for dark, 1 for light

        return np.where(end_phase == 1, 'light', 'dark')
    
    animal_id =  get_animal_id(nwb_file)   
    arena = get_arena_id(nwb_file)
    day = get_day(nwb_file)
    circ_phase = find_circ_phase(frame_ends)
    
    # Create metadata table
    epoch_metadata = pd.DataFrame({
        'animal_id' : animal_id,
        'arena': arena,
        'day': day,
        'circ_phase': circ_phase,
        'behavior_label': behavior,
        'beh_start_frame': frame_onsets,
        'beh_end_frame': frame_ends,
        'beh_dur_frame': frame_ends - frame_onsets,
        'beh_start_sample': behavior_onsets,
        'beh_end_sample': behavior_ends,
        'beh_dur_sample': behavior_ends - behavior_onsets
    })

    print(f'For {nwb_file} the day is: {day}')

    if bad_epochs:
        # Remove duplicate bad epoch indexes
        bad_epochs = np.unique(bad_epochs)
        print(f'Bad epochs for {nwb_file} listed: {bad_epochs}')

        # Create a mask for good epochs
        good_epochs_mask = np.ones(len(behavior_onsets), dtype=bool)
        good_epochs_mask[bad_epochs] = False

        # Filter out bad epochs from the EEG data
        cleaned_epochs = {location: data[location][good_epochs_mask] for location in data.keys()}
                    
        # Also, filter out the corresponding rows from the metadata dataframe
        cleaned_metadata = epoch_metadata[good_epochs_mask].reset_index(drop=True)
        print(f'Metadata: {cleaned_metadata}')

        # Return cleaned_epochs
        return mne.EpochsArray(
            data=np.stack(list(cleaned_epochs.values()), axis=1), 
            info=info,
            metadata=cleaned_metadata
        )


    else:
        print(f'Metadata: {cleaned_metadata}')

        return mne.EpochsArray(
                data=np.stack(list(cleaned_epochs.values()), axis=1), 
                info=info,
                metadata=cleaned_metadata
            )


if __name__ == '__main__':
    pass    
    # # Specify these
    # animal = '78244'
    # nwb_path = '../drd2_batch1/nwb_files'
    # epoch_output = 'epochs/'
    # behavior = 'social_sniff'
    # epoch_length = 1.5
    # relative_start = -0.5
    # ploss_threshold = 10

    # # # Or with sys if you are a command line enthusiast
    # # animal = sys.argv[1]
    # # nwb_path = sys.argv[2]
    # # animal = sys.argv[3]
    # # nwb_path = sys.argv[4]
    # # epoch_output =sys.argv[5]
    # # behavior = sys.argv[6]
    # # epoch_length = sys.argv[7]
    # # relative_start = sys.argv[8]
    # # ploss_threshold = sys.argv[9]

    # epochs_list = []
    # for i, file in enumerate(os.listdir(nwb_path)):
    #     if animal in file:
    #         print(f'Loading {os.path.join(nwb_path, file)}')

    #         filename = re.split('\.', file)[1]
    #         _, _, _, good_epochs = epoch_eeg(os.path.join(nwb_path, file), behavior, epoch_length, relative_start, ploss_threshold)
    #         good_epochs.save(f'{filename}-epo.fif')
    #         epochs_list.append(good_epochs)
            
    # print('Concatenating epochs')
    # all_epochs = mne.concatenate_epochs(epochs_list, add_offset=False)
    # all_epochs.save(f'{animal}_colonies-epo.fif')
    # print('Done')