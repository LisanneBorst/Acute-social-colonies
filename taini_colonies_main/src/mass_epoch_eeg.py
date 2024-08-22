'''
Epoch EEG and save for many NWB files
''' 

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
# from nwb_data_retrieval_functions import *
from scipy import signal
import re
import os
import json

from nwb_data_retrieval_functions import *
from analysis_epoch_eeg import epoch_eeg

if __name__ == '__main__':
    # Load settings
    with open('settings.json', "r") as f:
        settings = json.load(f)

    # Parse from settings
    nwb_path = settings['nwb_files_folder']
    epoch_output = settings['epochs_folder']

    behaviors_and_lens = {
        'social_sniff': 0.5,
        'social_approach': 1,
        'social_contact': 2 # social contact somewhere between 1 and 5
    }

    relative_start = 0
    ploss_threshold = 5

    for i, file in enumerate(os.listdir(nwb_path)):
        print(f'Loading {os.path.join(nwb_path, file)}')
        filename = os.path.splitext(file)[0]

        for behavior, epoch_length in behaviors_and_lens.items():
            # Check if epoch file already exists and if so skip it
            outname = f'{epoch_output}/{filename}_{behavior}-epo.fif'
            if os.path.exists(outname):
                print(f"{outname} exists. Skipping...")
                continue

            print(f'Making {behavior} epochs for {filename}')
            print(f"output folder: {epoch_output}")
            
            good_epochs = epoch_eeg(os.path.join(nwb_path, file), behavior, epoch_length, relative_start, ploss_threshold)
            if good_epochs != None:
                good_epochs.save(f'{epoch_output}/{filename}_{behavior}-epo.fif', overwrite = True)
            else:
                print(f"Did not save {filename}")
print('Done') 



