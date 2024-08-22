'''
TODO: 
- Change how EMG data are processed
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
from pynwb import NWBHDF5IO
import os
import json
import re
from ndx_events import LabeledEvents, AnnotatedEventsTable, TTLs
from taini_colonies_utils import str_sync_to_array
from filtering_functions import interpolate_nan, time_to_samples, filtering
# For compression
from hdmf.backends.hdf5.h5_utils import H5DataIO

# Load settings
with open('taini_colonies-main/settings.json', "r") as f:
    settings = json.load(f)

# Parse from settings
edf_folder = settings['edf_folder']
nwb_output_folder = settings['nwb_files_folder']
metadata_folder = settings['metadata']
experimenter = settings['experimenter']                         
lab = settings['lab']                                        
institution = settings['institution']
electrode_info = settings['electrode_info']
lcut = settings['lcut']
hcut = settings['hcut']
art = settings['art']
low_val = settings['low_val']
high_val = settings['high_val']

# Main 

# Load metadata file
metadata = pd.read_excel(metadata_folder, dtype={'mouseName':str, 'mouseId':str, 'cage':str})

for f in os.listdir(edf_folder):
    if not f.endswith(".edf"):
        continue
    try:
        # Get metadata info
        info = metadata[metadata['edf']==f].to_dict(orient='records')[0]
    except IndexError:
        print(f"No metadata record for {f}")
        continue

    # Prep NWB file metadata
    session_description = f"Animal {info['mouseId']} in the social colonies - Day{str(info['day'])[-1]}"
    print(f"session descriot: {session_description}")
    start_time = datetime.strptime('-'.join([info['date'], info['time']]), '%Y-%m-%d-%H-%M-%S').replace(tzinfo=tz.tzlocal())
    identifier = f'colonies_{info["mouseId"]}_Day{str(info["day"])[-1]}'
    session_id = f'{info["mouseId"]}_{info["sesId"]}'
    arena = f'Colony/Arena_{info["arena"]}_Position_{info["arena_position"]}'
    
    # Check if NWB exists and if so skip it
    outname = f'{nwb_output_folder}/{identifier}.nwb'
    if os.path.exists(outname):   # TODO tell VAS to update this line from 
        # if os.path.exists(nwb_output_folder):
        # to
        # if os.path.exists(outname):
        print(f"output folder: {nwb_output_folder}")
        print(f"{outname} exists. Skipping...")
        continue
    
    print('Creating NWB file...')
    # Create NWB file
    nwb = NWBFile(session_description=session_description,
              identifier=identifier,
              session_start_time=start_time,
              session_id=session_id,
              experiment_description = arena,                                
              experimenter=experimenter,                                   
              lab=lab,                                        
              institution=institution)
    
    # Add subject information
    nwb.subject = Subject(subject_id=info['mouseId'], # name that we give it
                    #   description=info['mouseName'], # unique animal id in the mouse card
                    # TODO: tell VAS, that they were swapped around here maybe??? im confused by
                            #the mouseName, mouseId, subjectId, etc.
                      species=info['species'], 
                      sex=info['sex'],
                      )
    
    print('Adding electrode information...')

    # Add device and electrode information
    device = nwb.create_device(
    name=str(info['transmitterId']),
    description=str(info['transmitterId']),
    manufacturer='TaiNi')

    nwb.add_electrode_column(name='label', description='label of electrode')

    for channel, details in electrode_info.items():
        location = details[0]
        AP = float(details[1])
        ML = float(details[2])
        DV = float(details[3])
        el_type = details[4]  

        # create an electrode group for this channel
        electrode_group = nwb.create_electrode_group(
        name=channel,
        description=f'{channel}_{el_type}_{location}',
        device=device,
        location=location
        )
        nwb.add_electrode(
                x=AP, y=ML, z=DV, imp=np.nan,
                location=location, 
                filtering='unknown',
                group=electrode_group,
                label=f'{el_type}_{location}'
            )
    print('Adding EEG data...')
    
    # Add raw EEG data
    raw = mne.io.read_raw_edf(f'{edf_folder}/{f}')
    sfreq = raw.info['sfreq']
    data = raw.get_data(picks=list(electrode_info.keys()))

    all_table_region = nwb.create_electrode_table_region(
    region=list(range(len(electrode_info.keys()))),  # reference row indices 0 to N-1
    description='all electrodes')

    raw_elec_series = ElectricalSeries(
        name='raw_EEG', 
        data=H5DataIO(data=data.T, compression=True), # to transpose the data because the (channels, data) format doesn't work lul
        electrodes=all_table_region, 
        starting_time=0.,  # relative to NWBFile.session_start_time
        rate=sfreq  # Sampling Frequency
        )
    nwb.add_acquisition(raw_elec_series)
    
    print('Adding EEG raw annotations...')

    # Filter EEG
    print('Filtering EEG')
    
    filt = []
    for channel in electrode_info.keys():
        print(f'Filtering channel {channel}')
        filt.append(filtering(raw[channel][0][0], sfreq, lcut, hcut, low_val, high_val, art))
    filt = np.array(filt)
    
    # Create new ElectricalSeries object to hold the filtered EEG, and add to nwb
    filt_elec_series = ElectricalSeries(
        name = 'filtered_EEG',
        data = H5DataIO(data=filt.T, compression=True),
        electrodes=all_table_region,
        starting_time = 0.,
        rate=sfreq,
        filtering = f'5th Order Bandpass butterwort Filter. Low:{lcut} Hz, High: {hcut}, low_val:{low_val}, high_val:{high_val}, art:{art}'
    )
    nwb.add_acquisition(filt_elec_series)

    # Add raw TTL annotations
    ttl_timestamps = raw.annotations.onset # Timestamps
    ttl_data = raw.annotations.description

    mapping = {}
    for i in range(16):
        mapping[f'SYNC_{i}'] = i

    ttl_data = np.array([mapping[element] for element in ttl_data])

    ttl_raw_events = TTLs( 
        name = 'raw_TTL',
        description = 'Raw TTL events from EEG annotations',
        timestamps = ttl_timestamps,
        data = ttl_data,
        labels = list(mapping.keys()))
    
    nwb.add_acquisition(ttl_raw_events)

    print('Parsing annotations and adding TTL onsets to NWB file...')

    # Parse TTL and add them to the NWB
    # Transform "SYNC" to bits
    output_array = np.array([str_sync_to_array(s) for s in raw.annotations.description])

    # Determine onsets for each TTL pulse
    onsets = {
        'TTL_1' : [],
        'TTL_2' : [],
        'TTL_3' : [],
        'TTL_4' : []
    }

    for i in range(output_array.shape[0]-1):
        x = output_array[i] - output_array[i+1]
        if x[0] == -1:
            onsets['TTL_1'].append(raw.annotations.onset[i+1])
        if x[1] == -1:
            onsets['TTL_2'].append(raw.annotations.onset[i+1])
        if x[2] == -1:
            onsets['TTL_3'].append(raw.annotations.onset[i+1])
        if x[3] == -1:
            onsets['TTL_4'].append(raw.annotations.onset[i+1])

    ttl_1_events = TTLs( 
        name = 'TTL_1',
        description = 'Processed TTL - Input 1',
        timestamps = onsets['TTL_1'],
        data = np.ones(len(onsets['TTL_1'])), # Useless??
        labels = ['TTL_1']
        )

    ttl_2_events = TTLs( 
        name = 'TTL_2',
        description = 'Processed TTL - Input 2',
        timestamps = onsets['TTL_2'],
        data = np.ones(len(onsets['TTL_2'])), # Useless??
        labels = ['TTL_2']
        )

    ttl_3_events = TTLs( 
        name = 'TTL_3',
        description = 'Processed TTL - Input 3',
        timestamps = onsets['TTL_3'],
        data = np.ones(len(onsets['TTL_3'])), # Useless??
        labels = ['TTL_3']
        )

    ttl_4_events = TTLs( 
        name = 'TTL_4',
        description = 'Processed TTL - Input 4',
        timestamps = onsets['TTL_4'],
        data = np.ones(len(onsets['TTL_4'])), # Useless??
        labels = ['TTL_4']
        )
    nwb.add_acquisition(ttl_1_events)
    nwb.add_acquisition(ttl_2_events)
    nwb.add_acquisition(ttl_3_events)
    nwb.add_acquisition(ttl_4_events)

    print('Saving file...')

    # Save the file
    with NWBHDF5IO(outname, 'w') as io:
        io.write(nwb)
    